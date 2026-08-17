# The `nvfp4_ds_mla` KV record, in full

The compact NVFP4 latent record is what lets GLM-5.2 fit 300K+ context per
DGX Spark rank: 368 bytes per token per layer against 656 for `fp8_ds_mla`.
The B12X reader for it is public (`b12x/attention/mla/traits.py`, `io.py`,
`decode_math.py`). Until now the writer was not: the other public GB10 stacks
load it from a `.so` extracted from a private image. This is the layout,
derived from the reader's dequant and proven byte-identical against it, so
anyone can write the record from Python.

## Layout (368 bytes, `KV_FP8_ROPE=1`)

| bytes | content | count | encoding |
|---|---|---|---|
| `[0, 256)` | NoPE latent, 512 dims | 512 x 4-bit | E2M1, two per byte. `cvt.rn.f16x2.e2m1x2` order: **low nibble = even dim, high nibble = odd dim** |
| `[256, 288)` | NoPE group scales | 32 x 1 byte | E4M3. Group `g` covers dims `[16g, 16g+16)` |
| `[288, 292)` | RoPE scale | 1 x 4 bytes | fp32, little-endian |
| `[292, 304)` | pad | 12 bytes | zero |
| `[304, 368)` | RoPE, 64 dims | 64 x 1 byte | E4M3 |

Dequant, exactly as the reader does it:

```
nope[d] = e2m1(nibble_d) * e4m3(scale[d // 16]) * latent_scale
rope[i] = e4m3(byte_i)   * rope_scale
```

`latent_scale` is the per-layer outer scale the attention backend passes in
(`layer._k_scale_float`; 1.0 for the compressed-tensors checkpoint). The writer
divides by the same value so writer and reader compose to identity.

The 432-byte variant (`KV_FP8_ROPE` unset) keeps `[0, 288)` unchanged, then
16 bytes of zero pad, then 64 bf16 RoPE values in `[304, 432)`.

## Quantization (writer side)

NoPE, per token, per group of 16:

1. `x = kv_c / latent_scale` in fp32.
2. `amax_g = max |x| over the group`; `scale_g = amax_g / 6` (6 is the E2M1 max).
3. Round `scale_g` to E4M3 **first**, then quantize the group with the *rounded*
   scale: `q = clamp(x / e4m3(scale_g), -6, 6)`. Quantizing with the unrounded
   scale drifts values across E2M1 boundaries.
4. Round `q` to the E2M1 grid `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (round to nearest,
   ties resolved by the midpoint compare, sign carried in bit 3).
5. Groups with `amax = 0` get scale `1e-8`, which rounds to E4M3 zero; the codes
   are all zero and the reader returns exact zeros.

RoPE, per token: `rope_scale = max |k_pe| / 448`, stored as fp32; values are
`e4m3(clamp(k_pe / rope_scale, -448, 448))`.

## E2M1 code table

| code | value | code | value |
|---|---|---|---|
| 0 | 0.0 | 8 | -0.0 |
| 1 | 0.5 | 9 | -0.5 |
| 2 | 1.0 | 10 | -1.0 |
| 3 | 1.5 | 11 | -1.5 |
| 4 | 2.0 | 12 | -2.0 |
| 5 | 3.0 | 13 | -3.0 |
| 6 | 4.0 | 14 | -4.0 |
| 7 | 6.0 | 15 | -6.0 |

## Accuracy

E2M1 with a per-16 E4M3 scale gives 8 to 10% relative RMS error on the NoPE
latent and cosine similarity above 0.994 per token against the bf16 input; the
RoPE half is under 3% RMS. Attention feels the cosine, not the RMS.

## The three ulp traps in a fused kernel

Making the Triton kernel byte-identical to the torch reference exposed three
places where "the same arithmetic" is not the same arithmetic:

1. **Triton's fp32 `/` is `div.full.f32`** (about 2 ulp), not IEEE. It flips
   values sitting on E2M1 rounding boundaries. Use `libdevice.div_rn` for
   tensor / tensor divisions.
2. **Multiply by reciprocal is not division.** `x * (1/s)` and `x / s` differ in
   the last ulp often enough to change codes.
3. **PyTorch treats a Python scalar divisor as multiply-by-reciprocal.**
   `tensor / 6.0` is computed as `tensor * fp32(1/6)`; `tensor / tensor` is a
   true division. A reference that uses both must be mirrored per site.

The self-test that catches all three: pack the same inputs through the torch
reference and the fused kernel and require `torch.equal` on the uint8 records,
across n in {1, 7, 64, 1024}, latent_scale in {1.0, 0.7}, exact-zero groups,
an outlier group, and slot -1 pad rows.

## Capture safety

The writer runs inside CUDA-graph capture (the KV update is part of the captured
decode step). No `.item()`, no `bool(tensor)`, no data-dependent shapes, no
host-to-device copies from pageable memory. Per-device constants are created
once outside capture and cached; `latent_scale` is read from a device tensor
inside the kernel; pad rows (`slot = -1`) are redirected to slot 0 of a null
block rather than filtered.

Reference implementation: `kernels/nvfp4_ds_mla_writer.py`
(`pack_records` = torch reference, `nvfp4_ds_mla_write_kernel` = fused,
`torch.ops.vllm.nvfp4_ds_mla_write` = registered op).
