"""nvfp4_ds_mla KV record writer for the b12x sparse-MLA reader.

The public b12x tree ships a complete READER for the compact GLM NVFP4 latent
record (traits.py, io.py, decode_math.py) but no WRITER, and stock vLLM has no
concat_and_cache path for it. This module is that writer, derived directly
from the reader's dequant so the two agree byte for byte.

Reader spec (b12x/attention/mla/decode_math.py, io.py; 368-byte KV_FP8_ROPE=1
record):

    [0,   256)  512 NoPE dims as E2M1, two per byte.
                cvt.rn.f16x2.e2m1x2 => LOW nibble = even dim, HIGH nibble = odd dim
    [256, 288)  32 E4M3 group scales, group g covers dims [16g, 16g+16)
    [288, 292)  fp32 rope_scale
    [292, 304)  zero pad
    [304, 368)  64 E4M3 RoPE values

    nope[d] = e2m1(nibble) * e4m3(scale[d // 16]) * latent_scale
    rope[i] = e4m3(byte)   * rope_scale

``latent_scale`` is the per-layer outer scale the backend passes to the kernel
(``layer._k_scale_float``, 1.0 for this compressed-tensors checkpoint). The
writer divides by the same value so reader and writer compose to identity.

The 432-byte layout (bf16 RoPE, KV_FP8_ROPE unset) is also supported.

Everything here is plain torch so it is correct first; a fused Triton version
can replace ``pack_records`` later without changing the layout contract.
"""
from __future__ import annotations

import torch

NOPE_DIM = 512
ROPE_DIM = 64
GROUP = 16
NUM_GROUPS = NOPE_DIM // GROUP          # 32
E2M1_MAX = 6.0
E4M3_MAX = 448.0
RECORD_FP8_ROPE = 368
RECORD_BF16_ROPE = 432

# E2M1 magnitude grid and the code for each grid point (positive half).
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Midpoints between adjacent grid points; used for round-to-nearest.
_E2M1_MID = (_E2M1_GRID[1:] + _E2M1_GRID[:-1]) / 2   # 0.25 .. 5.0

# CUDA-graph safety: this writer runs INSIDE graph capture (do_kv_cache_update
# is part of the captured decode step). Anything that syncs host<->device --
# .item(), bool(tensor), data-dependent shapes, H2D copies from pageable memory
# -- invalidates the capture (cudaErrorStreamCaptureInvalidated). So constants
# are moved to each device exactly once, outside capture, and cached.
_DEV_CONST: dict = {}


def _consts(device: torch.device, dtype: torch.dtype):
    key = (str(device), dtype)
    c = _DEV_CONST.get(key)
    if c is None:
        c = (_E2M1_MID.to(device=device, dtype=dtype), _E2M1_GRID.to(device=device))
        _DEV_CONST[key] = c
    return c


def _e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """float -> uint8 code in [0,15] on the E2M1 grid, round to nearest."""
    mag = x.abs()
    mid, _ = _consts(x.device, x.dtype)
    # bucketize against midpoints -> index into the 8-point grid
    idx = torch.bucketize(mag, mid, right=False)
    idx = idx.clamp_(0, 7).to(torch.uint8)
    sign = (x < 0).to(torch.uint8) << 3
    code = idx | sign
    # -0 is not a distinct value we want to emit; keep 0 as +0.
    code = torch.where(idx == 0, torch.zeros_like(code), code)
    return code


def _e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    """uint8 code -> float on the E2M1 grid (mirror of the PTX cvt)."""
    idx = (code & 0x7).long()
    _, grid = _consts(code.device, torch.float32)
    mag = grid[idx]
    neg = (code & 0x8) != 0
    return torch.where(neg, -mag, mag)


def _to_e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def _from_e4m3_bytes(b: torch.Tensor) -> torch.Tensor:
    return b.view(torch.float8_e4m3fn).to(torch.float32)


def pack_records(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    *,
    latent_scale: float = 1.0,
    fp8_rope: bool = True,
) -> torch.Tensor:
    """Quantize (N,512) latents + (N,64) rope into (N,368|432) uint8 records."""
    assert kv_c.shape[-1] == NOPE_DIM and k_pe.shape[-1] == ROPE_DIM
    n = kv_c.shape[0]
    dev = kv_c.device
    # latent_scale may be a python float or a 0-d/1-elem device tensor
    # (layer._k_scale). Dividing by the tensor is a device op -- no .item(),
    # no sync -- which keeps this capture-safe.
    if isinstance(latent_scale, torch.Tensor):
        x = kv_c.to(torch.float32) / latent_scale.to(torch.float32).reshape(-1)[0]
    else:
        x = kv_c.to(torch.float32) / float(latent_scale)

    # --- NoPE: per-16 group scale in E4M3, data on the E2M1 grid ---
    g = x.view(n, NUM_GROUPS, GROUP)
    amax = g.abs().amax(dim=-1)                                # (n,32)
    gs = amax / E2M1_MAX
    gs = torch.where(gs > 0, gs, torch.full_like(gs, 1e-8))
    gs_e4m3 = _to_e4m3_bytes(gs)                               # (n,32) uint8
    gs_used = _from_e4m3_bytes(gs_e4m3)                        # quantize with the ROUNDED scale
    gs_used = torch.where(gs_used > 0, gs_used, torch.full_like(gs_used, 1e-8))
    q = (g / gs_used.unsqueeze(-1)).clamp_(-E2M1_MAX, E2M1_MAX)
    codes = _e2m1_encode(q).view(n, NOPE_DIM)                  # (n,512) uint8 0..15
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)            # low nibble = even dim
    nope_bytes = torch.cat([packed, gs_e4m3], dim=1)           # (n,288)

    # --- RoPE ---
    r = k_pe.to(torch.float32)
    if fp8_rope:
        ramax = r.abs().amax(dim=-1)                           # (n,)
        rs = ramax / E4M3_MAX
        rs = torch.where(rs > 0, rs, torch.full_like(rs, 1e-8))
        rope_e4m3 = _to_e4m3_bytes((r / rs.unsqueeze(-1)).clamp_(-E4M3_MAX, E4M3_MAX))
        rs_bytes = rs.to(torch.float32).view(torch.uint8).view(n, 4)
        pad = torch.zeros(n, 12, dtype=torch.uint8, device=dev)
        rec = torch.cat([nope_bytes, rs_bytes, pad, rope_e4m3], dim=1)
        assert rec.shape[1] == RECORD_FP8_ROPE
    else:
        pad = torch.zeros(n, 16, dtype=torch.uint8, device=dev)
        rope_bf16 = r.to(torch.bfloat16).view(torch.uint8).view(n, ROPE_DIM * 2)
        rec = torch.cat([nope_bytes, pad, rope_bf16], dim=1)
        assert rec.shape[1] == RECORD_BF16_ROPE
    return rec.contiguous()


def unpack_records(
    rec: torch.Tensor, *, latent_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference dequant mirroring the b12x reader. Returns (nope f32, rope f32)."""
    n, w = rec.shape
    fp8_rope = w == RECORD_FP8_ROPE
    packed = rec[:, :256]
    codes = torch.empty(n, NOPE_DIM, dtype=torch.uint8, device=rec.device)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    vals = _e2m1_decode(codes).view(n, NUM_GROUPS, GROUP)
    gs = _from_e4m3_bytes(rec[:, 256:288]).unsqueeze(-1)
    nope = (vals * gs).view(n, NOPE_DIM) * float(latent_scale)
    if fp8_rope:
        rs = rec[:, 288:292].contiguous().view(torch.float32).view(n)
        rope = _from_e4m3_bytes(rec[:, 304:368]) * rs.unsqueeze(-1)
    else:
        rope = rec[:, 304:432].contiguous().view(torch.bfloat16).view(n, ROPE_DIM).to(torch.float32)
    return nope, rope


def _write_to_kv_cache_torch(
    kv_cache: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    latent_scale: float = 1.0,
) -> None:
    """Scatter freshly packed records into a paged uint8 KV cache.

    kv_cache: [num_blocks, block_size, record_bytes] (or any layout whose
    trailing dim is the record) -- flattened to [num_slots, record_bytes].
    slot_mapping: (N,) int64 physical slot ids; -1 entries are skipped.
    """
    rec_bytes = kv_cache.shape[-1]
    flat = kv_cache.view(-1, rec_bytes)
    if slot_mapping.numel() == 0:
        return
    rec = pack_records(kv_c, k_pe, latent_scale=latent_scale,
                       fp8_rope=(rec_bytes == RECORD_FP8_ROPE))
    # Padded rows carry slot -1 (PAD_SLOT_ID). vLLM V1 reserves block 0 as the
    # null block -- popped first, is_null=True, never handed to a request -- so
    # pointing pad rows at slot 0 keeps every index valid with no host sync and
    # any collision lands where nothing reads. This is what makes the write
    # legal inside CUDA graph capture.
    slots = slot_mapping.to(torch.long)
    slots = torch.where(slots >= 0, slots, torch.zeros_like(slots))
    flat.index_copy_(0, slots, rec)



# ---------------------------------------------------------------------------
# Fused Triton writer (default for the 368-byte record). One launch per call:
# py-spy showed the torch writer -- ~12 launches per layer per prefill chunk --
# as 54% of Python-visible prefill time on the fleet. Proven byte-identical to
# the torch reference above on all sizes and scales, CUDA-graph capturable, and
# ~50x faster per write. The torch path remains the reference and the fallback
# for the 432-byte layout.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False

if _HAVE_TRITON:

    _RECORD_C = tl.constexpr(368)


    @triton.jit
    def _e2m1_code(q):
        # q: fp32, already clamped to [-6, 6]. Return uint8 code 0..15.
        a = tl.abs(q)
        idx = (
            (a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
            + (a > 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32)
            + (a > 5.0).to(tl.int32)
        )
        sign = (q < 0).to(tl.int32) * 8
        code = tl.where(idx == 0, 0, idx + sign)
        return code.to(tl.uint8)


    @triton.jit
    def _e4m3_bytes(x):
        # fp32 -> E4M3 (round-nearest) -> raw byte
        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)


    @triton.jit
    def _e4m3_to_f32(b):
        return b.to(tl.float8e4nv, bitcast=True).to(tl.float32)


    @triton.jit
    def nvfp4_ds_mla_write_kernel(
        kv_c_ptr, kv_c_stride,          # (N, 512) latent
        k_pe_ptr, k_pe_stride,          # (N, 64) rope
        slot_ptr,                       # (N,) int64 slots, -1 = pad -> slot 0 (null block)
        cache_ptr,                      # flat (num_slots, 368) uint8
        scale_ptr,                      # 0-d fp32 tensor: latent_scale (loaded, no host sync)
        inv6_ptr, inv448_ptr,           # 0-d fp32: fp32(1/6), fp32(1/448) -- see host side
    ):
        row = tl.program_id(0)
        latent_scale = tl.load(scale_ptr)
        inv6 = tl.load(inv6_ptr)
        inv448 = tl.load(inv448_ptr)
        slot = tl.load(slot_ptr + row)
        slot = tl.where(slot < 0, 0, slot)
        rec = cache_ptr + slot.to(tl.int64) * _RECORD_C

        # ---------------- NoPE: [32 groups, 16 dims] as even/odd halves ----------
        g = tl.arange(0, 32)[:, None]           # group
        i = tl.arange(0, 8)[None, :]            # pair index within group
        src = kv_c_ptr + row * kv_c_stride
        # IEEE round-to-nearest division everywhere (libdevice.div_rn): Triton's
        # plain fp32 '/' is div.full.f32 (~2 ulp) and flips E2M1 boundary cases
        # against the torch reference. Divide by latent_scale, do not multiply by
        # its reciprocal, for the same reason.
        xe = libdevice.div_rn(tl.load(src + g * 16 + 2 * i).to(tl.float32), latent_scale)      # even dims
        xo = libdevice.div_rn(tl.load(src + g * 16 + 2 * i + 1).to(tl.float32), latent_scale)  # odd dims
        amax = tl.maximum(tl.max(tl.abs(xe), axis=1), tl.max(tl.abs(xo), axis=1))  # [32]
        # PyTorch quirk matched exactly: `tensor / python_scalar` is computed as
        # tensor * fp32(1/scalar) (reciprocal on CPU once, multiply on GPU), NOT an
        # IEEE division. tensor / tensor IS a true division. The reference uses the
        # scalar form for amax/6 and ramax/448, and the tensor form elsewhere.
        gs = amax * inv6
        gs = tl.where(gs > 0, gs, 1e-8)
        gs_b = _e4m3_bytes(gs)                                                     # [32] u8
        gs_used = _e4m3_to_f32(gs_b)
        gs_used = tl.where(gs_used > 0, gs_used, 1e-8)
        # The torch reference DIVIDES (g / gs_used); multiply-by-reciprocal can differ
        # in the last ulp, so true division is used here to stay byte-identical.
        qe = tl.minimum(tl.maximum(libdevice.div_rn(xe, gs_used[:, None]), -6.0), 6.0)
        qo = tl.minimum(tl.maximum(libdevice.div_rn(xo, gs_used[:, None]), -6.0), 6.0)
        ce = _e2m1_code(qe)
        co = _e2m1_code(qo)
        packed = ce | (co << 4)                                                    # [32,8] u8
        tl.store(rec + g * 8 + i, packed)
        tl.store(rec + 256 + tl.arange(0, 32), gs_b)

        # ---------------- RoPE: 64 dims, fp32 scale + E4M3 payload ---------------
        j = tl.arange(0, 64)
        r = tl.load(k_pe_ptr + row * k_pe_stride + j).to(tl.float32)
        ramax = tl.max(tl.abs(r), axis=0)
        rsv = tl.full([64], 1.0, tl.float32) * (ramax * inv448)              # [64], all equal
        rsv = tl.where(rsv > 0, rsv, 1e-8)
        rq = tl.minimum(tl.maximum(libdevice.div_rn(r, rsv), -448.0), 448.0)
        tl.store(rec + 304 + j, _e4m3_bytes(rq))
        # fp32 rope_scale at byte 288 (16-byte aligned: 368*slot and 288 are both /16)
        tl.store((rec + 288).to(tl.pointer_type(tl.float32)), tl.max(rsv, axis=0))
        # zero pad [292,304)
        z = tl.arange(0, 16)
        tl.store(rec + 292 + z, tl.zeros([16], dtype=tl.uint8), mask=z < 12)


    _TRITON_CONSTS: dict = {}


    def _triton_consts(device):
        """fp32(1/6), fp32(1/448) exactly as torch's scalar-divisor path forms them
        (a true fp32 division of 1 by the scalar), cached per device, moved once."""
        k = str(device)
        c = _TRITON_CONSTS.get(k)
        if c is None:
            one = torch.tensor(1.0, dtype=torch.float32)
            inv6 = (one / torch.tensor(6.0, dtype=torch.float32)).to(device)
            inv448 = (one / torch.tensor(448.0, dtype=torch.float32)).to(device)
            c = (inv6, inv448)
            _TRITON_CONSTS[k] = c
        return c


    def write_to_kv_cache_fused(
        kv_cache: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        latent_scale=1.0,
    ) -> None:
        """Drop-in for nvfp4_ds_mla_writer.write_to_kv_cache, one kernel launch."""
        n = kv_c.shape[0]
        if n == 0:
            return
        rec_bytes = kv_cache.shape[-1]
        assert rec_bytes == RECORD_FP8_ROPE, f"fused writer supports the 368-byte record, got {rec_bytes}"
        flat = kv_cache.view(-1, RECORD_FP8_ROPE)
        if isinstance(latent_scale, torch.Tensor):
            sc = latent_scale.to(torch.float32).reshape(-1)[0].contiguous()
        else:
            # cached per (device, value): a fresh CPU->GPU tensor inside CUDA graph
            # capture is illegal (H2D copy), and this path is on the captured step.
            k = (str(kv_c.device), float(latent_scale))
            sc = _TRITON_CONSTS.get(k)
            if sc is None:
                sc = torch.tensor(float(latent_scale), device=kv_c.device, dtype=torch.float32)
                _TRITON_CONSTS[k] = sc
        kv_c = kv_c if kv_c.stride(1) == 1 else kv_c.contiguous()
        k_pe = k_pe if k_pe.stride(1) == 1 else k_pe.contiguous()
        slots = slot_mapping.to(torch.int64)
        consts = _triton_consts(kv_c.device)
        nvfp4_ds_mla_write_kernel[(n,)](
            kv_c, kv_c.stride(0), k_pe, k_pe.stride(0), slots, flat, sc,
            consts[0], consts[1],
            num_warps=2,
        )



def write_to_kv_cache(kv_cache, kv_c, k_pe, slot_mapping, *, latent_scale=1.0):
    """Scatter freshly packed records into a paged uint8 KV cache.

    Fused Triton path for the 368-byte record; torch reference otherwise.
    """
    if _HAVE_TRITON and kv_cache.shape[-1] == RECORD_FP8_ROPE:
        return write_to_kv_cache_fused(kv_cache, kv_c, k_pe, slot_mapping, latent_scale=latent_scale)
    return _write_to_kv_cache_torch(kv_cache, kv_c, k_pe, slot_mapping, latent_scale=latent_scale)

# ---------------------------------------------------------------------------
# Self-test: round-trip against the reader-mirroring dequant.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    n = 4096
    kv_c = torch.randn(n, NOPE_DIM) * 2.0        # post-layernorm latent, O(1)
    k_pe = torch.randn(n, ROPE_DIM) * 1.5
    for fp8_rope, want in ((True, RECORD_FP8_ROPE), (False, RECORD_BF16_ROPE)):
        rec = pack_records(kv_c, k_pe, fp8_rope=fp8_rope)
        assert rec.shape == (n, want) and rec.dtype == torch.uint8, rec.shape
        nope, rope = unpack_records(rec)
        # E2M1 with per-16 E4M3 scale: expect low-single-digit % relative RMS error
        nope_rel = ((nope - kv_c).norm() / kv_c.norm()).item()
        rope_rel = ((rope - k_pe).norm() / k_pe.norm()).item()
        # cosine similarity is what attention actually feels
        cos = torch.nn.functional.cosine_similarity(nope, kv_c, dim=-1).mean().item()
        print(f"record={want:3d}B  nope_rel_rms={nope_rel:.4f}  nope_cos={cos:.5f}  rope_rel_rms={rope_rel:.5f}")
        assert nope_rel < 0.12, nope_rel      # E2M1 is coarse; ~8-10% RMS is normal
        assert cos > 0.99, cos
        assert rope_rel < (0.03 if fp8_rope else 0.01), rope_rel
        # Layout invariants the reader depends on
        assert (rec[:, 292:304] == 0).all(), "pad must be zero"
        # nibble order: dim 0 lives in the low nibble of byte 0
        c = _e2m1_encode((kv_c[:, :GROUP] / (kv_c[:, :GROUP].abs().amax(-1, keepdim=True) / 6)).clamp(-6, 6))
        assert ((rec[:, 0] & 0x0F) == c[:, 0]).float().mean() > 0.95, "low nibble != even dim"
    # zero-group and zero-rope robustness
    z = pack_records(torch.zeros(8, NOPE_DIM), torch.zeros(8, ROPE_DIM))
    zn, zr = unpack_records(z)
    assert torch.isfinite(zn).all() and torch.isfinite(zr).all() and zn.abs().max() < 1e-6
    # scatter path
    cache = torch.zeros(3, 64, RECORD_FP8_ROPE, dtype=torch.uint8)
    slots = torch.tensor([5, 70, 191, -1])
    write_to_kv_cache(cache, kv_c[:4], k_pe[:4], slots)
    back, _ = unpack_records(cache.view(-1, RECORD_FP8_ROPE)[[5, 70, 191]])
    assert torch.nn.functional.cosine_similarity(back, kv_c[:3], dim=-1).min() > 0.99
    # the -1 row must land in slot 0 (null block) and nowhere else
    assert (cache.view(-1, RECORD_FP8_ROPE)[0] != 0).any()
    assert (cache.view(-1, RECORD_FP8_ROPE)[1:5] == 0).all()
    # tensor latent_scale path (device op, capture-safe) must match the float path
    r1 = pack_records(kv_c[:8], k_pe[:8], latent_scale=1.0)
    r2 = pack_records(kv_c[:8], k_pe[:8], latent_scale=torch.tensor(1.0))
    assert torch.equal(r1, r2)
    print("ALL WRITER SELF-TESTS PASSED")
