"""Fused Triton writer for the b12x nvfp4_ds_mla KV record (368-byte layout).

Why: the torch writer is correct but costs ~12 kernel launches per layer per
prefill chunk; py-spy showed it as 54% of Python-visible prefill time on the
fleet. This is one launch: one program per token row does the whole record.

Byte-identical to nvfp4_ds_mla_writer.pack_records (the reader-derived torch
reference), which is enforced by the self-test at the bottom. Layout:

    [0,256)   512 NoPE dims as E2M1, 2/byte, low nibble = even dim
    [256,288) 32 E4M3 group-16 scales
    [288,292) fp32 rope_scale
    [292,304) zero pad
    [304,368) 64 E4M3 rope

Encoding rules replicated exactly from the torch reference:
  * group scale gs = amax/6 (or 1e-8 if amax==0), rounded to E4M3, then the
    ROUNDED scale (dequantised, again 1e-8-guarded) is what divides the group.
  * q = clamp(x/gs, -6, 6); E2M1 index = number of midpoints STRICTLY below |q|
    (torch.bucketize right=False semantics), midpoints 0.25 .. 5.0.
  * code = sign<<3 | idx, but idx==0 is stored as +0 (no negative zero).
  * rope: rs = amax/448 (1e-8 guard), rope bytes = E4M3(clamp(rope/rs)).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

RECORD = 368
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


_CONSTS: dict = {}


def _consts(device):
    """fp32(1/6), fp32(1/448) exactly as torch's scalar-divisor path forms them
    (a true fp32 division of 1 by the scalar), cached per device, moved once."""
    k = str(device)
    c = _CONSTS.get(k)
    if c is None:
        one = torch.tensor(1.0, dtype=torch.float32)
        inv6 = (one / torch.tensor(6.0, dtype=torch.float32)).to(device)
        inv448 = (one / torch.tensor(448.0, dtype=torch.float32)).to(device)
        c = (inv6, inv448)
        _CONSTS[k] = c
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
    assert rec_bytes == RECORD, f"fused writer supports the 368-byte record, got {rec_bytes}"
    flat = kv_cache.view(-1, RECORD)
    if isinstance(latent_scale, torch.Tensor):
        sc = latent_scale.to(torch.float32).reshape(-1)[0].contiguous()
    else:
        # cached per (device, value): a fresh CPU->GPU tensor inside CUDA graph
        # capture is illegal (H2D copy), and this path is on the captured step.
        k = (str(kv_c.device), float(latent_scale))
        sc = _CONSTS.get(k)
        if sc is None:
            sc = torch.tensor(float(latent_scale), device=kv_c.device, dtype=torch.float32)
            _CONSTS[k] = sc
    kv_c = kv_c if kv_c.stride(1) == 1 else kv_c.contiguous()
    k_pe = k_pe if k_pe.stride(1) == 1 else k_pe.contiguous()
    slots = slot_mapping.to(torch.int64)
    consts = _consts(kv_c.device)
    nvfp4_ds_mla_write_kernel[(n,)](
        kv_c, kv_c.stride(0), k_pe, k_pe.stride(0), slots, flat, sc,
        consts[0], consts[1],
        num_warps=2,
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/w")
    from nvfp4_ds_mla_writer import pack_records, write_to_kv_cache, unpack_records
    dev = torch.device("cuda")
    torch.manual_seed(0)
    for n in (1, 7, 64, 1024):
        for scale_val in (1.0, 0.7):
            kv_c = torch.randn(n, 512, device=dev, dtype=torch.bfloat16) * 2.0
            k_pe = torch.randn(n, 64, device=dev, dtype=torch.bfloat16) * 1.5
            # include exact zeros (guard paths) and a big outlier
            kv_c[0, :16] = 0
            if n > 3:
                kv_c[3, 40] = 250.0
                k_pe[2, :] = 0
            slots = torch.arange(64, 64 + n, device=dev, dtype=torch.int64)
            if n > 5:
                slots[-2:] = -1
            ref = torch.zeros(40, 64, RECORD, dtype=torch.uint8, device=dev)
            got = torch.zeros_like(ref)
            ls = torch.tensor(scale_val, device=dev)
            write_to_kv_cache(ref, kv_c, k_pe, slots, latent_scale=ls)
            write_to_kv_cache_fused(got, kv_c, k_pe, slots, latent_scale=ls)
            fr, fg = ref.view(-1, RECORD), got.view(-1, RECORD)
            # slot 0 is the null block (pad rows collide there nondeterministically); compare real slots
            same = torch.equal(fr[1:], fg[1:])
            if not same:
                bad = (fr[1:] != fg[1:]).any(dim=1).nonzero().flatten()[:5]
                r0 = int(bad[0]) + 1
                cols = (fr[r0] != fg[r0]).nonzero().flatten()[:8].tolist()
                print(f"MISMATCH n={n} scale={scale_val} rows={bad.tolist()} first cols={cols}")
                print("  ref:", fr[r0, cols].tolist()); print("  got:", fg[r0, cols].tolist())
                raise SystemExit(1)
            print(f"n={n:5d} latent_scale={scale_val}: fused == torch reference, byte-identical on all real slots")
    # timing: fused vs torch, realistic prefill chunk
    n = 4096
    kv_c = torch.randn(n, 512, device=dev, dtype=torch.bfloat16); k_pe = torch.randn(n, 64, device=dev, dtype=torch.bfloat16)
    slots = torch.arange(n, device=dev, dtype=torch.int64) + 64
    cache = torch.zeros(80, 64, RECORD, dtype=torch.uint8, device=dev)
    for fn, name in ((write_to_kv_cache, "torch"), (write_to_kv_cache_fused, "fused")):
        for _ in range(3): fn(cache, kv_c, k_pe, slots)
        torch.cuda.synchronize(); t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(20): fn(cache, kv_c, k_pe, slots)
        t1.record(); torch.cuda.synchronize()
        print(f"  {name:6s}: {t0.elapsed_time(t1)/20:.3f} ms per 4096-token write")
    # capture safety
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        for _ in range(2): write_to_kv_cache_fused(cache, kv_c, k_pe, slots)
    torch.cuda.current_stream().wait_stream(s_)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): write_to_kv_cache_fused(cache, kv_c, k_pe, slots)
    cache.zero_(); g.replay(); torch.cuda.synchronize()
    ref2 = torch.zeros_like(cache); write_to_kv_cache(ref2, kv_c, k_pe, slots)
    assert torch.equal(cache.view(-1, RECORD)[1:], ref2.view(-1, RECORD)[1:])
    print("  fused writer: CUDA-graph capturable, replay == eager")
    print("FUSED WRITER PROVEN")
