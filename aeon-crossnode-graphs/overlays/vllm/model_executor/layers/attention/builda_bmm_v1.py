# buildA kernel_v1 — Triton bmm for GLM-5.2 MLA-absorb decode, tuned per-shape on
# GB10 (sm_121a, 48 SMs) via in-graph sweep (see v1/bench_refine.py, 2026-08-12).
#
# v1 vs v0: same kernel body; per-shape launch configs re-tuned under CUDA-graph
# replay (the deployed regime) instead of eager events:
#   BMM1 (K=192, N=512): BLOCK_N=32, BLOCK_K=32, 2 warps, 2 stages  (v0: 128/64/4/2)
#     -> 16x16 = 256 CTAs; small tiles cut per-CTA prologue latency on this
#        latency-bound shape. 4.13 -> 3.02 us in-graph.
#   BMM2 (K=512, N=256): BLOCK_N=32, BLOCK_K=64, 4 warps, 3 stages  (v0: 64/64/4/2)
#     -> 16x8 = 128 CTAs, deeper pipeline over the 8 K-tiles. 5.67 -> 3.89 us.
# CUTE-DSL port was brought up (v1/cute_bmm.py, correct + capture-safe) but the
# SIMT path is 22.9/56 us; a competitive CUTE kernel needs the full tensor-core
# MMA pipeline — Triton retained per PLAN.md fallback clause.
#
# Interface identical to v0: builda::bmm_v1(a, b, out), arbitrary strides on A and
# out, bf16 in/out, fp32 accumulate, no host syncs (graph-capture safe).

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _bmm_tiny_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    saz, sam, sak,
    sbz, sbk, sbn,
    scz, scm, scn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_z = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_base = a_ptr + pid_z * saz
    b_base = b_ptr + pid_z * sbz
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ak = k0 + offs_k
        a = tl.load(
            a_base + offs_m[:, None] * sam + ak[None, :] * sak,
            mask=(offs_m[:, None] < M) & (ak[None, :] < K), other=0.0,
        )
        b = tl.load(
            b_base + ak[:, None] * sbk + offs_n[None, :] * sbn,
            mask=(ak[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)
    tl.store(
        c_ptr + pid_z * scz + offs_m[:, None] * scm + offs_n[None, :] * scn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _cfg(K: int, N: int):
    # Tuned in-graph on GB10 (v1). Keyed on K: 192 -> BMM1 shape, else BMM2-like.
    if K <= 192:
        return dict(BLOCK_N=32, BLOCK_K=32, num_warps=2, num_stages=2)
    return dict(BLOCK_N=32, BLOCK_K=64, num_warps=4, num_stages=3)


def bmm_v1(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for torch.bmm(a, b, out=out); arbitrary strides; capture-safe."""
    Z, M, K = a.shape
    _, _, N = b.shape
    if out is None:
        out = a.new_empty((Z, M, N))
    cfg = _cfg(K, N)
    BLOCK_M = max(8, triton.next_power_of_2(M))
    grid = (Z, triton.cdiv(N, cfg["BLOCK_N"]))
    _bmm_tiny_kernel[grid](
        a, b, out, M, K, N,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out


try:
    @torch.library.custom_op("builda::bmm_v1", mutates_args=("out",))
    def _op(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        bmm_v1(a, b, out)

    @_op.register_fake
    def _(a, b, out):
        return None
except Exception:
    pass


# ---- versioned dispatch (env VLLM_BUILDA_VER: 1 -> v1 (default), 0 -> v0) ----
_VER = os.environ.get("VLLM_BUILDA_VER", "1")

if _VER == "0":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kernel_v0 import bmm_v0 as _dispatch_impl
else:
    _dispatch_impl = bmm_v1


def builda_bmm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Version-dispatched entry point; VLLM_BUILDA_VER selects v1 (default) or v0."""
    return _dispatch_impl(a, b, out=out)
