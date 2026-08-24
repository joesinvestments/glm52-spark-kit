# SPDX-License-Identifier: Apache-2.0
"""Build A — Triton bmm kernel for the GLM-5.2 MLA-absorb decode GEMMs.

Drop-in replacement for the two cuBLAS-sm80-fallback torch.bmm call sites in
vllm/model_executor/layers/attention/mla_attention.py (BMM1 q_nope@W_UK_T,
BMM2 attn_out@W_UV). ~27% faster than torch.bmm on GB10 (sm_121a, 48 SMs) at
the deployed decode shapes: standalone 8.6-10.4us -> 6.2-8.2us per call,
B=3..6, flat in B (launch-latency bound). See buildA/PLAN.md for the audit.

Gating (all read ONCE at import time — no per-call getenv):
  VLLM_BUILDA_BMM=1        enable (default off: zero behavior change)
  VLLM_BUILDA_BMM_MAX_M=8  per-graph shape guard; batches larger than this
                           fall back to torch.bmm (correctness+perf envelope
                           was verified for M in 3..6, BLOCK_M=8)

CUDA-graph capture safety (audited):
  - launcher does no host<->device syncs: no .item()/.cpu()/.tolist(); grid,
    strides and block sizes are derived from python ints (tensor shapes),
    which are static per captured graph;
  - the M<=MAX_M / dtype dispatch branches are on shapes/dtypes only — static
    per captured graph;
  - Triton JIT compilation happens on the first eager call for a given
    (shape-specialization, strides) key; vLLM warms every capture size up
    eagerly before capturing (dummy runs), so no compilation can occur inside
    capture as long as VLLM_BUILDA_BMM is set BEFORE engine start (it is an
    image/launch env, not a runtime toggle).
"""

import os

import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get("VLLM_BUILDA_BMM", "0") == "1"
_MAX_M = int(os.environ.get("VLLM_BUILDA_BMM_MAX_M", "8"))


def builda_bmm_enabled() -> bool:
    return _ENABLED


@triton.jit
def _bmm_tiny_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    saz, sam, sak,          # A strides (batch, m, k)
    sbz, sbk, sbn,          # B strides
    scz, scm, scn,          # C strides (C may be a transposed view -> arbitrary strides)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_z = tl.program_id(0)        # batch (head) index, 0..15
    pid_n = tl.program_id(1)        # N tile

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

    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(
        c_ptr + pid_z * scz + offs_m[:, None] * scm + offs_n[None, :] * scn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _cfg(K: int, N: int):
    # 16 heads x (N/BLOCK_N) CTAs on 48 SMs:
    #   N=512, BLOCK_N=128 -> 64 CTAs (~1.3 waves)   [BMM1]
    #   N=256, BLOCK_N=64  -> 64 CTAs (~1.3 waves)   [BMM2]
    if N >= 512:
        return dict(BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2)
    return dict(BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2)


def bmm_v0(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """torch.bmm(a, b, out=out) for the MLA-absorb decode shapes.

    Accepts non-contiguous `a` and `out` (e.g. transpose(0, 1) views at both
    call sites). No host synchronization; safe under CUDA graph capture.
    """
    Z, M, K = a.shape
    _, _, N = b.shape
    cfg = _cfg(K, N)
    BLOCK_M = max(8, triton.next_power_of_2(M))
    grid = (Z, triton.cdiv(N, cfg["BLOCK_N"]))
    _bmm_tiny_kernel[grid](
        a, b, out,
        M, K, N,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out


# Custom-op registration: keeps the call compile/fake-tensor traceable if the
# surrounding code is ever captured by torch.compile. The two mla_attention.py
# call sites run inside the unified-attention custom op (eager), so the direct
# python call is the normal path; the op is belt-and-braces.
_HAS_OP = False
try:
    @torch.library.custom_op("builda::bmm_v0", mutates_args=("out",))
    def _op(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        bmm_v0(a, b, out)

    @_op.register_fake
    def _(a, b, out):
        return None

    _HAS_OP = True
except Exception:  # pragma: no cover — older torch; direct call still works
    pass


def builda_bmm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Dispatch wrapper used by the mla_attention.py call sites.

    Falls back to torch.bmm outside the verified envelope. All branch
    conditions are shape/dtype-derived -> static per captured CUDA graph.
    """
    if a.shape[1] > _MAX_M or a.dtype != torch.bfloat16:
        return torch.bmm(a, b, out=out)
    if _HAS_OP:
        torch.ops.builda.bmm_v0(a, b, out)
        return out
    return bmm_v0(a, b, out)
