# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers.

R17 port: pristine v0.27.1 sparse_attn_indexer.py used as the base (patched
full copy; see PORTS_SCHED_NOTES.md / PORTS_B12X_NOTES.md for the strategy
decision), with a B12X sparse-indexer CUDA execution path re-applied as an
additive, clearly-marked block, gated at runtime by
``VLLM_USE_B12X_SPARSE_INDEXER=1`` (read directly from ``os.environ`` --
matching the pattern used by the sibling port
``ports/vllm/v1/attention/backends/mla/indexer.py``, since
``envs.VLLM_USE_B12X_SPARSE_INDEXER`` does not exist in pristine 0.27.1's
``vllm/envs.py``).

Public surface (op registration, function signatures) is UNCHANGED from
0.27.1 -- no new parameters were added to ``sparse_attn_indexer``,
``sparse_attn_indexer_fake``, or ``SparseAttnIndexer``, since 0.27.1 model
callers (see ``deepseek_v32``/``deepseek_v4`` model files under
``work/src``) do not pass a b12x-related kwarg. The B12X path is selected
internally via the env gate, exactly like R9's production behavior when
``VLLM_USE_B12X_SPARSE_INDEXER`` is exported at process start.

Scope: this port targets R17 v1 (DCP1 only). DCP>1 B12X merge machinery
(``_merge_b12x_dcp_topk``, ``_prewarm_b12x_dcp_topk_merge``,
``_convert_b12x_dcp_local_topk_to_global`` and friends in the production
fork) is NOT ported; the B12X path raises ``NotImplementedError`` if it is
ever invoked with ``dcp_world_size > 1``. The non-b12x (upstream DeepGEMM)
path is untouched and remains fully DCP-capable via its existing
``_merge_dcp_topk_global`` (CuteDSL) mechanism -- unaffected by this patch.
"""

import os

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.pcp import maybe_gather_indexer_k
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32

# --- B12X sparse-indexer support (R9/prod port, DCP1 only) ---------------
# R9/prod read envs.VLLM_USE_B12X_SPARSE_INDEXER; that knob is not defined in
# the pristine 0.27.1 vllm/envs.py (the envs port is a separate PORT_MAP
# row), so read the process environment directly to keep this file
# self-contained -- mirrors ports/vllm/v1/attention/backends/mla/indexer.py.
_USE_B12X_SPARSE_INDEXER = os.environ.get(
    "VLLM_USE_B12X_SPARSE_INDEXER", "0"
).lower() in ("1", "true", "yes", "on")

_B12X_PAGED_INDEX_PAGE_SIZE = 64
_B12X_PAGED_INDEX_HEAD_DIM = 128
_B12X_PAGED_INDEX_SCALE_BYTES = 4
_B12X_PAGED_INDEX_PAGE_WIDTH = _B12X_PAGED_INDEX_PAGE_SIZE * (
    _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES
)
_B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT = 32768
_B12X_PAGED_INDEX_TILE_BLOCK_K = 512
_B12X_PREFILL_PAGED_ROUTE = "packed_contiguous"


def _b12x_sparse_indexer_requested() -> bool:
    """Env-gate only (DCP1 R17 scope). Prod also falls back to inspecting
    ``vllm_config.attention_config.backend == "B12X_MLA_SPARSE"``; that
    backend-name fallback is intentionally not carried here since this file's
    public signature takes no ``use_b12x_sparse_indexer`` kwarg for 0.27.1
    callers to thread through -- the env var is the single source of truth.
    """
    return _USE_B12X_SPARSE_INDEXER


def _ensure_b12x_sparse_indexer_supported() -> None:
    if not current_platform.is_cuda():
        raise RuntimeError("B12X sparse indexer/top-k requires CUDA.")
    if not current_platform.is_device_capability_family(120):
        raise RuntimeError("B12X sparse indexer/top-k currently requires an SM120 GPU.")


def _use_b12x_sparse_indexer() -> bool:
    if not _b12x_sparse_indexer_requested():
        return False
    _ensure_b12x_sparse_indexer_supported()
    return True


def _get_b12x_indexer_paged_supertile_k() -> int:
    raw = os.environ.get("B12X_PAGED_INDEX_SUPERTILE_K")
    tokens = _B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT if raw is None else int(raw)
    tokens = max(tokens, _B12X_PAGED_INDEX_TILE_BLOCK_K)
    return (
        (tokens + _B12X_PAGED_INDEX_TILE_BLOCK_K - 1)
        // _B12X_PAGED_INDEX_TILE_BLOCK_K
        * _B12X_PAGED_INDEX_TILE_BLOCK_K
    )


def _get_b12x_paged_indexer_profile_q_rows(q_rows: int) -> int:
    """Return the largest q chunk the real prefill chunker can hand to b12x."""
    max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
    tile_k = _get_b12x_indexer_paged_supertile_k()
    max_q_rows = max(1, max_logits_elems // max(1, tile_k))
    return min(max(1, int(q_rows)), max_q_rows)


def _get_b12x_paged_indexer_profile_k_rows(
    max_model_len: int,
    total_seq_lens: int,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> int:
    """K rows the paged-indexer scratch must hold on THIS rank at profile time.

    Under decode context parallelism a request's KV rows are sharded across the
    DCP group (interleaved in ``cp_kv_cache_interleave_size`` chunks), so a rank
    never holds more than ceil(rows / dcp) + one interleave slack per request.
    Reserving the unsharded ``max_model_len`` per rank over-allocates the paged
    indexer scratch by dcp_world_size (4x at DCP=4). On unified-memory GB10 that
    was enough to exhaust host+GPU memory during the profile run at 316K ctx
    (NV_ERR_NO_MEMORY, 2026-08-17). 0xdfi's exp1 notes record the same effect
    ("sparse-indexer scratch scaled with context x batch, not sharded by DCP").
    """
    known_k_rows = max(int(max_model_len), int(total_seq_lens), 0)
    if known_k_rows <= 0:
        return _get_b12x_indexer_paged_supertile_k()
    dcp = max(1, int(dcp_world_size))
    if dcp == 1:
        return known_k_rows
    slack = max(1, int(cp_kv_cache_interleave_size)) * 2
    return -(-known_k_rows // dcp) + slack


def _get_b12x_paged_indexer_profile_warmup_k_rows(profile_k_rows: int) -> int:
    raw = os.environ.get("B12X_PAGED_INDEX_PROFILE_WARMUP_K")
    cap = _get_b12x_indexer_paged_supertile_k() if raw is None else int(raw)
    cap = max(cap, _B12X_PAGED_INDEX_TILE_BLOCK_K)
    cap = (
        (cap + _B12X_PAGED_INDEX_TILE_BLOCK_K - 1)
        // _B12X_PAGED_INDEX_TILE_BLOCK_K
        * _B12X_PAGED_INDEX_TILE_BLOCK_K
    )
    return min(max(1, int(profile_k_rows)), cap)


def _b12x_profile_rows_or_empty(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    rows = max(1, int(rows))
    if int(tensor.shape[0]) >= rows:
        return tensor[:rows].contiguous()
    return torch.empty(
        (rows, *tuple(tensor.shape[1:])),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _b12x_profile_weights_2d(
    weights: torch.Tensor,
    *,
    q_rows: int,
    num_q_heads: int,
    device: torch.device,
) -> torch.Tensor:
    q_rows = max(1, int(q_rows))
    num_q_heads = int(num_q_heads)
    if (
        weights.ndim == 2
        and int(weights.shape[0]) >= q_rows
        and int(weights.shape[1]) == num_q_heads
        and weights.dtype == torch.float32
        and weights.device == device
    ):
        return weights[:q_rows].contiguous()
    return torch.empty((q_rows, num_q_heads), dtype=torch.float32, device=device)


def _assert_b12x_prefill_paged_route(obj: object, *, owner: str) -> None:
    route = getattr(obj, "route", None)
    if route is None:
        route = getattr(getattr(obj, "layout", None), "route", None)
    if route != _B12X_PREFILL_PAGED_ROUTE:
        raise RuntimeError(
            "B12X sparse prefill expected the b12x planner to resolve the paged "
            f"source to {_B12X_PREFILL_PAGED_ROUTE!r}, got {route!r} from {owner}."
        )


def _flatten_b12x_paged_index_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    expected_shape_tail = (
        _B12X_PAGED_INDEX_PAGE_SIZE,
        _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES,
    )

    if kv_cache.ndim != 3 or kv_cache.dtype != torch.uint8:
        raise RuntimeError(
            "b12x paged indexer cache must be rank-3 uint8 with "
            f"shape [num_blocks, {expected_shape_tail[0]}, "
            f"{expected_shape_tail[1]}], got shape={tuple(kv_cache.shape)} "
            f"dtype={kv_cache.dtype}."
        )
    if tuple(kv_cache.shape[1:]) != expected_shape_tail:
        raise RuntimeError(
            "b12x paged indexer cache has an unsupported shape, "
            f"got {tuple(kv_cache.shape)}; expected tail {expected_shape_tail}."
        )
    if kv_cache.stride(1) != expected_shape_tail[1] or kv_cache.stride(2) != 1:
        raise RuntimeError(
            "b12x paged indexer cache has an unsupported layout, "
            f"shape={tuple(kv_cache.shape)} stride={tuple(kv_cache.stride())}; "
            f"expected inner strides ({expected_shape_tail[1]}, 1)."
        )

    return kv_cache.as_strided(
        (int(kv_cache.shape[0]), _B12X_PAGED_INDEX_PAGE_WIDTH),
        (int(kv_cache.stride(0)), 1),
    )


def _run_b12x_paged_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    topk_scores: torch.Tensor | None = None,
    active_width: torch.Tensor | None = None,
    shared_page_table: bool = False,
) -> torch.Tensor:
    """Run b12x paged indexer top-k with caller-owned scratch.

    b12x sizes scratch from indexer K-cache rows/pages. ``active_width`` is
    the builder-computed live K-row window (a metadata tensor, not an
    in-kernel reduction); when None, b12x falls back to the capacity cap.
    """
    from b12x.attention.indexer import (
        INDEXER_SOURCE_LAYOUT_PAGED,
        PAGED_INDEX_PAGE_SIZE,
        B12XIndexerScratchCaps,
        index_topk_fp8,
        plan_indexer_scratch,
    )

    if int(PAGED_INDEX_PAGE_SIZE) != _B12X_PAGED_INDEX_PAGE_SIZE:
        raise RuntimeError(
            "b12x paged indexer page-size contract changed, got "
            f"{PAGED_INDEX_PAGE_SIZE}; expected {_B12X_PAGED_INDEX_PAGE_SIZE}."
        )

    index_k_cache = _flatten_b12x_paged_index_cache(kv_cache)
    expected_num_q_heads = int(q_fp8.shape[1])
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device=q_fp8.device,
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=expected_num_q_heads,
            max_q_rows=int(q_fp8.shape[0]),
            max_page_table_width=int(block_table.shape[1]),
            topk=int(topk_tokens),
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=bool(shared_page_table),
        )
    )
    if shared_page_table:
        _assert_b12x_prefill_paged_route(plan, owner="scratch plan")
    scratch = current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())
    binding = plan.bind(
        scratch=scratch,
        real_page_table=block_table,
        cache_seqlens_int32=seq_lens,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=expected_num_q_heads,
        shared_page_table=shared_page_table,
    )
    if shared_page_table:
        _assert_b12x_prefill_paged_route(binding, owner="binding")
    return index_topk_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
        page_size=PAGED_INDEX_PAGE_SIZE,
        expected_num_q_heads=expected_num_q_heads,
        out_indices=topk_indices,
        out_scores=topk_scores,
    )


def _reserve_b12x_paged_indexer_scratch(
    *,
    q_rows: int,
    num_q_heads: int,
    topk_tokens: int,
    total_k_rows: int,
    device: torch.device,
    shared_page_table: bool = False,
) -> None:
    from b12x.attention.indexer import (
        INDEXER_SOURCE_LAYOUT_PAGED,
        PAGED_INDEX_PAGE_SIZE,
        B12XIndexerScratchCaps,
        plan_indexer_scratch,
    )

    page_table_width = max(
        1,
        (max(1, int(total_k_rows)) + int(PAGED_INDEX_PAGE_SIZE) - 1)
        // int(PAGED_INDEX_PAGE_SIZE),
    )
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device=device,
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=int(num_q_heads),
            max_q_rows=max(1, int(q_rows)),
            max_page_table_width=page_table_width,
            topk=int(topk_tokens),
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=bool(shared_page_table),
        )
    )
    if shared_page_table:
        _assert_b12x_prefill_paged_route(plan, owner="scratch reservation plan")
    current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())


def _prewarm_b12x_paged_indexer_prefill(
    *,
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_tokens: int,
    profile_q_rows: int,
    profile_k_rows: int,
    page_table_k_rows: int | None = None,
) -> None:
    """DCP1-only prewarm for the shared-page-table (prefill) b12x route.

    Note: prod also prewarms the decode (non-shared) route and the
    packed-contiguous prefill variants (``_prewarm_b12x_contiguous_prefill_variants``,
    a separate optimization path not reachable from ``_run_b12x_paged_topk``);
    those are not ported here -- see PORTS_SCHED_NOTES.md B12X-DCP1 entry.
    Only the paged shared-page-table warmup this DCP1 prefill path needs is
    included; first invocation of the plain decode route may pay one-time
    JIT/allocation cost inside the first real decode step instead of during
    the dedicated profiling run.
    """
    q_rows = max(1, int(profile_q_rows))
    k_rows = max(1, int(profile_k_rows))
    page_table_rows = k_rows if page_table_k_rows is None else max(1, int(page_table_k_rows))
    num_q_heads = int(q_quant.shape[1])
    page_table_width = max(
        1,
        (page_table_rows + _B12X_PAGED_INDEX_PAGE_SIZE - 1) // _B12X_PAGED_INDEX_PAGE_SIZE,
    )
    q_warm = _b12x_profile_rows_or_empty(q_quant, q_rows)
    weights_warm = _b12x_profile_weights_2d(
        weights, q_rows=q_rows, num_q_heads=num_q_heads, device=q_quant.device
    )
    seq_lens = torch.full((q_rows,), k_rows, dtype=torch.int32, device=q_quant.device)
    block_table = torch.zeros(
        (1, page_table_width), dtype=torch.int32, device=q_quant.device
    ).expand(q_rows, page_table_width)
    kv_cache_warm = kv_cache
    if int(kv_cache_warm.shape[0]) <= 0:
        kv_cache_warm = torch.empty(
            (1, _B12X_PAGED_INDEX_PAGE_SIZE, _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES),
            dtype=torch.uint8,
            device=q_quant.device,
        )
    topk_indices = torch.empty((q_rows, int(topk_tokens)), dtype=torch.int32, device=q_quant.device)
    _run_b12x_paged_topk(
        q_fp8=q_warm,
        weights=weights_warm,
        kv_cache=kv_cache_warm,
        seq_lens=seq_lens,
        block_table=block_table,
        schedule_metadata=None,
        topk_indices=topk_indices,
        topk_tokens=int(topk_tokens),
        shared_page_table=True,
    )


def _assert_dcp1_for_b12x(dcp_world_size: int) -> None:
    # DCP>1 is supported by _merge_b12x_dcp_topk below; nothing to assert.
    return None


def _merge_b12x_dcp_topk(
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
) -> None:
    """DCP>1 merge for the B12X indexer path.

    Same contract as _merge_dcp_topk_global: each rank holds local top-K
    positions into its 1/N KV shard; a token in the global top-K must be in
    its owning rank's local top-K, so exchanging only the per-rank candidates
    is exact. The stock path packs (score, global_id) with a Triton kernel that
    looks scores up in the full logits row; b12x already returns the scores of
    exactly the selected indices (out_scores), so packing is pure arithmetic.
    Global id formula and -1/-inf padding are byte-identical to
    _pack_dcp_topk_candidates_triton_kernel. All device ops, no host sync, so
    it is CUDA-graph safe. Overwrites topk_indices with GLOBAL token ids.
    """
    if dcp_world_size <= 1:
        return
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        stable_topk_from_gathered_candidates_cutedsl,
    )
    valid = topk_indices >= 0
    idx = topk_indices.clamp(min=0).to(torch.int64)
    gid = (
        (idx // cp_interleave) * (dcp_world_size * cp_interleave)
        + dcp_rank * cp_interleave
        + idx % cp_interleave
    )
    gid = torch.where(valid, gid, torch.full_like(gid, -1))
    sc = topk_scores.to(torch.float32)
    sc = torch.where(valid, sc, torch.full_like(sc, float("-inf")))
    packed = torch.stack([sc, gid.to(torch.float32)], dim=-1).contiguous()
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


# --- end B12X sparse-indexer support --------------------------------------


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    use_b12x_indexer = _use_b12x_sparse_indexer()
    if use_b12x_indexer:
        _assert_dcp1_for_b12x(dcp_world_size)
        if use_fp4_cache:
            raise RuntimeError(
                "b12x sparse indexer currently requires the FP8 indexer cache; "
                "disable use_fp4_indexer_cache or disable "
                "VLLM_USE_B12X_SPARSE_INDEXER."
            )

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        if use_b12x_indexer:
            profile_q_rows = _get_b12x_paged_indexer_profile_q_rows(
                int(q_quant.shape[0])
            )
            profile_k_rows = _get_b12x_paged_indexer_profile_k_rows(
                max_model_len=max_model_len,
                total_seq_lens=total_seq_lens,
                dcp_world_size=dcp_world_size,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            )
            warmup_k_rows = _get_b12x_paged_indexer_profile_warmup_k_rows(
                profile_k_rows
            )
            _reserve_b12x_paged_indexer_scratch(
                q_rows=profile_q_rows,
                num_q_heads=int(q_quant.shape[1]),
                topk_tokens=int(topk_tokens),
                total_k_rows=profile_k_rows,
                device=q_quant.device,
                shared_page_table=False,
            )
            _reserve_b12x_paged_indexer_scratch(
                q_rows=profile_q_rows,
                num_q_heads=int(q_quant.shape[1]),
                topk_tokens=int(topk_tokens),
                total_k_rows=profile_k_rows,
                device=q_quant.device,
                shared_page_table=True,
            )
            _prewarm_b12x_paged_indexer_prefill(
                q_quant=q_quant,
                weights=weights,
                kv_cache=kv_cache,
                topk_tokens=int(topk_tokens),
                profile_q_rows=profile_q_rows,
                profile_k_rows=warmup_k_rows,
                page_table_k_rows=profile_k_rows,
            )
        else:
            values_spec, scales_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            current_workspace_manager().get_simultaneous(
                values_spec,
                scales_spec,
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )

            # Dummy allocation to simulate for peak logits tensor memory during
            # inference. FP8 elements so elements == bytes
            max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            _ = torch.empty(
                max_logits_elems, dtype=torch.uint8, device=hidden_states.device
            )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if use_pcp:
        num_tokens //= get_pcp_group().world_size
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            scale_fmt,
        )

    # The indexer and main MLA may classify the same short extend differently
    # because they use independent decode thresholds. Only the main MLA route
    # can determine whether the top-k indices will be consumed.
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, "prefill", None)
            if (
                getattr(prefill_metadata, "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Deliberately leave the buffer untouched. Dense MHA does not
                # consume top-k indices for this batch; clearing it would be
                # unnecessary work.
                return topk_indices_buffer

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache. Skipped entirely on the B12X path,
        # which reads K directly from the paged kv_cache and never gathers
        # into this workspace.
        if not use_b12x_indexer:
            workspace_manager = current_workspace_manager()
            values_spec, scales_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                values_spec,
                scales_spec,
            )
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            weights_slice = weights[chunk.token_start : chunk.token_end]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if use_b12x_indexer:
                # B12X paged top-k reads K directly from the paged kv_cache
                # via block_table + seq_lens; single-request chunks only, so
                # the (per-request) block table can be row-shared across the
                # chunk's query rows without an explicit expand/pack step.
                if chunk.local_total_seq_lens == 0:
                    topk_indices.fill_(-1)
                    continue
                row_has_no_kv = cu_seqlen_ke <= cu_seqlen_ks
                seq_lens = torch.where(
                    row_has_no_kv,
                    torch.zeros_like(cu_seqlen_ks),
                    cu_seqlen_ke - cu_seqlen_ks,
                )
                if chunk.num_reqs == 1:
                    # single request: row-share its block table (0xdfi's path)
                    block_table = chunk.block_table[:1].expand(
                        int(q_slice.shape[0]), int(chunk.block_table.shape[1])
                    )
                    shared_pt = True
                else:
                    # multi-request chunk: build a per-row block table by
                    # gathering each query row's request row. token_to_seq maps
                    # every row in the chunk to its request; one device gather,
                    # no host sync. This is the packing step the public port
                    # skipped, and it is what lets the B12X indexer serve
                    # concurrent prefills.
                    t2s = chunk.token_to_seq[: int(q_slice.shape[0])].to(torch.long)
                    block_table = chunk.block_table.index_select(0, t2s)
                    shared_pt = False
                need_dcp = dcp_world_size > 1
                topk_scores = (
                    torch.empty_like(topk_indices, dtype=torch.float32)
                    if need_dcp else None
                )
                _run_b12x_paged_topk(
                    q_fp8=q_slice.contiguous(),
                    weights=weights_slice.contiguous(),
                    kv_cache=kv_cache,
                    seq_lens=seq_lens,
                    block_table=block_table,
                    schedule_metadata=None,
                    topk_indices=topk_indices,
                    topk_tokens=topk_tokens,
                    topk_scores=topk_scores,
                    shared_page_table=shared_pt,
                )
                topk_indices.masked_fill_(row_has_no_kv[:, None], -1)
                if need_dcp:
                    _merge_b12x_dcp_topk(
                        topk_scores, topk_indices, topk_tokens,
                        dcp_rank, dcp_world_size, cp_kv_cache_interleave_size,
                    )
                continue

            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights_slice,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights_slice,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None

        if use_b12x_indexer:
            # B12X decode requires an unpadded rank-1 seq_lens contract (one
            # row per decode token); 0.27.1's seq_lens is always 2D (B,
            # next_n) or (B, 1). Reshape/repeat_interleave to the 1D
            # per-token contract when the tokens-per-row product matches --
            # mirrors the normalization prod applies before calling into
            # b12x. Padded/ragged batches (decode_metadata.requires_padding)
            # are refused rather than silently falling back to DeepGEMM.
            b12x_seq_lens = decode_metadata.seq_lens
            b12x_block_table = decode_metadata.block_table
            if b12x_seq_lens.dim() == 2:
                b12x_batch_size, b12x_next_n = b12x_seq_lens.shape
                if num_decode_tokens == b12x_batch_size * b12x_next_n:
                    b12x_seq_lens = b12x_seq_lens.reshape(-1).contiguous()
                    b12x_block_table = b12x_block_table.repeat_interleave(
                        b12x_next_n, dim=0
                    ).contiguous()
            # Padded batches are fine once seq_lens is rank-1: the flatten path
            # gives padded rows seq_len 0, and those rows are masked to -1
            # below (the prefill branch already does this). Only a genuinely
            # 2-D contract that could not be normalized is refused.
            if b12x_seq_lens.dim() != 1:
                raise RuntimeError(
                    "b12x sparse indexer decode requires a rank-1 seq_lens "
                    "contract after native-spec normalization; "
                    f"seq_lens_shape={tuple(decode_metadata.seq_lens.shape)}, "
                    f"normalized_seq_lens_shape={tuple(b12x_seq_lens.shape)}, "
                    f"num_decode_tokens={num_decode_tokens}."
                )

            seq_lens = b12x_seq_lens[:num_decode_tokens]
            block_table = b12x_block_table[:num_decode_tokens]
            topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
            # DSV4/C4-style compressed sources would need seq_lens/active_width
            # already converted to indexer K-row space by the metadata builder;
            # this port carries whatever `decode_metadata.active_width` the
            # ported indexer.py builder fills in (None if not applicable).
            need_dcp = dcp_world_size > 1
            topk_scores = (
                torch.empty_like(topk_indices, dtype=torch.float32)
                if need_dcp else None
            )
            _run_b12x_paged_topk(
                q_fp8=q_quant[:num_decode_tokens].contiguous(),
                weights=weights[:num_decode_tokens].contiguous(),
                kv_cache=kv_cache,
                seq_lens=seq_lens,
                block_table=block_table,
                schedule_metadata=getattr(decode_metadata, "schedule_metadata", None),
                active_width=getattr(decode_metadata, "active_width", None),
                topk_indices=topk_indices,
                topk_tokens=topk_tokens,
                topk_scores=topk_scores,
            )
            # padded / empty rows: no KV to attend, exactly like prefill's
            # row_has_no_kv masking.
            topk_indices.masked_fill_((seq_lens[:num_decode_tokens] == 0)[:, None], -1)
            if need_dcp:
                _merge_b12x_dcp_topk(
                    topk_scores, topk_indices, topk_tokens,
                    dcp_rank, dcp_world_size, cp_kv_cache_interleave_size,
                )
            return topk_indices_buffer

        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
        if use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        elif use_persistent_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.dense_mha_metadata_layer_name = ""
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        parallel_config = get_current_vllm_config().parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        if _b12x_sparse_indexer_requested():
            # Validate CUDA/SM120 support and the DCP1-only scope of this port
            # eagerly at construction time, rather than deep inside the first
            # forward pass.
            _ensure_b12x_sparse_indexer_supported()
            # DCP>1 supported via _merge_b12x_dcp_topk (concurrency-first port).
        elif current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
