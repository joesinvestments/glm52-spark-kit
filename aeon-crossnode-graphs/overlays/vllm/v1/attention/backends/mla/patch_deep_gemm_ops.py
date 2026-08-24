# SPDX-License-Identifier: Apache-2.0
"""SM12x compatibility: DeepGEMM's compiled sparse-indexer kernels do not
support sm_121a ("Assertion error ... Unsupported architecture" /
"DeepGEMM backend is unavailable"). Two 0.27 files independently import
these names from vllm.utils.deep_gemm at their own module-import time
(`vllm/v1/attention/backends/mla/indexer.py` and
`vllm/model_executor/layers/sparse_attn_indexer.py`), so patching the
source module alone does not reach either consumer's already-bound local
name -- same lesson as patch_flashmla_ops.py, applied to three more names.

Our own sm12x_deep_gemm_fallbacks.py (same lineage as the sparse-MLA Triton
kernels, already proven in production) already implements portable
replacements:
  - fp8_fp4_mqa_logits        <- _fp8_mqa_logits_sm12x        (exact signature match)
  - fp8_fp4_paged_mqa_logits  <- _fp8_paged_mqa_logits_sm12x   (adapter below: the
    sm12x 6-arg version has no `schedule_metadata` param -- our Triton path
    self-schedules and never asked for one, same as get_paged_mqa_logits_metadata
    below -- and no `clean_logits`, which the underlying Triton kernel handles
    internally via masking, same as the non-paged variant does)
  - has_deep_gemm / get_paged_mqa_logits_metadata: see inline comments.
"""
from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_APPLIED = False


def apply(force: bool = False) -> bool:
    global _APPLIED
    if _APPLIED and not force:
        return True

    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(120):
        return False

    import vllm.utils.deep_gemm as deep_gemm_mod
    from .sm12x_deep_gemm_fallbacks import (
        _fp8_mqa_logits_sm12x,
        _fp8_paged_mqa_logits_sm12x,
    )

    # NOT patching has_deep_gemm(): sparse_attn_indexer.py's Indexer.__init__
    # hard-requires it True just to construct the layer at all ("Sparse
    # Attention Indexer CUDA op requires DeepGEMM support"), a leftover
    # pre-Triton-fallback assumption unrelated to whether the COMPUTE calls
    # below actually reach native DeepGEMM. Leaving it real/True satisfies
    # that gate; the three functions below are patched directly so none of
    # them ever reach the sm_121a-incompatible compiled kernel regardless.

    def _sm12x_get_paged_mqa_logits_metadata(context_lens, block_size, num_sms):
        # `self.scheduler_metadata_buffer[:] = <this>` is a CUDA IntTensor
        # slice-assign -- None fails (TypeError), but a scalar 0 broadcasts
        # cleanly and is exactly as unused, since the Triton path
        # self-schedules and never reads this buffer.
        return 0

    def _sm12x_fp8_fp4_mqa_logits(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    ):
        return _fp8_mqa_logits_sm12x(
            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
        )

    def _sm12x_fp8_fp4_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,  # unused: self-scheduling Triton path
        max_model_len,
        clean_logits,  # unused: handled internally via masking
    ):
        return _fp8_paged_mqa_logits_sm12x(
            q, kv_cache, weights, context_lens, block_tables, max_model_len
        )

    patches = {
        "get_paged_mqa_logits_metadata": _sm12x_get_paged_mqa_logits_metadata,
        "fp8_fp4_mqa_logits": _sm12x_fp8_fp4_mqa_logits,
        "fp8_fp4_paged_mqa_logits": _sm12x_fp8_fp4_paged_mqa_logits,
    }

    for name, fn in patches.items():
        setattr(deep_gemm_mod, name, fn)

    for modname in (
        "vllm.v1.attention.backends.mla.indexer",
        "vllm.model_executor.layers.sparse_attn_indexer",
    ):
        try:
            import importlib

            mod = importlib.import_module(modname)
            for name, fn in patches.items():
                if hasattr(mod, name):
                    setattr(mod, name, fn)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("patch_deep_gemm_ops: could not patch %s: %s", modname, exc)

    _APPLIED = True
    logger.info(
        "Patched vllm.utils.deep_gemm (%s) in deep_gemm + indexer.py + "
        "sparse_attn_indexer.py: sm12x Triton/torch fallbacks active, no "
        "DeepGEMM native calls.",
        ", ".join(patches),
    )
    return True


try:
    apply()
except Exception as exc:  # pragma: no cover - never hard-fail import
    logger.warning("patch_deep_gemm_ops auto-apply failed: %s", exc)
