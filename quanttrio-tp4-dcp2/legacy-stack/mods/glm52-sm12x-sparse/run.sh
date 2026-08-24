#!/usr/bin/env bash
#
# ATTRIBUTION: This mod script was shared by NVIDIA forum user ciprianveg
# (thread 374125, post #34, "glm52-sparse.zip"):
# https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125/34
# It replicates the mods from CosmicRaisins' glm-5.2-gb10 repository
# (https://github.com/CosmicRaisins/glm-5.2-gb10), licensed Apache-2.0.
# Included here verbatim (this header added) for reproducibility.
#
# glm52-sm12x-sparse — install Triton sparse-MLA kernels + DeepGEMM bypass
#
# Replicates the CosmicRaisins/mods/glm52-sm12x-sparse mod:
#   1. Copies kernel .py files into vllm/v1/attention/backends/mla/
#   2. Creates vllm/v1/attention/ops/deepseek_v4_ops/ package from kernel files
#      (sm12x_mqa.py → sm12x_mqa.py, b12x_sparse_helpers.py → b12x_sparse_helpers.py)
#   3. Patches vllm/utils/deep_gemm.py to route to sm12x fallbacks on SM12x
#   4. Patches sparse_attn_indexer.py to not require has_deep_gemm() on SM12x
set -euo pipefail

MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
OPS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops"
DEEP_GEMM="/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py"
SPARSE_IDX="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py"
KERNELS="/root/models/models15/glm-triton"

echo "=== glm52-sm12x-sparse ==="

# ---------------------------------------------------------------------------
# Step 1: Copy kernel files into MLA backend directory
# ---------------------------------------------------------------------------
echo "Installing kernel files into $MLA ..."
for f in flashmla_sparse.py sm12x_deep_gemm_fallbacks.py sm12x_sparse_mla_attn.py \
         sparse_mla_kernels.py sparse_mla_env.py patch_flashmla_ops.py; do
    if [[ -f "$KERNELS/$f" ]]; then
        cp "$KERNELS/$f" "$MLA/$f"
        echo "  ✓ $f"
    else
        echo "  ⚠ $f not found in $KERNELS"
    fi
done

# ---------------------------------------------------------------------------
# Step 2: Create deepseek_v4_ops package from kernel files
# The sm12x_deep_gemm_fallbacks.py imports from deepseek_v4_ops.sm12x_mqa
# and deepseek_v4_ops.b12x_sparse_helpers. These are the SAME kernel files
# placed in a different directory to match the import path.
# ---------------------------------------------------------------------------
DVO="$OPS/deepseek_v4_ops"
mkdir -p "$DVO"

# __init__.py
cat > "$DVO/__init__.py" << 'EOF'
# Auto-created by glm52-sm12x-sparse mod — provides deepseek_v4_ops package
# for SM12x Triton sparse-MLA kernels.
EOF
echo "  ✓ deepseek_v4_ops/__init__.py"

# sm12x_mqa.py — the core Triton MQA logits kernels
if [[ -f "$KERNELS/sm12x_mqa.py" ]]; then
    cp "$KERNELS/sm12x_mqa.py" "$DVO/sm12x_mqa.py"
    echo "  ✓ deepseek_v4_ops/sm12x_mqa.py"
else
    echo "  ⚠ sm12x_mqa.py not found"
fi

# b12x_sparse_helpers.py — optional b12x decode integration
if [[ -f "$KERNELS/b12x_sparse_helpers.py" ]]; then
    cp "$KERNELS/b12x_sparse_helpers.py" "$DVO/b12x_sparse_helpers.py"
    echo "  ✓ deepseek_v4_ops/b12x_sparse_helpers.py"
else
    echo "  ⚠ b12x_sparse_helpers.py not found"
fi

# ---------------------------------------------------------------------------
# Step 3: Patch deep_gemm.py — route MQA logits to sm12x fallbacks on SM12x
# This is the CRITICAL fix: before _lazy_init() tries to import deep_gemm
# (which doesn't exist on SM12x), we intercept and set the sm12x fallbacks
# as the implementations.
# ---------------------------------------------------------------------------
echo "Patching $DEEP_GEMM ..."

if grep -q "glm52_sm12x_patch" "$DEEP_GEMM" 2>/dev/null; then
    echo "  ✓ Already patched"
else
    cp "$DEEP_GEMM" "${DEEP_GEMM}.bak"
    cat >> "$DEEP_GEMM" << 'PATCH_EOF'


# === glm52 SM12x DeepGEMM bypass — appended by glm52-sm12x-sparse mod ===
# On SM12x (GB10 / DGX Spark), the deep_gemm C extension is not installed.
# The CosmicRaisins Triton kernels provide equivalent implementations.
# This patch intercepts _lazy_init() to register the fallbacks before
# the _missing() gate fires.
try:
    from vllm.platforms import current_platform as _glm52_platform
    if _glm52_platform.is_cuda() and _glm52_platform.is_device_capability_family(120):
        import torch as _glm52_torch
        # Import the SM12x Triton fallbacks
        from vllm.v1.attention.backends.mla.sm12x_deep_gemm_fallbacks import (
            _fp8_mqa_logits_sm12x,
            _fp8_paged_mqa_logits_sm12x,
            _tf32_hc_prenorm_gemm_sm12x,
        )
        from vllm.v1.attention.ops.deepseek_v4_ops.sm12x_mqa import (
            fp8_mqa_logits_triton,
            fp8_paged_mqa_logits_triton,
            tf32_hc_prenorm_gemm_triton,
            _view_packed_fp8_paged_mqa_kv_cache,
        )

        # Override has_deep_gemm to return True so SparseAttnIndexer can init
        import vllm.utils.import_utils as _glm52_iu
        _orig_has_deep_gemm = _glm52_iu.has_deep_gemm
        def _glm52_has_deep_gemm() -> bool:
            return True
        _glm52_iu.has_deep_gemm = _glm52_has_deep_gemm
        has_deep_gemm = _glm52_has_deep_gemm

        # Register SM12x implementations BEFORE _lazy_init runs
        _fp8_fp4_mqa_logits_impl = _fp8_mqa_logits_sm12x

        def _glm52_paged_mqa_logits(q, kv_cache, weights, context_lens,
                                     block_tables, schedule_metadata,
                                     max_model_len, clean_logits=False):
            return _fp8_paged_mqa_logits_sm12x(
                q, kv_cache, weights, context_lens, block_tables, max_model_len
            )
        _fp8_fp4_paged_mqa_logits_impl = _glm52_paged_mqa_logits

        def _glm52_get_paged_mqa_logits_metadata(context_lens, block_size, num_sms):
            return _glm52_torch.zeros((num_sms + 1, 2), dtype=_glm52_torch.int32,
                                       device=context_lens.device)
        _get_paged_mqa_logits_metadata_impl = _glm52_get_paged_mqa_logits_metadata
        _tf32_hc_prenorm_gemm_impl = _tf32_hc_prenorm_gemm_sm12x

        # Monkey-patch the module-level names so _lazy_init fast-path triggers
        import vllm.utils.deep_gemm as _glm52_dg
        _glm52_dg._fp8_fp4_mqa_logits_impl = _fp8_fp4_mqa_logits_impl
        _glm52_dg._fp8_fp4_paged_mqa_logits_impl = _fp8_fp4_paged_mqa_logits_impl
        _glm52_dg._get_paged_mqa_logits_metadata_impl = _get_paged_mqa_logits_metadata_impl
        _glm52_dg._tf32_hc_prenorm_gemm_impl = _tf32_hc_prenorm_gemm_impl
        _glm52_dg.has_deep_gemm = _glm52_has_deep_gemm

        print("[glm52-sm12x-sparse] DeepGEMM bypass registered for SM12x")
except Exception as _e:
    print(f"[glm52-sm12x-sparse] patch failed: {_e}")

# === end glm52 SM12x DeepGEMM bypass ===
PATCH_EOF
    echo "  ✓ deep_gemm.py patched"
fi

# ---------------------------------------------------------------------------
# Step 4: Patch sparse_attn_indexer.py — remove has_deep_gemm() gate on SM12x
# The SparseAttnIndexer.__init__ raises if has_deep_gemm() is False.
# We patch it to skip the check on SM12x (where we use Triton fallbacks).
# ---------------------------------------------------------------------------
echo "Patching $SPARSE_IDX ..."

if grep -q "glm52_sm12x_patch" "$SPARSE_IDX" 2>/dev/null; then
    echo "  ✓ Already patched"
else
    cp "$SPARSE_IDX" "${SPARSE_IDX}.bak"
    # Replace the has_deep_gemm() gate with an SM12x-aware version
    python3 -c "
import re
with open('$SPARSE_IDX', 'r') as f:
    content = f.read()

# The gate is:
#   if current_platform.is_cuda() and not has_deep_gemm():
#       raise RuntimeError(...)
# Replace with: skip on SM12x
old = '''        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                \"Sparse Attention Indexer CUDA op requires DeepGEMM support in \"
                \"the current vLLM environment.\"
            )'''
new = '''        # glm52-sm12x-sparse: skip has_deep_gemm check on SM12x
        # (Triton fallbacks provide the ops via deep_gemm.py monkeypatch)
        if current_platform.is_cuda() and not has_deep_gemm():
            from vllm.platforms import current_platform as _cp
            if not (_cp.is_cuda() and _cp.is_device_capability_family(120)):
                raise RuntimeError(
                    \"Sparse Attention Indexer CUDA op requires DeepGEMM support in \"
                    \"the current vLLM environment.\"
                )'''
content = content.replace(old, new)

# Also mark as patched
content = content + '\n\n# glm52_sm12x_patch: applied by glm52-sm12x-sparse mod\n'

with open('$SPARSE_IDX', 'w') as f:
    f.write(content)
print('  ✓ sparse_attn_indexer.py patched')
"
fi

echo "=== glm52-sm12x-sparse complete ==="
