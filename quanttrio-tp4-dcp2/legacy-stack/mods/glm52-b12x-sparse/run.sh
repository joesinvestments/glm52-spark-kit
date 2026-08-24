#!/usr/bin/env bash
#
# ATTRIBUTION: This mod script was shared by NVIDIA forum user ciprianveg
# (thread 374125, post #34, "glm52-sparse.zip"):
# https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125/34
# It replicates the mods from CosmicRaisins' glm-5.2-gb10 repository
# (https://github.com/CosmicRaisins/glm-5.2-gb10), licensed Apache-2.0.
# Included here verbatim (this header added) for reproducibility.
#
# glm52-b12x-sparse — install b12x for graph-safe sparse MLA decode
#
# b12x provides a CUDA-graph-safe decode kernel for GLM-5.2 sparse MLA.
# WITHOUT it, cudagraph_mode: FULL crashes (torch.full under capture).
# WITH it, FULL works and gives ~20 t/s decode on 4× GB10.
#
# If b12x is not available (pip, network), gracefully skip — the Triton
# fallback in sm12x_sparse_mla_attn.py handles decode without b12x.
set -euo pipefail

echo "=== glm52-b12x-sparse ==="

# Try to install b12x
if python3 -c "import b12x" 2>/dev/null; then
    echo "  ✓ b12x already installed"
else
    echo "  Installing b12x ..."
    if pip install --no-deps b12x==0.23.0 2>&1; then
        echo "  ✓ b12x pip install succeeded"
    else
        echo "  ✗ b12x pip install failed"
    fi
fi

# Verify b12x import works after install (with debug on failure)
B12X_IDX=""
if python3 -c "import b12x; print(b12x.__file__)" 2>/tmp/b12x_import_err.txt; then
    B12X_IDX=$(python3 -c "import b12x; print(b12x.__file__)" | sed 's|__init__.py||')
    echo "  ✓ b12x import OK at: ${B12X_IDX}"
else
    echo "  ✗ b12x import FAILED after install:"
    cat /tmp/b12x_import_err.txt 2>/dev/null || true
    echo "  Debug: python3 -c 'import site; print(site.getsitepackages())'"
    python3 -c "import site; print(site.getsitepackages())" 2>/dev/null || true
    echo "  Debug: pip show b12x"
    pip show b12x 2>/dev/null || true
    echo "  Debug: ls /usr/local/lib/python3.12/dist-packages/b12x*"
    ls /usr/local/lib/python3.12/dist-packages/b12x* 2>/dev/null || true
fi

# Patch b12x fused_indexer for GLM score mode (if b12x is installed)
if [[ -n "$B12X_IDX" && -f "${B12X_IDX}fused_indexer.py" ]]; then
    if grep -q "glm52_patch" "${B12X_IDX}fused_indexer.py" 2>/dev/null; then
        echo "  ✓ fused_indexer.py already patched"
    else
        cp "${B12X_IDX}fused_indexer.py" "${B12X_IDX}fused_indexer.py.bak"
        # GLM uses ReLU scores, not softmax — patch score mode
        python3 -c "
with open('${B12X_IDX}fused_indexer.py', 'r') as f:
    c = f.read()
# Add GLM score mode support
if 'glm52_patch' not in c:
    c += '''

# glm52_patch: GLM score mode (ReLU, not softmax)
def _glm52_score_mode():
    import os
    return os.getenv(\"GLM52_B12X_SCORE_MODE\", \"relu\")

'''
    with open('${B12X_IDX}fused_indexer.py', 'w') as f:
        f.write(c)
print('  ✓ fused_indexer.py patched for GLM')
"
    fi
else
    # NOTE (issue #5): this is EXPECTED and HARMLESS on b12x==0.23.0, which ships
    # no fused_indexer.py. This optional patch only adds a GLM ReLU-score-mode
    # helper; it is NOT what provides fused_indexer_q_rope_quant. That symbol
    # comes from the vendored kernels/sparse_attn_indexer.py overlay (bind-mounted
    # by launch.sh), NOT from b12x. If you hit a fused_indexer_q_rope_quant
    # ImportError, this skip is a red herring — check that kernels/ is the matched
    # vendored set on every node (see README step f), not your b12x install.
    echo "  ⓘ b12x 0.23.0 has no fused_indexer.py — skipping OPTIONAL GLM score-mode helper (expected; harmless — see issue #5)"
fi

echo "=== glm52-b12x-sparse complete ==="
