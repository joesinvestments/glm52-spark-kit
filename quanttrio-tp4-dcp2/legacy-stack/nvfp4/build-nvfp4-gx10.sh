#!/usr/bin/env bash
# GLM-5.2 NVFP4 ds_mla — per-node image build + overlay dir.
# Run on EVERY node (deterministic rebuild; no image transfer needed).
# Inputs (in $PORT_DIR): apply_nvfp4_plumbing.py, nvfp4_glm_kernels.py,
#                        b12x-nvfp4/{traits.py,kernel.py,b12x_sparse_helpers.py,...}
# Produces: image glm52-legacy:challenger-nvfp4
#           overlay dir /var/tmp/glm-triton-nvfp4 (KNOWNGOOD /var/tmp/glm-triton untouched)
set -euo pipefail

BASE_IMAGE="glm52-legacy:challenger-v2b"
OUT_TAG="glm52-legacy:challenger-nvfp4"
PORT_DIR="${PORT_DIR:-$HOME/nvfp4-port/port}"
OVERLAY_SRC="/var/tmp/glm-triton"
OVERLAY_DST="/var/tmp/glm-triton-nvfp4"
SP="/usr/local/lib/python3.12/dist-packages"
RUN_GPU_SELFTEST="${RUN_GPU_SELFTEST:-0}"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

[ -f "$PORT_DIR/apply_nvfp4_plumbing.py" ] || die "missing apply_nvfp4_plumbing.py in $PORT_DIR"
[ -f "$PORT_DIR/nvfp4_glm_kernels.py" ] || die "missing nvfp4_glm_kernels.py in $PORT_DIR"
[ -d "$PORT_DIR/b12x-nvfp4" ] || die "missing b12x-nvfp4/ in $PORT_DIR"
[ -d "$OVERLAY_SRC" ] || die "missing KNOWNGOOD overlay dir $OVERLAY_SRC"

say "1/5 overlay dir: $OVERLAY_DST (copy of KNOWNGOOD + nvfp4 patches)"
rm -rf "$OVERLAY_DST"
cp -r "$OVERLAY_SRC" "$OVERLAY_DST"
python3 "$PORT_DIR/apply_nvfp4_plumbing.py" --overlay "$OVERLAY_DST"
# agent-patched b12x_sparse_helpers.py (dual-mode fp8/nvfp4) replaces the overlay copy
cp "$PORT_DIR/b12x-nvfp4/b12x_sparse_helpers.py" "$OVERLAY_DST/b12x_sparse_helpers.py"
python3 -m py_compile "$OVERLAY_DST"/*.py

say "2/5 build container from $BASE_IMAGE"
docker rm -f nvfp4build 2>/dev/null || true
docker run -d --name nvfp4build --entrypoint sleep "$BASE_IMAGE" inf >/dev/null

say "3/5 apply in-image patches"
docker cp "$PORT_DIR/apply_nvfp4_plumbing.py" nvfp4build:/tmp/
docker cp "$PORT_DIR/nvfp4_glm_kernels.py" "nvfp4build:$SP/vllm/v1/attention/backends/mla/nvfp4_glm_kernels.py"
for f in "$PORT_DIR"/b12x-nvfp4/*.py; do
  base="$(basename "$f")"
  case "$base" in
    b12x_sparse_helpers.py) ;;  # overlay-mounted at runtime, handled in step 1
    traits.py|kernel.py) docker cp "$f" "nvfp4build:$SP/b12x/attention/mla/$base" ;;
    fp4.py)              docker cp "$f" "nvfp4build:$SP/b12x/cute/$base" ;;
    *)                   docker cp "$f" "nvfp4build:$SP/b12x/attention/mla/$base" ;;
  esac
done
docker exec nvfp4build python3 /tmp/apply_nvfp4_plumbing.py --image "$SP/vllm" \
  | tee /tmp/nvfp4-plumbing.log
grep -q "PLUMBING OK" /tmp/nvfp4-plumbing.log || die "plumbing patcher failed"

say "4/5 smoke tests (imports + dtype registration; no GPU needed)"
docker exec nvfp4build python3 - <<'PY'
import ast, sys
sp = "/usr/local/lib/python3.12/dist-packages"
# compile-check every file we touched
for f in ["vllm/config/cache.py", "vllm/utils/torch_utils.py",
          "vllm/v1/kv_cache_interface.py",
          "vllm/model_executor/layers/attention/mla_attention.py",
          "vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py",
          "vllm/v1/attention/backends/mla/nvfp4_glm_kernels.py",
          "b12x/attention/mla/traits.py", "b12x/attention/mla/kernel.py",
          "b12x/attention/mla/io.py", "b12x/attention/mla/io_mg.py",
          "b12x/attention/mla/prefill_mg.py", "b12x/attention/mla/nvfp4.py"]:
    ast.parse(open(f"{sp}/{f}").read(), filename=f)
# dtype registered end-to-end
from vllm.config.cache import CacheDType
import typing
assert "nvfp4_ds_mla" in typing.get_args(CacheDType), "CacheDType missing nvfp4_ds_mla"
from vllm.utils.torch_utils import is_quantized_kv_cache, STR_DTYPE_TO_TORCH_DTYPE
assert is_quantized_kv_cache("nvfp4_ds_mla")
assert STR_DTYPE_TO_TORCH_DTYPE["nvfp4_ds_mla"].itemsize == 1
print("SMOKE OK")
PY

if [ "$RUN_GPU_SELFTEST" = "1" ]; then
  say "4b/5 GPU kernel selftest (store/gather roundtrip, tiny tensors)"
  docker exec nvfp4build python3 "$SP/vllm/v1/attention/backends/mla/nvfp4_glm_kernels.py" \
    || die "GPU selftest FAILED"
fi

say "5/5 commit $OUT_TAG"
docker commit \
  --change 'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]' \
  --change 'CMD []' \
  nvfp4build "$OUT_TAG" >/dev/null
docker rm -f nvfp4build >/dev/null
# Verify the tag we actually produced. Previously this grepped a hardcoded 'nvfp4-v1' left
# over from a rename, so under `set -e` a GOOD build exited 1 on a clean node, and on a node
# holding a stale nvfp4-v1 image it printed the OLD image and declared success. Reported by
# ThinkCode in issue #3. A verification step must verify the thing it claims to.
docker image inspect "$OUT_TAG" >/dev/null 2>&1 || die "commit did not produce $OUT_TAG"
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -F "$OUT_TAG"
echo "BUILD OK on $(hostname)"
