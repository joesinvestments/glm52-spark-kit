#!/usr/bin/env bash
# Build the collab candidate images on gx10-1.
# Usage: bash build_images.sh          (from the collab-work/ directory on the Mac)
#
# Safe to run while ornith_slot serves: docker build does not touch running containers.
# Disk cost: ~2 image layers on top of the existing vllm/vllm-openai:v0.27.0 base.
set -Eeuo pipefail

MAC_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # collab-work/
REMOTE=gx10-1
REMOTE_DIR='~/glm52-collab-build'

echo "== syncing build context to $REMOTE:$REMOTE_DIR"
KIT_DIR="$(cd "$MAC_DIR/../glm52-spark-kit" && pwd)"
ssh "$REMOTE" "mkdir -p $REMOTE_DIR"
rsync -a --delete "$MAC_DIR"/images "$MAC_DIR"/patches "$REMOTE:$REMOTE_DIR/"
rsync -a --delete \
  --include='MANIFEST.json' --include='overlays/***' --exclude='*' \
  "$KIT_DIR"/ "$REMOTE:$REMOTE_DIR/kit/"
scp -q "$MAC_DIR"/images/bake_overlays.py "$REMOTE:$REMOTE_DIR/kit/"

echo "== verifying base image present"
ssh "$REMOTE" 'docker image inspect vllm-b12x:v0.27.0-pinned --format "base {{.Id}}"'

for ps in "${@:-b4}"; do
  echo "== building glm52-collab:$ps (PATCHSET=$ps)"
  ssh "$REMOTE" "cd $REMOTE_DIR && docker build \
    --build-arg PATCHSET=$ps --build-arg RAY_SPEC=${RAY_SPEC:-ray[cgraph]} \
    -f images/Dockerfile \
    -t glm52-collab:$ps ."
done

echo "== smoke import test (tiny container, ~200MB, safe alongside ornith_slot)"
for tag in b0 b2; do
  ssh "$REMOTE" "docker run --rm --entrypoint python3 glm52-collab:$tag \
    -c 'import vllm.config.compilation; print(\"$tag imports OK\")'"
done

echo "== done. Images ready on gx10-1:"
ssh "$REMOTE" 'docker images glm52-collab --format "{{.Repository}}:{{.Tag}} {{.Size}}"'