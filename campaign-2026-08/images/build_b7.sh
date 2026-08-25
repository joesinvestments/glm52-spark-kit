#!/usr/bin/env bash
# Stage + build glm52-collab:b7 (D2 parity candidate) on gx10-1.
# RUN ONLY inside an approved between-boot downtime slot (Window K rider).
# Disk: ~2-4G unique layers over the shared vllm-b12x base; 41G free as of staging.
# Usage: build_b7.sh <path-to-glm52-spark-kit-checkout>
set -Eeuo pipefail
REPO="$(cd "$1" && pwd)"
IMG_DIR="$REPO/campaign-2026-08/images"
SRV=gx10-1

echo "== gate: disk headroom on $SRV"
FREE=$(ssh "$SRV" 'df -BG --output=avail / | tail -1 | tr -dc "0-9"')
[ "${FREE%G}" -ge 10 ] || { echo "FAIL-CLOSED: only ${FREE}G free (need >=10G)"; exit 3; }

echo "== syncing build context"
ssh "$SRV" 'sudo -n rm -rf /tmp/b7-context; mkdir -p /tmp/b7-context/kit'
rsync -a "$IMG_DIR/Dockerfile.b7" "$IMG_DIR/bake_overlays.py" "$SRV:/tmp/b7-context/"
rsync -a "$REPO/MANIFEST.json" "$SRV:/tmp/b7-context/kit/"
rsync -a --delete "$REPO/overlays/" "$SRV:/tmp/b7-context/kit/overlays/"
ssh "$SRV" 'mkdir -p /tmp/b7-context/patches'
rsync -a --delete "$REPO/campaign-2026-08/patches/"*.patch "$SRV:/tmp/b7-context/patches/"

echo "== building glm52-collab:b7"
ssh "$SRV" 'cd /tmp/b7-context && docker build -f Dockerfile.b7 -t glm52-collab:b7 . 2>&1 | tail -8'

echo "== import-gate smoke"
ssh "$SRV" 'docker run --rm --entrypoint python3 glm52-collab:b7 -c "
from b12x.attention._shared.mla.api import (
    sparse_mla_decode_forward, sparse_mla_extend_forward)
from b12x.attention.sparse_mla._scratch import (
    B12XSparseMLAScratchCaps, plan_sparse_mla_scratch)
import vllm.config.compilation
print(\"b7 SMOKE PASS\")"' 2>&1 | grep -v Triton | tail -3

echo "== staged image:"
ssh "$SRV" 'docker images glm52-collab --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep b7'
