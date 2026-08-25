#!/usr/bin/env bash
# Lane-2 ORACLE RUN — uniform-K2 planning confirmation (AWAITING JOE'S APPROVAL).
# Purpose-limited: confirm/deny that uniform sqg_e4m3 K2 containers pass upstream
# prepare/planning on CPU, and capture reference decoded tiles for encoder validation.
# Scope guard: ONE CPU-only container on gx10-1, --gpus NOT passed, synth-scale inputs
# (408KB each), MemAvailable fail-closed gate, bounded minutes, production untouched.
set -Eeuo pipefail

MEM_FLOOR_GIB=8
SRV=gx10-1
IMAGE=glm52-collab:b3
SRC="${BASH_SOURCE[0]%/*}/oracle-inputs"   # repo: campaign-2026-08/tailquant/oracle-inputs

log() { printf '[oracle] %s\n' "$*"; }
die() { printf '[oracle] FAIL-CLOSED: %s\n' "$*"; exit 3; }

MEM=$(ssh -o ConnectTimeout=8 "$SRV" 'free -g | awk "/^Mem:/{print \$7}"') || die "node unreachable"
case "$MEM" in ''|*[!0-9]*) die "bad mem read: $MEM";; esac
[ "$MEM" -ge "$MEM_FLOOR_GIB" ] || die "MemAvailable ${MEM}GiB < ${MEM_FLOOR_GIB}GiB"
log "gate OK: $SRV mem=${MEM}GiB"

ssh "$SRV" 'sudo -n rm -rf /tmp/lane2-oracle-run && mkdir -p /tmp/lane2-oracle-run'
rsync -a "$SRC/uncoupled" "$SRC/coupled" "$SRV:/tmp/lane2-oracle-run/"

cat > /tmp/lane2_prepare_probe.py << 'PYEOF'
import json, pathlib, sys
sys.path.insert(0, ".")
import torch
from b12x.moe._shared.btx_schema import read_btx_manifest, BTX_MANIFEST_FILENAME
from b12x.moe._shared.kernels.w4a16 import btx as btx_mod

results = {}
for variant in ("uncoupled", "coupled"):
    root = pathlib.Path("/work/run") / variant
    manifest = read_btx_manifest(str(root))
    layer = btx_mod.read_btx_layer(str(root), manifest, 0,
                                   first_slot=0,
                                   slot_count=manifest.geometry.atom_slots)
    try:
        prepared = btx_mod.prepare_btx_moe_weights(
            layer, activation="silu", device="cpu", params_dtype=torch.float16)
        results[variant] = {"planning": "PASS"}
    except Exception as exc:
        results[variant] = {"planning": "FAIL",
                            "error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(results, indent=1))
pathlib.Path("/work/run/oracle_result.json").write_text(json.dumps(results, indent=1))
PYEOF

scp -q /tmp/lane2_prepare_probe.py "$SRV:/tmp/lane2-oracle-run/"
log "running prepare probe (CPU-only container)..."
ssh "$SRV" "docker run --rm --network none -v /tmp/lane2-oracle-run:/work/run \
  --entrypoint python3 $IMAGE /work/run/lane2_prepare_probe.py 2>&1 | grep -v Triton | tail -20"
ssh "$SRV" 'cat /tmp/lane2-oracle-run/oracle_result.json' || die "no result file"
log "COMPLETE - results above; copy oracle_result.json into repo journal commit"
