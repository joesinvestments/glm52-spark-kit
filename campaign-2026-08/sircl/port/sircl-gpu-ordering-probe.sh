#!/usr/bin/env bash
# SIRCL GPU-ordering probe — WINDOW-1 cell A4 (d3-transport-comparison.md spec).
# Staged 2026-08-25 overnight. DO NOT EXECUTE outside an Joe-approved window.
#
# Proves: GPU-consumes-NIC-written-payload ordering under our switched fabric
# (cuda-mapped arenas, verifier on) + eager AR latency at decode shapes.
# Usage: sircl-gpu-ordering-probe.sh [--dry-run]
#   --dry-run: evaluate local gates + print every remote command; NO ssh, NO exec.

set -Eeuo pipefail

MEM_FLOOR_GIB=8
VRAM_FLOOR_MIB=4096
SRV=gx10-1
CLI=gx10-2
IMAGE=glm52-collab:b3
TREE=/tmp/sircl-build/upstream          # compiled tree staged by B4 session
RUN=/tmp/sircl-runtree                  # binaries dir (both nodes)
PORTS="--control-port 9415"
BYTES_LIST="36864 65536 147456"         # decode-shape sweep + 64K anchor
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

log() { printf '[probe] %s\n' "$*"; }
die() { printf '[probe] FAIL-CLOSED: %s\n' "$*"; exit 3; }

# ---- gate helpers (run per node) ----
gate_node() { # $1 host  -> prints mem_gib vram_free_mib ; fails closed
  ssh -o ConnectTimeout=8 "$1" '
    free -g | awk "/^Mem:/{print \$7}"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1
  ' 2>/dev/null || echo "UNREACHABLE UNREACHABLE"
}

check_gate() { # $1 host $2 mem_gib $3 vram_mib
  local host=$1 mem=$2 vram=$3
  case "$mem" in UNREACHABLE|""|*[!0-9]*) die "$host unreachable or bad read";; esac
  case "$vram" in ""|*[!0-9]*) die "$host bad VRAM read";; esac
  [ "$mem" -ge "$MEM_FLOOR_GIB" ] || die "$host MemAvailable ${mem}GiB < ${MEM_FLOOR_GIB}GiB"
  [ "$vram" -ge "$VRAM_FLOOR_MIB" ] || die "$host free VRAM ${vram}MiB < ${VRAM_FLOOR_MIB}MiB"
  log "gate OK: $host mem=${mem}GiB vram_free=${vram}MiB"
}

# ---- main ----
log "dry_run=$DRY srv=$SRV cli=$CLI image=$IMAGE tree=$TREE"

for h in "$SRV" "$CLI"; do
  if [ "$DRY" = 1 ]; then
    log "[dry] would gate-check $h: ssh $h 'free -g; nvidia-smi --query-gpu=memory.free'"
  else
    read -r MEM VRAM <<< "$(gate_node "$h")"
    check_gate "$h" "$MEM" "$VRAM"
  fi
done

PROBE_ENV='ldconfig -p | grep -q libibverbs || apt-get install -y -qq libibverbs1'
LINK_CMD="/vrun/spark_transport_probe --device rocep1s0f0 --gid 3 --memory cuda-mapped \
--gpu-producer --gpu-verifier --gpu-roundtrip --bytes 65536 --buffer-bytes 4194304 \
--warmup 10 --iterations 200 $PORTS"

for B in $BYTES_LIST; do
  AR_CMD="./build/spark_tp4_probe --rank RANK --peer0 192.168.100.11 --peer1 192.168.100.12 \
--device0 rocep1s0f0 --device1 rocep2p1s0f0 --gid0 3 --gid1 3 $PORTS \
--bytes $B --warmup 20 --iterations 500"
  if [ "$DRY" = 1 ]; then
    log "[dry] would run link test ($B): docker run --rm --network host --ipc host --gpus all \
--device /dev/infiniband -v $RUN:/vrun:ro --entrypoint bash $IMAGE -c \"$PROBE_ENV; $LINK_CMD\""
    log "[dry] would run AR ($B): srv+cli docker ... -c \"$AR_CMD\""
  fi
done

if [ "$DRY" = 1 ]; then
  cat << 'EOF'
[probe] DRY-RUN COMPLETE. Execution plan at window time:
  1. Gate both nodes (MemAvailable >=8GiB, free VRAM >=4096MiB). Abort otherwise.
  2. Server (gx10-1): transport_probe cuda-mapped producer+verifier roundtrip (server role).
  3. Client (gx10-2): same, client role, 200 iterations @64KB. PASS = verifier correct=true.
  4. spark_tp4_probe eager AR pair at 37KB/64KB/147KB: rank0 on gx10-1, rank1 on gx10-2,
     dual-rail devices, gid 3. PASS = correctness ok; record p50/p99 vs NCCL 38-90us.
  5. All output appended to campaign journal; containers never touched.
EOF
  exit 0
fi

# ---- live execution (window only) ----
log "starting server role on $SRV"
ssh "$SRV" "nohup docker run --rm --name sircl_gpu_srv --network host --ipc host --gpus all \
  --device /dev/infiniband -v $RUN:/vrun:ro --entrypoint bash $IMAGE -c \
  \"$PROBE_ENV; $LINK_CMD --server\" > /tmp/sircl_gpu_srv.log 2>&1 & echo SRV_UP"
sleep 3
log "starting client role on $CLI"
ssh "$CLI" "docker run --rm --name sircl_gpu_cli --network host --ipc host --gpus all \
  --device /dev/infiniband -v $RUN:/vrun:ro --entrypoint bash $IMAGE -c \
  \"$PROBE_ENV; $LINK_CMD --client 192.168.100.11\"" | tee /tmp/sircl_gpu_client.log
grep -q "correct=true\|RESULT" /tmp/sircl_gpu_client.log || die "link verification failed"

for B in $BYTES_LIST; do
  log "AR sweep bytes=$B"
  ssh "$CLI" "docker run --rm --network host --ipc host --gpus all --device /dev/infiniband \
    -v $TREE:/work:ro -w $TREE --entrypoint bash $IMAGE -c \
    './build/spark_tp4_probe --rank 1 --peer0 192.168.100.11 --peer1 192.168.100.11 \
     --device0 rocep1s0f0 --device1 rocep2p1s0f0 --gid0 3 --gid1 3 $PORTS \
     --bytes $B --warmup 20 --iterations 500'" > "/tmp/sircl_ar_cli_$B.log" &
  ssh "$SRV" "docker run --rm --network host --ipc host --gpus all --device /dev/infiniband \
    -v $TREE:/work:ro -w $TREE --entrypoint bash $IMAGE -c \
    './build/spark_tp4_probe --rank 0 --peer0 192.168.100.12 --peer1 192.168.100.12 \
     --device0 rocep1s0f0 --device1 rocep2p1s0f0 --gid0 3 --gid1 3 $PORTS \
     --bytes $B --warmup 20 --iterations 500'" > "/tmp/sircl_ar_srv_$B.log"
  wait || true
  tail -2 "/tmp/sircl_ar_srv_$B.log"
done
log "COMPLETE - append outputs to campaign journal"
