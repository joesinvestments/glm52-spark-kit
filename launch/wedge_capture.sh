#!/usr/bin/env bash
# wedge_capture.sh: run ONCE per detected wedge, before any recovery action.
# Implements the discriminator from vllm-project/vllm#51921 (marksunner):
#   1. NCCL RAS per-rank collective op counts, sampled twice >= 90 s apart
#      (RAS listens on 127.0.0.1:28028 inside each rank process; NCCL >= 2.24).
#      frozen counts, spread 0-1 ops, no in-flight collective  -> kernel-upstream
#      livelock (#49026 class); the wedged step's collective was never launched.
#      counts advanced with a rank parked mid-collective        -> genuine divergence.
#   2. nvidia-smi on all four nodes at both samples: ~96 % util at ~18 W with 0 %
#      memory utilisation and zero movement is a spin loop reporting busy.
#   3. Container/engine state and the last 200 log lines per rank.
# Everything lands in $OUT (default ~/.gx10-watchdog/wedges/<utc-stamp>/) so a
# recovery that follows cannot destroy the evidence.
#
# A plain RAS status query reads "RUNNING OK" during this wedge (8/8 of their
# captures); only the op-count DELTA between the two samples discriminates.
set -uo pipefail
NODES=(${NODES:-192.168.100.11 192.168.100.12 192.168.100.13 192.168.100.14})  # rail IPs: the head cannot ssh its own hostname
CONTAINER="${CONTAINER:-vllm_slot}"
GAP="${GAP:-95}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="${OUT:-$HOME/.gx10-watchdog/wedges/$STAMP}"
mkdir -p "$OUT"
NCCLRAS_HOST=/var/tmp/ncclras            # built from NCCL src/ras/client.cc on each node
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$OUT/capture.log"; }

sample() {  # $1 = sample tag (s1|s2)
  local tag=$1
  for h in "${NODES[@]}"; do
    if hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "$h"; then RUN="bash -c"; else RUN="ssh -o ConnectTimeout=15 -o BatchMode=yes $h"; fi
    $RUN "
      echo '### nvidia-smi'; nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,memory.used --format=csv,noheader 2>&1
      echo '### ras (host client, ports 28028.. inside net=host container are host-visible)'
      if [ -x $NCCLRAS_HOST ]; then timeout 25 $NCCLRAS_HOST -f json -vv 2>&1; else echo 'ncclras missing on host'; fi
      echo '### container'; docker inspect $CONTAINER --format '{{.State.Status}} oom={{.State.OOMKilled}} pid={{.State.Pid}}' 2>&1
    " > "$OUT/$tag.$h.txt" 2>&1 &
  done
  wait
  log "sample $tag written"
}

log "wedge capture start -> $OUT"
running=$(curl -s -m 5 "http://${API_HOST:-127.0.0.1}:${API_PORT:-8210}/metrics" 2>/dev/null | awk '/^vllm:num_requests_running/{s+=$2} END{print s+0}')
log "requests running at capture start: ${running:-?} (frozen counters are only a wedge signature when this is > 0; an idle fleet is frozen by definition)"
sample s1
for h in "${NODES[@]}"; do
  if hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "$h"; then docker logs --tail 200 $CONTAINER > "$OUT/logtail.$h.txt" 2>&1 &
  else ssh -o ConnectTimeout=15 -o BatchMode=yes "$h" "docker logs --tail 200 $CONTAINER 2>&1" > "$OUT/logtail.$h.txt" 2>&1 & fi
done; wait
log "waiting ${GAP}s for the second RAS/nvidia-smi sample"
sleep "$GAP"
sample s2

# Verdict helper: per-rank collective_counts deltas between the two RAS JSON samples.
for h in "${NODES[@]}"; do
  python3 - "$OUT/s1.$h.txt" "$OUT/s2.$h.txt" "$h" <<'PYX' | tee -a "$OUT/capture.log"
import json,sys
def load(p):
    raw=open(p).read(); i=raw.find("{"); j=raw.rfind("}")
    try: return json.loads(raw[i:j+1])
    except Exception: return None
a,b,h=load(sys.argv[1]),load(sys.argv[2]),sys.argv[3]
if not a or not b: print(f"{h}: RAS json missing in a sample"); sys.exit(0)
def counts(d):
    out={}
    for c in d.get("communicators",[]):
        for r in c.get("ranks",[]):
            out[(c.get("hash"),r.get("rank"))]=sum(int(v) for v in r.get("collective_counts",{}).values())
    return out
ca,cb=counts(a),counts(b)
moved=sum(1 for k in ca if cb.get(k,0)!=ca[k]); tot=len(ca)
comms={k[0] for k in cb}
spread=max((max(cb[k] for k in cb if k[0]==c)-min(cb[k] for k in cb if k[0]==c)) for c in comms) if cb else 0
print(f"{h}: ranks x comms={tot} moved={moved} max_intra_comm_spread={spread}  ({'FROZEN -> kernel-upstream livelock signature' if moved==0 else 'ADVANCED -> not a frozen livelock'})")
PYX
done
log "verdict rule: all counters UNCHANGED + high util at low power + 0% mem util = kernel-upstream livelock (#49026 class); counters ADVANCED with one rank parked = real collective divergence"
echo "$OUT"
