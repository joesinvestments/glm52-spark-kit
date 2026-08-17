#!/usr/bin/env bash
# Wrapper: resolve RoCEv2 GID dynamically on all nodes, verify agreement, inject into
# launch_gx10.sh, then exec it. GID indexes are NOT stable across boots (10h outage lesson).
set -euo pipefail
# single-flight: two recovery brains (a cron relauncher, the watchdog, a human) must never stack 98 GB loads
exec 9>/var/tmp/glm_launch.lock
if ! flock -n 9; then echo "another GLM launch is already in progress (lock /var/tmp/glm_launch.lock); refusing to stack"; exit 3; fi
# in-progress guard: never tear down a boot that is still coming up
age=$(docker inspect vllm_slot --format "{{.State.StartedAt}}" 2>/dev/null || true)
if [ -n "$age" ] && [ "$(docker inspect vllm_slot --format "{{.State.Status}}" 2>/dev/null)" = "running" ]; then
  started=$(date -u -d "${age%%.*}" +%s 2>/dev/null || echo 0); now=$(date +%s)
  if [ "$started" -gt 0 ] && [ $((now-started)) -lt 3600 ]; then
    if curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:8210/v1/models | grep -q 200; then echo "already serving; nothing to do"; exit 0; fi
    echo "a boot is in progress (container up $(( (now-started)/60 )) min < 60); refusing to tear it down. Set FORCE_RELAUNCH=1 to override."
    [ "${FORCE_RELAUNCH:-0}" = "1" ] || exit 4
  fi
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
NODES=(LOCAL gx10-2 gx10-3 gx10-4)
resolve() {
  RCMD='
  B=/sys/class/infiniband/rocep1s0f0/ports/1
  IP=$(ip -4 -o addr show enp1s0f0np0 | awk "{print \$4}" | cut -d/ -f1); [ -n "$IP" ] || { echo NOIP; exit; }
  SUF=$(printf "%02x%02x:%02x%02x" $(echo $IP | tr "." " "))
  for i in $(ls $B/gids 2>/dev/null | sort -n); do
    [ "$(cat $B/gid_attrs/types/$i 2>/dev/null)" = "RoCE v2" ] || continue
    [ "$(cat $B/gids/$i 2>/dev/null)" = "0000:0000:0000:0000:0000:ffff:$SUF" ] && { echo $i; exit; }
  done; echo NOTFOUND'
  if [ "$1" = LOCAL ]; then bash -c "$RCMD"; else ssh -o BatchMode=yes "$1" "$RCMD"; fi
}
G=""
for n in "${NODES[@]}"; do
  g=$(resolve "$n"); case "$g" in NOIP|NOTFOUND|"") echo "FATAL: $n GID unresolvable ($g)"; exit 2;; esac
  [ -z "$G" ] && G="$g"
  [ "$g" != "$G" ] && { echo "FATAL: GID disagreement ($G vs $g on $n)"; exit 2; }
done
echo "GID index = $G (all nodes agree)"
sed -i "s|NCCL_IB_GID_INDEX=[0-9]*\"|NCCL_IB_GID_INDEX=$G\"|" "$HERE/launch_gx10.sh"
# DRY_RUN passes straight through (recovery-fidelity diffs need the argv, not a boot).
if [ "${DRY_RUN:-0}" = "1" ]; then exec bash "$HERE/launch_gx10.sh" "$@"; fi

# Sequential boot retry. The DCP init path dies to a gloo race on ~30-40% of
# launches; a single attempt therefore fails a third of the time, and the
# sentinel used to make exactly one attempt then wait 30 minutes. This loop is
# the ONE place every launcher (sentinel, RECOVER.sh, manual) goes through, so
# all of them inherit it. Strictly sequential: concurrent retries each load
# ~98GB and once drove all four nodes to memory exhaustion (power cycle).
NODES_ALL="gx10-2 gx10-3 gx10-4"
API=http://127.0.0.1:8210/v1/models
GEN=http://127.0.0.1:8210/v1/chat/completions
MAX_ATTEMPTS=${LAUNCH_MAX_ATTEMPTS:-4}
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "launch attempt $attempt/$MAX_ATTEMPTS $(date -u +%H:%M:%S)"
  docker rm -f vllm_slot o14_test >/dev/null 2>&1 || true
  sudo -n find /dev/shm -maxdepth 1 \( -name "psm_*" -o -name "sem.mp-*" \) -delete 2>/dev/null || true
  for n in $NODES_ALL; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" 'docker rm -f vllm_slot o14_test >/dev/null 2>&1; sudo -n find /dev/shm -maxdepth 1 \( -name "psm_*" -o -name "sem.mp-*" \) -delete 2>/dev/null; true' >/dev/null 2>&1 &
  done; wait
  low=$(free -g | awk 'NR==2{print $7}')
  for n in $NODES_ALL; do
    m=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" 'free -g | awk "NR==2{print \$7}"' 2>/dev/null)
    [ -n "$m" ] && [ "$m" -lt "$low" ] && low=$m
  done
  if [ "${low:-0}" -lt 100 ]; then echo "  memory not reclaimed (lowest ${low}Gi); waiting 30s"; sleep 30; continue; fi
  bash "$HERE/launch_gx10.sh" "$@" || true
  ok=""
  for i in $(seq 1 240); do   # 60 min: a cold-cache boot after a reboot takes ~27 min; never kill a live load
    if curl -s --max-time 5 "$API" 2>/dev/null | grep -q '"id"'; then
      # /v1/models answers 200 with a dead engine; require a real generation.
      if curl -s --max-time 120 "$GEN" -H 'Content-Type: application/json' \
          -d '{"model":"glm-5.2-quanttrio","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' 2>/dev/null \
          | grep -q '"completion_tokens"'; then ok=1; break; fi
    fi
    st=$(docker ps -a --filter name=vllm_slot --format '{{.Status}}' 2>/dev/null)
    case "$st" in Up*) ;; *) echo "  attempt $attempt died before serving: $st"
        # keep the evidence before the next attempt removes the container
        for n in LOCAL gx10-2 gx10-3 gx10-4; do
          if [ "$n" = LOCAL ]; then docker logs --tail 120 vllm_slot > "$HOME/glm_death_attempt${attempt}_gx10-1.log" 2>&1
          else ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" "docker logs --tail 120 vllm_slot 2>&1" > "$HOME/glm_death_attempt${attempt}_$n.log" 2>/dev/null; fi
        done
        echo "  evidence: $HOME/glm_death_attempt${attempt}_*.log"
        grep -hE "Error|error|Traceback|Assertion|Killed|OOM|timeout|Timeout" "$HOME"/glm_death_attempt${attempt}_*.log 2>/dev/null | grep -viE "Duplicate NCCL|import_utils|Error ignored|deep_ep|ModuleNotFound|SymmMem" | tail -6 | cut -c1-200 | sed "s/^/    /"
        break;; esac
    sleep 15
  done
  [ -n "$ok" ] && { echo "SERVING after $attempt attempt(s) $(date -u +%H:%M:%S)"; exit 0; }
done
echo "LAUNCH FAILED after $MAX_ATTEMPTS attempts"; exit 1

