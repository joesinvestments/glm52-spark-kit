#!/usr/bin/env bash
# screen_027.sh v2 — unattended 0.27 variable screen with the v3 (work-based) predicate.
# The CONTROL runs first every campaign: if the control does not wedge, the trigger is not
# reproducing and every downstream "SURVIVED" would be meaningless. Fail loud in that case.
set -uo pipefail
OUT=~/Desktop/GLM52-RESTORE-BUNDLE/window-20260810/screen027.jsonl
TRIG=~/Desktop/GLM52-RESTORE-BUNDLE/window-20260810/wedge_trigger.py
say(){ printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"; }
boot_cell(){  # $1 = launcher basename (without .sh), returns 0 if serving
  for n in gx10-1 gx10-2 gx10-3 gx10-4; do
    ssh -o BatchMode=yes "$n" 'docker rm -f vllm_slot >/dev/null 2>&1; sudo -n /usr/local/sbin/gx10-rails.sh >/dev/null 2>&1; sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; true' 2>/dev/null
  done
  ssh -o BatchMode=yes gx10-1 "cd ~/glm-legacy-stack && sed -i 's|launch_gx10.sh|$1.sh|' resolve_gid_and_launch.sh && ./resolve_gid_and_launch.sh >/dev/null 2>&1; sed -i 's|$1.sh|launch_gx10.sh|' resolve_gid_and_launch.sh" 2>/dev/null
  # Stage 1: HTTP up. NOT readiness — /v1/models answers during cudagraph warmup, and a
  # completion then legitimately takes minutes. Judging a cell before it can serve is how
  # warmup gets recorded as a wedge (learned the hard way 2026-08-12).
  local http=0
  for i in $(seq 1 30); do
    sleep 30
    curl -s --max-time 5 http://192.168.1.16:8210/v1/models 2>/dev/null | grep -q '"id"' && { http=1; break; }
    ssh -o BatchMode=yes gx10-1 'docker logs vllm_slot 2>&1 | grep -qE "EngineCore failed|Unsupported architecture|AttributeError|ValueError" && echo X' 2>/dev/null | grep -q X && return 1
  done
  [ "$http" = 1 ] || return 1
  # Stage 2: READY = the engine actually completes a small request. Up to 15 min of warmup.
  READY_S=0
  for i in $(seq 1 15); do
    if curl -s --max-time 60 http://192.168.1.16:8210/v1/chat/completions \
         -H 'Content-Type: application/json' \
         -d '{"model":"glm-5.2-quanttrio","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"chat_template_kwargs":{"enable_thinking":false}}' 2>/dev/null \
         | grep -q '"completion_tokens"'; then
      READY_S=$((i*60)); say "  ready after ~${READY_S}s of warmup"; return 0
    fi
    sleep 15
  done
  say "  NEVER became ready (HTTP up, no completion in 15 min)"
  return 2
}
ssh -o BatchMode=yes gx10-4 'sudo -n systemctl stop gx10-sentinel.service' 2>/dev/null
trap 'ssh -o BatchMode=yes gx10-4 "sudo -n systemctl start gx10-sentinel.service" 2>/dev/null' EXIT

say "=== CONTROL (must wedge, else the trigger is not reproducing) ==="
boot_cell launch_gx10_v027_51538; rc=$?
if [ $rc = 2 ]; then
  echo '{"label":"CONTROL","verdict":"WEDGED_AT_BOOT","detail":"HTTP up, no completion in 15 min"}' | tee -a "$OUT"
  say "CONTROL wedges at boot — trigger confirmed reproducing, continuing"
elif [ $rc = 0 ]; then
  v=$(python3 "$TRIG" CONTROL 1 90 2>/dev/null | tail -1); echo "$v" | tee -a "$OUT"
  echo "$v" | grep -q '"verdict": "WEDGED"' || { say "CONTROL DID NOT WEDGE — aborting, trigger unreliable"; exit 1; }
else
  echo '{"label":"CONTROL","verdict":"BOOT_FAIL"}' | tee -a "$OUT"; exit 1
fi

for cell in "$@"; do
  say "=== CELL $cell ==="
  boot_cell "screen_$cell"; rc=$?
  if [ $rc = 0 ]; then
    say "  serving, running trigger"
    python3 "$TRIG" "$cell" 2 90 2>/dev/null | tail -1 | tee -a "$OUT"
  elif [ $rc = 2 ]; then
    echo "{\"label\":\"$cell\",\"verdict\":\"WEDGED_AT_BOOT\",\"detail\":\"HTTP up but no completion within 15 min of warmup\"}" | tee -a "$OUT"
  else
    err=$(ssh -o BatchMode=yes gx10-1 'docker logs vllm_slot 2>&1 | grep -oE "[A-Za-z]+Error: .*" | sort -u | head -1 | cut -c1-100' 2>/dev/null)
    echo "{\"label\":\"$cell\",\"verdict\":\"BOOT_FAIL\",\"err\":\"${err:-timeout}\"}" | tee -a "$OUT"
  fi
done
say "SCREEN COMPLETE"
