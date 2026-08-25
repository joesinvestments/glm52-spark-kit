#!/usr/bin/env bash
# win3_identity_diff.sh - POST-RESTORE IDENTITY DIFF (G-rail, Joe 2026-08-25).
# Compares LIVE vllm_slot argv against EXPECTED identity. Exit 0=verified 1=DIVERGE.
# Usage: win3_identity_diff.sh <expected_k> <expected_ladder_csv> [host]
set -Eeuo pipefail
EXP_K="$1"; EXP_LADDER="$2"; HOST="${3:-gx10-1}"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
ssh -o ConnectTimeout=10 "$HOST" 'docker inspect vllm_slot --format "{{json .Args}}{{println}}{{.Config.Image}}"' > "$TMP" \
  || { echo "FAIL: node/container unreachable"; exit 2; }
python3 - "$EXP_K" "$EXP_LADDER" "$TMP" << 'PYEOF'
import json, sys
exp_k, exp_ladder, path = sys.argv[1], sys.argv[2].replace(" ", ""), sys.argv[3]
lines = open(path).read().splitlines()
args = json.loads(lines[0])
image = lines[1].strip() if len(lines) > 1 else "?"
k = ladder = "?"
for a in args:
    if "num_speculative_tokens" in a:
        k = a.split("num_speculative_tokens\":")[1].split(",")[0]
    if "cudagraph_capture_sizes" in a:
        ladder = a.split("[", 1)[1].split("]", 1)[0].replace(" ", "")
print(f"[identity] live: k={k} ladder=[{ladder}] image={image}")
fail = 0
if k != exp_k:
    print(f"DIVERGE: k live={k} expected={exp_k}"); fail = 1
if ladder != exp_ladder:
    print(f"DIVERGE: ladder live=[{ladder}] expected=[{exp_ladder}]"); fail = 1
if not (image == "vllm-b12x:v0.27.0-pinned" or image.startswith("glm52-collab:b")):
    print(f"DIVERGE: unexpected image {image}"); fail = 1
if fail:
    sys.exit(1)
print(f"[identity] VERIFIED: matches expected ({exp_k} / {exp_ladder})")
PYEOF
