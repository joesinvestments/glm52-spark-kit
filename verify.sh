#!/usr/bin/env bash
# Verify every manifest file's md5 at overlay_root, locally or on each host given.
# Usage: verify.sh [overlay_root] [host ...]      exit 1 on any mismatch (fail closed)
set -uo pipefail
ROOT="${1:-$(cd "$(dirname "$0")" && pwd)/overlays}"; shift || true
MF="$(cd "$(dirname "$0")" && pwd)/MANIFEST.json"
expected=$(python3 -c 'import json,sys
for f in json.load(open(sys.argv[1]))["files"]: print(f["md5"], f["overlay_file"])' "$MF")
rc=0
check() { # $1 = host or "" ; reads "md5 path" pairs
  local host="$1" bad=0
  while read -r h p; do
    if [ -n "$host" ]; then got=$(ssh -o ConnectTimeout=10 "$host" "md5sum '$ROOT/$p' 2>/dev/null" | cut -d' ' -f1)
    else got=$(md5sum "$ROOT/$p" 2>/dev/null | cut -d' ' -f1); fi
    [ "$got" = "$h" ] || { echo "MISMATCH ${host:-local}: $p (want $h got ${got:-absent})"; bad=1; }
  done <<< "$expected"
  [ $bad = 0 ] && echo "OK ${host:-local}: all $(echo "$expected" | wc -l | tr -d ' ') files match"
  return $bad
}
if [ $# -eq 0 ]; then check "" || rc=1; else for h in "$@"; do check "$h" || rc=1; done; fi
exit $rc
