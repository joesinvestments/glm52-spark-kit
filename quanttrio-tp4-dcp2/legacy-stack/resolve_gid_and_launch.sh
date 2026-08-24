#!/usr/bin/env bash
# Wrapper: resolve RoCEv2 GID dynamically on all nodes, verify agreement, inject into
# launch_gx10.sh, then exec it. GID indexes are NOT stable across boots (10h outage lesson).
set -euo pipefail
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
exec bash "$HERE/launch_gx10.sh" "$@"
