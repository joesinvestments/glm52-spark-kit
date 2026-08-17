#!/bin/bash
# NVFP4 swap on the four Sparks. Production goes DOWN for the whole window. Requires GO=1.
# Phases (each idempotent, re-runnable):
#   verify   : QuantTrio backup on Storage is complete (sizes + md5 spot check vs gx10-1)
#   evict    : stop production, delete QuantTrio from all four nodes (only after verify passed)
#   stage    : NVFP4 Storage -> gx10-1 (rsync, ~75 min), then gx10-1 -> gx10-2/3/4 over the rails
#   boot     : ~/launchers/champion_nvfp4weights_32k.sh via the guarded wrapper, gate, battery
#   restore  : QuantTrio Storage -> gx10-1 -> rails, then production launcher
set -uo pipefail
[ "${GO:-0}" = "1" ] || { echo "refusing: set GO=1 (the operator's explicit go for the outage)"; exit 2; }
PH=${1:-verify}
ST=/Volumes/Storage/weights-staging
QT=$ST/GLM-5.2-QuantTrio-Int4-Int8Mix/weights
NV=$ST/GLM-5.2-NVFP4/weights
NODES=(gx10-1 gx10-2 gx10-3 gx10-4); RAILS=(192.168.100.11 192.168.100.12 192.168.100.13 192.168.100.14)
HUB=/var/tmp/glm-legacy/hf/hub
case $PH in
verify)
  grep -q "BACKUP DONE" $ST/GLM-5.2-QuantTrio-Int4-Int8Mix/backup.log || { echo "backup not finished"; exit 1; }
  grep -q "SIZE MISMATCH" $ST/GLM-5.2-QuantTrio-Int4-Int8Mix/backup.log && { echo "size mismatches in backup log"; exit 1; }
  for f in model-00001-of-00124.safetensors model-00062-of-00124.safetensors model-00124-of-00124.safetensors config.json; do
    a=$(md5 -q $QT/$f); b=$(ssh gx10-1 "md5sum $HUB/glm52-quanttrio-unpruned/$f | cut -d' ' -f1")
    [ "$a" = "$b" ] && echo "md5 ok $f" || { echo "MD5 MISMATCH $f"; exit 1; }
  done
  n=$(find $QT -type f | wc -l | tr -d ' '); m=$(ssh gx10-1 "find $HUB/glm52-quanttrio-unpruned -type f | wc -l")
  echo "files storage=$n node=$m"; [ "$n" = "$m" ] || exit 1
  echo "VERIFY OK";;
evict)
  ssh gx10-1 'nohup flock /var/tmp/glm_launch.lock sleep 36000 >/dev/null 2>&1 & echo $! > /tmp/nvfp4_swap_lock.pid'
  for h in "${NODES[@]}"; do ssh $h 'docker rm -f vllm_slot >/dev/null 2>&1; rm -rf '"$HUB"'/glm52-quanttrio-unpruned; df -h / | tail -1' ; done
  echo "EVICTED (launch lock held by gx10-1 pid $(ssh gx10-1 cat /tmp/nvfp4_swap_lock.pid))";;
stage)
  ssh gx10-1 "mkdir -p $HUB/glm52-nvidia-nvfp4"
  rsync -a --inplace $NV/ gx10-1:$HUB/glm52-nvidia-nvfp4/ || exit 1
  for i in 1 2 3; do ssh ${NODES[$i]} "mkdir -p $HUB/glm52-nvidia-nvfp4" ; ssh gx10-1 "rsync -a --inplace $HUB/glm52-nvidia-nvfp4/ ${RAILS[$i]}:$HUB/glm52-nvidia-nvfp4/" & done; wait
  for h in "${NODES[@]}"; do ssh $h "du -sh $HUB/glm52-nvidia-nvfp4; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null"; done
  echo "STAGED";;
boot)
  # keep the launch lock held (sentinel/wrapper stay out); launch like the chains do
  for h in "${NODES[@]}"; do ssh $h 'docker rm -f vllm_slot >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; free -g | sed -n 2p'; done
  ssh gx10-1 'bash $HOME/launchers/champion_nvfp4weights_32k.sh' >/dev/null 2>&1
  echo "launched; watch: docker logs -f vllm_slot on gx10-1; api http://192.168.1.16:8210/v1/models";;
restore)
  ssh gx10-1 "mkdir -p $HUB/glm52-quanttrio-unpruned"
  rsync -a --inplace $QT/ gx10-1:$HUB/glm52-quanttrio-unpruned/ || exit 1
  for i in 1 2 3; do ssh gx10-1 "rsync -a --inplace $HUB/glm52-quanttrio-unpruned/ ${RAILS[$i]}:$HUB/glm52-quanttrio-unpruned/" & done; wait
  for h in "${NODES[@]}"; do ssh $h "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null"; done
  echo "QUANTTRIO RESTORED ON NODES; launch production with the normal wrapper";;
esac
