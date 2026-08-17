#!/bin/bash
# 4-node data-parallel DSpark speculator finetune on OUR captures. Run on gx10-1 (rank 0).
# Adapted from bird/GLM-spark speculator-training/dspark-ddp.sh (Apache-2.0) to this fleet:
#   the node user, rails 192.168.100.11-14, image vllm-b12x:v0.27.0-pinned, our paths.
# Needs the GPUs: production must be DOWN (hold /var/tmp/glm_launch.lock while it runs).
# Env: CAPD (capture dir, default all sets under dspark-capture), SPEC (drafter dir), STEPS, STITCH=1,
#      MIN_ANCHOR, K (3), WINDOW (1024), LR.
set -euo pipefail
CAPD=${CAPD:-/var/tmp/glm-legacy/hf/dspark-capture/dcp4_redhat}
SPEC=${SPEC:-/home/<node-user>/.cache/huggingface/hub/dspark-quanttrio-ft}
OUT=${OUT:-$HOME/dspark-ft-out}
IMAGE=${IMAGE:-vllm-b12x:v0.27.0-pinned}
WORKERS=(${WORKER_IPS:-192.168.100.12 192.168.100.13 192.168.100.14})
MASTER=${MASTER_ADDR:-192.168.100.11}
NN=$((1 + ${#WORKERS[@]}))
SSHO="-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8"
echo "capture: $CAPD  spec: $SPEC  out: $OUT  nnodes: $NN  steps: ${STEPS:-2000} k=${K:-3} W=${WINDOW:-1024}"
mkdir -p "$OUT"
rm -f /tmp/ddp-sync-fail.*
for ip in "${WORKERS[@]}"; do
  ( ssh $SSHO $ip "mkdir -p $CAPD $HOME/dspark-ft-out $HOME/w" \
    && rsync -a --inplace -e "ssh $SSHO" "$CAPD/" "$ip:$CAPD/" \
    && scp -q $SSHO $HOME/w/dspark_finetune.py "$ip:$HOME/w/dspark_finetune.py" \
    && echo "synced $ip" || touch /tmp/ddp-sync-fail.$ip ) &
done
wait
ls /tmp/ddp-sync-fail.* >/dev/null 2>&1 && { echo "SYNC FAILED: $(ls /tmp/ddp-sync-fail.*)"; exit 1; }
# drop page cache on every node: the captures we just copied count against nothing here, but keep the habit
for ip in "${WORKERS[@]}" ""; do
  if [ -z "$ip" ]; then sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; else ssh $SSHO $ip 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'; fi
done
echo "=== data synced; launching ranks ==="
run_node() {  # $1 ip ('' = local), $2 node rank
  local cmd="docker rm -f dspark_ddp >/dev/null 2>&1; nohup docker run --rm --gpus all --network host --name dspark_ddp \
 --cap-add IPC_LOCK --ulimit memlock=-1:-1 --ipc host --shm-size 10gb \
 --device /dev/infiniband:/dev/infiniband \
 -e STEPS=${STEPS:-2000} -e ACCUM=${ACCUM:-1} -e SAVE_EVERY=${SAVE_EVERY:-250} -e CUTOFF_MTIME=0 \
 -e K=${K:-3} -e WINDOW=${WINDOW:-1024} -e LR=${LR:-1e-5} \
 -e STITCH=${STITCH:-} -e MIN_ANCHOR=${MIN_ANCHOR:-8} \
 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
 -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
 -e NCCL_IB_GID_INDEX=3 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 \
 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
 -v $CAPD:/data:ro -v $SPEC:/spec:ro -v $OUT:/out -v $HOME/w/dspark_finetune.py:/ft.py:ro \
 --entrypoint python3 $IMAGE \
 -m torch.distributed.run --nnodes $NN --node-rank $2 \
 --master-addr $MASTER --master-port 29777 --nproc-per-node 1 /ft.py \
 > $HOME/dspark-ddp.log 2>&1 < /dev/null & echo launched-rank-$2"
  if [ -z "$1" ]; then bash -c "$cmd"; else ssh $SSHO "$1" "$cmd"; fi
}
r=1
for ip in "${WORKERS[@]}"; do run_node "$ip" $r; r=$((r + 1)); done
run_node "" 0
echo "DDP-LAUNCHED (logs: ~/dspark-ddp.log on each node; output: $OUT/model.safetensors on rank 0)"
