#!/bin/bash
# 4-node data-parallel speculator finetune. Runs on spark-4 (rank 0 / master).
# Env knobs: CAPD (capture dir), CUT_OVERRIDE (0 = no mtime filter),
#            STITCH=1 (stream-stitched loader), MIN_ANCHOR, STEPS.
set -e
CAPD=${CAPD:-$HOME/dspark-capture}
CUT=${CUT_OVERRIDE:-$(stat -c %Y ~/dspark-ft.log)}
WORKERS=(${WORKER_IPS:-10.0.0.2 10.0.0.3 10.0.0.4})  # EDIT or export WORKER_IPS
NN=$((1 + ${#WORKERS[@]}))
echo "capture dir: $CAPD  cutoff: $CUT  stitch: ${STITCH:-0}  nnodes: $NN"
SSHO="-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8"

rm -f /tmp/ddp-sync-fail.*
for ip in "${WORKERS[@]}"; do
  ( ssh $SSHO bird@$ip "mkdir -p $CAPD $HOME/dspark-ft-out" \
    && rsync -a -e "ssh $SSHO" $CAPD/ bird@$ip:$CAPD/ \
    && scp -q $SSHO $HOME/dspark_finetune.py bird@$ip:$HOME/dspark_finetune.py \
    && echo "synced $ip" || touch /tmp/ddp-sync-fail.$ip ) &
done
wait
ls /tmp/ddp-sync-fail.* >/dev/null 2>&1 && { echo "SYNC FAILED: $(ls /tmp/ddp-sync-fail.*)"; exit 1; }
echo "=== data synced; launching ranks ==="

run_node() {  # $1 ip ('' = local), $2 node rank
  local cmd="docker rm -f dspark_ddp >/dev/null 2>&1; nohup docker run --rm --gpus all --network host --name dspark_ddp \
 --cap-add IPC_LOCK --ulimit memlock=-1:-1 --ipc host --shm-size 10gb \
 --device /dev/infiniband:/dev/infiniband \
 -e STEPS=${STEPS:-2000} -e ACCUM=1 -e SAVE_EVERY=250 -e CUTOFF_MTIME=$CUT \
 -e STITCH=${STITCH:-} -e MIN_ANCHOR=${MIN_ANCHOR:-8} \
 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
 -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
 -e NCCL_IB_GID_INDEX=3 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 \
 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
 -v $CAPD:/data:ro \
 -v $HOME/.cache/huggingface/hub/glm52-speculator-dspark:/spec:ro \
 -v $HOME/dspark-ft-out:/out \
 -v $HOME/dspark_finetune.py:/ft.py:ro \
 --entrypoint python3 vllm-node-dspark:latest \
 -m torch.distributed.run --nnodes $NN --node-rank $2 \
 --master-addr ${MASTER_ADDR:-10.0.0.1} --master-port 29777 --nproc-per-node 1 /ft.py \
 > $HOME/dspark-ddp.log 2>&1 < /dev/null & echo launched-rank-$2"
  if [ -z "$1" ]; then bash -c "$cmd"; else ssh $SSHO bird@$1 "$cmd"; fi
}
r=1
for ip in "${WORKERS[@]}"; do
  run_node "$ip" $r
  r=$((r + 1))
done
run_node "" 0
echo "DDP-LAUNCHED"
