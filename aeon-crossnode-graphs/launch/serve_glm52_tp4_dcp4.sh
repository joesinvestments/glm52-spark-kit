#!/usr/bin/env bash
# GLM-5.2 (QuantTrio Int4-Int8Mix, unpruned) on 4x DGX Spark, TP=4 + DCP=4, on the platform image
# (AEON v0.27.1 sm_121a base + b12x + overlays baked). This is the exact profile that captured
# PIECEWISE + FULL decode CUDA graphs cross-node and passed the correctness gate; see ../FINDING.md.
# Run on the head node. Fill in the four rail IPs and the weights dir for your fleet.
set -uo pipefail
NODES=(${NODE_IPS:-10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4})   # 200G rail IPs, head first
IMAGE="${IMAGE:-glm52-spark-platform:aeon-0.27.1}"
NAME="vllm_glm52"
PORT=${PORT:-8210}
MASTER_PORT=29501
WEIGHTS_DIR="${WEIGHTS_DIR:-$HOME/.cache/huggingface}"   # holds hub/glm52-quanttrio-unpruned and hub/nccl-2.30.4/libnccl.so.2
MODEL_DIR=/cache/huggingface/hub/glm52-quanttrio-unpruned
NNODES=4
HEAD="${NODES[0]}"
ENVV=(
  -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
  -e "LD_PRELOAD=/cache/huggingface/hub/nccl-2.30.4/libnccl.so.2"   # NCCL 2.30.4, the known-good line for multi-node GB10 RoCE
  -e "HF_HOME=/cache/huggingface"
  -e "TRITON_CACHE_DIR=/cache/huggingface/.tritoncache"
  -e "HF_HUB_OFFLINE=1"
  -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  -e "TORCH_CUDA_ARCH_LIST=12.1a"
  -e "NCCL_NET=IB" -e "NCCL_IB_DISABLE=0"
  -e "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0"                # both QSFP rails
  -e "NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0"
  -e "GLOO_SOCKET_IFNAME=enp1s0f0np0"
  -e "NCCL_IB_GID_INDEX=3"                                 # RoCEv2 GID; verify with show_gids on your nodes
  -e "NCCL_MAX_NCHANNELS=4" -e "NCCL_MIN_NCHANNELS=4"
  -e "NCCL_CROSS_NIC=1" -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_IGNORE_CPU_AFFINITY=1" -e "NCCL_DEBUG=WARN"     # the measured run logged INFO/INIT,NET to a host file; WARN is fine for serving
  -e "VLLM_MARLIN_USE_ATOMIC_ADD=1"                        # Marlin small-batch reduce (measured on GB10)
  -e "VLLM_ONE_GPU_PER_NODE=1"                             # DCP init: one GPU per node, skips the intra-node gloo race
  -e "B12X_CUTE_COMPILE_CACHE_DIR=/cache/huggingface/.b12x_cute_cache"
  -e "CUTE_DSL_CACHE_DIR=/cache/huggingface/.cute_dsl_cache"
  -e "KV_FP8_ROPE=1"
  -e "PYTHONUNBUFFERED=1"
  -e "VLLM_MOE_MARLIN_ATOMIC_ADD=1"                        # unknown to vLLM 0.27 (harmless warning); kept so the argv matches the measured run
  -e "R17_DRAFT_TEMP_SCALE=1.0"                            # 0xdfi overlay knob; 1.0 is a no-op
  -e "VLLM_BUILDA_BMM=0"
  -e "VLLM_USE_B12X_SPARSE_INDEXER=1"
)
BASE=(
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --network host --ipc host --shm-size 10gb --gpus all
  --device /dev/infiniband:/dev/infiniband
  -v "$WEIGHTS_DIR:/cache/huggingface"
)
SERVE=(
  "$MODEL_DIR"
  --served-model-name glm-5.2-quanttrio --host 0.0.0.0 --port "$PORT"
  --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --enable-prefix-caching
  --attention-backend B12X_MLA_SPARSE
  --disable-custom-all-reduce                              # removes a 300 s MNNVL/TCPStore stall at init on this fleet
  --tensor-parallel-size 4 --pipeline-parallel-size 1
  --decode-context-parallel-size 4 --dcp-comm-backend a2a
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_tensor_parallel_size":1,"attention_backend":"B12X_MLA_SPARSE","quantization":"compressed-tensors","draft_sample_method":"probabilistic"}'
  --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048
  --gpu-memory-utilization 0.90
  --kv-cache-dtype nvfp4_ds_mla
  --distributed-executor-backend mp
  --cpu-distributed-timeout-seconds 1800                   # workers wait for the head's weight load; 600 is too short at 4 nodes
  --safetensors-prefetch-num-threads 16
  --watermark 0.02
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[3,6,9,12]}'   # multiples of 1+k for MTP k=2
)
docker_run_cmd() {  # rank headless
  local rank="$1" headless="$2"
  local cmd=(docker run -d --name "$NAME" "${BASE[@]}" "${ENVV[@]}"
             -e "NODE_RANK=$rank" -e "MASTER_ADDR=$HEAD" -e "VLLM_HOST_IP=${NODES[$rank]}"
             --entrypoint vllm "$IMAGE" serve "${SERVE[@]}"                # AEON image entrypoint is /bin/bash
             --nnodes "$NNODES" --node-rank "$rank" --master-addr "$HEAD" --master-port "$MASTER_PORT")
  [ "$headless" = 1 ] && cmd+=(--headless)
  local out="" t; for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done; echo "$out"
}
for h in "${NODES[@]}"; do ssh -o BatchMode=yes "$h" "docker rm -f $NAME >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null" ; done
# workers first (reverse rank), head last: gloo connectFullMesh retries only 3x
for r in 3 2 1; do ssh -o BatchMode=yes "${NODES[$r]}" "$(docker_run_cmd $r 1)"; sleep 5; done
eval "$(docker_run_cmd 0 0)"
echo "launched; watch: docker logs -f $NAME  |  ready when: curl -s http://127.0.0.1:$PORT/v1/models"
