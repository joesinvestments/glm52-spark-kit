#!/usr/bin/env bash
set -uo pipefail
NODES=(${NODES:?"set NODES=\"ip1 ip2 ip3 ip4\" (rail IPs, rank order)"})
SSH_HOSTS=(gx10-1 gx10-2 gx10-3 gx10-4)
IMAGE="vllm-b12x:v0.27.0-pinned"
NAME="vllm_slot"
PORT=8210
MASTER_PORT=29501
WEIGHTS_DIR="/var/tmp/glm-legacy/hf"
OVERLAY_DIR="/var/tmp/o14-test-overlay"
NNODES=4
HEAD="${NODES[0]}"

ENVV=(
  -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
  -e "LD_PRELOAD=/cache/huggingface/hub/nccl-2.30.4/libnccl.so.2"
  -e "HF_HOME=/cache/huggingface"
  -e "TRITON_CACHE_DIR=/cache/huggingface/.tritoncache_o14test"
  -e "HF_HUB_OFFLINE=1"
  -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  -e "TORCH_CUDA_ARCH_LIST=12.1a"
  -e "NCCL_NET=IB" -e "NCCL_IB_DISABLE=0"
  -e "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0"
  -e "NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0"
  -e "GLOO_SOCKET_IFNAME=enp1s0f0np0"
  -e "NCCL_IB_GID_INDEX=3"
  -e "NCCL_MAX_NCHANNELS=4" -e "NCCL_MIN_NCHANNELS=4"
  -e "NCCL_CROSS_NIC=1" -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_IGNORE_CPU_AFFINITY=1" -e "NCCL_DEBUG=WARN"
  -e "VLLM_MARLIN_USE_ATOMIC_ADD=1"
  -e "VLLM_ONE_GPU_PER_NODE=1" -e "B12X_CUTE_COMPILE_CACHE_DIR=/cache/huggingface/.b12x_cute_cache" -e "CUTE_DSL_CACHE_DIR=/cache/huggingface/.cute_dsl_cache"
  -e "KV_FP8_ROPE=1"
  -e "PYTHONUNBUFFERED=1"
  -e "VLLM_MOE_MARLIN_ATOMIC_ADD=1"
  -e "R17_DRAFT_TEMP_SCALE=1.0"
  -e "VLLM_BUILDA_BMM=1"
  -e "KV_FP8_ROPE=1"
  -e "PYTHONUNBUFFERED=1"
  -e "VLLM_USE_B12X_SPARSE_INDEXER=1"
)

MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
BACKENDS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends"
FM="/usr/local/lib/python3.12/dist-packages/vllm/model_executor"
CFG="/usr/local/lib/python3.12/dist-packages/vllm/config"
COMP="/usr/local/lib/python3.12/dist-packages/vllm/compilation"
OPS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops"
KMOUNTS=(
  -v "$OVERLAY_DIR/registry.py:$BACKENDS/registry.py:ro"
  -v "$OVERLAY_DIR/parallel_state.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.py:ro"
  -v "$OVERLAY_DIR/shm_broadcast.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py:ro"
  -v "$OVERLAY_DIR/torch_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/torch_utils.py:ro"
  -v "$OVERLAY_DIR/kv_cache_interface.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py:ro"
  -v "$OVERLAY_DIR/sparse_mla_scratch.py:/usr/local/lib/python3.12/dist-packages/b12x/integration/sparse_mla_scratch.py:ro"
  -v "$OVERLAY_DIR/nvfp4_ds_mla_writer.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/nvfp4_ds_mla_writer.py:ro"
  -v "$OVERLAY_DIR/b12x_mla_sparse.py:$MLA/b12x_mla_sparse.py:ro"
  -v "$OVERLAY_DIR/indexer.py:$MLA/indexer.py:ro"
  -v "$OVERLAY_DIR/cache.py:$CFG/cache.py:ro"
  -v "$OVERLAY_DIR/b12x_capture.py:$COMP/b12x_capture.py:ro"
  -v "$OVERLAY_DIR/patch_deep_gemm_ops.py:$MLA/patch_deep_gemm_ops.py:ro"
  -v "$OVERLAY_DIR/sm12x_deep_gemm_fallbacks.py:$MLA/sm12x_deep_gemm_fallbacks.py:ro"
  -v "$OVERLAY_DIR/sm12x_mqa.py:$MLA/sm12x_mqa.py:ro"
  -v "$OVERLAY_DIR/deepseek_v4_ops:$OPS/deepseek_v4_ops:ro"
  -v "$OVERLAY_DIR/marlin_moe.py:$FM/layers/fused_moe/experts/marlin_moe.py:ro"
  -v "$OVERLAY_DIR/speculator.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/speculator.py:ro"
  -v "$OVERLAY_DIR/mla_attention.py:$FM/layers/attention/mla_attention.py:ro"
  -v "$OVERLAY_DIR/logits_processor.py:$FM/layers/logits_processor.py:ro"
  -v "$OVERLAY_DIR/builda_bmm_v0.py:$FM/layers/attention/builda_bmm_v0.py:ro"
  -v "$OVERLAY_DIR/builda_bmm_v1.py:$FM/layers/attention/builda_bmm_v1.py:ro"
)

BASE=(
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --network host --ipc host --shm-size 10gb --gpus all
  --device /dev/infiniband:/dev/infiniband
  -v "$WEIGHTS_DIR:/cache/huggingface"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
)

# First test: eager mode, no speculative decoding, modest context. Isolate
# whether base sparse-MLA-on-0.27 boots before adding complexity.
SERVE=(
  /cache/huggingface/hub/glm52-quanttrio-unpruned
  --served-model-name glm-5.2-quanttrio --host 0.0.0.0 --port "$PORT"
  --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --enable-prefix-caching
  --attention-backend B12X_MLA_SPARSE --disable-custom-all-reduce
  --tensor-parallel-size 4 --pipeline-parallel-size 1
  --decode-context-parallel-size 4
  --dcp-comm-backend a2a
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_tensor_parallel_size":1,"attention_backend":"B12X_MLA_SPARSE","quantization":"compressed-tensors","draft_sample_method":"probabilistic"}'
  --max-model-len 315968 --max-num-seqs 16 --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.90
  --kv-cache-dtype nvfp4_ds_mla
  --distributed-executor-backend mp
  --cpu-distributed-timeout-seconds 1800
  --safetensors-prefetch-num-threads 16
  --watermark 0.02
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[3,6,9,12]}'
)

docker_run_cmd() {
  local rank="$1" headless="$2"
  local cmd=(docker run -d --name "$NAME" "${BASE[@]}" "${ENVV[@]}" "${KMOUNTS[@]}"
             -e "NODE_RANK=$rank" -e "MASTER_ADDR=$HEAD"
             -e "VLLM_HOST_IP=${NODES[$rank]}"
             "$IMAGE" "${SERVE[@]}"
             --nnodes "$NNODES" --node-rank "$rank" --master-addr "$HEAD" --master-port "$MASTER_PORT")
  [ "$headless" = 1 ] && cmd+=(--headless)
  local out="" t
  for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done
  printf '%s' "${out# }"
}

# DRY_RUN=1 prints the head argv and launches NOTHING. This guard MUST stay
# above the worker loop: a version that guarded only the head still ran the
# worker loop, which killed and relaunched the live workers on three nodes.
if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '%s\n' "$(docker_run_cmd 0 0)"
  exit 0
fi
echo "== GLM-5.2 champion launch: B12X + DCP=4 + MTP2, port $PORT =="
# NOTE: master-first ordering was TESTED and did not help (4 clean / 4 failed,
# vs 5 failed / 13 before). Reverted to the original ordering that produced every
# validated measurement in this campaign. The gloo race remains unfixed.
for ((rank=1; rank<NNODES; rank++)); do
  w="${SSH_HOSTS[$rank]}"
  run="$(docker_run_cmd "$rank" 1)"
  shell="docker rm -f $NAME 2>/dev/null; $run"
  echo "   worker $w rank=$rank"
  ssh -o BatchMode=yes "$w" "$shell" || echo "WORKER LAUNCH FAILED on $w"
done
run="$(docker_run_cmd 0 0)"
shell="docker rm -f $NAME 2>/dev/null; $run"
echo "   head $HEAD rank=0"
# Topology-aware: this launcher is run BOTH from a workstation (where gx10-1 is
# an ssh-config alias) and ON the head itself by the sentinel's recovery path
# (where that alias does not resolve). Detect whether the head rail IP is local.
if ip -4 -o addr show 2>/dev/null | grep -q " ${HEAD}/"; then
  bash -c "$shell" || echo "HEAD LAUNCH FAILED"
else
  ssh -o BatchMode=yes "${SSH_HOSTS[0]}" "$shell" || echo "HEAD LAUNCH FAILED"
fi
echo "== launched, container IDs above =="
