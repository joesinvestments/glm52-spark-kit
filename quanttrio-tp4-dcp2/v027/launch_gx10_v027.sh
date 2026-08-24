#!/usr/bin/env bash
#
# launch.sh — start GLM-5.2 (TP=4) across a 4-node GB10 / DGX Spark cluster.
#
# Derived from launch.sh in CosmicRaisins/glm-5.2-gb10
# (https://github.com/CosmicRaisins/glm-5.2-gb10), Copyright (c) CosmicRaisins,
# licensed under the Apache License, Version 2.0. Adaptations for the unpruned
# QuantTrio Int4-Int8Mix checkpoint at 200K context. See NOTICE.
#
# Self-contained: a plain `docker run` per node, no external harness. Multi-node
# is vLLM's NATIVE mechanism (--nnodes/--node-rank/--master-addr/--master-port);
# the workers (rank >= 1) start headless, then the head (rank 0) serves the API.
# There is NO Ray and NO shared-filesystem requirement — each node just needs the
# weights present locally (see WEIGHTS_DIR) and the kernels deployed to
# KERNELS_DIR (see README step "Kernels").
#
# Run from the HEAD node (NODES[0]). Workers are reached over key-based SSH.
#
#   ./launch.sh            # launch
#   ./launch.sh --dry-run  # print the docker commands without running them
#   ./launch.sh --stop     # docker rm -f the container on every node
#
# License: Apache-2.0.
set -uo pipefail

# ============================================================================
# CONFIG — edit these for your cluster
# ============================================================================
# EDIT: RoCE rail IPs, rank 0 (head) FIRST. Run this script from the head node.
# (Our fabric: a CRS812 switch on 192.168.192.0/24, MTU 9000.)
NODES=(192.168.100.11 192.168.100.12 192.168.100.13 192.168.100.14)   # rail IPs: MASTER_ADDR/rank identity
SSH_HOSTS=(gx10-1 gx10-2 gx10-3 gx10-4)                                # ssh targets (head resolves these)

# EDIT: SSH key the head node uses to reach the workers (key-based, no prompt).
SSH_KEY="$HOME/.ssh/id_ed25519"

# EDIT: how to derive the SSH username for a given node IP. If every node uses
# the same user, just `echo` it. Our cluster numbers users by the last octet
# (192.168.192.1 -> tonyspark1, .2 -> tonyspark2, ...), i.e.:
#   node_user() { echo "tonyspark${1##*.}"; }
node_user() { echo "${NODE_USER:-<node-user>}"; }

IMAGE="glm52-v027:fix2"
NAME="vllm_slot"                         # container name on every node
PORT=8210                                # OpenAI API port on the head node
MASTER_PORT=29501                        # vLLM cross-node rendezvous port

# Weights. A directory THAT EXISTS ON EVERY NODE, holding the HF hub layout:
#   $WEIGHTS_DIR/hub/glm52-int4-int8mix       (QuantTrio Int4-Int8Mix weights,
#                                              symlink -> ../glm52-int4-int8mix)
#   $WEIGHTS_DIR/hub/nccl-2.30.4/libnccl.so.2 (LD_PRELOADed NCCL, see README)
# Mounted writable at /cache/huggingface (HF_HOME) so JIT/Triton caches stay local.
# How the weights get onto each node is YOUR choice — per-node copy, rsync, or a
# shared mount pointed here. This script does not assume any of them.
WEIGHTS_DIR="/var/tmp/glm-legacy/hf"

# Per-node directory holding the 10 Triton sparse-MLA kernel .py files. Bound
# file-by-file over the vLLM tree, read-only. A KNOWN-GOOD, MATCHED set of these
# 10 files is now VENDORED in this repo under kernels/ (see README step f) — put
# them here on every node with:  rsync -a kernels/ <user>@$node:~/glm-triton/
# Do NOT mix these with a fresh upstream checkout: sparse_attn_indexer.py and
# deepseek_v2.py are a matched pair (the indexer must export
# fused_indexer_q_rope_quant, which deepseek_v2.py imports). See issue #5.
KERNELS_DIR="/var/tmp/glm-triton-nvfp4"
# ============================================================================

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

DRYRUN=0; STOP=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRYRUN=1 ;;
    --stop)    STOP=1 ;;
    *) die "unknown arg: $a (use --dry-run or --stop)" ;;
  esac
done

[ "${#NODES[@]}" -ge 1 ] || die "NODES is empty"
NNODES="${#NODES[@]}"
HEAD="${NODES[0]}"

if [ "$STOP" = 1 ]; then
  say "stopping '$NAME' on all ${NNODES} nodes"
  for ip in "${SSH_HOSTS[@]}"; do
    ssh -o BatchMode=yes "$ip" "docker rm -f $NAME 2>/dev/null" \
      && printf '   stopped on %s\n' "$ip"
  done
  exit 0
fi

# ----------------------------------------------------------------------------
# Container env. The NCCL_IB_HCA / *_SOCKET_IFNAME values are RoCE-fabric-
# specific: set them for YOUR cluster (HCAs via `ibdev2netdev`, interfaces via
# `ip link`). Marked EDIT.
ENVV=(
  -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
  -e "HF_HOME=/cache/huggingface"
  -e "TRITON_CACHE_DIR=/cache/huggingface/.tritoncache"
  -e "HF_HUB_OFFLINE=1"
  -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  -e "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256"
  -e "GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192"
  -e "TORCH_CUDA_ARCH_LIST=12.1a"
  -e "NCCL_NET=IB"
  -e "NCCL_IB_DISABLE=0"
  # NOTE non-uniform clusters: if a node's fabric identity is NOT the same port
  # on every host (e.g. head on rocep1s0f1/enp1s0f1np1, workers on
  # rocep1s0f0/enp1s0f0np0), lift these three -e lines out of ENVV and inject
  # them per rank in docker_run_cmd() from arrays indexed by node-rank, e.g.:
  #   HCAS=(rocep1s0f1 rocep1s0f0 rocep1s0f0 rocep1s0f0)
  #   IFACES=(enp1s0f1np1 enp1s0f0np0 enp1s0f0np0 enp1s0f0np0)
  -e "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0"          # EDIT: your RoCE HCA (`ibdev2netdev`)
  -e "NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0"  # EDIT: your fabric interface (`ip link`)
  -e "GLOO_SOCKET_IFNAME=enp1s0f0np0"  # EDIT: same fabric interface
  -e "NCCL_IB_GID_INDEX=3"             # EDIT if your RoCEv2 GID differs (`show_gids`)
  -e "NCCL_MAX_NCHANNELS=4"
  -e "NCCL_MIN_NCHANNELS=4"
  -e "NCCL_CROSS_NIC=1"
  -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_IGNORE_CPU_AFFINITY=1"
  -e "NCCL_DEBUG=WARN"
  -e "VLLM_MARLIN_USE_ATOMIC_ADD=1"
  -e "VLLM_USE_BREAKABLE_CUDAGRAPH=0"
  -e "VLLM_USE_DEEP_GEMM=0"
)

# Triton sparse-MLA kernels, bound read-only over the vLLM tree (matches
# GLM52_BIND_HOST_TRITON=1). Paths are inside the image's vLLM install.
MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
OPS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/deepseek_v4_ops"
LAYERS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"
MODELS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"
KMOUNTS=()

# ----------------------------------------------------------------------------
# Kernel preflight — FAIL CLOSED (issue #5). deepseek_v2.py imports
# fused_indexer_q_rope_quant from sparse_attn_indexer.py; if the indexer overlay
# is missing or version-skewed and doesn't export it, EVERY rank dies at model
# registration with a cryptic ImportError before any API comes up. Catch it here
# on the head instead of discovering it four containers deep. The known-good
# MATCHED set is vendored in this repo under kernels/ (rsync it to $KERNELS_DIR
# on every node — README step f). NOTE: this validates the HEAD's $KERNELS_DIR
# only; make sure the SAME set is present on all nodes.
say "0.27 native tree: no kernel overlay, no preflight needed"

# docker run base — IB passthrough is REQUIRED (without --device=/dev/infiniband
# + IPC_LOCK + memlock, NCCL silently drops to TCP: ~12 vs 30+ tok/s).
#
# --restart unless-stopped (on EVERY node): if the engine dies mid-serve — e.g.
# the long-context Triton fault in issue #6, where the head process exits 0 and
# would otherwise stay dead forever — Docker relaunches the container. Because
# every node carries the policy, all four re-rendezvous fresh instead of leaving
# a half-dead cluster. Recovery is a full reload (~15-20 min over the fabric),
# so this is a self-heal safety net, not a substitute for root-causing #6. To
# take the cluster down deliberately, use `--stop` — it `docker rm -f`s the
# container on every node, and a removed container is not restarted.
BASE=(
  --restart unless-stopped
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --entrypoint vllm --network host --ipc host --shm-size 10gb --gpus all
  --device /dev/infiniband:/dev/infiniband
  -v "$WEIGHTS_DIR:/cache/huggingface"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
)

# The serving command, with {port} resolved.
# NOTE: --max-num-seqs 6 requires the indexer MTP-overhang patch baked into the
# image (patches/fix-indexer-mtp-overhang.py, README step h) — unpatched vLLM
# crashes at >= 3 concurrent requests with MTP enabled.
SERVE=(
  serve /cache/huggingface/hub/glm52-quanttrio-unpruned
  --served-model-name glm-5.2-quanttrio --host 0.0.0.0 --port "$PORT"
  --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --enable-prefix-caching
  --hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
  --async-scheduling
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_tensor_parallel_size":1,"quantization":"compressed-tensors","draft_sample_method":"probabilistic"}'
  --tensor-parallel-size 4 --pipeline-parallel-size 1
  --max-model-len 200000 --max-num-seqs 6 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.91 --kv-cache-memory-bytes 10950000000
  --kv-cache-dtype fp8_ds_mla
  --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[3,6,9,12,15,18],"max_cudagraph_capture_size":18}'
)

# Optional: decode-aware prefill scheduling (see mods/decode-aware-scheduler).
# Fixes the prefill-induced decode stall: a long prefill freezing active decode
# streams (head-of-line blocking). OFF by default (zero change to the above).
# REQUIRES the mod baked into the image (bash mods/decode-aware-scheduler/run.sh);
# without it these flags are rejected. Enable + tune via env:
#   DECODE_AWARE=1 [DPTB=256] [LPT=2048] [IPTB=8192] ./launch.sh
# DPTB (decode-prefill-token-budget) is the dial: lower = smoother decode under
# load / higher prefill TTFT; higher = the reverse. See mods/ README for numbers.
if [ "${DECODE_AWARE:-0}" = 1 ]; then
  SERVE+=(
    --enable-chunked-prefill
    --enable-decode-aware-prefill
    --long-prefill-token-threshold "${LPT:-2048}"
    --decode-prefill-token-budget "${DPTB:-256}"
    --idle-prefill-token-budget "${IPTB:-8192}"
  )
  say "decode-aware prefill ENABLED (budget ${DPTB:-256}, threshold ${LPT:-2048}, idle ${IPTB:-8192})"
fi

# Build the full `docker run` for a given rank, as a single shell-quoted string.
docker_run_cmd() {
  local rank="$1" headless="$2"
  local cmd=(docker run -d --name "$NAME" "${BASE[@]}" "${ENVV[@]}" "${KMOUNTS[@]}"
             -e "NODE_RANK=$rank" -e "MASTER_ADDR=$HEAD"
             "$IMAGE" "${SERVE[@]}"
             --nnodes "$NNODES" --node-rank "$rank" --master-addr "$HEAD" --master-port "$MASTER_PORT")
  [ "$headless" = 1 ] && cmd+=(--headless)
  # printf %q on each token yields a paste-safe, correctly-quoted command line.
  local out="" t
  for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done
  printf '%s' "${out# }"
}

say "GLM-5.2 launch: ${NNODES} nodes, head=$HEAD:$PORT, image=$IMAGE"
[ "$DRYRUN" = 1 ] && echo "   (dry-run — nothing will be executed)"

# Workers first (rank 1..N-1, headless), then the head (rank 0).
for ((rank=1; rank<NNODES; rank++)); do
  w="${SSH_HOSTS[$rank]}"
  run="$(docker_run_cmd "$rank" 1)"
  shell="docker rm -f $NAME 2>/dev/null; $run"
  if [ "$DRYRUN" = 1 ]; then
    printf '\n# worker %s (rank %d, headless)\nssh %s %q\n' "$w" "$rank" "$w" "$shell"
  else
    printf '   worker %s rank=%d (headless)\n' "$w" "$rank"
    ssh -o BatchMode=yes "$w" "$shell" \
      || die "worker launch failed on $w"
  fi
done

run="$(docker_run_cmd 0 0)"
shell="docker rm -f $NAME 2>/dev/null; $run"
if [ "$DRYRUN" = 1 ]; then
  printf '\n# head %s (rank 0)\n%s\n' "$HEAD" "$shell"
  exit 0
fi
printf '   head %s rank=0\n' "$HEAD"
bash -c "$shell" || die "head launch failed"

say "launched"
echo "   poll:  curl -s http://localhost:$PORT/v1/models"
echo "   logs:  docker logs -f $NAME   (on the head node)"
echo "   stop:  ./launch.sh --stop"
echo "   Ready in ~12 min load + ~10 min cudagraph warmup; serves as 'glm-5.2'."
