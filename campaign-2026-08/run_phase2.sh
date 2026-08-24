#!/usr/bin/env bash
# Phase 2 experiment runbook - glm52-collab.
# Stages are deliberately separate commands; each one expects human eyes before the next.
# NEVER run stages concurrently. Ornith must be stopped before any stage that touches GPUs.
#
# Grid (one variable per cell), updated 2026-08-21 after digesting issue #1 thread:
#   B0  public baseline      golden DCP4 1M@2048 argv, BOTH rails, kit HEAD overlays.
#                            Carries the deepseek_mtp.py packed-modules mapping that
#                            stock v0.27.0 lacks (draft-quant fix) - measure accepted/draft
#                            and per-draft-forward cost FIRST; if spec overhead ~32ms/step,
#                            the decode gap is already closed here.
#   B1  = B0 + VLLM_ADAPTIVE_SPEC_DEPTHS=2             (depth-2 pin; +9% peak C4 n=3 on
#                                                        switch fabric; verify transfers here)
#   B0n = B0 with NO speculative-config                (base step-time reference; the
#                                                        spec-overhead A/B answers whether
#                                                        the draft-quant disease applies here:
#                                                        public-tree disease = ~68ms/step added;
#                                                        partner parity = ~32ms/step added)
#   B1b OPTIONAL = B0 + VLLM_FORCE_FUSE_GEMM_COMMS=1   (measured FLAT by repro team - MoE-off
#                                                        by design at O2. Only run if bored.)
#
# PHASE 3 arms (b12x-upgrade port, NVFP4 backend A/B, TailQuant, drafter swap,
# pattern sweep) are specced in collab-work/PHASE3-PLAN.md - read that before
# planning any boot beyond B1.
#
# Expectation from issue #1 + kit COMM_FLOOR: this fleet has the fast (dual-HCA) fabric,
# so B0 should land NEAR published matrix numbers, not at ecohash's 0.25x-prefill outlier.
set -Eeuo pipefail

ENDPOINT_HOST=192.168.100.11     # rank0 rail-A IP (API host)
PORT=8211                        # NOTE: ornith currently serves 8211 too - do not overlap
KIT=$HOME/Desktop/glm52-spark-kit
PROBE="$HOME/Desktop/GLM Collab/collab-work/benchmarks/probe.py"
LAUNCH_REPO="$HOME/Desktop/GLM Collab/glm-5.2-dgx-spark-vllm027"

NODES=(gx10-1 gx10-2 gx10-3 gx10-4)
FABRIC_IPS=(192.168.100.11 192.168.100.12 192.168.100.13 192.168.100.14)
MODEL=/var/tmp/glm-legacy/hf/hub/glm52-quanttrio-unpruned

# Fleet fabric facts (verify uverbs2 index at first use: ls /dev/infiniband/ per node)
RDMA_IFNAME=enp1s0f0np0
IB_HCA="rocep1s0f0,roceP2p1s0f0"
RDMA_DEV_UVERBS=/dev/infiniband/uverbs0
RDMA_DEV_UVERBS2=/dev/infiniband/uverbs1   # CONFIRM which uverbs maps to roceP2p1s0f0

CPUSET_CPUS=5-9,15-19
CPUSET_MEMS=0

# DCP4 1M@2048 preset (recipes/presets.md), allocator budget re-derived per sizing rule
DCP_SIZE=4 MAX_MODEL_LEN=1000000 KV_CACHE_MEMORY_BYTES=7995534848 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 DECODE_PREFILL_TOKEN_BUDGET=1024 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 PROFILE_DRAFTER_CAP=1

stage0_comm_check() {
  echo "== nccl_micro vs reference table (kit docs/COMM_FLOOR-2026-08-19.md)"
  echo "   expect ~90us AR150KB / ~49us AG55KBx4 / ~38us A2A50KB. 2x worse => stop, fix fabric."
  ssh gx10-1 'docker run --rm --network host --gpus all \
    -v "$HOME/Desktop/glm52-spark-kit:/kit:ro" glm52-collab:b0 \
    python3 /kit/benchmarks/comm/nccl_micro.py' 2>&1 | tail -20
}

preflight() {
  for n in "${NODES[@]}"; do
    echo "-- $n: model dir, sidecar, devices, free mem"
    ssh "$n" "test -r $MODEL/config.json && echo '   config OK'
      ls $MODEL/lmhead_w8v2_sidecar.safetensors >/dev/null 2>&1 && echo '   sidecar OK' \
        || echo '   SIDECAR MISSING - regenerate per issue #1 before B-series boots'
      ls /dev/infiniband/ | tr '\n' ' '; echo
      free -g | awk '/Mem:/{print \"   avail \" \$7 \"G\"}'
      sudo -n docker ps --format '{{.Names}} {{.Status}}' | grep -v workbench || true"
  done
}

smoke_32k() {  # first boot of any new image: 32K context, correctness gate, THEN full preset
  local tag=$1 extra_env=${2:-}
  echo "== 32K smoke boot of $tag $extra_env"
  # intentionally minimal; fill in from partner launch.sh invocation pattern at run time
  echo "   RUN_ID=collab-$tag-smoke IMAGE_TAG=glm52-collab:$tag MODEL_PATH=$MODEL \\"
  echo "   NODE1=${NODES[0]} ... FABRIC_IP_0=${FABRIC_IPS[0]} ... RDMA_IFNAME=$RDMA_IFNAME \\"
  echo "   IB_HCA='$IB_HCA' RDMA_DEV_UVERBS=$RDMA_DEV_UVERBS RDMA_DEV_UVERBS2=$RDMA_DEV_UVERBS2 \\"
  echo "   DCP_SIZE=1 MAX_MODEL_LEN=32000 KV_CACHE_MEMORY_BYTES=1023232000 MAX_NUM_SEQS=1 \\"
  echo "   MAX_NUM_BATCHED_TOKENS=256 CUDAGRAPH_CAPTURE_SIZES=6 CPUSET_CPUS=$CPUSET_CPUS CPUSET_MEMS=$CPUSET_MEMS \\"
  echo "   $extra_env bash recipes/launch.sh launch"
  echo "== then: gate x4/x8 (kit benchmarks/correctness_probe.py) BEFORE promoting to full preset"
}

battery() {
  local label=$1
  echo "== full battery -> results/${label}.jsonl"
  mkdir -p "$HOME/Desktop/GLM Collab/collab-work/results"
  python3 "$PROBE" --endpoint "http://$ENDPOINT_HOST:$PORT" --model glm-5.2 \
    --label "$label" --out "$HOME/Desktop/GLM Collab/collab-work/results/$label.jsonl" battery
}

spec_overhead() {
  echo "== spec-overhead A/B analysis (run after B0 and B0n batteries)"
  python3 - << 'PY'
import json, pathlib
rows={}
for f in pathlib.Path("$HOME/Desktop/GLM Collab/collab-work/results").glob("*.jsonl"):
    for line in f.read_text().splitlines():
        r=json.loads(line); rows.setdefault(f.stem,{}).update(r)
def g(label,key):
    return rows.get(label,{}).get(key)
b0_c1,b0n_c1=g("b0","prose_c1"),g("b0n","prose_c1")
acc=g("b0","accepted_per_draft")
if b0_c1 and b0n_c1:
    step_spec=1000.0/b0n_c1; step_both=1000.0/b0_c1
    print(f"B0n (no spec) prose C1 {b0n_c1:.2f} -> {step_spec:.0f} ms/step base")
    print(f"B0 (spec on) prose C1 {b0_c1:.2f} -> {step_both:.0f} ms/step with spec")
    print(f"spec overhead: {step_both-step_spec:.0f} ms/step  (partner parity ~32ms, public disease ~68ms)")
    if acc: print(f"accepted/draft: {acc}  -> per-draft-forward cost = overhead/(k+1)")
else:
    print("need b0 + b0n battery results first")
PY
}

case "${1:-}" in
  stage0|commcheck)  stage0_comm_check ;;
  preflight)         preflight ;;
  smoke32k)          smoke_32k "${2:?b0|b2}" "${3:-}" ;;
  battery)           battery "${2:?label}" ;;
  specab)            spec_overhead ;;
  *) sed -n '2,30p' "$0"; exit 64 ;;
esac