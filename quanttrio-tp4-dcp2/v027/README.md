# GLM-5.2 on vLLM 0.27: the frontier, mapped as far as one fleet could take it

Nobody had run GLM-5.2 on vLLM 0.27 when this was written. This directory is everything
learned getting it to first tokens on 4x DGX Spark, and exactly where it still breaks, so
the next person starts where this fleet stopped.

## What works

The native path exists and engages: `GlmMoeDsaForCausalLM` is in the mainline registry,
capability 12.1 auto-selects `FLASHINFER_MLA_SPARSE_SM120` for sparse decode with
FLASH_ATTN MLA prefill, and completions come back in ~1.3s single-stream. No kernel
overlays, no community patch chains. The entire overlay era this repo documents becomes
unnecessary on this tree, once the bugs below are gone.

## The three walls, in the order you will hit them

1. **Entrypoint**: the official image's entrypoint is the vllm CLI. Pass `serve ...` args,
   not `vllm serve ...` (or override the entrypoint; see launch_gx10_v027.sh).
2. **DeepGEMM "Unsupported architecture"** at the DSA indexer's paged-MQA call: v0.27.0's
   default DeepGEMM pin dropped the sm_12x branches. Fix: Dockerfile.deepgemm-fix here,
   which runs vLLM's own tools/install_deepgemm.sh at ref 2fd67329. Traps inside: the
   runtime image has no git and no CUDA headers (borrowed multi-stage from any full CUDA
   image), and uv needs UV_SYSTEM_PYTHON=1. This hits GLM, not just DeepSeek (vllm #51758).
3. **masked_mha_available AttributeError**: the new SM120 backend impl never sets an
   attribute the prefill dispatcher reads; the engine dies at startup and crash-loops.
   Fix: Dockerfile.fix2 + patch_masked_mha.py here. Filed upstream: vllm #51920.

## The wall still standing

With all three fixed, the engine serves its first requests and then permanently stalls
after about a minute of engine idleness: requests never reach the scheduler, the API stays
up, EngineCore repeats "No available shared memory broadcast block", and only a fleet
restart recovers. Reproduced 3/3. Two-node setups reportedly do not hit this (vllm #51758's
soak was clean); 4-node TP=4 does, every time. Filed upstream with full repro: vllm #51921.

Until that one is fixed, 0.27 is not serviceable for multi-node production on this
hardware, and this repo's production config (the capacity build in the parent directory)
remains the recommendation. When 0.27.1 or a fix lands, everything needed to retest is in
this directory: the launcher, both fix layers, and the probe battery in window-data/.

Take it further than I did and I will put your name in this README.
