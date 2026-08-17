# Platform: AEON's sm_121a vLLM 0.27.1 rebuild as base

`glm52-spark-platform:aeon-0.27.1` = `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-08-16-v0.27.1`
+ `b12x` at the exact commit the fleet runs (`local-inference-lab/b12x@334a2d75`, cutlass-dsl 4.6.0,
which matches his pin) + this kit's overlays baked into `/usr/local/lib/python3.12/site-packages/`
(AEON installs vLLM into site-packages, not dist-packages).

Overlay compatibility, checked file by file against stock v0.27.0 and his image (2026-08-17):
- 14 targets are new files (b12x integration, our writer, sm12x fallbacks, adaptive dynamic/): copied as-is.
- 13 targets are files he did not change from stock: our versions replace his.
- 6 targets are files he ALSO modified: `vllm/config/speculative.py`, `vllm/utils/torch_utils.py`,
  `vllm/v1/core/sched/scheduler.py`, `vllm/v1/kv_cache_interface.py`,
  `vllm/v1/worker/gpu/cudagraph_utils.py`, `vllm/v1/worker/gpu/spec_decode/speculator.py`.
  Replacing them wholesale breaks his carries (his `torch_utils.py` provides
  `nvfp4_kv_cache_split_views` for the Triton NVFP4-KV path). For the production (V1) profile:
  `torch_utils.py` and `kv_cache_interface.py` are 3-way merged here (`merged/`, base = stock 0.27.0,
  ours = overlay, theirs = AEON; `git merge-file` clean, deltas do not overlap: ours 2 and 15 lines,
  his 76 and 28); the other four keep his versions (they only matter for the adaptive-MTP profile,
  which needs its own merge pass).

Build: `docker build -t glm52-spark-platform:aeon-0.27.1 .` with `overlays/` = the kit overlays with
the four adaptive files removed and the two merged files substituted (see `../scratch` notes in
docs/OVERNIGHT-2026-08-17.md). Launch: `launch/champion_platform_32k.sh` pattern: no overlay mounts
(baked), `--entrypoint vllm ... serve` because his image's entrypoint is `/bin/bash`.
