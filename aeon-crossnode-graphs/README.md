# GLM-5.2 on 4x DGX Spark, on the AEON vLLM 0.27.1 sm_121a image, with cross-node CUDA graphs

A small, self-contained hand-back to the AEON `vllm-ultimate-dgx-spark` project.

Your v0.27.1 release notes say: *cross-node graphs still broken upstream, bring up TP=2 with
`--enforce-eager`.* On four DGX Sparks (GB10, sm_121a) with your image as the base, GLM-5.2 (744B,
unpruned, QuantTrio Int4-Int8Mix) at TP=4 + DCP=4 captured PIECEWISE and FULL decode CUDA graphs across
all four nodes, passed a correctness gate, and served at parity with the stock v0.27.0 stack the same
fleet runs in production. `FINDING.md` has the claim, the evidence, and the flags that matter.

Bonus, `NVFP4.md`: nvidia's full-NVFP4 GLM-5.2 checkpoint (`nvidia/GLM-5.2-NVFP4`, modelopt) also
boots and serves on your image at TP=4 with the `FLASHINFER_CUTLASS` NvFp4 MoE backend, which as far
as we can find has not been shown on four Sparks before. It is not the fast path, and the memory
arithmetic in that file says why.

## What is here

| path | what |
|---|---|
| `Dockerfile` | `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-08-16-v0.27.1` + `b12x@334a2d75` + `overlays/` baked into `site-packages` |
| `overlays/` | 30 files (b12x sparse-MLA integration for GB10, sm12x fallbacks, an NVFP4 MLA KV writer, DCP-aware indexer); origin of every file in `overlays/PROVENANCE.md`; two of them are 3-way merges with your v0.27.1 versions |
| `launch/serve_glm52_tp4_dcp4.sh` | the exact serving profile that captured cross-node graphs (fill in rail IPs and weights dir) |
| `FINDING.md` | claim vs evidence, flags, numbers |
| `NVFP4.md` | full-NVFP4 GLM-5.2 on your image: what fits, what does not, and why |
| `NOTICE` | credits |

## Build and run

```bash
docker build -t glm52-spark-platform:aeon-0.27.1 .
NODE_IPS="<head-rail-ip> <n2> <n3> <n4>" WEIGHTS_DIR=/path/holding/hub launch/serve_glm52_tp4_dcp4.sh
```

Prerequisites on each node are the usual ones for this fleet class: 200G RoCEv2 rails between the four
Sparks (MTU 9000), NCCL 2.30.x, the QuantTrio checkpoint under `hub/glm52-quanttrio-unpruned`, and
enough free unified memory at boot (drop the page cache first; the launcher does).

Everything measured here was measured with the same probes as the fleet's production stack; nothing in
this repo is a projection.
