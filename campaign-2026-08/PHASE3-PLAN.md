# PHASE 3 PLAN - exhaust every outcome (2026-08-22)

Everything below waits on Phase 2's boots (B0/B0n/B1 batteries) only for DATA,
not for permission. Each arm is independently bootable; order by information gain.
Cluster facts: QuantTrio restored+verified on all 4 nodes 2026-08-22; Ornith holds
the GPUs until operator says otherwise; images build on gx10-1 without registry pulls.

## Arms, ranked by expected value

### 3A. b12x upgrade port (arm B3) - NEW LEAD, biggest single unknown
Upstream moved 222 commits past our pin (v0.30.2@334a2d75 -> v1.2.x). Changelog maps
1:1 onto our gaps: tiny-decode inactive routes (#228), route-packed W4A16 scaling (#219),
graph-safe route histograms (#150), prefill-tail arena reuse (#226), SM121 paged-attn opts,
topology-scoped fused PCIe all-reduce (#133), NATIVE exact DCP top-k owner exchange,
MX-FP6 W6A8 MoE stack, NVFP4 KV + compact FP8-RoPE writer.
HYPOTHESIS: partner's private-wheel speed = newer b12x lineage. One boot answers it.
PORT COST (audited 2026-08-22): full restructure, no drop-in.
  b12x.cute.{compiler,utils,fp4}   -> b12x._lib.{compiler,intrinsics}   (+fp4 renamed)
  b12x.integration.scratch(_layout)-> b12x._lib.scratch(_layout)
  b12x.integration.mla/sparse_mla_scratch -> b12x.attention.sparse_mla._scratch
  b12x.attention.indexer           -> b12x.attention.nsa_indexer.*
  b12x.attention.workspace         -> b12x.attention._shared.workspace
Steps: (1) minimal-overlay-set determination - which of the 35 kit overlays are now
upstream-fixed (indexer width fix? #50365 sparse_utils? scratch sharding?) so we port
ONLY still-missing features; (2) import-surface rewrite of survivors; (3) pin either
release tag 1.2.3 (80715bed5, Aug 10) or Aug-20 HEAD 36bce2c15 (needs CUTLASS DSL 4.6.2);
(4) 32K smoke gate then battery vs B0. Watch JIT cache isolation (RUN_ID-keyed already).

### 3B. NVFP4 backend A/B - zero new code, one boot each way
Verified: marlin_moe accepts float4_e2m1f, so nvidia/GLM-5.2-NVFP4 runs through BOTH
marlin_moe AND FLASHINFER_CUTLASS MoE. Same checkpoint, flip backend, battery both:
quantifies CUTLASS large-M prefill advantage vs Marlin small-M decode leanness on OUR
fabric. Memory rules apply (gmu step-up, no MTP co-boot, watchdog armed - see Aug 17
post-mortem in kit HANDOFF). Also produces the prefill number nobody has published.

### 3C. Marlin/CUTLASS regime forensics - analysis, free
Decision axis: decode=bandwidth-bound (bytes win: int4 g128 ~4.1 bits/w beats nvfp4 ~4.5)
vs prefill=computebound (tensor-core FP4 wins). If B0 prefill lands well under partner's
694 despite identical fabric, diff the large-M GEMM path first. Feeds 3D format choice.

### 3D. TailQuant - frequency-aware mixed-precision experts (the flagship)
Insight: ~95% of weight bytes are experts; decode touches ~9%/layer. Spend precision
where routing spends time. NO PRUNING - every expert stays.
Pipeline: (1) router-hit histograms - harvest from FIRST B0 battery via patch 004
(TODO: finalize hook at MarlinExperts.apply call sites, env VLLM_TAILQUANT_PROFILE_DIR);
(2) per-layer split hot/cold at histogram knee; (3) cold tail -> 2-bit wna16 g128
(GPTQ-style calib on own captures) OR MX-FP6 via new b12x stack as safer middle tier;
(4) serve via two-tensor expert partition + two grouped-GEMM launches + moe_sum merge
(scheduler-level code, no kernel-format surgery); (5) optional recovery tune via kit
dspark-training rig. Math: 64 hot @4b + 192 cold @2b ~= 2.5 bits/w avg -> weights
~107 -> ~62 GiB/rank -> decode FASTER than QuantTrio baseline + multi-M-token DCP4.
Quality risk concentrates in tail = gated by correctness_probe + perplexity harness.

### 3E. Drafter arms
- dspark-ft A/B: b1rd finetuned drafter ALREADY on Storage (GLM-5.2-speculator.dspark-
  quanttrio-ft). Swap drafter, measure accepted/step vs MTP k=2 baseline (kit
  spec_metrics.py). bird measured +25% accepted/step same body.
- Depth ladder: VLLM_ADAPTIVE_SPEC_DEPTHS sweep {2}, {2,4}, {2,4,5} - config-only;
  depth-2 pin already promoted to Phase 2 B1 (+9% peak C4 on switch fabric, n=3).
- k escalation: ONLY if 3A shows draft-forward cost back near ~6ms (partner economics),
  deep-k becomes net-positive again - retest k=5 ladder.

### 3F. index_topk_pattern sweep - config-only, untested by anyone
HF_OVERRIDES index_topk_pattern='FFFSSSF...' controls per-layer fresh-vs-shared indexer
top-k. Current recipe: 3 fresh then mostly every-4th. Sweeping share density directly
trades indexer passes per step - interacts with whatever 3A does to indexer cost.

### 3G. Comm path - supersede parked work
Kit's parked fused-DCP-gather (correctness-diverged, ~1.5% ceiling) is SUPERSEDED by
new b12x native PCIe DCP top-k owner exchange + fused all-reduce paths (3A). Do not
resume old approach; adopt upstream instead. Custom RDMA collective floor (~12-20us
vs NCCL 50-90us, kit COMM_FLOOR) remains the endgame IF 3A's PCIe paths underdeliver.

## Sequencing logic
Phase 2 batteries -> (B0 data: histograms for 3D, spec-overhead A/B, prefill baseline)
-> 3A boot (upgrade) -> whichever of {3B, 3E-drafter} the data ranks next -> 3D build
(only real code project, starts after histograms land) -> 3F/3C fill idle boots.
Every promotion: n>=2 repeats, correctness gate x4/x8, stability-first winner.

## Open questions for partner (with results, not before)
- His message claims DCP1 400K KV + 36 t/s C1 / 80 agg / 660 prefill - reconcile vs
  repo matrix (320K capacity ceiling at mnbt2048; records 40.47/78-79/694).
- Which b12x rev his wheel carries; whether his prefill edge survives 3A.
