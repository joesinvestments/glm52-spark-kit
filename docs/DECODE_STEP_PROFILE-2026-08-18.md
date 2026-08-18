# Decode-step profile, production GLM-5.2 on 4x DGX Spark (2026-08-18 02:50 UTC)

Config: the production identity (QuantTrio Int4-Int8Mix unpruned, V1, MTP k=2, TP=4 + DCP=4 a2a,
B12X_MLA_SPARSE, nvfp4_ds_mla KV, 315,968 ctx, seqs 16, FULL_AND_PIECEWISE graphs) plus
`--profiler-config '{"profiler":"torch","torch_profiler_dir":...,"torch_profiler_with_stack":false}'`.
Two windows via `/start_profile` `/stop_profile`: C1 (one 452-token prompt, 96 new tokens) and C4 (four).
Traces per rank; analyzer `benchmarks/profile/analyze_trace.py` buckets GPU kernel time by name.

## Step time
Steps counted two ways (DCP a2a pack kernels / 78 layers; all-reduce kernels / 158) agree: 38 steps in
the 5.0 s C1 window, 46 in the 10.5 s C4 window. Roughly 110-130 ms per step at C1, ~200 ms at C4
(the C4 window mixes 1- to 4-request batches). This matches 11.8 tok/s per request at C4 with 2.3
tokens per step. Earlier notes that said "~50 ms per step" were wrong; that was an aggregate figure.

## Where the time goes (C4 window, rank 0; ranks 1-3 within 1%)
| slice | share | ~ms/step | kernels/step |
|---|---|---|---|
| MoE expert GEMMs (Marlin int4 `marlin_moe_wna16::Marlin`, `marlin::Marlin`) | 57-58% | ~115 | ~970 (12 per layer), ~0.12 ms each |
| NCCL all-reduce (`AllReduce_Sum_bf16_RING_LL`) | 10-11% | ~22 | 158, ~0.14 ms each (~150 KB messages) |
| NCCL all-gather + send/recv (DCP a2a, indexer gather) | 15% | ~30 | |
| dense GEMMs (attention projections, lm_head; cutlass wmma bf16, gemvx) | 8% | ~16 | |
| sparse MLA attention + indexer (b12x `mgUnifiedPrefillMGKernel` and friends) | 4% | ~9 | |
| norm / rope / elementwise | 4% | ~9 | ~1,460 |
| sampler / MTP misc | <1% | | |
GPU busy 97% of the window at C4 (92% at C1); idle 3-8%. CUDA graphs are working.

## Reading
Bandwidth floor for the experts actually touched per step (12 tokens x 8 of 256 experts, int4) is
~35 ms at C4 and ~11 ms at C1; Marlin spends 115 / 67 ms. The expert path is 3-6x off bandwidth
because it is launch/latency-bound at small M, not byte-bound. Communication is a quarter of the step
and also latency-bound (tiny messages on RING_LL). Attention is not the bottleneck at these depths.

## What this points at, in order
1. Expert parallel for the experts (`--enable-expert-parallel`): 64 whole experts per node, 4x fewer
   and 4x larger expert GEMMs per step, dispatch/combine a2a instead of the per-layer all-reduce.
   Attacks both top slices structurally. One boot to find out.
2. Marlin small-M: fewer launches per layer (grouped/persistent MoE), the builda_bmm line.
3. NCCL algo/proto sweep for ~150 KB messages on the rails.

Raw traces: 8 files (4 ranks x 2 windows), kept off-repo (127 MB); reproduce with
`benchmarks/profile/profile_drive.sh` against a server booted with the profiler flag.
