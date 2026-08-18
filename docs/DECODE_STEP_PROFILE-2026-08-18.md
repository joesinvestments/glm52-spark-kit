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
| MoE expert GEMMs (Marlin int4 `marlin_moe_wna16::Marlin`) + dense int8 linears (`marlin::Marlin`) | 57-58% | ~115 | 2 MoE launches per layer (w13 ~0.9 ms, w2 ~0.45 ms at C4) plus ~4 dense w8a16 launches per layer |
| NCCL all-reduce (`AllReduce_Sum_bf16_RING_LL`) | 10-11% | ~22 | 158, ~0.14 ms each (~150 KB messages) |
| NCCL all-gather + send/recv (DCP a2a, indexer gather) | 15% | ~30 | |
| dense GEMMs (attention projections, lm_head; cutlass wmma bf16, gemvx) | 8% | ~16 | |
| sparse MLA attention + indexer (b12x `mgUnifiedPrefillMGKernel` and friends) | 4% | ~9 | |
| norm / rope / elementwise | 4% | ~9 | ~1,460 |
| sampler / MTP misc | <1% | | |
GPU busy 97% of the window at C4 (92% at C1); idle 3-8%. CUDA graphs are working.

## Reading (corrected the same night; the first version of this file said the expert path was launch-bound, it is not)
Per layer the expert path is exactly two Marlin MoE launches (w13, w2). With the real dims (256 experts,
6144 x 2048, int4 g128, TP-sharded 4 ways: ~4.85 MB per touched expert per rank) the C1 window comes
out at roughly 75-80% of the node's memory bandwidth. The expert kernel is close to the byte floor
already; a custom small-M expert kernel would buy a fraction of the 57% slice, not multiples.
The dense int8 linears (`marlin::Marlin` w8a16: q/kv/o projections, shared expert) are 8-20% of the
step depending on batch and have not been checked against bandwidth yet.
Communication is a quarter of the step and IS latency-bound: five collectives per layer (two TP
all-reduces ~60 us, two indexer all-gathers ~85 us for the DCP top-k path, one send/recv ~50 us for the
DCP KV all-to-all), ~78 x 340 us. Attention itself is 4%.

## What this points at, in order
1. Communication fusion on the DSA/DCP path: fuse the two indexer all-gathers, pack the a2a into the
   same launch, overlap the all-reduce of layer L with the norm/router of L+1. Biggest non-bandwidth
   slice, no model-quality cost, nobody has done it for DSA on DCP.
2. Tokens per byte: the experts are bandwidth-bound, so every accepted draft token is free. A drafter
   finetuned on captures from this exact quant and stack (bird measured +25% on the same body).
3. Check the dense w8a16 linears against bandwidth at M=3..12; second kernel-shape target if short.
4. Expert parallel: does not change bytes per node, trades the all-reduces for a dispatch a2a; one cheap
   boot, not expected to be the win.

Raw traces: 8 files (4 ranks x 2 windows), kept off-repo (127 MB); reproduce with
`benchmarks/profile/profile_drive.sh` against a server booted with the profiler flag.
