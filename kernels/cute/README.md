# CuTe DSL decode kernels for GLM-5.2 on GB10 (work in progress)

Started 2026-08-19 from the decode-step profile (docs/DECODE_STEP_PROFILE-2026-08-18.md): the dense
W8A16 projections (o_proj, q_b, kv_b, qkv_a, shared expert; 212M params per layer, ~4.2 GB of int8 reads
per rank per step at TP=4) run at an effective ~161 GB/s under the Marlin w8a16 path at M=3, against a
273 GB/s node. These kernels are the CUTLASS (CuTe DSL 4.6, same toolchain as b12x) replacement for the
small-M decode shapes.

| kernel | shape tested | correctness | GB/s | note |
|---|---|---|---|---|
| `w8a16_smallm_gemv.py` v3 | o_proj layer 10, N=6144 K=16384 M=3 (real checkpoint weights) | max rel err 1.9e-3 (bf16 output rounding vs fp32 reference) | 214 | warp-per-8-rows, 16 B weight loads, group-128 bf16 scales, fp32 accumulate, butterfly reduce |

Run inside the serving image on a node with the checkpoint (extract weights once with the snippet in the
file header): `python3 w8a16_smallm_gemv.py --weights /w --M 3`.

Next: reach ~90% of peak (activation loads to smem or registers per K-slice, split-K for occupancy at
small N), M buckets 4/8/16, then the same design for the int4 g128 expert path (`marlin_moe` w13/w2),
then wire in as a vLLM linear method behind a flag with the byte-identity gate.
