# CuTe DSL decode kernels for GLM-5.2 on GB10 (work in progress)

Started 2026-08-19 from the decode-step profile (docs/DECODE_STEP_PROFILE-2026-08-18.md): the dense
W8A16 projections (o_proj, q_b, kv_b, qkv_a, shared expert; 212M params per layer, ~4.2 GB of int8 reads
per rank per step at TP=4) run at an effective ~161 GB/s under the Marlin w8a16 path at M=3, against a
273 GB/s node. These kernels are the CUTLASS (CuTe DSL 4.6, same toolchain as b12x) replacement for the
small-M decode shapes.

| kernel | shape tested | correctness | GB/s | note |
|---|---|---|---|---|
| `w8a16_smallm_gemv.py` v5 | o_proj layer 10, N=6144 K=16384 M=3 (real checkpoint weights) | max rel err 1.9e-3 (bf16 output rounding vs fp32 reference) | 224 | split-K, smem-staged activations, 8 rows/warp, 16 B loads |
| vLLM Marlin w8a16 (`marlin_w8a16_baseline.py`, `ops.marlin_gemm`) | same weight, same shape | (repack layout in the harness is not yet bit-correct; timing is representative) | **239** | the production kernel |
| cuBLAS bf16 (2x bytes) | same shape | | 227-236 | practical DRAM ceiling on GB10 is ~235-240 GB/s |

Run inside the serving image on a node with the checkpoint (extract weights once with the snippet in the
file header): `python3 w8a16_smallm_gemv.py --weights /w --M 3`.

**Result (2026-08-19, honest):** Marlin's w8a16 kernel is already at the practical DRAM ceiling on this
shape (239 GB/s vs ~235-240 achievable); the CuTe DSL kernel reaches parity (224) and cannot beat it by
more than a few percent. The "dense slice 1.7x off floor" reading in the profile doc was an undercount
of the bytes that slice reads per step, not kernel inefficiency. Conclusion: **the GEMM path (dense and,
by the same measurement logic, the Marlin MoE experts) is at the hardware floor; there is no matmul
kernel win on this body.** The recoverable time in a decode step is the ~390 latency-bound collectives
(~30 ms/step at C1) and ~5-10 ms of launch/serialization between ~400 small kernels.
The harness stays for other shapes/dtypes; the work moves to the communication path.
