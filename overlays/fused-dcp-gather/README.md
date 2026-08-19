# Fused DCP gather (decode): one collective instead of two per layer

Per decode step at TP=4 + DCP=4, every layer issues two data-independent all-gathers on the DCP group
back-to-back: the sparse indexer's top-k candidate gather (`_merge_dcp_topk_global`, ~48 KB/rank at C1)
and the MLA query-head gather in `MLAAttention.forward` right before `forward_mqa` (~55 KB/rank). Each
is ~85 us of ring latency on the RoCE rails. This change makes them ONE all-gather:

- indexer (decode path, no padding): packs candidates as before, then DEFERS the gather (module-level
  side channel), leaving `topk_indices` (a live view of the shared buffer) untouched;
- MLA decode: concatenates the raw bytes of `[q | candidates]`, does one `all_gather`, splits per rank,
  finishes the stable top-k into the same buffer, and proceeds with the gathered q.

Same data, same kernels, one transport. CORRECTION 2026-08-19: GLM-5.2 runs the indexer in only ~22 of
78 layers per step (the rest reuse the previous layer's top-k, `skip_topk`), so this removes ~22 gathers
per step, ~2 ms at C1 (~1.5%), not 78. Production trace: 102.7 AllGather/step = 78 q + ~25 indexer. Enabled by `VLLM_FUSED_DCP_GATHER=1`; off = stock.

Files: `overlays/vllm/model_executor/layers/attention/mla_attention.py` (consumer) and the indexer.
Production mounts the STOCK indexer file (the D-config indexer overlay is not adopted), so the test
variant is `sparse_attn_indexer.stock+fused.py` = stock v0.27.0 + this patch; the same patch also lives
in the D-config overlay `overlays/vllm/model_executor/layers/sparse_attn_indexer.py`.

Evaluated and rejected alternative: vLLM's `VLLM_DCP_Q_REPLICATE=1` removes the q gather by holding all
DCP-group heads of `q_b_proj` on every rank (+~25 MB int8 reads per layer per rank, ~+2 GB per step):
a wash on a bandwidth-bound step, and it costs KV pool.

Test plan: 32K, seqs 4, production identity + flag (`launch/champion_fusedgather_32k.sh`), gate x4/x8,
battery vs baseline-32K (35.9 / 11.2 / 566), and a decode-step trace to count AllGather kernels per step
(expect 78 fewer). Then 316K, then production, each on the operator's go.

## Test result 2026-08-19 01:02 UTC (32K, seqs 4, DCP4, gmu 0.88)
- Mechanism confirmed: trace on rank 0 shows AllGather/step 81.4 vs 156 baseline (78 removed), AllReduce
  158 and SendRecv 78 unchanged.
- Correctness NOT identical: gate x4 passed, x8 FAILED; greedy acceptance 0.63/step vs 1.15 baseline;
  per-request decode 8.2 vs 11.2; prefill figure nonsensical (1892). The fused path changes attention
  results. NOT adoptable as is. Suspects: the deferred top-k finish landing after a consumer already read
  the buffer for that layer (backend metadata, or `skip_topk` layers sharing top-k), or the fp32
  candidates-as-bytes reassembly. Next step: single-layer tensor diff of both paths off the serving
  path before any further boot. Production restored (gate PASS, 38.35 / 11.91 / 601).

## 2026-08-19 03:50 diagnosis so far
- 4-rank test over the rails (`/var/tmp/gather_equiv.py`): the byte-packed single gather reproduces both
  gathered tensors exactly on all ranks. The CuTe `stable_topk` returns the same SET of ids as a torch
  reference but in a run-to-run varying order (fine for attention; it just makes exact-equality tests
  too strict).
- Fused-run trace: indexer kernels present at ~22/step as in production; AllGather 81.4/step (= 78 q
  gathers + ~3), i.e. the ~22 indexer gathers were fused. So the mechanism is right and the divergence
  is somewhere in what the model reads. Verify mode `VLLM_FUSED_DCP_GATHER=2` now computes the original
  two-gather result alongside and set-compares per row on-device, logging the first mismatches; one 32K
  boot with it localizes the bug.

## 2026-08-19 04:44: PARKED
Verify mode (=2) failed all four boot attempts with a `c10 DistError` at distributed init (the reference
all-gather issued inside the indexer during warm-up breaks the DCP group's init sequence), while mode 1
booted first try but diverged. With the measured payoff at ~22 collectives per step (~1.5% at C1) the
remaining diagnosis is not worth further fleet boots. Kept for the record and for anyone who wants to
finish it: the mechanism is proven and the divergence is somewhere between the deferred finish and the
consumer of the shared top-k buffer. The larger comm items (78 query gathers, 156 all-reduces per step)
are structural to TP+DCP and are the real targets (RDMA-direct 4-rank collectives, all-reduce overlap).
Production restored: gate PASS, 36.51 / 11.34 / 603.
