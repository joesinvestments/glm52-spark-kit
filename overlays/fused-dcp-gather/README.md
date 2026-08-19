# Fused DCP gather (decode): one collective instead of two per layer

Per decode step at TP=4 + DCP=4, every layer issues two data-independent all-gathers on the DCP group
back-to-back: the sparse indexer's top-k candidate gather (`_merge_dcp_topk_global`, ~48 KB/rank at C1)
and the MLA query-head gather in `MLAAttention.forward` right before `forward_mqa` (~55 KB/rank). Each
is ~85 us of ring latency on the RoCE rails. This change makes them ONE all-gather:

- indexer (decode path, no padding): packs candidates as before, then DEFERS the gather (module-level
  side channel), leaving `topk_indices` (a live view of the shared buffer) untouched;
- MLA decode: concatenates the raw bytes of `[q | candidates]`, does one `all_gather`, splits per rank,
  finishes the stable top-k into the same buffer, and proceeds with the gathered q.

Same data, same kernels, one transport: results are byte-identical by construction. Removes 78
collectives per step (~6-7 ms at C1, ~5% of the step). Enabled by `VLLM_FUSED_DCP_GATHER=1`; off = stock.

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
