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
