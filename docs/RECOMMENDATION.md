# GLM-5.2 on 4x DGX Spark: what to run, and why

## Update 2026-08-16 (late): production is now nvfp4 at 315,968 context

Production launcher (`launch_gx10.sh`, recovery inherits it): **B12X_MLA_SPARSE +
TP4 + DCP4 + a2a + MTP depth 2 + `nvfp4_ds_mla` (368-byte record, source-form
Triton writer) + 315,968 context + `--max-num-seqs 16` + dense capture ladder +
gmu 0.90 + `VLLM_ONE_GPU_PER_NODE=1`**. Boots first try; gate passes at C8.

Measured today, same probes, all at 315,968 context and seqs 16:

| stack | KV pool | 2K C4 e2e | 2K C16 e2e | 14K C4 e2e | cold prefill |
|---|---|---|---|---|---|
| legacy FLASHMLA_SPARSE nvfp4 DCP1 | 317K (1.00x) | 41.88 | 75.22 | 18.52 | 841 |
| DCP2 nvfp4 B12X | 429K (1.36x) | | | 16.76 | 787 |
| **production DCP4 nvfp4 B12X** | **959K (3.03x)** | 37.79 | 61.59 | 14.87 | 618 |

DCP1 is 10-25% faster on every throughput cell. DCP4 holds three resident 316K
sessions where DCP1 holds one. **Decided 2026-08-17: DCP4 stays production.**
The workload (autonomous coding agents) is bimodal: most turns are short bursts, the
sessions that matter run past 200K, and every turn in a deep session lands on
the prefix cache (cold 100K prefill 175 s, cache hit 1.7 s). Evicting one deep
session costs more than the DCP1 speed edge returns in an hour of bursts. All
throughput work targets making the DCP4 config faster, not trading the pool.

Fused nvfp4 writer vs the torch writer, live: +6.8% short prefill, flat at 14K.
The 51x kernel speedup was CPU time that overlaps GPU work at depth. Kept for
being public, upstreamable source; not claimed as a throughput lever.

---


Rewritten 2026-08-16 after an exhaustive lever sweep. Every number below was
measured on this fleet and correctness-gated. Where a lever could not be
measured, that is stated rather than guessed.

## The configuration

**B12X_MLA_SPARSE + TP=4 + DCP=4 + `--dcp-comm-backend a2a` + MTP depth 2 +
fp8_ds_mla + 200K context + `--max-num-seqs 16` +
`cudagraph_capture_sizes [3,6,9,12]` + `--gpu-memory-utilization 0.90` +
`--cpu-distributed-timeout-seconds 90`**, on stock `vllm/vllm-openai:v0.27.0-aarch64`
plus `pip install b12x` and `patch_deep_gemm_ops.py`.

| metric | previous | this config |
|---|---|---|
| prose C4 | 44.97 | **50.34** (+12.7%) |
| prose C8 | 51.94 | **71.82** (+38%) |
| prose C16 | not usable | **99.84** |
| KV pool @200K | ~600,000 (3.05x) | 612,531 (3.06x) |

## The two findings that matter

### 1. The concurrency ceiling was a bug, not hardware

`cudagraph_capture_sizes` **must be a dense ladder of multiples of
`1 + num_speculative_tokens`** (here: 3, 6, 9, 12). A gap means some batch size
cannot be served without padding; padding rows carry `decode_len = 0`, the batch
becomes non-uniform, and it takes a broken `torch.repeat_interleave` branch in
the sparse MLA indexer that assumes DCP-sharded and global block-table widths
match. Engine dies on the first affected request.

Scaling capture sizes up proportionally with `max_num_seqs` -- the natural thing
to do -- creates exactly such a gap. That single mistake made "concurrency is
broken with DCP" look true for the entire campaign.

The legacy production launcher already encoded the correct rule as
`[3,6,9,12,15,18]`. Full diagnosis and a suggested upstream fix are in
the upstream issue thread (vllm-project/vllm#51921).

### 2. Cold prefill barely matters, because the prefix cache is worth ~100x

| prefix | cold TTFT | turn 2+ | speedup |
|---|---|---|---|
| 16K | 25.3 s | 1.30 s | 19.5x |
| 100K | 175.4 s | 1.66 s | **~100x** |

Cold prefill is pinned at ~580 tok/s and **three separate levers failed to move
it**. It does not need to move: an agent pays it once per conversation, then
every subsequent turn is ~1.7 s even at 100K context.

**Measured capacity: 16 sessions at 16K, or 6 sessions at 100K, all resident.**
Six 100K sessions occupy 600K tokens of a 612K pool -- ~98% utilization with
zero evictions.

**The operating rule that follows: protect the KV pool above all else.** Evicting
a cached session costs ~100x. No throughput lever measured in this campaign moved
more than 5% in either direction, except one that was catastrophically negative.

## Do not change these

| lever | why |
|---|---|
| `--all2all-backend` | DeepEP is **37-85% slower** than the default `allgather_reducescatter` on this fabric. Largest effect measured all campaign, and negative. |
| `--enable-dbo` | Requires a DeepEP-family backend, so it inherits the above. Closed. |
| `--gpu-memory-utilization` | 0.90 is correct here. Cutting to 0.87 evicts deep sessions (~100x cost) to buy a sub-noise throughput change. Zero Xid errors observed at 0.90. |
| `--max-num-seqs` beyond 16 | 32 measured identical to 16. No gain. |

## Cannot be used on this hardware

| lever | reason |
|---|---|
| `--prefill-context-parallel-size` | Multiplies world size. TP=4 x PCP=2 needs 8 GPUs, PCP=4 needs 16. The fleet has 4. Permanent. |
| cascade attention | vLLM overrides the flag: "not yet compatible with async speculative decoding". Would require dropping MTP. |
| `--max-num-batched-tokens 8192` | 3/3 boot failures at DCP=4 (production runs it fine at DCP=1, which never creates that process group). Unresolved. |

## Measured no-ops (all inside the +/-5% noise floor)

`--performance-mode throughput`, `--async-scheduling`, `--optimization-level 3`,
`B12X_MLA_FORCE_SINGLE_PASS`, `draft_tensor_parallel_size 4`.

The last one is inert by design: MTP shares the target's embedding and lm_head,
so the drafter already rides TP=4 and has nothing left to shard.

## Untested

- `--enable-expert-parallel` -- 2/2 gloo boot failures. Expect a regression given
  the DeepEP result; do not adopt without the full battery.
- `--enable-bf16x3-router-gemm`, `--dcp-kv-cache-interleave-size` -- not reached.

## Operational requirements

- **Boot retry is mandatory.** The DCP group init hangs on ~40% of launches:
  `initialize_model_parallel` -> `in_the_same_node_as` -> gloo barrier timeout,
  which is caught and ignored but leaves node topology unresolved so group
  creation fails. `--cpu-distributed-timeout-seconds 90` converts a 30-minute
  silent hang into a fast failure. Three fixes have been hypothesized and
  **all three disproved by measurement** (`suppress(OSError)` asymmetry,
  a longer timeout, master-first launch ordering). Retry, do not theorize.
- **Never stack boot retries.** Concurrent retries each load 98GB and drove all
  four nodes to memory exhaustion once, requiring a physical power cycle. One
  config at a time, memory verified reclaimed between runs.
- **Correctness-gate every config**, including concurrent phases. A config that
  passes sequential single requests can still die on the first 4-concurrent
  batch -- `max-num-seqs 16` with a gapped capture ladder did exactly that.
- **Discard the first benchmark batch after a boot.** Measured ~14% cold-start
  penalty, nearly 3x the noise floor.
- `VLLM_MARLIN_USE_ATOMIC_ADD=1` is a **correctness** requirement on SM121, not
  a tuning knob. The MoE path resolves to MARLIN WNA16, not NVFP4.

## Honest gap to 0xdfi's published numbers

His prose C4 is 54.68; ours is 50.34, or **92%** (up from 88%). The remaining gap
is structural, not tuning: he runs DCP=1 where we pay DCP=4 for ~3x concurrency
at depth, he serves the 368-byte `nvfp4_ds_mla` KV record where we use the
656-byte `fp8_ds_mla`, and his measured build uses a private native wheel his own
manifest marks as `"serving_image_rebuilt_from_public_inputs": false`.

Ours runs entirely on public inputs.
