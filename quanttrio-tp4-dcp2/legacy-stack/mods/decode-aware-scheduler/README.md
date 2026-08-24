# Decode-Aware Prefill Scheduler (optional mod)

Fixes the **prefill-induced decode stall**: when a long prompt (~15k+ tokens)
arrives while another request is actively decoding, the decode stream freezes
for several seconds while the long prefill monopolizes the scheduler step
(classic head-of-line blocking / decode starvation). On this recipe's 4x GB10
TP=4 setup, a ~15k-token prompt landing on an active decode stream froze it for
about **9.4 seconds**, every time.

This is the same "TTFT-under-load wart" noted in `SPEED-NIGHT-FINDINGS.md`. The
built-in remedy there (`--max-num-partial-prefills 2`) throws
`NotImplementedError: Concurrent Partial Prefill` on the pinned vLLM commit
(`ab666069`). **This mod takes a different path that works on that exact pin**:
chunked prefill plus a decode-aware token budget, so a long prefill is metered
in small chunks and the active decode stream keeps flowing.

## Attribution

- **Scheduler patch:** NVIDIA Developer Forums user **penguinchang** (2026-07-15).
- **Distribution:** **ciprianveg**'s `gb10-glm-5.2` repo (`mods/decode-aware-scheduler`),
  https://github.com/ciprianveg/gb10-glm-5.2 , Apache-2.0. Files included verbatim.
- **Port + validation + integration into this recipe:** **OsakaTX** (verified
  clean apply against the `ab666069` pin; benchmarks below).

## How it works

Dynamic prefill budgets that adapt to decode activity:

- **No active decode** -> prefill gets the full idle budget (`idle-prefill-token-budget`,
  default 16384, capped by `--max-num-batched-tokens`). Prefill throughput stays high.
- **Active decode** -> all prefill work shares a small budget
  (`decode-prefill-token-budget`, default 1024, tunable). Decode keeps progressing
  at reduced but non-zero rate instead of freezing.
- At most `max-long-prefills-per-step` long prefills per step, selected by
  least-recently-scheduled age (fairness across waiting requests).

It is a pure scheduling-layer change (patches `vllm/config/scheduler.py`,
`vllm/engine/arg_utils.py`, `vllm/v1/core/sched/scheduler.py`). It does not touch
the attention backend, kernels, or weights, so it is backend-agnostic: validated
on this recipe's `FLASHMLA_SPARSE` build, and it applies unchanged to the
NVFP4-KV / 300K variant (same pin, same scheduler substrate).

## Enable it

1. Bake the mod into the image at build time (runs the patch against the
   installed vLLM), same pattern as the other `mods/*/run.sh`:
   ```
   bash mods/decode-aware-scheduler/run.sh
   ```
   Confirm the startup log shows:
   `Decode-aware prefill scheduling enabled with decode_prefill_token_budget=...`
   (if that line is absent, the flag did not reach SchedulerConfig and the mod is inert).
2. Turn it on at serve time via `launch.sh` (OFF by default):
   ```
   DECODE_AWARE=1 ./launch.sh
   ```
   Tunables (env, with defaults): `DPTB=256` (decode-prefill-token-budget),
   `LPT=2048` (long-prefill-token-threshold), `IPTB=8192` (idle-prefill-token-budget).

## Validation (OsakaTX, this recipe, 4x GB10 TP=4, 200K, MTP k=4, pin ab666069)

Concurrent test: one steady decode stream + a ~15k-token prefill landing mid-decode.

| Config | Decode stall (median MAX) | Decode ITL p95 | Prefill TTFT (under load) |
|--------|---------------------------|----------------|---------------------------|
| baseline (no mod) | 9368 ms | 152 ms | 18.6 s |
| DPTB=256 | 634 ms | 571 ms | 33.5 s |
| DPTB=512 | 903 ms | 837 ms | 24.7 s |
| DPTB=1024 | 1451 ms | 1340 ms | 20.6 s |

- **~15x reduction** in the worst-case decode stall at DPTB=256 (9.4 s -> 0.6 s).
- **No single-stream regression**: uncontended steady decode was 24.7 t/s / 45%
  MTP accept vs 23.7 / 41% baseline, tool-calling 15/15 clean. The throttle only
  engages when a decode stream and a long prefill actually compete.
- **Tradeoff**: the long prefill's TTFT roughly doubles under contention, and
  decode gets choppier (higher ITL p95). The budget is the dial -- lower =
  smoother decode / higher prefill TTFT, higher = the reverse. 256 is the most
  decode-protective; 512 is a balanced middle.
- **Regime caveat**: validated under low KV-cache pressure (peak KV usage ~15%
  of the pool during tests). Behavior under heavy long-context / KV-preemption
  load is not characterized here.
