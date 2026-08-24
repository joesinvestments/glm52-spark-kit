# Cold prefill is ~580 tok/s and immovable. It also barely matters.

Measured 2026-08-16, same fleet and config as `../concurrency`.

## The measurement that reframed the whole campaign

Every throughput probe in this repo (and most published ones anywhere)
nonce-busts its prompts so each request pays full cold prefill. That is correct
for measuring raw prefill, and it is the **worst case** for real agent traffic,
which re-sends a growing conversation, a stable system prompt, and the same file
contents over and over.

With `--enable-prefix-caching`, those tokens are already resident:

| prefix | cold TTFT | turn 2 | turn 3 | speedup |
|---|---|---|---|---|
| 16K | 25.3 s | 1.30 s | 1.37 s | **19.5x** |
| 100K | 175.4 s | 1.66 s | 1.78 s | **~100x** |

The prompt keeps growing across turns (93,362 -> 93,404 -> 93,441 tokens) and
TTFT does not move. **A 100K-context agent waits 175 s once, then 1.7 s per
turn.**

## Capacity: how many sessions stay resident

Open N sessions at a given depth, then revisit the OLDEST one. If its prefix
survived, TTFT stays ~1-2 s; if evicted, it returns toward cold.

| ctx | sessions | total tokens | oldest revisit | verdict |
|---|---|---|---|---|
| 16K | 16 | 256K | 1.12 s | resident |
| 100K | 3 | 300K | 1.70 s | resident |
| 100K | **6** | **600K** | **1.67 s** | **resident** |

600K tokens resident in a 612K pool -- ~98% utilization, zero evictions.

**Operating envelope: 16 concurrent sessions at 16K, or 6 at 100K, all warm.**

## Why this changes what you tune

Cold prefill sits at 524-615 tok/s across every configuration and depth measured,
and three separate levers failed to move it:

| lever | result |
|---|---|
| `--max-num-batched-tokens 8192` | will not boot at DCP=4 (3/3 attempts) |
| `--prefill-context-parallel-size` | impossible: multiplies world size, needs 8-16 GPUs for a 4-GPU fleet |
| `B12X_MLA_FORCE_SINGLE_PASS` | +0.4-2.6%, inside the +/-5% noise floor |

It does not need to move. **The lever that matters is KV pool size**, because
evicting a cached session converts a 1.7 s turn into a 175 s cold prefill -- a
~100x penalty. No throughput knob measured in this campaign moved more than 5%
in either direction, with one exception that was catastrophically negative
(see `../fabric`).

Corollary: **do not shrink `--gpu-memory-utilization` to buy headroom.** At 0.90
the 6x100K case fits with ~12K tokens to spare; 0.87 breaks it. Zero Xid errors
were observed fleet-wide at 0.90.

## Measuring this yourself

`prefill_tok_s` and `max_ttft_s` require separating prefill from decode, which
needs streaming time-to-first-token:

```
prefill_s     = TTFT
decode_s      = wall - TTFT
decode tok/s  = completion_tokens / decode_s
prefill tok/s = prompt_tokens / TTFT
```

Without that split, `output_tokens / wall` at 32K+ context is a prefill number
wearing a decode label. Probes are in the fleet-kit repo.
