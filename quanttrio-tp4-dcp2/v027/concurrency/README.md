# The concurrency ceiling was a cudagraph bug, not the hardware

Measured 2026-08-16 on 4x DGX Spark (GB10, sm_121a), vLLM 0.27.0, B12X_MLA_SPARSE,
TP=4 + DCP=4, MTP depth 2, fp8_ds_mla, 200K context.

## Result

Raising `--max-num-seqs` from 4 to 16 -- while leaving the cudagraph capture
ladder alone -- moved aggregate throughput substantially, and unlocked a
concurrency level that previously killed the engine:

| | max-num-seqs 4 | max-num-seqs 16 |
|---|---|---|
| prose C4 | 44.97 tok/s | **50.34** (+12.7%) |
| prose C8 | 51.94 | **71.82** (+38%) |
| prose C16 | not serviceable | **99.84** |
| KV pool @200K | ~600,000 tok (3.05x) | 612,531 tok (3.06x) |

At C8 the per-request profile changes shape entirely: `max-num-seqs 4` produces
four requests at ~13 tok/s and four at ~6.5 (two waves), while 16 produces eight
at 9.0-9.7 tok/s (one wave). Engine steps/s rises 22.4 -> 31.0.

`max-num-seqs 32` measured identical to 16 (48.4 vs 50.3 at C4, 99.6 vs 99.8 at
C16). 16 is the useful setting.

## The rule that makes it work

**`cudagraph_capture_sizes` must be a DENSE ladder of multiples of
`1 + num_speculative_tokens`, with no gaps below the largest captured size.**

With MTP depth 2, `n` active requests is `n x (1+2) = 3n` decode query rows. If
`3n` falls below the largest capture size but is not itself a capture size, the
batch pads. Padding rows carry `decode_len = 0`, which makes the batch
non-uniform, which routes it into a `torch.repeat_interleave` branch in the
sparse MLA indexer that assumes the DCP-sharded and global block-table widths
match. They do not, and the engine dies:

```
vllm/v1/attention/backends/mla/indexer.py _prepare_decode_tensors
RuntimeError: The expanded size of the tensor (782) must match the existing
size (3126) at non-singleton dimension 1.
```

- `3126 = ceil(200000/64) + 1` -- the global block table width
- `782  = ceil(3125/4) + 1`   -- the DCP=4 sharded per-rank width

Measured behaviour, identical configs except the ladder:

| capture sizes | max-num-seqs | result |
|---|---|---|
| `[3,6,9,12]` | 4 | works |
| `[3,6,9,12]` | 8 | works |
| `[3,6,9,12]` | 16 | works |
| `[6,12,18,24]` | 8 | **dies on request 1** (3 rows pad to 6) |
| `[3,6,12,24,48]` | 16 | passes single requests, **dies on the first 4-concurrent batch** (9 rows pad to 12) |

Batches *larger* than the biggest capture size are safe -- they fall back to
eager without padding. That is why a ladder stopping at 12 serves 24-row batches
fine.

**The trap:** scaling capture sizes up proportionally with `max-num-seqs` is the
natural thing to do, and it creates exactly the gaps that break it. That single
mistake made "DCP cannot do concurrency" look true.

The legacy stack's launcher already encoded the correct dense ladder
(`[3,6,9,12,15,18]`). The rule had simply never been written down.

## Reproduce

```bash
# works
--max-num-seqs 16 \
--compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[3,6,9,12]}'

# dies on the first concurrent batch
--max-num-seqs 16 \
--compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[3,6,12,24,48]}'
```

Upstream write-up with the full mechanism and a suggested patch is in
`INDEXER-DCP-MTP-BUG.md`.
