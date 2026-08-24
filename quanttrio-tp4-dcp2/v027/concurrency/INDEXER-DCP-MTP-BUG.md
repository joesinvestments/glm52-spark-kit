# Sparse MLA indexer crashes with DCP>1 + MTP on a padded decode batch

Full mechanism, reproduction matrix, and suggested fix. Filed-ready.

**Environment**
- vLLM 0.27.0 (stock `vllm/vllm-openai:v0.27.0-aarch64`)
- 4x NVIDIA DGX Spark GB10 (sm_121a), TP=4, `--decode-context-parallel-size 4`
- GLM-5.2 (compressed-tensors WNA16, MARLIN MoE), MTP `num_speculative_tokens=2`
- `--kv-cache-dtype fp8_ds_mla`, `--max-model-len 200000`, block size 64
- `cudagraph_mode: FULL`

**What happens**

The server boots normally, allocates KV, and answers `/v1/models`. The **first
generation request** kills the engine:

```
vllm/v1/attention/backends/mla/indexer.py:823 in _prepare_decode_tensors
    self.expanded_block_table_buffer[:actual_expanded] = (
        torch.repeat_interleave(block_table, decode_lens, dim=0,
                                output_size=actual_expanded))
RuntimeError: The expanded size of the tensor (782) must match the existing
size (3126) at non-singleton dimension 1.
Target sizes: [3, 782].  Tensor sizes: [3, 3126]
```

followed by `scheduler.py:1761 KeyError: 'chatcmpl-...'` and
`EngineDeadError` (the KeyError is a downstream symptom: the worker died, so
the request is missing from `model_runner_output.req_id_to_index`).

**Analysis**

The two widths identify the mismatch exactly:
- `3126 = ceil(200000/64) + 1` is the **global** block table width.
- `782  = ceil(3125/4) + 1` is the **DCP=4 sharded** per-rank width.

`expanded_block_table_buffer` is allocated in `__init__` as
`(scheduler_config.max_num_batched_tokens, block_table_width)`, where
`block_table_width` comes from `get_block_table_width(...)` over
`kv_cache_spec.max_num_blocks_per_req(...)` — i.e. the DCP-sharded per-rank
count. The write in the variable-length branch copies a global-width table into
it.

`_prepare_decode_tensors` has two branches for the spec-decode flatten path:

- `min_decode_len == max_decode_len` (**uniform**) dispatches to the triton
  `_prepare_uniform_decode_kernel`, which is passed `block_table.stride(0)` and
  `self.expanded_block_table_buffer.stride(0)` as *separate* arguments and so
  handles the differing widths correctly.
- otherwise (**variable**) uses `torch.repeat_interleave` assigned directly into
  `expanded_block_table_buffer[:actual_expanded]`, which requires the widths to
  be equal. **Only this branch is broken.**

**Trigger**

Padding requests carry `decode_len = 0` (see the function's own worked example,
`decode_lens [3, 1, 4, 0]` where "the final req is padding"). So any decode
batch that must be padded up to a cudagraph capture size becomes non-uniform and
takes the broken branch.

With `num_speculative_tokens=2`, `n` active requests is `n * (1 + 2) = 3n` query
rows. A batch crashes when `3n` is below the largest capture size but is not
itself a capture size, because it then pads.

Three configs, identical except for `cudagraph_capture_sizes`, on the same
hardware and model:

| capture sizes | max_num_seqs | result |
|---|---|---|
| `[3,6,9,12]` | 4 | works |
| `[3,6,9,12]` | 8 | works (C1/C4/C8, repeated batteries) |
| `[6,12,18,24]` | 8 | **dies on request 1**: `[3, 782]` vs `[3, 3126]` (3 rows pad to 6) |
| `[3,6,12,24,48]` | 16 | passes single requests, **dies on the first 4-concurrent batch**: `[9, 782]` vs `[9, 3126]` (9 rows pad to 12) |

The `actual_expanded` value in each error names the exact batch that padded: 3
rows (one request) in the first case, 9 rows (three requests) in the second.

Batches *larger* than the biggest capture size are safe, since they fall back to
eager/piecewise without padding. That is why `[3,6,9,12]` with `max_num_seqs 8`
serves 24-row batches without trouble.

**Workaround**

`cudagraph_capture_sizes` must be a *dense* ladder of multiples of
`1 + num_speculative_tokens` from the smallest reachable batch up to the largest
captured size, with no gaps. Scaling capture sizes up proportionally with
`max_num_seqs` (the natural thing to do) introduces exactly such gaps.

On our fleet this made a **+12.7% prose C4 / +38% C8 throughput win** look
like "concurrency is broken with DCP" until the ladder was identified as the
real variable.

**Suggested fix**

In the variable-length branch, copy per-row into the buffer's own width (or
slice the source to `block_table_width`) rather than assigning the full-width
tensor — matching the stride handling the uniform triton path already does.
