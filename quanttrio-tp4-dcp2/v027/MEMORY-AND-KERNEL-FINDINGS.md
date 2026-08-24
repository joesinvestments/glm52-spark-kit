# Memory pressure, attention/MoE backend overrides, and a live stuck-kernel capture

Follow-up to `SCREENING.md`. Same trigger, same harness (`wedge_trigger.py`), extended with
two forensic snapshots taken the instant WEDGED fires: an NCCL RAS query per rank
(`echo "verbose status" | nc localhost 28028`, on by default since NCCL 2.24) and `free -h` /
`nvidia-smi` / `/proc/meminfo` per node. Neither gates the verdict. Both are single-shot, no
loop, logged raw.

## Cudagraph mode: none of the four are safe

| mode | verdict | RAS |
|---|---|---|
| FULL | WEDGED | clean |
| PIECEWISE | WEDGED | clean |
| NONE | WEDGED | **mismatched** (3 ranks at op 1356, 1 rank at 1355) |
| eager (`--enforce-eager`) | WEDGED | clean |

All four wedged with the identical storm-phase-timeout signature. Cudagraph mode does not gate
the failure, including NONE and eager, where there is no graph to diverge on. RAS caught real
divergence in exactly one of four cells; the other three read clean. That asymmetry turned out
to be misleading, see below.

## The memory campaign

Every wedge that day, cudagraph or otherwise, coincided with a fresh `NV_ERR_NO_MEMORY` (driver
error `0x00000051`) burst in dmesg, fleet-wide, landing within a ~10-90 second window of the
verdict. The fleet runs with `gpu-memory-utilization 0.91` and a fixed `kv-cache-memory-bytes`,
which on this unified-memory hardware leaves under 1.5 GiB "available" system-wide on every
node, always, wedged or healthy. Reasonable to suspect memory pressure was the whole story.

It wasn't, but getting to a clean answer took several real, wrong turns worth recording:

- **Dropping `gpu-memory-utilization` alone did almost nothing.** 0.91 → 0.80 → 0.65 moved
  available memory from under 500 MiB to about 1 GiB. Still nowhere near healthy.
- **The reason: `--kv-cache-memory-bytes`, when set explicitly, is documented by vLLM's own
  boot log to not respect `gpu-memory-utilization` at all.** Every gmu variant that day left KV
  cache sizing untouched. Whatever effect gmu did have came from activation buffers, not the
  dominant fixed allocation.
- **`--kv-cache-memory-bytes` has a hard floor tied to `--max-model-len`.** At `max-model-len
  200000` the floor is 10.19 GiB; the production value (10.95 GiB) sits barely above it.
  Requesting less than the floor is a clean, fast boot-time `ValueError`, not a runtime failure.
- **The only way to free real memory was to also drop `max-model-len`.** At 64000 (still >3x
  the trigger's 20K-token deep-prefill, so the test stays valid) the KV-cache floor drops to
  about 3.3 GiB, and `--kv-cache-memory-bytes 4294967296` (4.0 GiB) boots clean with **4-8 GiB
  genuinely free on every node**, confirmed via `/proc/meminfo`, not inferred.

**Result under real, verified headroom, production `max-num-batched-tokens 8192`, NONE
cudagraph mode: WEDGED anyway.** Same storm-phase-timeout signature, same `shm_broadcast`
reader-starvation log line, no acute memory event within the window. Memory pressure, acute or
chronic, is demoted from necessary condition to (at most) an amplifier of something else.

## A second, different bug found and closed along the way

At the same healthy-headroom config but with `--max-num-batched-tokens` reduced to 2048 (a
setting introduced for the memory campaign itself, not a production value), the engine did not
hang. It **crashed**, reproducibly, twice, with an exact stack:

```
File "vllm/v1/core/sched/scheduler.py", line 1761, in update_from_output
    req_index = model_runner_output.req_id_to_index[req_id]
KeyError: 'chatcmpl-<per-request id>'
```

Root cause, confirmed against the installed source: `sparse_mla_attention.py`'s `forward_mha`
takes a `force_dense` branch straight into the attention base class's unimplemented default
(`vllm/v1/attention/backend.py:1052`, `raise NotImplementedError`) whenever
`prefill_max_seq_len <= topk_tokens`. Under chunked prefill with `max-num-batched-tokens=2048`,
every chunk of the trigger's 20K-token deep-prefill request can satisfy that condition. All four
ranks hit the identical exception simultaneously; the scheduler's `KeyError` on rank 0 is a pure
downstream artifact of workers dying before producing output, not a scheduler bug.

**Confirmed as an artifact, not the production bug**: re-run at production's
`max-num-batched-tokens 8192`, same everything else, and the crash does not occur. The original
silent hang returns instead. This is a real vLLM bug (an unguarded fallthrough to an
unimplemented method), worth its own upstream report, but it is not #51921.

## Live capture: stacks from a genuinely stuck fleet

With `SYS_PTRACE` added to the container and `py-spy`/`cuda-gdb` installed, the same
healthy-headroom, production-batched-tokens config was reproduced once more. `py-spy dump`
against all four ranks' host-visible PIDs, sampled twice, several minutes apart:

- **EngineCore (rank 0's scheduler)**: blocked in `acquire_read` on the shared-memory broadcast
  queue (`shm_broadcast.py:795`), via `zmq.poll()`, waiting for a worker response that never
  arrives. Exactly what the recurring `shm_broadcast.py:802` log line says, confirmed at the
  Python-stack level instead of inferred from a log message.
- **Worker rank 0**: stuck inside `moe_wna16_marlin_gemm`, the Marlin MoE GEMM kernel.
- **Worker rank 1**: stuck inside a different AOT-inductor-compiled kernel entirely.
- **Worker ranks 2 and 3**: stuck identically inside `triton_convert_req_index_to_global_index`,
  FlashInfer's sparse-MLA index-conversion Triton kernel.

Zero movement across two samples on any rank. No NCCL frame visible in any Python stack. The
important part is not "stuck in a kernel", it's that **all four are stuck in different kernels**.
TP-sharded ranks doing the same op should hang at the same point together if a single kernel
were the whole story; instead they had already diverged before any of them stalled, consistent
with whichever rank finishes first blocking on a collective buried inside a later kernel launch
that a stalled sibling never reaches.

`cuda-gdb -p <pid>` inside the container could not complete an attach against the live process
within several minutes and was killed rather than left running. That the debugger itself
couldn't interrupt the CUDA context is a data point, not a null result: it's more consistent
with a genuine device-side stall than a pure host-side Python wait.

## Attention and MoE backend overrides: both closed, not just untried

Checked against the installed v0.27.0 source before spending a boot cycle, then confirmed live:

- **Attention backend.** For device capability 12 (GB10), vLLM's own platform code lists exactly
  two MLA candidates: `TRITON_MLA` and `FLASHINFER_MLA_SPARSE_SM120`. Forcing `TRITON_MLA`
  produces a clean, fast, boot-time rejection: `Reason: ['kv_cache_dtype not supported', 'sparse
  not supported']`. `'sparse not supported'` is independent of the kv_cache_dtype reason, so
  this isn't a config nuance, GLM-5.2's DSA sparse indexer cannot run on `TRITON_MLA` at all. Both
  sparse-specific alternatives (`FLASHMLA_SPARSE`, `FLASH_ATTN_MLA_SPARSE`) already explicitly
  exclude capability 12 in their own gates. For GB10 plus a sparse-indexer model on 0.27,
  `FLASHINFER_MLA_SPARSE_SM120` is not our choice, it is the only one that exists.
- **MoE backend.** Less locked down: Triton WNA16 MoE has no capability gate at all, only
  quant-detail conditions. It got further than the attention attempt, weight loading completed
  fully, then failed with a clean assertion once it reached the MTP drafter's own layer
  construction: `assert weight_quant.strategy == "group"` in
  `compressed_tensors_moe_wna16.py:122`. The drafter's quantization strategy isn't `"group"`;
  Marlin's construction path doesn't hit this assertion. Marlin remains the only MoE backend
  this checkpoint boots under.

## Where this leaves #51921

Eliminated as necessary conditions, each with a positive control, not just an absence of
evidence: speculative decoding, the full upstream `#51538` patch, cudagraph mode (all four),
acute and chronic memory pressure, and both attention/MoE backend alternatives available in this
vLLM version. What remains: a confirmed, reproducible, multi-minute stuck state with all four
ranks individually blocked in different native kernels, no NCCL frame visible in any Python
stack, and a debugger that cannot itself interrupt the target. That's the shape of the bug now,
not a guess about it.
