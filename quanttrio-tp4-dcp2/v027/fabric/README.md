# Do not change --all2all-backend on a 1-GPU-per-node RoCE fleet

Measured 2026-08-16. This is the largest single effect found in the entire
sweep, and it is negative.

`--enable-dbo` (dual batch overlap) is refused at config validation with the
default backend:

```
Microbatching currently only supports the deepep_low_latency,
deepep_high_throughput, and nixl_ep all2all backends.
allgather_reducescatter is not supported.
```

`deep_ep` and `nixl` are both present in the image, so the suggested fix boots.
It should not be used:

| shape | default (`allgather_reducescatter`) | `deepep_low_latency` | change |
|---|---|---|---|
| 2K x C4 decode | 48.40 tok/s | 20.20 | **-58%** |
| 2K x C16 decode | 99.56 | 62.13 | -38% |
| 16K x C4 decode | 49.88 | 7.66 | **-85%** |
| 16K x C4 prefill | 580.9 | 202.3 | **-65%** |
| 16K x C4 TTFT | 103.2 s | 295.9 s | **2.9x worse** |

Correctness passed throughout -- this is purely a performance collapse, which is
what makes it dangerous: nothing errors, it is simply three to seven times
slower.

## Why

DeepEP is built for NVLink/InfiniBand multi-GPU expert parallelism with fp8/nvfp4
MoE. This fleet is one GB10 per node over 200G RoCEv2, and the MoE path resolves
to MARLIN WNA16 (`compressed-tensors` -> `CompressedTensorsWNA16MoEMethod` ->
`Using 'MARLIN' WNA16 MoE backend`), not NVFP4. Expert dispatch that assumes a
fast intra-node fabric becomes a cross-node round trip per layer.

## Consequences

- **DBO is closed on this hardware** -- it cannot run without a DeepEP-family
  backend, and that backend is a severe regression here.
- **Keep `--all2all-backend` at its default.** The `--dcp-comm-backend a2a`
  setting is a *different* knob and remains correct (it reduces NCCL calls from
  3 to 2 per layer for MLA).
- Related: `--enable-expert-parallel` is untested (lost 2/2 boots to an
  unrelated init race) but is expected to regress for the same reason. Do not
  adopt it without measuring.
