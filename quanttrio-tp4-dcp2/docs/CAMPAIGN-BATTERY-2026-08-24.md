# Campaign battery 2026-08-24: reproduced baseline, k5-adaptive falsified, determinism advantage

Three-day autonomous session against the production identity on this fleet
(QuantTrio Int4-Int8Mix, B12X_MLA_SPARSE, nvfp4_ds_mla KV, TP4+DCP4,
315,968 ctx, mp backend, V1 runner, MTP k=2 fixed). Full record in the
glm52-campaign repository.

## Reproduced baseline (two independent boots, all gates passed)

| probe | boot 1 | boot 2 (clean slate) |
|---|---|---|
| determinism temp-0 | 0/3 diverged | 0/3 diverged |
| prose C1 | 12.1 / 11.9 tok/s | 12.22 / 10.89 tok/s |
| prose C4 agg | 29.0 tok/s | 29.08 tok/s |
| peak C1 | 15.5 tok/s | 15.14 tok/s |
| peak C4 agg | 37.6 tok/s | 38.54 tok/s |
| prefill gate @161K | PASS 187.9 tok/s | PASS 188.3 tok/s |

Determinism note: this stack reproduces byte-identical temp-0 output across
boots where other stacks on the same model report 2/3 diverged.

## Experiment: k=5 adaptive MTP - falsified on this stack

Configuration: num_speculative_tokens=5 with adaptive window enabled
(speculative.py overlay exposing the adaptive field), graph ladder [6,12,18,24].
Boot healthy, gate ALL PASSED including concurrent x8. Battery:

| probe | k=2 baseline | k=5 adaptive |
|---|---|---|
| prose C1 | 12.1 / 11.9 | 8.92 / 6.3 |
| prose C4 agg | 29.0 | ~19.4 |
| accepted/draft | ~1.15 | up to 1.75 |

Reading: deeper speculation accepts more tokens per draft exactly as expected,
but verifying six tokens per step costs more than it saves on the V1-runner /
mp-backend public stack. Net 35 to 45 percent slower. Adaptive-k5 economics do
not transfer without the runtime stack that originally measured them.

## Operational root causes behind the "boot wedge" epidemic

All four are now fixed and documented in spark-fleet-guard:

1. earlyoom (-m 4 --prefer python3|VLLM|vllm|ray::) SIGTERMs workers during
   CUDA graph capture; systemd presets re-enable it after every reboot.
2. watchdog.service hard-resets nodes when its userspace feeder stalls under
   capture memory pressure; also re-enables after reboot.
3. Orphaned VLLM workers escape containers after failed boots and hold memory
   plus CUDA context invisible to container cleanup.
4. Rule: one clean boot per node-reboot cycle when wedges appear. Retry-
   rolling a poisoned driver state burns hours and confounds every variable.

## Where the comparison to published numbers stands

Direct comparison between this stack and external headline figures is a
category error until three axes match: content class (peak predictable-code vs
ordinary prose), context envelope (DCP1-399K-old-runtime vs DCP4-316K), and
speculation regime (adaptive k5 private runtime vs fixed k2 public). The
campaign repo carries like-for-like anchors for each axis so future claims can
be tested instead of guessed at.
