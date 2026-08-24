# Campaign findings 2026-08-23/24: three-day autonomous session

Condensed record of the joint session with the partner runtime (0xdfi b4
recipe) as comparison substrate. Full detail lives in the glm52-campaign repo.

## Reproduced production baseline (two boots, all gates passed)

k=2 fixed, DCP4, 315,968 ctx, mp backend, V1 runner, kit overlays mounted:

| probe | result |
|---|---|
| determinism temp-0 | 0/3 diverged (his stack reports 2/3 diverged) |
| prose C1 | 12.1 / 11.9 / 12.22 / 10.89 tok/s |
| prose C4 agg | 29.0 / 29.08 tok/s |
| peak C1 | 15.5 / 15.14 tok/s |
| peak C4 agg | 37.6 / 38.54 tok/s |
| prefill gate @161K | PASS 187.9 / 188.3 tok/s |

## Root causes of the boot-wedge epidemic (all operational, not kernels)

1. earlyoom (`-m 4 --prefer python3|VLLM|vllm|ray::`) SIGTERMs workers during
   graph capture. Re-enables via presets after every reboot.
2. watchdog.service resets whole nodes when its feeder stalls under capture
   memory pressure. Also re-enables after reboot.
3. Orphaned `VLLM::Worker` processes escape containers on failed teardown,
   hold memory + CUDA context, and starve later boots on their node.
4. Empirical rule: one clean boot per node-reboot cycle when wedges appear.
   Retry-rolling on a poisoned driver state burns hours.

Both daemons now disabled and masked fleet-wide; disarm script added to
spark-fleet-guard (node/fleet_disarm.sh) plus orphan_sweep.sh.

## Experiment verdicts

| experiment | verdict |
|---|---|
| Build A v0 kernel A/B (BUILDA_BMM 1 vs 0) | flat at prose C1 on our shapes. Kept =1 matching O14. |
| k=5 adaptive MTP on public mp/V1 stack | FALSIFIED: C1 drops to 6.3-8.92 tok/s vs 12.1 at k=2 despite higher accepted/draft (up to 1.75). Verify cost dominates without his MRv2 runtime. |
| fuse_gemm_comms:true | no-op on his stack too (confirmed by him in writing). Retired. |
| single-request indexer fast path merge | already present in our overlay, extended with multi-request packing and DCP merge his version lacks. Nothing to port. |
| new-b12x upgrade arm | he confirmed his wheel lacks upstream tiny-decode fixes, route-packing scaling, PCIe DCP owner-exchange kernels. Leapfrog candidate, port pending. |

## TailQuant pipeline status

Format resolved: cold experts move to Trellis K3 atoms via the BTX container;
hot stay K4/NVFP4-class. Upstream mixed_trellis already serves one-launch
mixed-rate; per-expert-pair rates are first-class manifest features. No custom
serving kernel needed.

Built and verified this session:
- split.py: histograms to per-layer hot/cold plan (partition-tested)
- btx_writer.py: plan to mixed P44/P33 rate tables through upstream
  write_btx_checkpoint; round-trip validated by BtxManifest parser
- capture-safe profiler overlay deployed dormant on all four ranks

Remaining for flagship result: real-weight trellis encode step, offline
quality gate, serving battery on BTX weights.
