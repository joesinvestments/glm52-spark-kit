# WINDOW-2 MANIFEST — D6 capture boot (drafter finetune data harvest)

Status: DRAFT window request for Joe (build-directive-2 Lane 1b). Nothing runs until
approved. Build state feeding this window: Lane 1a loop PROVEN offline (hook builder ->
cap-*.pt -> rig ingest -> optimizer step -> checkpoint, commit-recorded); hook format
unit-tested (P2); capture-boot launcher wired dormant (CAPD env).

## What boots

DSpark-ring serving identity (champion_dspark_ring_dcp4_32k.sh lineage), production
weights, hook ARMED via CAPD=<dir> (VLLM_DSPARK_CAPTURE_DIR + VLLM_DSPARK_CAPTURE_EVERY).
This is a DIFFERENT serving path than MTP production (ring drafter, proven bootable
2026-08-17); traffic during the window trains the drafter, it does not serve users.

## Two design fixes this window REQUIRES (both code-side, offline-tested before boot)

1. Min-T guard in the capture hook: skip saves when num_target_tokens < 64 (rig loader
   hard-drops files below max(64, MIN_A+K+2); organic decode steps at C16 are ~48
   tokens -> without the guard most captures would be silently discarded). One-line
   overlay change + unit test.
2. Append-safe writes: already true for this hook (each cap-*.pt is written whole at
   capture time - no atexit dependency; the tailquant-histogram lesson does not apply
   here, recorded explicitly). Add host-side periodic `sync` in the driver loop anyway.

## Rate math (why 3.5 hours)

Target ~600K (hidden-state, token) pairs. With min-T guard active and C>=22 sustained:
avg T~80/file -> 7.5K files x ~4.9MB ~= 37GB total.
VLLM_DSPARK_CAPTURE_EVERY=5 => ~2.9K files/hour ~= 230K tokens/hour => ~3h for 600K;
budget 3.5h window. EVERY=50 (organic default) would need ~33h - rejected.

## Traffic

NATURAL prose-class only: live Hermes/routing traffic if present, else battery-prompt
REPLAY at C24 sustained (replay text is real prose prompts, not synthetic probe text;
the never-probe-text rule targets repetitive/synthetic content poisoning calibration -
battery prompts are the approved prose class used by every prior battery).
Driver: looped probe.py batteries labeled win2-capture-NC (non-comparable, ignored).

## G-rails (same as window 1)

G1 P0 handoff: pause chain cleanly before takedown; resume after end-state.
G2 relauncher hold: sentinel currently disabled (pre-existing); launch-lock held for
window duration EXCEPT during our own launches; re-verified after every takedown.
G3 abort ladder: identical (one retry after reboot-cycle; two skips abort; node-loss =
immediate stop-clean; end states: done/partial-with-restore, or stopped-clean+incident).

Rollback: relaunch production MTP identity via proven launcher (config untouched);
capture dir is append-only side data, deletable.

## Cells

W2-0 Pre-flight gates (mem/disk/daemons/orphans) - 10m
W2-1 Ring boot healthy: gate ALL PASSED at DCP4 ring config, acceptance sanity vs
     2026-08-17 record (~2.1 accepted/step bird-parity target) - 35m
W2-2 Capture hour x3.5: EVERY=5, min-T guard on, mixed prose-class load, periodic sync,
     per-hour file+byte count logged (expect ~2.9K files/hour) - 3h30m
W2-3 Harvest: rsync caps to Mac + Storage staging, sha256 manifest, spot-load 20 files
     through rig loader (T>=64 enforced), delete node copies - 25m
W2-4 Restore production identity clean, gate, hand-back - 40m

Total: ~5h15m.

Post-window (separate windows, same rails): W3 DDP finetune run(s) from staged -ft init
(rig, 4-node, hours); W4 acceptance pairing boot (gate + battery + GSM8K-vs-floor +
acceptance-at-production-temperature vs 1.15 baseline; distribution-preserving check:
any quality shift under the new drafter is a bug, not a trade).

Lane-1(e) note: veloGB10 src/dflash2/ and src/dspark/ are the read-only reference
oracles for drafter/capture semantics diffing before W3 training starts.
