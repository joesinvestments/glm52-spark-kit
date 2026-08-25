# WINDOW-1 MANIFEST — first acceptance window (every cell is a built thing being accepted)

Status: FINAL-FORM DRAFT for Joe's morning approval 2026-08-25. Nothing in this document
runs until Joe picks the start hour. Production stays up until the window opens.

Estimated total wall time: **~5h15m** (phase A 2h50m, phase B 2h05m, slack 20m).

Standing rules honored: battery runs on NON-instrumented boots only; every boot packs its
cells; earlyoom/watchdog stay masked (structural, fleet_disarm.sh); after any failed boot,
full node reboot before retry (one-clean-boot-per-reboot-cycle lesson); first benchmark
batch after any boot is discarded (cold-start ~14%); commit author joesinvestments only.

## Phase A — clean boots (non-instrumented)

### A0. Pre-flight (fleet up, no boot) — 10 min
Commands (read-only + gates):
- `free -g` + `df -h /` on all four nodes (floor: MemAvailable >= 8GiB, disk <= 96%)
- `systemctl is-enabled earlyoom watchdog` == masked/inactive everywhere
- orphan sweep check: `pgrep -f "VLLM::"` == empty on all nodes
Pass: all green. Fail: abort window, report.

### A1. b3 parity gate (candidate: glm52-collab:b3, production config, k=2) — 60 min
Boot: stop vllm_slot fleet-wide, pkill -9 -f "VLLM::", drop caches,
`FORCE_RELAUNCH=1` with image pinned to glm52-collab:b3 (new-b12x 1.2.6 @36bce2c15,
ray[cgraph]==2.47.1, overlays baked kit-light, patch 001 only).
Cells in this one boot:
1. correctness gate (correctness_probe.py) — must be ALL PASSED
2. determinism temp-0 x3 — must be 0/3 diverged
3. full battery -> results/win1-a1.jsonl (discard first batch)
GSM8K cell: accuracy vs tonight's baseline JSON, McNemar vs noise floor.
Pass: gate PASS + determinism clean + battery within noise floor of b0 baseline
(prose C1 12.1 band) + GSM8K delta not significant (McNemar p>0.05).
Fail: capture outgoing log tail BEFORE docker rm -f; rollback = FORCE_RELAUNCH on
glm52-collab:b0 (known-good); investigate off-window.

### A2. Ladder extension cell [3,6,9,12,24,48] on accepted b3 (config-only) — 45 min
Motivation (D5+, measured): production dominant state is 16 running requests => ~48-token
steps vs ladder cap 12 => most live decode runs off-graph today.
Boot: same as A1 + ladder arg extended to 3,6,9,12,24,48.
KV-DELTA CHECK FIRST-CLASS: record KV pool allocation + host free at boot;
adoption requires KV-delta <= agreed budget AND host MemAvailable floor intact AND
prefix-cache hit rate unchanged (canary). If 48 breaks budget: retry once with
[3,6,9,12,24]; if that passes, adopt partial.
Cells: battery incl explicit C8 concurrency cell + C16 spot; verify from logs that
batches >12 now hit captured graphs (grep capture-size hits / eager counter == 0).
GSM8K cell vs baseline.
Pass: no regression beyond noise floor vs A1 + off-graph decode eliminated at observed
concurrency + KV delta within budget.
Rollback: revert ladder arg, relaunch (config-only).

### A3. k=4 packed cell (bonus, ONE slot) — 45 min
Per ASK-RESULTS ask7 verdict: no recorded A/B exists at production shape; 14.7 is a
reason to test, not a target.
Boot: k=4 with dense-capture ladder [5,10,15,20].
Cells: battery (same harness) + stability watch through capture (k>=3 ladders were the
historical instability class; earlyoom/watchdog must be confirmed masked pre-boot).
Pass: beats k=2 winner beyond noise floor on prose AND peak cells, zero errors under
concurrency. Fail on EITHER: keep k=2, record falsification at production shape.
Rollback: k=2 config relaunch.

### A4. SIRCL GPU-ordering probe (containers DOWN between A3 and phase B) — 15 min
The decisive unknown from d3-transport-comparison.md. Exact spec there; summary:
spark_transport_probe --memory cuda-mapped --gpu-producer --gpu-verifier --gpu-roundtrip
(two nodes) + spark_tp4_probe two-node eager AR at 37KB/147KB decode shapes.
MemAvailable gate >=8GiB host + >=4GiB free VRAM both nodes immediately before launch.
Pass: verifier true + latency p50 recorded vs NCCL 38-90us baselines.
Fail: B4 stays blocked-pending-analysis; NCCL remains the collective; no further SIRCL
work until root-caused. Rollback: n/a (standalone binaries, containers untouched).

### A5. Synth-BTX stage 3 (toy prepare probe in b3 container) — 5 min
Runs the already-staged toy probe's [prepare] stage inside glm52-collab:b3
(campaign-2026-08/tailquant/probe_btx_mixed_toy.py; 655KB container, CPU-only).
Pass/fail IS the data either way: OK => mixed P44/P33 loads+prepares (mixed-rate thesis
alive); fail-closed error => record verbatim, uniform-K fallback thesis stands
(uniform coupled K2 sqg_e4m3 = production-qualified structure).
Rollback: n/a (ephemeral container).

## Phase B — instrumented (NON-COMPARABLE, labeled loudly)

### B1. Profiler-armed capture hour (TailQuant harvest) — 75 min
Relaunch production identity with patch-004 router profiler ARMED
(VLLM_TAILQUANT_PROFILE_DIR set; overlay mounted runtime-style per current practice -
confirm exact mount line with Joe at window open).
Traffic: one hour MIXED-CLASS live traffic including real-shape prompts (not
battery-prose only); NO battery numbers recorded from this boot (non-comparable label).
Deliverables: router_hist.json harvested -> split.py plan committed ->
unblocks B2 real-weight encode (S3a) and the exllamav3-adapted quantizer work.
Rollback: none needed (env-gated dormant instrumentation; relaunch without env).

### B2. Hand-back boot (winner config from A1-A3 votes) — 40 min
Clean relaunch of the winning identity; discard first batch; quick sanity battery;
production handed back. GSM8K chain resumes client-side after this boot is healthy.

## Post-window (no boot): accept/reject memo per candidate + journal + push; winners
become the new production identity ONLY on Joe's explicit bless.
