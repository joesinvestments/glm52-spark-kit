# GLM-5.2 on 4x DGX Spark GB10 - Campaign State

Central record of the 3-day campaign: everything learned, everything built,
every finding, and the road forward. Companion artifacts live in this repo
(patches, tools, results, diffs). Fleet: gx10-1..4, DGX Spark GB10, 121GB
unified each, dual 200G RoCE rails.

---

## 1. Where the system stands right now

Production GLM-5.2 is UP and stable on all four nodes:
- Identity: QuantTrio Int4-Int8Mix unpruned, B12X_MLA_SPARSE attention,
  nvfp4_ds_mla KV, TP=4 + DCP=4, MTP k=2 fixed, ladder [3,6,9,12],
  mp backend / V1 runner, 315,968-token context, ~953K KV pool.
- Launcher: ~/glm-legacy-stack/launch_gx10.sh via resolve_gid_and_launch.sh
  (flock, memory gate, retry, sentinel integration).
- Gate ALL PASSED twice including concurrent x8. Determinism clean (0/3).

### Reproduced baseline battery (two independent boots agree)

| probe | result | accepted/draft |
|---|---|---|
| determinism (temp-0) | 0/3 diverged | |
| prose C1 | 12.1 / 11.9 / 12.22 / 10.89 tok/s across runs | ~1.15 |
| prose C4 agg | 29.0 / 29.08 tok/s | ~0.94-1.14 |
| prose C2 agg | 17.3 / 19.4 tok/s | ~1.0-1.18 |
| peak C1 | 15.5 / 15.14 tok/s | ~1.8 |
| peak C4 agg | 37.6 / 38.54 tok/s | ~1.87 |
| prefill gate @161K | PASS, 187.9 / 188.3 tok/s | |

Results files: results/prod-k2-dcp4-316k.jsonl, results/prod-clean-final.jsonl.

## 2. Root causes found and permanently fixed

These were killing boots silently for days before this campaign named them:

1. **earlyoom** ran with `-m 4 --prefer python3|VLLM|vllm|ray::` - executed
   workers the moment free memory dipped below 4%, which happens during CUDA
   graph capture. Disabled + removed on all nodes.
   Lesson: any "hang during capture" on this fleet - check earlyoom first.
2. **watchdog.service** hard-reset entire nodes when capture memory pressure
   stalled its feeder. Disabled fleet-wide. Same symptom family as (1).
3. **Orphaned VLLM workers** survive failed boots outside containers and hold
   memory/CUDA contexts. Cleanup procedure: pkill -9 -f "VLLM::" on every node
   after any failed boot, then drop caches.
4. **k=5 adaptive MTP requires multi-size FULL-graph ladders whose capture is
   unstable on public kernels.** Stable identity is k=2 + [3,6,9,12].
5. **Ray 2.57 unpinned** showed placement livelock + head flakiness;
   ray 2.47.1 pinned in image b5.
6. **/tmp on nodes purges periodically** - probes/scripts must live in ~ or be
   re-deployed per session.

Meta-pattern (empirical, tonight): after a failed boot, subsequent boots keep
failing until nodes reboot; one clean boot per reboot cycle was common. When
grinding experiments: reboot between failed attempts rather than retry-rolling.

## 3. What we already had (and nearly forgot)

All of it proved decisive once re-read:

- **glm52-spark-kit**: 35-file overlay set, launchers, gates, probes, docs.
  The production stack mounts exactly these. Byte-parity with partner's wheel
  confirmed for marlin_moe.py, deepseek_mtp.py, builda v0/v1, logits_processor.
- **docs/HANDOFF**: full ops knowledge incl. earlyoom history, boot rules,
  storage discipline, sentinel design.
- **docs/RECOMMENDATION.md**: DCP decision rationale, capture-ladder rule
  (dense multiples of 1+k), concurrency-crash post-mortem (vllm#51921).
- **docs/COMM_FLOOR + kernels/rdma/ag4_proto.c**: NCCL sits 3-4x above the
  physical RDMA floor at decode message sizes; prototype AG exists.
- **benchmarks/**: correctness_probe, nccl_micro, profile tooling.
- **dspark-training rig** + bird's finetuned drafter weights on Storage.
- **results/k5adaptive-partial.jsonl**: falsification evidence (below).

## 4. Partner intel (0xdfi), integrated and verified

His answers resolved every open question about his claims:

1. His wheel base b12x pin == ours (334a2d75). NOT newer upstream.
   Three B12X mods only: tiled_topk i64 indexing, compiler cache keys,
   scratch NVFP4 plumbing. Manifest published.
2. The custom kernel behind his historical O14 numbers is Build A v0 -
   byte-identical copy exists in our overlays. A/B on our production shapes:
   FLAT at prose C1 (11.76/12.45 vs 12.1/11.9). Kept enabled (=O14 config).
3. Headline figures (36.63 C1 / 80.56 C4 / 661 prefill) were PEAK-class
   predictable-code on an OLDER DCP1-399K runtime (~12.79GB KV/rank) that even
   he cannot reproduce today ("no single proven explanation for lost capacity").
   Ordinary cold-prose there was ~25.4-25.6. Not comparable to our DCP4-316K.
4. Quantized W8A16 drafter mapping confirmed active on his stack AND ours
   (deepseek_mtp.py overlays identical). Parity, not gap.
5. fuse_gemm_comms:true is a no-op on his current runtime too (dropped before
   engine init). None of his performance credits it. Patch 002 retired.
6. Publication boundary admitted: no OCI image, no private sidecar published,
   no clean public rebuild+benchmark reproduction claimed by him.

## 5. Experiments run and their verdicts

| experiment | verdict |
|---|---|
| Build A v0 kernel A/B (BUILDA_BMM 0 vs 1) | FLAT on our shapes at prose C1. Kept =1 to match O14. Full C4/peak cells pending. |
| k=5 adaptive MTP (window 32, ladder [6,12,18,24]) on public mp/V1 stack | FALSIFIED: healthy boot, gate passed, but prose C1 dropped to 6.3-8.92 (vs 12.1) and C4 to ~19.4 (vs 29). Deeper speculation accepts more (acc up to 1.75) but verify cost dominates. Evidence: results/k5adaptive-partial.jsonl |
| fuse_gemm_comms enablement | dead end - no-op on his stack and ours |
| indexer single-request fast path "merge" | already present in our kit (superset: multi-request packing + DCP merge). No action needed. |
| new-b12x (222 commits ahead) contains tiny-decode fixes, DCP top-k owner exchange, MX-FP6/Trellis tiers | CONFIRMED he does not have these. This is the leapfrog arm. |

## 6. Built this campaign (all in this repo)

- patches/: 001 index-share diagnostic log; 002 env-forced fuse_gemm_comms
  (retired from plan but documents the upstream plumbing bug); 003 launcher
  hardening (RDMA_DEV_ALL second-rail mounts, DROP_CACHES preflight,
  EXTRA_SERVE_ARGS + STRIP_SPEC); 004 TailQuant router-frequency profiler
  v2 CAPTURE-SAFE (index_add_ only, atexit dump, deployed dormant in
  production overlays on all nodes).
- images/: Dockerfile (parameterized b12x ref / DSL ver / ray spec),
  build_images.sh, sidecar_regen.py. Image lineage b0(old pin baseline,
  gate-passed) b2 b3(new b12x) b4(b3+profiler) b5(ray 2.47.1).
- benchmarks/probe.py: protocol-complete battery harness
  (/metrics deltas, idle-refusal, contamination self-check).
- tailquant/: split.py (histograms -> per-layer hot/cold plan, tested);
  btx_writer.py (plan -> mixed P44/P33 per-expert-pair rate tables ->
  write_btx_checkpoint; round-trip verified against BtxManifest parser);
  probe_btx_mixed.py; DESIGN.md v2 (architecture resolved: BTX-native mixed
  K4/K3 via upstream mixed_trellis path - zero serving patches needed).
- comm/floor_bench.sh + findings: decode-shape RDMA floor bench built and run;
  blocked at rdma_cm connect (event 8) between standalone apps while NCCL works
  - needs GID-index/TOS pinning matching production NCCL (next-session item).
- o14-diff/: overlay-diff.json + content diffs. Shared paths 24 (15 IDENTICAL
  incl MoE/drafter/lm-head parity at source level); his-only 49 categorized
  (flash_attn_cute 31, fmha_sm100 11, spec_decode_mrv2 5, misc 2); ours-only 11
  (nvfp4 writer, deep_gemm patches, sparse_utils backports, shm_broadcast fix).
- results/: three battery JSONLs + falsification evidence.
- JOURNAL.md, PHASE3-PLAN.md, run_phase2.sh, supv5.sh automation lineage.

## 7. Findings that change the picture

1. His published matrix rows and our measured baseline measure DIFFERENT
   THINGS: peak-class vs ordinary-class content, DCP1-399K-old-runtime vs
   DCP4-316K-current, k5-adaptive-private vs k2-public. Direct number
   comparison is category error. Like-for-like anchors now exist.
2. Determinism: OUR stack 0/3 diverged repeatedly; his reports 2/3 diverged.
   For agent workloads requiring reproducibility, ours is strictly better.
3. The capture-instability root causes were operational daemons, not kernels.
   With earlyoom+watchdog disarmed, k=2 boots reliably; k=5 ladders remain
   genuinely unstable on public kernels (open problem, forensics armed).
4. Adaptive-k5 economics do NOT transfer to public mp/V1 stack (falsified).
   If adaptive matters, it must come with his MRv2 runtime - see roadmap D1.
5. Upstream b12x moved 222 commits past both wheels: tiny-decode inactive-
   route fixes, route-packing scaling, graph-safe histograms, DCP top-k owner
   exchange over PCIe, MX-FP6 tier, Trellis K3/K4 mixed-rate serving with BTX
   containers. He has none of it. Neither does production yet. That is where
   leapfrog lives.

## 8. Roadmap from here (ranked)

D1. **TailQuant calibration + real-weight container** (flagship)
    Needs: one production restart window with profiler env armed + ~1h traffic
    (battery suffices); then split.py -> btx_writer with REAL encoded payloads
    (encode step = dequant experts + trellis atom assignment using upstream
    prepare paths + SQG/MCG codebooks shipped in-tree); offline quality gate;
    serving boot on BTX weights; full battery.
    Payoff if coverage model holds: ~25-35% weight-byte cut -> decode above any
    uniform-quant baseline + tens of GB freed for context capacity.
D2. **New-b12x port** (capability leapfrog)
    Minimal set first: route-packing fixes (#228/#219/#150/#226), PCIe DCP
    owner-exchange, topology-scoped AR. Requires porting kit overlays onto
    namespaced tree (mapping table already produced) OR running kit-light.
    Payoff: possibly closes remaining decode/prefill gap AND adds comm wins.
D3. **RDMA floor bench completion**
    Blocker precisely known: standalone rdma_cm needs GID-index/TOS pinning to
    match production NCCL context. Fix ag4_proto init (set IBV GID index 3,
    TOS), rerun sweep. Go/no-go data for custom collective investment.
D4. **Spec-path forensics** (only if k5 economics still wanted)
    py-spy sidecar method proven tonight; capture stacks of the dying worker
    during multi-size ladder capture would name the deadlock.
D5. **index_topk_pattern sweep** - config-only, zero risk, untried anywhere.

## 8b. Roadmap additions (2026-08-24, ideation-lane handoff adopted)

D6. **Drafter finetune targeting prose acceptance** - candidate #2 behind D1.
    Baseline: peak acceptance ~1.8 vs prose ~1.15. At k=2 tokens/step ~= the
    acceptance number, so prose decode runs at ~60 percent of peak-class.
    Moving prose acceptance 1.15 -> 1.5 takes prose C1 ~12 -> ~15.5 with zero
    serving changes, and STACKS with TailQuant. Rig + finetuned drafter
    confirmed staged (ASK-RESULTS-20260824.md ask 3). Ornith corollary: an
    uncalibrated donor head measured WORSE than no spec-decode (7.1 percent),
    so calibration to real traffic is the entire game. Measure acceptance at
    production temperature, matched content class.

D5+. **Cudagraph ladder coverage at real concurrency** - config lever for the
    next packed window. Ladder [3,6,9,12] covers C4 (needs 12) but NOT C8
    (needs 24): concurrent x8 likely decodes on a non-captured path today.
    VERIFY from metrics/logs whether production serves C8; if yes and KV
    headroom allows, extend ladder to [3,6,9,12,24] and check the KV delta.

WINDOW RULE. Clean boots are the scarce resource (one clean boot per reboot
cycle). Every restart window gets PACKED: profiler env armed + sweep cells +
ladder-coverage cell + standard battery, manifest designed before asking for
the window. Battery runs on NON-instrumented boots only; label loudly if a
profiler-armed boot must carry a battery.

## 9. Operating notes for whoever runs the next session

- Production restart procedure: stop vllm_slot containers fleet-wide,
  pkill -9 -f "VLLM::", drop caches, FORCE_RELAUNCH=1 resolve_gid_and_launch.sh.
- earlyoom/watchdog MUST stay disabled; check after every node reboot
  (they re-enable themselves via systemd presets).
- Probes persist in ~/probes/ on gx10-1 (probe.py, correctness_probe.py with
  MODEL patched to glm-5.2-quanttrio).
- Battery command: python3 ~/probes/probe.py --endpoint http://127.0.0.1:8210
  --model glm-5.2-quanttrio --label <name> --out <file> battery
- First benchmark batch after any boot: discard (cold-start penalty ~14%).
- Images glm52-collab:b0 (gate-passed baseline) and :b0oldpin exist on gx10-1;
  b3/b4/b5 were removed in disk cleanup but rebuild from images/Dockerfile.
- Disk discipline: nodes sit at 90%+; large artifact builds belong on /tmp of
  gx10-1 only, cleaned immediately (btx-full synthetic build alone was 57G).

## 10. What "done" looks like

The mission beat-line remains: exceed the published numbers with clean,
reproducible, fully-public inputs. Tonight established the honest floor
(production identity, reproduced), killed the wrong hypotheses with data, and
built the two engines (BTX pipeline, upstream-port path) that get us past him
rather than merely up to him.
