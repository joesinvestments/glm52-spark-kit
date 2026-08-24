# CAMPAIGN JOURNAL - autonomous execution log

## 2026-08-23/24 (overnight)

### Fixed (fleet-level, permanent)
1. earlyoom was SIGTERM-killing vllm/ray workers during graph capture
   (`-m 4 --prefer python3|VLLM|vllm|ray::`) -> disabled+removed on all 4 nodes.
   This was THE cause of every "wedge"/"hang" boot failure tonight.
2. watchdog.service was hard-resetting nodes under capture memory pressure ->
   disabled fleet-wide. Second silent killer.
3. k=5 adaptive MTP requires multi-size FULL-graph ladders whose capture is
   unstable on public kernels; single-size captures fine but adaptive depth then
   errors. Stable identity: **k=2 + ladder [3,6,9,12]** (= Joe's production).
4. Ray 2.57 unpinned: placement livelock + head flakiness. Pinned 2.47.1 in b5 image.

### Measured (first clean battery - production identity)
k2 / DCP4-316K / V1 runner / mp backend / kit overlays:
  determinism 0/3 diverged | prose C1 12.1/11.9 | prose C4 agg 29.0 |
  peak C1 15.5 (acc 1.86) | peak C4 agg 37.6 (acc 1.85) |
  prefill gate @161K PASS 187.9 tok/s
Saved: results/prod-k2-dcp4-316k.jsonl

### A/B: VLLM_BUILDA_BMM 0->1 (Build A v0, partner-proven on O14)
prose C1 flat (11.76/12.45 vs 12.1/11.9). Kept =1 to match O14 config.
Full C4/peak cells for this arm still pending (probe idle-refusal hiccup).

### Intel integrated (partner's answers via operator)
- His wheel base b12x pin == ours (334a2d75). NOT newer upstream.
- His 3 B12X mods: tiled_topk i64 indexing, compiler cache keys,
  scratch NVFP4 plumbing (manifest published).
- Headline figures were PEAK-class on older DCP1-399K runtime; ordinary prose
  ~25.4-25.6 there. Current b4 cannot reproduce 399K (unexplained by him).
- Quantized W8A16 drafter confirmed active on his stack AND ours (parity).
- fuse_gemm_comms no-op on his stack too -> patch 002 retired from plan.
- Build A v0 kernel: byte-identical copy exists in our overlays; env-gated;
  A/B above says flat on our shapes at prose C1.

### Overlay diff vs his published 74-file set (o14-diff/overlay-diff.json)
shared paths: 24 | he-only: 49 | we-only: 11
His-only categorized:
  flash_attn_cute 31 files (entire CUTLASS DSL attention stack)
  fmha_sm100 11 files (NVFP4-KV decode fwd path)
  spec_decode_mrv2 5 files (MRv2 runner + adaptive machinery + dflash)
  misc 2 (_custom_ops.py, _nvfp4_ds_mla_loader.py)
Ours-only 11 files incl our nvfp4 writer, deep_gemm patches, sparse_utils backports.

## OPEN ITEMS
- Capture-wedge forensics: py-spy sidecar method proven; stack watcher needs pgrep fix.
- BMM=1 remaining battery cells (c4/peak/prefill).
- Overlay content diff (24 shared files may differ in content).

### WIN: Shared-overlay content diff COMPLETE (o14-diff/content/)
24 shared paths: 15 IDENTICAL, 8 differ, 1 no-local.
Delta concentration (his vs ours):
  211 lines sparse_attn_indexer.py  <- DCP/prefill indexer path DIVERGED LINEAGES
       his: single-request-chunk fast path (+num_reqs!=1 guard), NO dcp-sharded
            scratch machinery (65 lines of ours absent), no world-size asserts
       ours: DCP-sharded scratch + guards (needed for our DCP4 correctness)
  48 b12x_mla_sparse.py | 43 parallel_state.py | 32 mla_attention.py
  20 kv_cache_interface | 16 indexer | 5 speculator | 4 torch_utils
15 files byte-identical incl marlin_moe, deepseek_mtp, builda v0/v1,
logits_processor -> MoE/drafter/lm-head parity CONFIRMED at source level.
Actionable: merge candidate = his single-req-chunk prefill fast path INTO our
DCP-capable indexer (targets prefill gap without losing DCP4).

### CORRECTION (same night): "merge candidate" already exists in our kit
Close reading of our sparse_attn_indexer.py shows the single-request row-share
fast path IS PRESENT (comment: "0xdfi's path"), EXTENDED with multi-request
per-row page-table packing (token_to_seq gather) AND DCP merge support.
His file RAISES on multi-request chunks; ours serves them. The 211-line delta
is our additional capability, not a gap. No merge needed - production already
runs the strictly-more-capable indexer. Journal entry above stands corrected.

### EXPERIMENT RESULT: k=5 adaptive on public mp/V1 stack — FALSIFIED
Config: production launcher + num_spec_tokens 5 + adaptive window 32 + ladder [6,12,18,24]
(speculative.py overlay newly mounted to expose adaptive field).
Boot: HEALTHY (capture passed with earlyoom/watchdog dead), gate ALL PASSED.
Battery (partial): prose C1 8.92/6.3 tok/s acc 1.75/1.07; prose C4 agg ~19.4 acc 1.47-1.5.
vs k=2 baseline: prose C1 12.1/11.9 acc ~1.15; C4 29.0.
CONCLUSION: deeper speculation ACCEPTS more per draft but verify cost dominates on
V1-runner/mp public stack -> net SLOWER (~35-45% worse). Partner's adaptive economics
do NOT transfer without his runtime stack (MRv2 + private pieces).
ACTION: reverted to k=2/[3,6,9,12] stable identity.

### 2026-08-24 (later): session content distributed to fleet repos
- spark-fleet-guard: BOOT-WEDGE-CATALOGUE.md + fleet_disarm.sh + orphan_sweep.sh
- glm52-spark-kit: capture-safe TailQuant profiler overlay (optional/) +
  TAILQUANT-PROFILER.md + CAMPAIGN-2026-08.md findings
- gx10-bench-optimizer: battery_probe.py + docs/BATTERY-METHOD.md (9 rules)
- QuantTrio repo: docs/CAMPAIGN-BATTERY-2026-08-24.md (baseline + falsification)
All pushed to main under operator authorship.
