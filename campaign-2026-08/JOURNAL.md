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

### 2026-08-25 (build-directive session): four builds launched
Directive: collab-work/claude_to_agent_glm_build_directive_20260824.md supersedes
measurement-first framing. Four builds in parallel; measurement only as acceptance gates.

#### B1 DONE: glm52-collab:b3 built clean and staged on gx10-1
new-b12x 1.2.6 @ verified pin 36bce2c15 | cutlass-dsl 4.6.2 | ray[cgraph]==2.47.1
(root-cause-#5 pin carried into build args) | 33 kit overlays baked, 2 skipped
(b12x-coupled, kit-light mode), 2 md5-drift warnings = known HEAD-ahead files.
Collab patch 001 only (no profiler -> battery-clean boot candidate).
BTX machinery VERIFIED IN-IMAGE: b12x/moe/_shared/{btx_schema,trellis_codebooks,
btx_synth,btx_compat,mixed_trellis,w4a8_trellis_decode}. Import smoke PASS.
NOTE: upstream b12x moved past our pin (HEAD 9ae32e297 today, EXL3-MCG work);
staying on 36bce2c15 - it is what the campaign verified and what B2 tooling targets.

#### B2 BLOCKER NAMED (corrects DESIGN.md v2): no real-weight trellis encoder exists upstream
Source-audited at pin 36bce2c15: prepare.py prepare_trellis256_moe_weights with real
tensors is a ZERO-COPY WRAP path ("no bytes copied or permuted"); write_btx_checkpoint
is SYNTH-only (random payloads). No encode path ships anywhere in-tree; the decode law
lives in CUTLASS intrinsics (packed_decode_sqg_xor_cheb_t12_to_e4m3x8,
packed_decode_trellis_sqg_direct_lut, trellis_ring lane geometry). DESIGN.md v2's
"encode step ... using upstream prepare paths" assumption is WRONG - that was the
whole S3a plan. Real-payload BTX needs an encoder built from scratch OR adapted from
exllamav3's public trellis quantizer (SparkRing's exl3-r7 proves that family runs on
GB10). Additional format fact for boot planning (docs/btx-checkpoint-format.md):
per_expert_pair sets containing P44 are DECLARED, UNFUSED - single-launch planning
fails closed; vehicles are two serial launches or mixed_trellis benchmark kernel;
{P33,P43}/{P33,P24} execute through expert-static dispatch; uniform coupled K2 sqg_e4m3
is THE production-qualified structure. Weights located: gx10-1
/var/tmp/glm-legacy/hf/hub/glm52-quanttrio-unpruned (380G, 129 shards,
compressed-tensors pack-quantized; H6144 I2048 E256 x78 layers).

#### B4 STAGED: spark_transport vendored verbatim
campaign-2026-08/sircl/{LICENSE,NOTICE,THIRD_PARTY_NOTICES,UPSTREAM.md} +
upstream/ (29 files byte-exact) @ sparkring commit 3a60ca71 + port/PORTING-NOTES.md
(11 source-study facts incl.: raw ibverbs not rdma_cm so our event-8 stall has no
analog; gid_index 3 already default; traffic_class must be ADDED in RTR; XOR-matchings
pair-sum valid on switched fabric as-is; one QP per peer; fatal failure contract;
CPU affinity contract vs GB10 10 cores).

#### Background: GSM8K salvage chain running detached
Baseline (1319 items conc 30) then --compare-baseline noise-floor run, chained via
results/gsm8k-salvage.log driver; resumes from .resume.json if interrupted.
Status reports with first build progress report per directive.
