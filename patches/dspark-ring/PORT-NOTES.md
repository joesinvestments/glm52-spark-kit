# DSpark ring-buffer draft KV: port of bird/vllm-lil@dspark-ring-1m-20260711 onto vLLM 0.27.0 MRv2

Source commits (Apache-2.0, bird), base June e232d262; our target is v0.27.0 (Aug 10). The delta
between his files and 0.27.0 is ~4,000 lines but most is upstream drift; the logic to port is the
patch series here (~1,000 lines net):

| patch | what | 0.27.0 target |
|---|---|---|
| 0001 08ced786 | VLLM_DSPARK_DRAFT_WINDOW opt-in windowed draft (speculators/dspark) | config/speculative.py, spec_decode/dspark/utils.py |
| 0002 2a0e74d7 | ring-buffer draft KV (VLLM_DSPARK_DRAFT_RING=1) for dspark/dflash | spec_decode/dflash/speculator.py, spec_decode/dspark/speculator.py, models/qwen3_dflash.py |
| 0003 7c6ad9e2 | deterministic ingest: keep only trailing-W per row | same |
| 0005 a344d0d3 | kill per-step host sync in bookkeeping | same |
| 0006 3f7b6181 | sync-free ingest via trash-slot redirect | same |
| 0011 2dc1673c | skip mask rebuild in steady long-context state | same |
| 0012-0014 d457a49b, 9cc7a3aa, 8f0dec4b | env-gated training-data capture (VLLM_DSPARK_CAPTURE_DIR): raw per-layer aux hiddens + token ids | spec_decode/dspark/speculator.py, model_runner.py |
| 0015 fd0c1011 | trim prefix-cache hits by the draft window (VLLM_DSPARK_CACHE_TRIM) | core/kv_cache_manager.py, core/sched/scheduler.py |
| 0016 ecd43757 | confidence-gated variable draft submission (bs=1) | dspark/speculator.py |
| 0017 b7abab51 | hoist slot addressing out of the per-layer ingest loop; fuse per-layer context k-norm | qwen3_dflash.py |
| 0090 dd26ab3a | cp_utils: exempt spec-less (ring) draft layers from DCP attention checks | v1/worker/cp_utils.py (4 lines) |
| 0091 2f2e04b2 | kv_cache: host non-MLA draft layers in the full-MLA group at DCP=1; VLLM_KV_SKIP_LAYERS_DTYPE | v1/core/kv_cache_utils.py, layers/attention/attention.py |
| 0092 5b79ea9c | weight_utils: tolerate non-dict hf_overrides in get_quant_config (our wall 1) | model_loader/weight_utils.py (9 lines) |

Why it matters (measured 2026-08-17 on our fleet): stock 0.27.0 DSpark with the quant-tuned drafter
serves correctly at DCP1 but at 0.755 accepted/step (bird: ~2.1 with the same drafter, ring on).
The dense drafter cannot do DCP at all without the ring (TRITON_ATTN has no DCP decode path);
0090 only helps once draft layers own no allocator KV.

Numbering follows the series as extracted (0004/0007 duplicate 0090/0092; 0008-0010 are dflash quant-aware ingest, needed only for quantized drafts).

Order of work: 0092 (trivial) -> 0091 -> 0001/0002 (the ring, re-derived onto 0.27.0's
dflash/dspark speculators, which moved since June) -> 0008 (cache trim; without it every
prefix-cache hit collapses acceptance) -> 0090 -> validate at DCP1 for acceptance parity with bird
(~2.1) -> DCP4 -> 0007 capture hook (unblocks training our own head) -> 0009/0010 polish.

## Port status 2026-08-17 12:50 UTC (overlays/dspark-ring/, source tree ~/Desktop/O14-BUILD/vllm-0.27.0 branch ring-port)

Re-derived onto v0.27.0 MRv2: 0092, 0001, 0002, 0003, 0005, 0006, 0011, 0012-0014 (capture, final
form: raw per-layer aux + target-batch ids), 0015 (cache trim), 0090 (as a `get_kv_cache_spec is None`
exemption in `check_attention_cp_compatibility`; 0.27.0 has no `dcp_replicated` path at all).
Not ported: 0091 (targets the DeepseekV4-only grouping path GLM never takes, and is moot once the ring
owns the draft KV), 0016 (bs=1 confidence gating, WIP with debug prints), 0017 (k-norm fuse + slot hoist
polish; next after acceptance parity).

Structural differences vs bird's June tree that needed hand work:
- `DFlashQwen3Attention` uses stock `Attention` in 0.27.0; the ring needs the draft layers to register NO
  allocator KV, so a `_DFlashDraftAttention(Attention)` subclass returns spec None in ring mode (MRv2's
  `attn_utils.get_kv_cache_spec` skips None specs; `bind_kv_cache` only iterates allocated layers).
- `prepare_dflash_inputs` gained `temperature`/`seeds` args; the ring branch calls it with the scratch slot
  sinks + `_ring_dummy_block_table` and `max_model_len` as the "block size" (every lookup -> row 0).
- `_layer_group_idx` / `_group_causal` defaults must be set BEFORE the ring early-return in `set_attn`.
- 0xdfi's MRv2 runner passes `num_speculative_tokens=` to `propose`; the ported file accepts it.
- The capture body (0012/0013) never applies by fuzz on 0.27.0; inserted by hand at the
  `self.hidden_states[:num_target_tokens].copy_` anchor.

Files: qwen3_dflash.py, dflash_speculator.py, kv_cache_manager.py, cp_utils.py, speculators_algos.py.
Launchers: launch/champion_dspark_ring_32k.sh (DCP1), champion_dspark_ring_dcp4_32k.sh.
Env: VLLM_DSPARK_DRAFT_RING=1 VLLM_DSPARK_DRAFT_WINDOW=1024 (VLLM_DSPARK_CACHE_TRIM defaults to W;
VLLM_DSPARK_CAPTURE_DIR / _EVERY for training data).
