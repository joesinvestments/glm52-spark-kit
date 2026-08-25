# LANE 3 PORT MAP — kit overlays onto new-tree GLM_NSA surface (pin 36bce2c15)

Status: v1 2026-08-25. Source-verified against raw tree + vendored files. This is the
work plan for rewriting the kit's ONE affected overlay
(overlays/vllm/v1/attention/backends/mla/b12x_mla_sparse.py, 1081 lines).

## Verified import rewrites (the A1 blocker)

OLD (dead at 36bce2c15):
    from b12x.integration.mla import (
        sparse_mla_decode_forward, sparse_mla_extend_forward)
    from b12x.integration.sparse_mla_scratch import (
        B12XSparseMLAScratchCaps, plan_sparse_mla_scratch)

NEW (source-verified same names, new homes):
    from b12x.attention._shared.mla.api import (
        sparse_mla_decode_forward,   # api.py:372
        sparse_mla_extend_forward)   # api.py:427
    from b12x.attention.sparse_mla._scratch import (
        B12XSparseMLAScratchCaps,    # _scratch.py:40
        plan_sparse_mla_scratch)     # _scratch.py:523, in __all__

## Contract deltas (field-level audit)

1. Caps: new B12XSparseMLAScratchCaps has kv_dtype (not kv_cache_dtype), NO
   scale_format field, ADDS mode ("decode"/"extend"/"verify"/"draft_extend" -
   our _make_plan's mode arg maps directly), max_batch, page_size etc.
2. NVFP4-KV scale plumbing MOVED from caps to forward kwargs:
   sparse_mla_decode_forward(..., scale_format: int|None = None) - api.py:389/441;
   "NVFP4 (scale_format=2)" first-class at api.py:587. Port = drop both kwargs from
   caps_kwargs; thread self._b12x_scale_format into each forward call.
3. Container/bind contract UNCHANGED: B12XSparseMLAScratch exposes exactly the
   duck-typed fields our overlay relies on (tmp_output/tmp_lse/output_buffer/
   final_lse/num_chunks_ptr/set_split_chunk_config; bind() at :142/:495).
4. plan_sparse_mla_scratch(caps) -> Plan shape preserved.

## Remaining audit items (next work blocks)

- sparse_mla_decode_forward FULL signature vs our two call sites (q/kv/indices args,
  return metadata MLASparseDecodeMetadata vs prior tuple).
- extend path + prefill strategy (_resolve_mla_prefill_strategy auto/single/split).
- Other overlays touching b12x integration surfaces (grep shows only this file).
- GLM_NSA batch=1 claim reproduction spec for the parity boot (verdict doc).

## Offline gates before any boot request (per directive)

b3-successor image build with rewritten overlay -> import-chain clean -> unit/shape
tests green (upstream tests/attention/* run CPU-side where possible) -> parity-boot
window request.

## AUDIT COMPLETE (2026-08-25 late): the port COLLAPSED to ~4 lines

Full signature audit vs our overlay's actual call sites:
1. Import paths: 2 lines (the only dead code). Same function names, new modules.
2. Caps construction: DELETE kv_cache_dtype= and scale_format= from caps_kwargs
   (moved to forward kwargs; our _b12x_nvfp4_kwargs(latent_scale) ALREADY threads
   scale_format per-call - written for this API generation).
3. Plan.bind(): signature IDENTICAL to our call sites INCLUDING scratch= caller-owned
   storage (matches workspace-manager get_simultaneous pattern verbatim).
   B12XSparseMLAScratch exposes all duck-typed fields (tmp_output/tmp_lse/
   output_buffer/final_lse/num_chunks_ptr/set_split_chunk_config).
4. Forward contracts: keyword-only binding/kv_cache/sm_scale/v_head_dim/
   forced_num_splits/return_lse/lse_scale("natural")/scale_format/fp8_rope -
   ALL present with same semantics. Returns Tensor|tuple matching our lse handling.
   fp8_rope param exists (production runs KV_FP8_ROPE=1).

CONSEQUENCE: the "days-class D2 port" estimate was wrong in OUR FAVOR - the overlay was
already authored against this API generation (binding-based plan/bind/run); window-1's
A1 failure was purely the stale import paths. Remaining D2 scope = b3-successor image
build with rewritten overlay + import gate + shape tests, then parity boot where the
GLM_NSA 1.2-1.6x claim gets verified. Rewrite itself: trivial, next work block.
