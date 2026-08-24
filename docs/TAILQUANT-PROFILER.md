# TailQuant router-frequency profiler overlay (optional, not in MANIFEST)

File: `overlays/optional/tailquant/marlin_moe_profiler.py`

Drop-in replacement for `overlays/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`
that adds a capture-safe router-frequency histogram recorder. Purpose: collect
per-layer expert-hit distributions from live serving traffic as the empirical
basis for mixed-rate expert quantization (hot experts stay K4/NVFP4-class,
cold tail moves to cheaper atoms).

## Safety properties (why this version is boot-safe under FULL cudagraph)

v1 of this hook used `.item()` (host sync) and per-call allocations inside
`MarlinExperts.apply`. Under cudagraph FULL capture both are illegal and the
hook killed workers at exactly the capture phase. Three production boots were
lost to this before the redesign.

v2 rules enforced in the hot path:
- preallocated buffers sized on first warmup call (before capture starts)
- only `index_add_` into the fixed buffer afterwards
- no `.item()`, no host syncs, no allocations, no logging after first call
- dump happens at process exit via atexit, outside any capture

## Use

1. Deploy the file over the mounted marlin_moe overlay on every rank.
2. Arm with env: `VLLM_TAILQUANT_PROFILE_DIR=/var/tmp/tailquant`
   Unset env = byte-for-byte dormant except two dict lookups per call.
3. Drive representative traffic (battery or real corpus).
4. After shutdown, JSON files land in that dir, one per layer instance:
   `{layer_key: [hit_count_per_expert]}`.
5. Feed to `tailquant/split.py` (see glm52-campaign repo) to produce the
   hot/cold split plan.

## Status

Deployed dormant on all four production ranks since 2026-08-24. Awaiting a
restart window with env armed plus battery traffic for the first real corpus.
