# TailQuant — frequency-aware mixed-precision MoE experts
Status: design v1 (2026-08-23). Owner: collab. Prereq data: patch 004 histograms.

## Thesis

~95% of GLM-5.2's weight bytes are MoE experts (78 layers x 256 experts x ~4.85 MB/rank).
At decode each token touches ~9% of them per layer (~23/256). Routing frequency is heavily
heavy-tailed. Today every expert is stored at one precision (int4 g128 in QuantTrio,
nvfp4 in the NVIDIA checkpoint) — paying full bytes for ballast that mostly never fires.

**TailQuant stores experts at precision proportional to measured routing frequency.**
Every expert stays (no pruning — operator constraint). Hot experts keep 4-bit; the cold
tail re-quantizes to 2-bit wna16 g128 (or MX-FP6 via new b12x as a safer middle tier).

## Expected math (per rank)

| split | avg bits/w | weights | vs today |
|---|---|---|---|
| all-4bit (status quo NVFP4) | ~4.5 | 106.9 GiB | baseline |
| top-64 hot @4b + 192 cold @2b | ~2.5 | **~62 GiB** | −42% |
| top-64 @4b + 192 @3b(eq) | ~3.25 | ~80 GiB | −25% |

Consequences: (a) KV headroom returns to QuantTrio-plus levels on the NVFP4 lineage;
(b) decode gets FASTER than any uniform build because most touched-expert bytes are the
cheap kind; (c) prefill bursts hit cold experts hardest → watch compute-bound regressions
there (3C forensics informs format choice for the tail).

## Pipeline (5 stages)

1. **Harvest** — patch 004 (`VLLM_TAILQUANT_PROFILE_DIR=/var/tmp/tailquant`) records
   per-layer expert-hit histograms from live traffic. Run during the standard battery +
   a Hermes-shaped replay segment. Output: router_hist.json per boot.
2. **Split** — per layer, sort experts by hits; knee detection (largest gap in sorted
   CDF, floor of 32 hot experts, cap 96) → hot set H_l, cold set C_l.
3. **Requant** — from source checkpoint: dequantize experts, GPTQ-style 2-bit (g128)
   calibration on our own capture corpus; emit a two-tensor layout per layer:
   W_hot[E_h, ...] @4bit-native-format, W_cold[E_c, ...] @2bit-packed.
   Tooling: modelopt dequant + calib harness in dspark-training rig style.
4. **Serve without kernel surgery** — partition experts into two ID-disjoint tensors;
   run the existing Marlin grouped-GEMM twice per layer (hot pass, cold pass), merge
   via moe_sum. Router masks route tokens to whichever tensor owns their expert.
   Requires: marlin_moe.py dual-tensor extension (~150 lines, env-gated
   VLLM_TAILQUANT_LAYOUT=<json>), plus quant-config carrying per-range bit widths.
   Fallback if Marlin rejects 2-bit packing: MX-FP6 tier through b12x 1.2.x stack
   (upstream already ships fp6 MoE serving kernels).
5. **Gate & tune** — correctness_probe identity gates, perplexity delta <0.5%, agent-task
   spot evals; optional recovery LoRA on the cold tail via the kit DDP finetune rig.

## Risks / open questions

- Marlin wna16 2-bit pack support in our overlay lineage (asserts {4,8} today) —
  decide 2-bit-pack vs FP6-tier after histograms land and a kernel feasibility spike.
- Quality cliff location unknown until stage 3; the split point is tunable post-hoc.
- Two GEMM launches per layer add fixed overhead (~launch cost) — acceptable only
  because both launches are inside existing CUDA graphs.

## Milestones

M1 histograms harvested (needs one healthy boot with b4 image)
M2 split tooling + offline perplexity of requant layers on single GPU
M3 dual-tensor marlin_moe extension boots at 32K, gate PASS
M4 full-shape battery vs B0 baseline -> promote/deny

## ARCHITECTURE REVISION v2 (2026-08-23 late): BTX-native, zero serving patches

Source audit of b12x 36bce2c15 (mixed_trellis.py, btx.py) shows:
- mixed_trellis = ONE-launch execution; route packer assigns experts to a combined
  namespace; per-tile dispatch picks bitrate-specialized K3/K4 decoders.
- btx-atoms-v1 containers carry RATE_STRUCTURE_PER_EXPERT_PAIR - per-expert bit
  rates are a first-class manifest feature ("the pair finalizer for per-expert ones").

=> TailQuant serving integration collapses to: WRITE A BTX CONTAINER whose manifest
assigns K4 atoms to hot experts and K3 atoms to cold experts, then boot with
--quantization/plugin pointing at it (mechanism: same path AEON/QSRT checkpoints use).
The dual-tensor marlin_moe patch is DROPPED (was higher-risk, duplicated upstream).

New stage list:
S3a BTX writer: QuantTrio int4 experts -> dequant -> per-expert trellis K4 (hot) /
    K3 (cold) atom encode -> btx-atoms-v1 manifest+shards.
    Reuse: prepare_trellis256_moe_weights reference paths + SQG/MCG codebooks shipped
    in-tree; calibration corpus from dspark captures.
S3b Feasibility spike BEFORE S3a: encode one layer's 256 experts as mixed K3/K4 BTX,
    load via btx.py on a single node, compare outputs vs bf16 reference (offline).
S4 Serving boot with BTX weights through existing mixed-trellis path (env-gated).
Quality note: K3 trellis (learned codebooks) historically >= GPTQ-3bit quality;
cold-set-only assignment concentrates any loss where routing already spends few hits.
