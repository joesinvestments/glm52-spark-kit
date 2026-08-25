# TailQuant DESIGN v3 — real-weight BTX encode via adapted exllamav3 quantizer
Status: v3 draft 2026-08-25 overnight (build-directive P1b). Supersedes v2's S3a seam
claim (v2 wrongly assumed upstream prepare paths encode; they wrap/synth only).

## 0. Headline finding (de-risks everything below)

exllamav3's codebook constant `codebook_mcg_mult = 0xCBAC1FED`
(vendor/exllamav3/modules-quant/exl3_lib/quantize.py:18) is IDENTICAL to b12x `mcg`'s
multiplier (b12x/moe/_shared/trellis_codebooks.py docstring). SparkRing's EXL3 lineage
and b12x's trellis family share the MCG codebook. The adaptation targets MCG first;
SQG_e4m3 second.

## 1. Decode laws (both sides, now fully specified)

### MCG (shared family)
CUDA reference: vendor/exllamav3/ext-quant-src/codebook.cuh `decode_3inst<cb=1>`:
    x *= 0xCBAC1FED (mod 2^32)
    x = lop3(x, 0x8FFF8FFF, 0x3B603B60, imm=0x6A)  # (x & mask) | or_mask
    value = fp16(high16(x)) + fp16(low16(x))
Torch port plan: uint32 mul via int64 arithmetic & 0xFFFFFFFF; lop3 as bitwise
and/or on uint32; view as two uint16 -> reinterpret fp16 -> add.
VALIDATION GATE: build 256-entry table, compare against exl3 kernel behavior via
its own tests (if runnable off-GPU) and against b12x mcg consumers; document any
discrepancy verbatim.

### SQG_e4m3 (b12x-specific; exllamav3 has NO equivalent)
Pure-torch builder ships upstream: b12x/_lib/quant/sqg_e4m3.py (CPU LUT builders,
no CUDA import needed):
    codeword16 = history(16-K bits) || branch(K bits)
    phase   = mix(history, 0x65AF, 0x16BF, shifts (6,4,5))     # width-masked mix
    syndrome= mix(history ^ 0x5105, 0x8693, 0x2A21, (2,4,4)) & branch_mask
    stratum = (7 * (reversed_branch ^ syndrome)) & branch_mask
    rank    = (stratum << width) | phase
    label   = T12_lut[rank]           # e4m3 byte, NaN(0x80)/Inf(low 7F) excluded
Trellis state = history; branch = emitted K-bit index. Encoder search space per
group = 2^K branches under a carried history state (true trellis).

## 2. Emit-path mapping (exllamav3 -> BTX)

| exllamav3 stage | file | BTX counterpart | gap |
|---|---|---|---|
| LDLQ/faux-LDLQ optimal ordering | quantize.py block_ldl/ldlq/fallback_quant | none (BTX stores plain tiles) | reuse as-is pre-transform |
| Hadamard preapply | preapply_had_l/r | rotations/suh/svh tables + rotation_draws | must EMIT rotation tensors instead of only transforming |
| tile quantization (GPU CUDA) | ext-quant-src/quantize*.cu via quantize_tiles() | NEEDS CPU twin targeting MCG/SQG laws above | THE core work item |
| indices [tiles,256] u16 | pack_trellis() | atoms rows: bundle = gate||up||down planes [hidden/16][16*bits]i16 K-major | repack/layout writer (have layout spec from btx-checkpoint-format.md + synth writer) |
| per-tile scales (codebook_scale 1.24371088, g_scale search) | sample_scale_tiles/g_scale_gss | BTX has NO embedded scale sections (format doc) | OPEN QUESTION #1 below |
| signs packing | pack_signs() | sign bit inside codewords? (MCG law output sign from high bit) | verify against decode law numerically |

## 3. Scale handling — OPEN QUESTION #1 (blocks nothing tonight)

EXL3 quantizes scaled tiles; b12x BTX containers carry no scales. Hypotheses:
(a) b12x serving applies suh/svh/rotation tables AND implicit unit scale => encoder
    must fold scale into weights pre-encode (lossy for wide dynamic range);
(b) scales ride outside the container in the vLLM quant-config/plugin path
    (mechanism used by AEON/QSRT checkpoints per DESIGN v2 note).
Resolution path: read how mixed_trellis/btx.py dequant consumes scales at serve
time (single-node probe, window-gated) OR find scale plumbing in btx_schema
metadata. Recorded as question; do not guess.

## 4. Round-trip protocol for P1d (one expert pair)

Ground truth decoder = pure-torch ports of section 1 (NOT the CUDA kernels; those
need GPU). Metric recorded verbatim whatever it is: relative Frobenius error
||W - D(E(W))||_F / ||W||_F per matrix (gate/up/down), plus max-abs error, at
K4 (hot) and K3 (cold). No curation.

## 5. Substep log (updated as work completes)

- [x] a. vendored (commit d52b94f)
- [x] b. this document
- [x] c. encode path PROVEN at tile level on REAL checkpoint weights
      (expert 0 down_proj/gate_proj, layer 5, gx10-1 pair copied to Mac 26MB;
       compressed-tensors unpack semantics from helpers.py unpack_from_int32:
       LSB-first nibbles, signed offset -8, symmetric g128 scales)
- [x] d. round-trip metric RECORDED VERBATIM (K=4 naive greedy, no transforms):
      gate_proj tiles n=8: relF p50 0.3799 mean 0.4062 worst 0.4942,
        maxAbsErr p50 0.0335 worst 0.0519 (orig units), scale p50 69.9
      down_proj tiles n=8: relF p50 0.4456 mean 0.4576 worst 0.5381,
        maxAbsErr p50 0.0437 worst 0.0835, scale p50 74.1
      gaussian sanity tile (no outliers): relF 0.27
      MCG LUT property: 65536 windows -> 2585 distinct values, range +-3.997,
      empty bands near +-3.5 and +-2.5 (staircase structure)
      Gap vs serving-grade = exactly the mapped reuse items: Hadamard
      pre-transform (outlier spreading), LDLQ ordering, beam>greedy, g_scale -
      all present in vendored exllamav3 pipeline, none yet ported.
- [ ] e. batch harness: encoder VECTORIZED (encode_vec2.py, 11ms/tile,
      bit-identical to scalar greedy on cross-check tiles => full matrix
      ~2.5h single-thread); harness launch deferred until quality levers land - full-matrix encode
      at current CPU throughput is ~hours/matrix and the quality levers above
      change the numbers anyway; run it after those land

## LANE-2A VERDICT (2026-08-25, directive 2)
Uniform-K2-sqg_e4m3 requires NO calibration input: confirmed from btx_synth config
surface (uniform rates declare bits, not per-expert tables; nothing consumes
activations) - router histograms were solely the hot/cold placement consumer, now dead.
Encoder input = weights alone. B1 stays deprioritized per directive.

Blocking unknown named (work item, has oracle): exact SQG trellis STATE TRANSITION for
CPU encode - stored bits per group are K-bit branches, history is carried state;
sqg_e4m3.py gives rank=frozen_graph(history,branch)->staircase label exactly, but the
history-update rule + bit-order across the 16-channel group must be pinned before the
harness emits servable bytes. ORACLE available offline:
tests/quantization/test_sqg_e4m3.py validates builders against
sqg_xor_cheb_t12_direct_lut_cpu() - candidate state machines verify without GPU.
K2 constraint note: mcg supports K3-K6 only => K2 MUST be sqg_e4m3 (no mcg shortcut).
Coupled-vs-uncoupled: production-QUALIFIED is coupled K2; uncoupled acceptance by
reader/planning unverified - second open item, same oracle bootstraps the check.

## LANE-2 ACCEPTANCE REFRAME (2026-08-25, Joe-endorsed)
Two-sided acceptance for the uniform-K2 encoder, both halves required:
1. VALUE side (proven): round-trip decode returns correct numbers (P1c/d).
2. LAYOUT side (pending): bytes accepted by upstream chain - synth-writer layout +
   extents + manifest + fused PLANNER pass at container scale. A5 proved a
   value-correct container can still be refused by planning (mixed kinds).
Corollary recorded: "uniform-K2 passes planning" remains an INFERENCE from A5's
rejection message structure, not a demonstrated pass. The oracle run's FIRST output =
demonstrate uniform-K2 planning PASS (or fail-closed reshape) BEFORE the 380G grind is
too deep to cheaply redirect.
Gate flag: prepare chain needs b3 kernel import chain => one CPU-only container on
gx10-1 (--gpus none, MemAvailable-gated, bounded minutes, synth-scale input), purpose-
limited to uniform-planning confirmation + oracle capture. AWAITING JOE'S WORD -
standing rule keeps node actions gated; all oracle inputs prepared locally meanwhile.
