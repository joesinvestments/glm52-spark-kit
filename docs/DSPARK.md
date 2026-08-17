# DSpark block drafting on GLM-5.2, 4x DGX Spark, vLLM 0.27.0

Status 2026-08-17 16:30 UTC. bird's ring-buffer draft KV is PORTED onto 0.27.0 MRv2
(`overlays/dspark-ring/`, notes in `patches/dspark-ring/PORT-NOTES.md`) plus two fixes of ours
(rejected draft tokens -> ring trash slot; window visibility = seq_len - num_rejected). Ring DSpark boots
first try and passes the correctness gate at DCP1 AND DCP4 (first DSpark at DCP>1 on this stack; the
draft owns no allocator KV, so the 32K pool is 1.43-1.52M tokens vs 1.11M for MTP k=2), and the
training-data capture hook is live (`VLLM_DSPARK_CAPTURE_DIR`).

## Acceptance: measured, same probes, greedy unless stated (accepted draft tokens / step, k=3)

| config | greedy essays 4x512 | natural docs (py stdlib) C4 24x256 | battery temp 1.0 |
|---|---|---|---|
| ring DCP1, bird ft drafter | 0.92 (gate) | | 0.93 |
| ring DCP4, bird ft | 0.92 | | |
| ring DCP4, target KV fp8_ds_mla (bird's dtype) | 0.85 | | 0.95 |
| ring DCP4, RedHat base drafter (3L, current HF revision) | 0.98 | | 1.06 |
| ring DCP4 + rejected->trash + mask fix | 1.04 | 0.68 | 1.04 |
| production MTP k=2 DCP4 (reference) | 1.28 (63.9%/draft) | 1.42 (71%/draft), 19.5 tok/s e2e vs ring 11.7 | 1.15 |

Offline per-position draft accuracy (bird's trainer `EVAL=1`, teacher-forced) of the ft drafter on aux
hidden states captured from OUR target during the natural drive: p1=0.575 p2=0.431 p3=0.341 (chains to
~0.9/step). The served path therefore realizes ~75% of what this drafter can do on our target, and the
drafter itself is far below RedHat's published p0~0.75 (FP8 verifier) and bird's 2.03-2.23 (his stack,
his corpus). An earlier EVAL of 0.69/0.53/0.43 was on the battery's repeated-sentence filler and is void.

Ruled out by experiment: sampling temperature (greedy probe), target KV dtype (fp8 vs nvfp4 identical),
drafter revision (RedHat base and bird ft within 0.1 of each other), and by inspection: aux-layer capture
semantics, 1+N fill-in readout, weight loading, grouped rms_norm, mask token id, RoPE theta (8e6), the
prepare kernel (byte-identical to bird's). Two ring asymmetries vs the paged path found and fixed (+15%).

Speed at this acceptance (32K s4, 1K C4): ring DSpark DCP4 30.3-31.0 e2e / 9.6-9.7 per-req vs MTP k=2
DCP4 36-37.5 / 11.5. DSpark k=3 needs >~1.6 accepted/step to beat MTP k=2 per request on this stack;
bird's 24.7 vs 19.6 vs 13.7 was measured against NO speculation, not against MTP.

Path to make it pay: (1) finetune the drafter on OUR captures (pipeline adapted: launch/dspark_ddp_finetune.sh,
natural-corpus drive scratchpad natural_drive.py; needs the GPUs ~75 min); bird measured +25% from a
matched finetune. (2) the remaining served-vs-offline gap is not cleanly established: the offline EVAL is
teacher-forced on document text while served acceptance is on the model's own continuation; a fair
number needs EVAL on captures of the served decode tail (capture keeps every step at EVERY=1; the
trainer's loader drops files <64 tokens, so stitch decode captures per request before EVAL). At depth
the block's own ring slots hold the oldest window tokens (valid context), so that is not a defect. (3) k=7 (verify width 8) once
acceptance is up, on the b12x path only powers of two are legal.

## What DSpark is, in this stack

A standalone 5-layer dense drafter (speculators format `DSparkDraftModel` -> vLLM
`Qwen3DSparkModel`) that reads the target's aux hidden states from layers 8/23/39/55/70
(`aux_hidden_state_layer_ids`; GLM's class supports Eagle3-style aux export), drafts a block
of 8 in one parallel pass, then sequentially adds a low-rank Markov bias
(`markov_w1`: vocab x 256 embedding of the previously sampled token, `markov_w2`: 256 -> vocab)
so the block is coherent. `num_speculative_tokens` is `block_size - 1` = 7 max; on the b12x
path the verify width `1+k` must be a power of two, so k = 3 or 7 (bird: k=3 optimal, verify
width is MoE-bandwidth priced at ~10.5 ms/token).

Drafters that exist:
- `RedHatAI/GLM-5.2-speculator.dspark`: trained against FP8 GLM hidden states, B200-validated.
- `b1rd/GLM-5.2-speculator.dspark-quanttrio-int4-ft`: RedHat's, finetuned on the QuantTrio
  Int4-Int8Mix target's hidden states captured while serving on 4x GB10 (bird: ~1.45 -> ~2.1
  accepted/step, ~19.6 -> ~24.7 tok/s C1 at 1M ctx). This is the drafter under test.

## The walls, in order, and the fix for each (all in `launch/champion_dspark_32k.sh`)

1. **`hf_overrides must be a dict for get_quant_config`.** vLLM builds the speculators-format
   draft config with a callable `hf_overrides` (the `update_dspark` transform); the draft has
   no `quantization_config`; when the spec config leaves `quantization` empty the draft
   inherits the target's `compressed-tensors`, and `get_quant_config` falls through to the
   `hf_overrides` path and raises. `"quantization":"fp8"` hits the same path.
   Fix: give the drafter's local `config.json` an explicit `quantization_config`. Online fp8
   (`{"quant_method":"fp8","activation_scheme":"dynamic"}`) loads but then dies in the first
   draft forward with `BFloat16 != Float8_e4m3fn` (the Markov/aux heads are bf16 modules;
   `ignored_layers` names must match module paths exactly). Robust fix: an EMPTY
   compressed-tensors config (`config_groups: {}`, `ignore: []`, `format: dense`) so the
   drafter is bf16 everywhere (7.6 GB per rank; fine at 32K) and inheritance is satisfied.
   Upstream #49133 ("build draft under its own model/quant config") is the DSv4-flavored
   version of this problem and is not in v0.27.0.
2. **`TRITON_ATTN is not valid ... kv_cache_dtype not supported`.** The fleet-wide
   `--kv-cache-dtype nvfp4_ds_mla` is an MLA record; a dense drafter cannot use it.
   Fix: `"kv_cache_dtype":"auto"` in the speculative config (0.27.0 honors a per-draft KV
   dtype; `spec_decode/dspark/utils.py` passes `speculative_config.kv_cache_dtype`).
3. **`Decode Context Parallelism (DCP) requires attention implementations to return the
   softmax LSE during decode`.** DCP shards every attention layer's KV, including the
   drafter's dense layers, and `TRITON_ATTN` cannot serve DCP decode. This is the real
   integration work: bird's branch keeps the draft KV in a private ring buffer (last 1024
   tokens) outside the paged allocator, which makes the draft DCP-agnostic and, per his
   measurements, also fixes acceptance depth-collapse. Until that path is ported, DSpark on
   this stack runs at DCP=1 only.

Also required for the boot path: MRv2 (`VLLM_USE_V2_MODEL_RUNNER=1`, DSpark is MRv2-only),
0xdfi's MRv2 runner overlays (`mrv2/`), `--hf-overrides '{}'`, capture ladder multiples of
`1+k` (k=3: 4,8,12,16), `--disable-custom-all-reduce`.

## Measurement plan
Same probes as everything else (2K C4 / 2K C16 / 14K C4, gate) plus `benchmarks/spec_metrics.py`
for mean accepted tokens per step from `/metrics`. Fair pair at DCP=1: DSpark k=3 vs MTP k=2
(baseline-dcp1). DCP=4 comparison waits on the ring-buffer port.

## Training our own head
`dspark-training/` (bird's pipeline, provenance recorded). Blocker: his data-capture hook
(`VLLM_DSPARK_CAPTURE_DIR`) lives in his vLLM branch; port onto our MRv2 DSpark speculator is
the first task of that track.
