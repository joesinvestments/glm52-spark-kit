# Full-NVFP4 GLM-5.2 on four DGX Sparks, on the AEON image: it boots, it is correct, it is not the fast path

Date: 2026-08-17. Same fleet and image as `FINDING.md`. Checkpoint: `nvidia/GLM-5.2-NVFP4` (modelopt,
`quant_algo=NVFP4`, 433 GiB, 47 shards). As far as public recipes go, the unpruned 744B model with
NVFP4 weights at TP=4 on four Sparks has not been shown before: the published 4x recipes run int4 weights
(with NVFP4 only in the KV cache), and the published NVFP4-weight recipe is the REAP-pruned 469B on
three Sparks with pipeline parallelism.

## What happened

- Launcher: `launch/serve_glm52_tp4_dcp4.sh` with the model path swapped to the NVFP4 checkpoint,
  `--gpu-memory-utilization 0.92`, `--max-num-batched-tokens 1024`, no speculative config, capture sizes
  `[1,2,4]`. Everything else identical.
- The MoE resolved to the real FP4 kernel, verbatim:
  `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTEDSL_BATCHED', 'FLASHINFER_C...`
  and `Using MoEPrepareAndFinalizeNoDPEPModular`. The experts carry per-input global scales
  (`*.input_scale` F32 + `weight_scale` F8_E4M3 + `weight_scale_2` F32), so CUTLASS is eligible.
- Memory, verbatim: `Model loading took 106.86 GiB memory` per rank. At gmu 0.90:
  `Available KV cache memory: -0.12 GiB` and the engine refuses (`No available memory for the cache
  blocks`). At gmu 0.92: `GPU KV cache size: 136,704 tokens` at 32K, boot first try, correctness gate
  pass (4 and 8 concurrent, 17*23 -> 391).
- Speed, no speculation: 1K prompt C4: 25.5 tok/s e2e, 7.2 per request, prefill 551; 14K C4: 11.8 e2e.
  The same fleet on QuantTrio int4 + MTP k=2: 37.7 e2e, 11.6 per request, prefill 605-626.
- Adding the MTP drafter on top of the 107 GiB body at gmu 0.92 pushed all four nodes past physical
  memory during weight load; on unified memory that thrashes rather than fails. Do not run that
  combination.

## Why it is not the decode path on this hardware

- This checkpoint quantizes only the experts. Every `self_attn*`, `mlp.shared_experts*`, and layers 0-1
  stay bf16 (`hf_quant_config.json` exclude list). That is why it is 433 GiB, not smaller than int4.
- Even a fully quantized NVFP4 body cannot beat int4 on size for a 744B model: NVFP4 is 4-bit values plus
  an FP8 scale per 16 weights, about 4.5 bits per weight; QuantTrio's int4 group-128 is about 4.1.
  `RedHatAI/GLM-5.2-NVFP4-FP8` (FP8 attention, NVFP4 experts) is 446.8 GiB, larger still.
- GB10 decode is memory-bandwidth-bound (AEON's notes say the same). Speed follows bytes read per token,
  and NVFP4 reads more of them than int4 here. The CUTLASS FP4 kernel is real and it works; it cannot
  win the byte count.
- With 107-112 GiB of weights per rank there is no room for a speculative drafter or a useful KV pool
  on 121 GiB nodes without pruning experts, which trades away model quality.

So: a working, correct full-NVFP4 GLM-5.2 on four Sparks on the AEON image, useful as a platform for
NVFP4 kernel work, not as a serving configuration. The int4 body with MTP stays faster per request.
