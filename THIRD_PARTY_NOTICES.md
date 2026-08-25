# Third-party notices and credit chain

Aggregated attribution for everything this repository carries, derives from, or
measures against. Each consolidated subdirectory also keeps its own NOTICE;
this table is the one place that lists everything. If you find your work here
uncredited or miscredited, open an issue and it gets fixed first, argued never.

| author / project | what we carry or derive | where in this repo | license | link |
|---|---|---|---|---|
| 0xdfi | O14 overlay set (per-file identity in MANIFEST.json), probe battery protocol, answered campaign questions | overlays/, campaign-2026-08/benchmarks/probe.py, campaign-2026-08/o14-diff/ | Apache-2.0 | https://github.com/0xdfi/GLM-5.2-Harness-O14-4x-DGX-Spark |
| CosmicRaisins | adaptive-MTP overlays carried verbatim (commit-pinned in file headers); Triton sparse-MLA kernels; draft-quant packed-mapping patch; adaptive-MTP forward-port | overlays/vllm/v1/spec_decode/dynamic/, quanttrio-tp4-dcp2/legacy-stack/kernels/, quanttrio-tp4-dcp2/legacy-stack/draft-quant.patch | Apache-2.0 | https://github.com/CosmicRaisins/glm-5.2-gb10 |
| Luke Alonso / local-inference-lab | acceptance-length controller the adaptive overlays forward-port; b12x CuTe-DSL sparse-MLA + indexer kernels (pip package); llm-decode-bench benchmark tool referenced in campaign comparisons | lineage of overlays/vllm/v1/spec_decode/dynamic/; b12x at runtime; campaign records | Apache-2.0 / MIT (llm-decode-bench) | https://github.com/local-inference-lab |
| Aiden Le | acceptance-length adaptation in the adaptive-MTP lineage | via CosmicRaisins forward-port | Apache-2.0 | credited via upstream headers |
| ciprianveg | sparse-MLA + DeepGEMM-bypass mod scripts; decode-aware scheduler lineage | quanttrio-tp4-dcp2/legacy-stack/ | Apache-2.0 | NVIDIA forum thread 374125 |
| tonyd2wild | kernel redistribution with attribution intact; indexer-MTP overhang fix; NVFP4 KV-cache port | quanttrio-tp4-dcp2/legacy-stack/, quanttrio-tp4-dcp2/patches/ | Apache-2.0 | GLM-5.2-NVFP4-KV-4x-DGX-Spark |
| penguinchang | decode-aware scheduler lineage | quanttrio-tp4-dcp2/legacy-stack/mods/ | Apache-2.0 | credited via upstream |
| danielwoz | E2M1 store kernel under the NVFP4 KV port | via tonyd2wild's port | Apache-2.0 | vllm-dspark-nvfp4 |
| BTankut | 380K recipe, pinned-pool discipline, external deep-prefill reference numbers | methodology, quanttrio-tp4-dcp2/ docs | Apache-2.0 | glm-5.2-4x-dgx-spark |
| drowzeys | independent parallel nvfp4-MLA KV implementation (reference) | quanttrio-tp4-dcp2/ docs | Apache-2.0 | credited via NOTICE |
| AEON-7 | vllm-ultimate-dgx-spark sm_121a rebuild the aeon record's image builds FROM | aeon-crossnode-graphs/ | Apache-2.0 | vllm-ultimate-dgx-spark |
| bird / b1rd | GLM-spark speculator-training pipeline; weight_utils tolerance overlay; DSpark ring-drafting; quant-matched finetuned drafter checkpoint | dspark-training/ (PROVENANCE.md), aeon-crossnode-graphs/overlays/ | Apache-2.0 | https://github.com/bird/GLM-spark |
| FujitsuPolycom | SparkRing SIRCL transport, vendored full-tree with upstream LICENSE / NOTICE / THIRD_PARTY_NOTICES intact (includes NVIDIA NCCL portions) | campaign-2026-08/sircl/ | Apache-2.0 | https://github.com/FujitsuPolycom/sparkring |
| vLLM project | base serving engine (image vllm/vllm-openai:v0.27.0), overlay targets, sm12x fallback lineage | throughout | Apache-2.0 | https://github.com/vllm-project/vllm |
| NVIDIA | NCCL portions inside the vendored SIRCL patches; modelopt NVFP4 GLM-5.2 checkpoint | campaign-2026-08/sircl/spark_transport/nccl/; weights | Apache-2.0 / model license | https://github.com/NVIDIA/nccl |
| QuantTrio | GLM-5.2 Int4-Int8Mix unpruned checkpoint (the production weights) | served weights | model license | HuggingFace: QuantTrio |
| brandonmusic | GLM-5.2 EXL3 3.5-bpw quant (referenced in campaign comparisons; staged for evaluation) | campaign records | model license | HuggingFace: brandonmusic |
| exllamav3 project | trellis quantizer VENDORED (modules/quant + ext quant sources, MIT LICENSE verbatim) as the B2 encode-path base | campaign-2026-08/tailquant/vendor/exllamav3/ | MIT | https://github.com/turboderp-org/exllamav3 |

Every Apache-2.0 subtree vendored here keeps its upstream LICENSE and NOTICE
files in place, per Apache-2.0 section 4. Nothing in this table transfers
copyright; upstream files remain their authors'.
