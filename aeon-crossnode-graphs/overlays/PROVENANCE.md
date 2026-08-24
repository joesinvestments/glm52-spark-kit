# Overlay provenance

Every file baked into the image, with origin. "0xdfi (identical)" = unmodified copy of a file from the public 0xdfi GLM-5.2 4x DGX Spark work; "derived from 0xdfi overlay, modified here" = that lineage plus changes in this repo; "vLLM/DeepSeek-V4 sm12x lineage" = files from public vLLM pull requests carried unchanged; "ours" = written for this repo's parent kit.

| file | md5 | provenance |
|---|---|---|
| `b12x/attention/indexer/tiled_topk.py` | `b99382a527416267c31c3e17daabb57b` | 0xdfi (identical): b12x/b12x/attention/indexer/tiled_topk.py |
| `b12x/integration/sparse_mla_scratch.py` | `0a0964882512c3bc81ef8774155d7129` | derived from 0xdfi overlay, modified here |
| `vllm/compilation/b12x_capture.py` | `5dcdaa63e05b1325ddb711888085cf9a` | 0xdfi (identical): vllm/vllm/compilation/b12x_capture.py |
| `vllm/config/cache.py` | `f697993e7ccf0438719124b3dcfed41c` | 0xdfi (identical): vllm/vllm/config/cache.py |
| `vllm/config/scheduler.py` | `5ed5ee8cf9f8b418caf6f831ab2cc19b` | 0xdfi (identical): vllm/vllm/config/scheduler.py |
| `vllm/config/vllm.py` | `81732d73c5afb76d08ab65cf5a72288c` | 0xdfi (identical): vllm/vllm/config/vllm.py |
| `vllm/distributed/device_communicators/shm_broadcast.py` | `3407e99def761bdf1499cf12ad4900db` | ours |
| `vllm/distributed/parallel_state.py` | `9e9283d47546f07c136c30180bb70fb5` | derived from 0xdfi overlay, modified here |
| `vllm/engine/arg_utils.py` | `4033e6f28cadc04968855161608b0049` | 0xdfi (identical): vllm/vllm/engine/arg_utils.py |
| `vllm/model_executor/layers/attention/builda_bmm_v0.py` | `4ba5d699babc0b1cfbd484fd88f37825` | 0xdfi (identical): vllm/vllm/model_executor/layers/attention/builda_bmm_v0.py |
| `vllm/model_executor/layers/attention/builda_bmm_v1.py` | `4fe97b938b0948dd061228a045eccdfe` | 0xdfi (identical): vllm/vllm/model_executor/layers/attention/builda_bmm_v1.py |
| `vllm/model_executor/layers/attention/mla_attention.py` | `d964ffd099df063b3d7ea8cd4dd634c8` | 0xdfi (identical): vllm/vllm/model_executor/layers/attention/mla_attention.py |
| `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` | `b7bcdc8239c4857413548681e187b8d9` | 0xdfi (identical): vllm/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py |
| `vllm/model_executor/layers/logits_processor.py` | `a35f610b34b9bc8f954ca9a8db91ceae` | 0xdfi (identical): vllm/vllm/model_executor/layers/logits_processor.py |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | `2d739e36043d07ba6b6939b85f84e8a8` | derived from 0xdfi overlay: + DCP>1 top-k merge (_merge_b12x_dcp_topk), multi-request chunks, padded decode, DCP-sharded profile-time scratch reservation |
| `vllm/model_executor/model_loader/weight_utils.py` | `ec59b01f17b303e9d6703a40a4821d40` | 9-line tolerance for non-dict hf_overrides in get_quant_config (from bird/vllm-lil dspark-ring-1m-20260711, Apache-2.0) |
| `vllm/model_executor/models/deepseek_mtp.py` | `9ce74b3a832309a7d84845d62669888e` | 0xdfi (identical): vllm/vllm/model_executor/models/deepseek_mtp.py |
| `vllm/utils/torch_utils.py` | `6e6c0f4ae7916471912dbe2bec6b3c34` | 3-way merge: base stock v0.27.0, ours (0xdfi lineage, 2-line delta), theirs = AEON v0.27.1 (76 lines, keeps `nvfp4_kv_cache_split_views` for the Triton NVFP4-KV path); `git merge-file` clean |
| `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` | `64ef5546a816ed5bc6526d7a565b5f3f` | derived from 0xdfi overlay, modified here |
| `vllm/v1/attention/backends/mla/indexer.py` | `ab472dba11caf698f077c2059447437c` | derived from 0xdfi overlay, modified here |
| `vllm/v1/attention/backends/mla/nvfp4_ds_mla_writer.py` | `be148c45226f0eb4d5b09b5f57244f1c` | ours |
| `vllm/v1/attention/backends/mla/patch_deep_gemm_ops.py` | `74cf36eff87f97ab5c9761d39ef2fb84` | vLLM/DeepSeek-V4 sm12x lineage (public PRs), carried unchanged |
| `vllm/v1/attention/backends/mla/sm12x_deep_gemm_fallbacks.py` | `e198dbf2a9170c7196c4073c6b76ab70` | vLLM/DeepSeek-V4 sm12x lineage (public PRs), carried unchanged |
| `vllm/v1/attention/backends/mla/sm12x_mqa.py` | `28e6dd13bbaa05598d5aff5d6285b2c5` | vLLM/DeepSeek-V4 sm12x lineage (public PRs), carried unchanged |
| `vllm/v1/attention/backends/registry.py` | `df49aead5287a8f151963ddecd5ccc2e` | 0xdfi (identical): vllm/vllm/v1/attention/backends/registry.py |
| `vllm/v1/attention/ops/deepseek_v4_ops/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | derived from 0xdfi overlay, modified here |
| `vllm/v1/attention/ops/deepseek_v4_ops/sm12x_mqa.py` | `28e6dd13bbaa05598d5aff5d6285b2c5` | vLLM/DeepSeek-V4 sm12x lineage (public PRs), carried unchanged |
| `vllm/v1/kv_cache_interface.py` | `2b43fb164e5a9a2e74ebfbb01c48f683` | 3-way merge: base stock v0.27.0, ours (0xdfi lineage, 15-line delta), theirs = AEON v0.27.1 (28 lines); `git merge-file` clean |
| `vllm/v1/spec_decode/dynamic/acceptance_length.py` | `0920040a9d387bff195a911dfa066340` | 0xdfi (identical): vllm/vllm/v1/spec_decode/dynamic/acceptance_length.py |
| `vllm/v1/spec_decode/dynamic/depth_ladder.py` | `01799728e21b3a73635fc68993e6f926` | 0xdfi (identical): vllm/vllm/v1/spec_decode/dynamic/depth_ladder.py |
