# GLM-5.2 on 4x DGX Spark: source-form runtime kit

Everything needed to run GLM-5.2 (QuantTrio W4A16 / Int8Mix) on four DGX Spark
nodes (GB10, sm_121a, one GPU per node, RoCEv2) on stock
`vllm/vllm-openai:v0.27.0-aarch64` plus `pip install b12x`, as a set of overlay
files mounted over site-packages. No rebuilt image, no private wheels, no donor
binaries. Every file has an md5 and a provenance line in `MANIFEST.json`.

Measured on this fleet, correctness-gated, and reproducible from the launcher in
`launch/`. Numbers are cold, cache-busted, streamed, with the first post-boot
batch discarded.

## What is in here that is not published anywhere else

| piece | file | what it does | status |
|---|---|---|---|
| **nvfp4_ds_mla KV writer, source form** | `kernels/nvfp4_ds_mla_writer.py` | Packs the 368-byte compact record (E2M1 nope nibbles, E4M3 group-16 scales, fp32 rope_scale, E4M3 rope) that the B12X reader consumes. Fused Triton kernel, one launch per call. Registered as `torch.ops.vllm.nvfp4_ds_mla_write` with a fake impl. | byte-identical to the torch reference on every tested shape; CUDA-graph capturable; 51x faster than the torch reference (0.765 ms to 0.015 ms per 4096-token write); `torch.library.opcheck` passes |
| **B12X sparse indexer at DCP > 1** | `overlays/vllm/model_executor/layers/sparse_attn_indexer.py` | The public B12X indexer raises `NotImplementedError` above DCP 1. This adds the DCP top-k merge (`_merge_b12x_dcp_topk`): pack local (score, global_id), all-gather, stable top-k. | pack path byte-identical to the stock Triton kernel; proven live at DCP 4: gate pass, +3 to +8% at 32K vs the fallback path, slower at 316K (pool -27%), so documented and not adopted for production |
| **sparse-MLA indexer width fix** | `overlays/vllm/v1/attention/backends/mla/indexer.py` | Padded decode rows with `decode_len = 0` reached a `repeat_interleave` branch that assumed the DCP-sharded and global block-table widths match. Engine died on the first affected request. Fix reads the buffer's real width. | proven: a gapped capture ladder that used to kill the engine now serves |
| **one-GPU-per-node topology short-circuit** | `overlays/vllm/distributed/parallel_state.py` | `VLLM_ONE_GPU_PER_NODE=1` skips the gloo `in_the_same_node_as` probe that stalled 91 s on one rank against a 90 s timeout and caused ~40% boot failures at DCP > 1. | in production |
| **DCP-sharded indexer scratch** | `overlays/vllm/model_executor/layers/sparse_attn_indexer.py` | The profile run reserved the paged-indexer scratch for the full `max_model_len` on every rank. Under DCP a rank holds ~1/dcp of a request's rows; at DCP 4 the reservation was 4x too large and exhausted GB10 unified memory at 316K (`NV_ERR_NO_MEMORY`). Now `ceil(rows / dcp) + interleave slack`. | 316K/DCP4 profile reservation 315,968 to 78,994 rows |
| **node protection** | `launch/node_protect.sh` | SBSA hardware watchdog daemon (load-only test; no ping, no min-memory: both produced false resets), sshd/dockerd OOM immunity, `plymouth-quit-wait` masked. earlyoom is installed but disabled (it killed a healthy boot during graph capture). Replaces the power button when a load thrashes the box. | live on all four nodes; see the spark-fleet-guard repo for the current versions |
| **persistent dual-rail addressing** | `launch/rails_netplan.sh` | Static RoCEv2 rail IPs via netplan (rail A 192.168.100.x, rail B 192.168.101.x, MTU 9000). Rail IPs used to die with every reboot and recovery failed at GID resolution. | live on all four nodes |
| **boot retry with real-generation health check** | `launch/resolve_gid_and_launch.sh` | Sequential retry (never concurrent: concurrent retries loaded 98 GB four times and forced a power cycle), memory-reclaim gate between attempts, health = a real completion, not `/v1/models` 200. | in production |

The nvfp4 writer is the piece the other public GB10 stacks ship only as a
binary `.so` extracted from a private image. This one is Python you can read,
diff, and upstream.

Three exactness lessons from making the fused writer byte-identical, each caught
by the byte-compare test rather than by reasoning:

1. Triton fp32 `/` is `div.full.f32` (about 2 ulp) and flips E2M1 rounding
   boundaries. Tensor/tensor divisions use `libdevice.div_rn`.
2. Multiply-by-reciprocal is not division at the ulp level.
3. PyTorch computes `tensor / python_scalar` as `tensor * fp32(1/scalar)`, not as
   a division, while `tensor / tensor` is a true division. The reference uses
   both. The kernel mirrors each exactly.

## Measured against 0xdfi's carried kernels (honest)

`builda_bmm` v0/v1 (Triton MLA-absorb BMM) at our decode shapes on the stock
0.27.0 image: cuBLAS `torch.bmm` 6.2 us vs 10.1 us for M <= 24 (23-67% slower),
equal at M 36-48, faster only at M >= 96. v1 == v0 within noise everywhere. Both
are carried in the overlay for completeness and **disabled** (`VLLM_BUILDA_BMM=0`)
in the production launcher. His measured +27% was on a different torch/cuBLAS build.

## Carried from 0xdfi's O14 harness (identical, credited in the manifest)

B12X capture prewarm, `builda_bmm` v0/v1 Triton MLA-absorb BMM, Marlin MoE
atomic-add gate, adaptive MTP depth ladder (scheduler, config, drafter, telemetry),
`mla_attention` and `logits_processor` overlays, `cache.py`, `tiled_topk.py`,
`registry.py`. Files we changed are marked "derived" with the reason.
`NOTICE` carries attribution.

## Layout

```
MANIFEST.json      every overlay file: site-packages target, md5, sets, provenance
apply.sh           emits docker -v mounts for a set (core | indexer | adaptive | all)
verify.sh          md5-checks the overlay on this host or on every host given
overlays/          files mirrored at their site-packages paths
kernels/           the nvfp4 writer (also mounted from overlays/) and its standalone twin
tests/             byte-identity, CUDA-graph replay, opcheck, DCP pack identity
launch/            production launcher, retry wrapper, staged experiment launchers
benchmarks/        probes: correctness gate, mid-context, long-context A/B/C, shared prefix
docs/              RECOMMENDATION, DSPARK, NVFP4_DS_MLA_RECORD, SESSION-2026-08-17
```

## Install on a fleet

```bash
for h in node1 node2 node3 node4; do rsync -a overlays/ $h:/var/tmp/glm52-overlay/; done
./verify.sh /var/tmp/glm52-overlay node1 node2 node3 node4     # fails closed on any mismatch
docker run ... $(./apply.sh core /var/tmp/glm52-overlay) ... vllm/vllm-openai:v0.27.0-aarch64 ...
```

`launch/launch_gx10.sh` is the full production command (attention backend
`B12X_MLA_SPARSE`, TP 4, DCP 4, `--dcp-comm-backend a2a`, MTP depth 2,
`--kv-cache-dtype nvfp4_ds_mla`, 315,968 context, `--max-num-seqs 16`,
dense capture ladder). Set `DRY_RUN=1` to print the argv without touching the fleet.

## Tests

```bash
docker run --rm --gpus all -v $PWD/kernels:/w:ro -v $PWD/tests:/t:ro --entrypoint python3 IMAGE /t/test_nvfp4_writer_identity.py
docker run --rm --gpus all -v $PWD/kernels:/w:ro -v $PWD/tests:/t:ro --entrypoint python3 IMAGE /t/test_nvfp4_writer_opcheck.py
```

Expected: `dispatcher(fused) == torch reference: byte-identical`, `captured +
replay == reference`, `opcheck PASSED`.

## Results

See `docs/RECOMMENDATION.md` for the measured configuration and the trade
between the DCP 4 warehouse and DCP 1 speed, and `docs/SESSION-2026-08-17.md` for the
session log including the negative results.


## State as of 2026-08-17 evening, and where to pick up

Read in this order: `docs/SESSION-2026-08-17.md` (the session log, every result with the config it
was measured at), `docs/DSPARK.md` (the ring port, the acceptance matrix, why
it does not beat MTP k=2 yet), `docs/RECOMMENDATION.md` (why production stays DCP=4), `platform/README.md`
(the AEON-base image), `docs/NVFP4_DS_MLA_RECORD.md` (the KV record and the writer).

Facts a newcomer needs:
- Production body: QuantTrio Int4-Int8Mix unpruned, V1 runner, MTP k=2, TP=4 + DCP=4, B12X sparse MLA,
  nvfp4_ds_mla KV. Launchers in `launch/`. It is the memory-optimal full-intelligence body for 121 GB nodes.
- Full-NVFP4 (nvidia modelopt) boots on the platform image with the CUTLASS FP4 MoE kernel; 107 GB/rank,
  not the decode path (see the AEON hand-back repo, `NVFP4.md`). Pruning is not on the table.
- Open work, in priority order: a decode-step profile at C1 and C4 (we run ~50 ms where the bandwidth bound
  says ~15; nobody has published this breakdown), then attack the top slice (comm fusion, Marlin small-M,
  graph coverage); a drafter finetuned on captures from this stack (`launch/dspark_ddp_finetune.sh`,
  capture hook in `overlays/dspark-ring/`); PRs upstream for the DCP-aware indexer and the writer.
- Companion repos: `spark-fleet-guard` (the failsafes that stopped the power cycles) and
  `glm52-aeon-crossnode-graphs` (the AEON hand-back: platform Dockerfile, cross-node graph finding).

## License

Apache-2.0. Attribution for carried work in `NOTICE`.
