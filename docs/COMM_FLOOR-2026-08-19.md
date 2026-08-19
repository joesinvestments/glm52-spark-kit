# Communication floor on the 4x Spark rails vs what NCCL delivers (2026-08-19)

Motivation: the decode-step profile puts ~390 collectives per step (78 query all-gathers, 156 TP
all-reduces, ~25 indexer gathers, 78 DCP all-to-alls) at ~25% of the step, all latency-bound. This
measures the physical floor and what NCCL actually achieves, 4 ranks, one GPU per node, 200G RoCEv2.

## Raw RDMA write latency (perftest `ib_write_lat`, one rail, one QP, node to node)
| message | t_typical |
|---|---|
| 8 KB | 4.8 us |
| 64 KB | 10.1 us |
| 160 KB | 17.5 us |

## NCCL 2.30.4, 4 ranks, torch.distributed, per call (300-iter average, `docs/../benchmarks/comm/nccl_micro.py`)
| collective (size as in a decode step) | default (RING_LL) | LL128 | Simple | Tree |
|---|---|---|---|---|
| all_reduce 150 KB (12 tok x 6144 bf16, C4 TP all-reduce) | 90.4 | 76.7 | 84.1 | error |
| all_reduce 37 KB (C1) | 50.3 | 78.2 | 71.5 | error |
| all_gather 55 KB/rank x4 (q heads, C1) | 49.4 | 48.3 | 55.3 | error |
| all_to_all 50 KB (DCP partials, C1) | 37.9 | 38.1 | 35.8 | error |
NCCL point-to-point emulation (batch_isend_irecv to 3 peers + local reduce): all_reduce 150 KB 74 us,
37 KB 54 us, all_gather 67 us: no better (per-op host overhead ~20 us).

## Reading
A direct 4-rank collective (each rank one-sided-writes its chunk to the three peers in parallel, GPU-side
completion flags, local reduce) has a floor of ~12-20 us at these sizes: 3-4x below NCCL ring, and no
NCCL algo/proto reaches it. Per decode step that is on the order of 14 ms of a ~120 ms step at C1
(~12% single stream), more at C4 where messages are larger and comm share is higher. It requires a
custom RDMA transport (ibverbs QPs over both rails, MR registration of the unified-memory buffers,
RDMA WRITE with immediate, GPU-visible completion for CUDA-graph safety), i.e. a real systems project,
not a flag. Nothing else on the comm path is worth a fleet boot; the fused indexer gather was measured
at ~1.5% and parked.
