# RDMA-direct 4-rank collectives for the Spark fleet (prototype, 2026-08-19)

Why: docs/COMM_FLOOR-2026-08-19.md. ~390 latency-bound NCCL collectives per decode step; the raw RDMA
write floor on the rails is 5-17 us while NCCL costs 38-90 us at our sizes.

## Step 1 result: host-memory all-gather over ibverbs (`ag4_proto.c`)
Full-mesh RC QPs via rdma_cm over rail A, RDMA WRITE with immediate into each peer's slot, CPU polls
the CQ. Single rail, signaled sends, no rail striping (unoptimized).

| all-gather, 4 ranks | this prototype | NCCL 2.30.4 (same fleet) |
|---|---|---|
| 8 KB | 18.4 us | |
| 37 KB (C1 sizes) | 17.9 us (best 13.9) | 49-50 us |
| 147 KB (C4 sizes) | 42.3 us (best 39.4) | 90 us |
Content verified each run. 2.1-2.8x below NCCL; the ~18 us floor at small sizes is polling/signaling
overhead, not the wire (4.8 us at 8 KB), so more is available (unsignaled sends, both rails striped,
busy-poll on a flag instead of CQ).

Build/run on a node: `gcc -O2 -o ag4_proto ag4_proto.c -libverbs -lrdmacm`;
`ag4_proto <rank> 4 <chunk_bytes> <iters> <ip0> <ip1> <ip2> <ip3>` on all four (rail A IPs).

## Step 2 design (not built): make it usable from the decode graph
NCCL runs WITHOUT GPUDirect RDMA on GB10 ("GPU Direct RDMA Disabled for HCA" in its logs) and stages
through host memory; on unified memory that is the same DRAM, so this design does the same:
- receive buffers = pinned host memory registered with the NIC, written by peers with LL-style flags
  (payload + flag per 16 bytes, or a tail flag), so a GPU kernel can spin on the flags directly
  (graph-capturable receive, no CPU in the receive path);
- send side = a proxy thread per node that watches a GPU-written ready flag and posts the RDMA writes
  to the three peers (NCCL's proxy design, minus the ring);
- all-reduce = all-gather of the full buffer + local reduce kernel; all-to-all = one write per peer.
Estimated per-step gain at the step-1 numbers: ~9 ms at C1 (~7.5%), ~12 ms at C4 (~6%); more with the
optimizations above. Cost: a week-class systems build (QP setup over both rails, MR registration,
LL packing/unpacking kernels, proxy thread, torch custom op, then vLLM's GroupCoordinator hook).
