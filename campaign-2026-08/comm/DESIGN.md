# Graph-safe RDMA collectives for TP4/DCP4 decode - design v1
Status: 2026-08-23. Grounding: kit docs/COMM_FLOOR-2026-08-19.md, kernels/rdma/ag4_proto.c,
b12x 1.2.x comm/pcie kernel family.

## Problem

A decode step issues ~390 latency-bound collectives (78 q-head all-gathers, 156 TP
all-reduces, ~25 indexer gathers, 78 DCP a2a). NCCL ring delivers 50-90 us per call at
these sizes; the one-sided RDMA floor is 12-20 us. That gap is ~14 ms of every ~120 ms
C1 step (~12%), more at C4. Kit decode profile: communication is 25% of the step.

## Approach: build on b12x 1.2.x PCIe kernels, not from scratch

The new tree already carries exact DCP top-k owner exchange (PCIe), topology-scoped
fused all-reduce (#133), bounded-degree AR, CuTe-DSL PCIe comm. Our work:

1. INVENTORY which 1.2.x collective kernels cover our five call sites
   (q gather / TP AR x2 / indexer merge / DCP a2a) at our shapes (37-160 KB).
2. GAP ANALYSIS vs COMM_FLOOR numbers; prototype the largest-delta site first
   using their plan/bind/run API.
3. CUDA-GRAPH SAFETY CONTRACT: capture-safe only - fixed buffers, no host sync,
   GPU-visible completion; verify under FULL cudagraph with replay-count checks.
4. FALLBACK LADDER: env-gated per call site (VLLM_COMM_IMPL_<site>=nccl|b12x);
   NCCL stays the fallback; sites flip independently.

## Prototype track (no vLLM integration needed to start)

kernels/rdma/ag4_proto.c proves W-rank RDMA WRITE-with-immediate all-gather at line
rate host-side. Extend into a bench matrix matching decode shapes exactly (37 KB,
55 KB/rank, 150 KB; 4 ranks; both rails) for our own achievable-floor table.
G0 GATE: if neither ag4-style nor b12x kernels beat NCCL by >=2x at >=2 shapes on
our fabric, deprioritize this arm in favor of TailQuant.

## Integration sketch (vLLM side)

Sites: parallel_state TP-group ops + B12X indexer merge (_merge_b12x_dcp_topk)
+ MLA q-head gather. Mechanism: persistent MRs over unified-memory buffers
allocated at graph capture; QPs pre-established at init on both rails; completion
via immediate-data CQ polled by spin-flag in device memory (ag4_proto pattern).
UMA makes registration pinning free; reuse activation/workspace tensors.

## Gates and milestones

G0 floor-bench >=2x NCCL at >=2 shapes -> M0 static inventory (no GPU needed)
G1 single-site swap boots 32K, gate PASS, trace shows fewer/faster collectives
G2 full-shape battery n>=2 vs baseline; promote outside noise band only
