# D3 transport comparison — SIRCL (SparkRing spark_transport) vs veloGB10 doorbell

Overnight audit 2026-08-25 (amended addendum: two candidates, six criteria, one verdict).
Sources read-only, Mac-local: campaign vendored tree `campaign-2026-08/sircl/upstream/`
(@3a60ca71), fresh clone of github.com/sf-stav/veloGB10 (Apache-2.0, nothing vendored),
and campaign intel note velogb10-transport-intel.md (23bd823).

## C1. Visibility correctness on GB10 (CAN_FLUSH_REMOTE_WRITES=0)

**veloGB10 — SAFE by explicit design.** native/tp_doorbell.h:16-18 (I5): "the GPU may NOT
consume NIC-written payload" directly; the CPU proxy observes peer_committed, issues a full
fence, RELEASE-stores cpu_done; the GPU ACQUIRE-loads cpu_done and only then reads payload.
Flag segregation TP_F_CPU_DONE "CPU-written: GPU release gate (I5 — v1 receive)"
(tp_doorbell.h:68,88). Author documents a real 4-node frozen-mid-reduce wedge from a retired
8-byte inline commit stuck in flush limbo (tp_doorbell.h:42-43,106) that forced
deadline-bounded waits everywhere.

**SIRCL — SAFE by construction, same classification, different mechanism.**
The NIC never writes device memory anywhere in the transport:
- The single MR covers the arena's HOST pointer:
  sircl/upstream/src/verbs_endpoint.cpp:51-54 `ibv_reg_mr(PD, buffer_.host_data(), ...)`.
- The session arena is `cudaHostAllocMapped` (+optional WriteCombined) with a device alias
  obtained via `cudaHostGetDevicePointer`:
  src/memory_buffer.cu:84-90.
- The GPU worker binds exactly those device ALIASES of host memory:
  src/gpu_tp4_tensor.cu:751-758 `round0/1_mapped_device_buffer` (+ExchangeBufferLayout);
  the only cudaMallocs are small graph-state structs (:790-797) and the legacy eager
  probe's private buffers (src/gpu_tp4.cu:200-205, not an MR target).
Ordering between "NIC done writing payload" and "GPU reads payload" is CPU-mediated with
proper memory ordering: the verbs progress thread polls completions then publishes
doorbell/observed-sequence words with `__atomic_store_n(..., __ATOMIC_RELEASE)`
(src/tp4_session.cpp:118) while device-side consumers `__atomic_load_n(...,
__ATOMIC_ACQUIRE)` (:114), and graph mode additionally GATES on
`cudaDevAttrHostNativeAtomicSupported` (:513-521, :671-673). Under IB RC semantics the
sender-side CQ completion certifies remote delivery before the release-store, so the GPU
never reads payload ahead of its commit — the veloGB10 I5 pattern expressed through a
mapped-arena doorbell instead of a separate cpu_done flag.
Residual risk (not a correctness gap): relaxed Grace C2C means payload bytes themselves
must not be read by the GPU before the ACQUIRE — satisfied because kernels read payload
only after acquiring the sequence word published AFTER completion polling. Our own
two-node link test exercised HOST-memory mode only, so this remains unproven EMPIRICALLY
on our fabric -> see probe spec below (window-1).

Classification: BOTH criterion-(a)-equivalent (CPU-mediated visibility with fence +
release/acquire). No GDRCopy / flush-remote-writes reliance found in either.

## C2. rdma_cm dependence

Both: NONE.
- SIRCL: raw ibverbs + TCP control-channel rendezvous (src/control_channel.cpp; zero
  rdma_cm references). Matches our root-caused finding that rdma_cm is broken on this
  RoCE fabric and the correct pattern is TCP QP-handshake (velogb10-transport-intel §1).
- veloGB10: TCP QP handshake by design ("rdma_cm fails on this RoCE", intel doc).
Our ag4_proto event-8 blocker is moot for both candidates.

## C3. Topology fit for our switched dual-rail fabric

- SIRCL ships a COMPLETE 4-rank all-reduce today: two-round XOR-matchings pair-sum
  (src/tp4_schedule.cpp make_tp4_round_plan), one HCA per round, valid unchanged on a
  switch (all pairs directly reachable). Known limit: one QP per peer => <=200G/direction
  per collective unless rail aggregation is added later.
- veloGB10's doorbell is PAIRWISE with TP=4 composition layered above; his TP=4 is newer
  ("quad campaign target", DEFAULT OFF flag per intel doc) and unpublished-here.
On a switch both map naturally; SIRCL gives the full collective algorithm now, veloGB10
gives a composable primitive whose 4-rank orchestration is younger.

## C4. Integration surface (what a vLLM GroupCoordinator hook must touch)

- SIRCL: 5,418-line C++17/CUDA static lib + SHARED C API (spark_tp4_create ladder,
  spark_tp4_all_reduce, spark_tp4_capture_q1_all_reduce, graph status flags -
  include/spark_transport/tp4_c_api.h:99-183) AND a shipped vLLM integration subtree
  (integrations/vllm/: sitecustomize backend hook, collective/cudagraph-bucket audits,
  tests). Heavier codebase, but the engine seam exists and is documented.
- veloGB10: ~2,300-line self-contained C shim (native/net_shim.c) + tp_doorbell.h
  invariant contract + Rust engine glue with NO vLLM integration surface - every vLLM
  hook point would be ours to build. Smaller audit surface, larger build-out.

## C5. Proven-ness

- SIRCL: SparkRing's production GLM serving on GB10 (his EXL3-r7 profile numbers) + our
  two-node host-memory link test PASS (p50 14.6us @64KB, verify correct=true).
- veloGB10: published live TP=2 traces + endurance report; TP=4 young/DEFAULT OFF;
  nothing of it has executed on our fleet.

## C6. MTP-under-TP desync guard (DESIGN REQUIREMENT for us: we run k=2 on TP=4)

- veloGB10: SHIPS a lockstep agreement channel - ranks exchange (step, accept_count,
  hash) per step over the doorbell, because drafted-token desync pairs mismatched epochs
  permanently (intel doc; src/dflash*, dspark trees reference implementations).
- SIRCL: NO equivalent (pure transport, no acceptance/epoch concept anywhere in-tree).
  Adding one: natural seam is the existing TCP control_channel (out-of-band, off the hot
  path) or a reserved slot in the mapped command ring; estimated ~100-200 LOC + tests on
  top of the session API, no kernel changes. Either candidate needs this before MTP-over-
  custom-collective serving; only veloGB10 has it designed and exercised today.

## VERDICT

Continue B4 on the SIRCL base. It is the only candidate that is simultaneously (i) a
complete 4-rank AR, (ii) proven in GB10 production service, (iii) safe-by-construction
on C1 with release/acquire discipline already in-code, and (iv) equipped with an engine
integration seam. Adopt from veloGB10 as PORT REQUIREMENTS (design rules, not code):
I5-style deadline-bounded waits with cooperative abort + full counter re-init (his I8/I9),
plain-load poll loops (I6), 64-byte flag segregation with non-relaxed MR ordering (I7),
graph-capture ctx-pointer hygiene, and the C6 MTP lockstep agreement channel (build it on
control_channel; veloGB10 is the working reference).

DECISIVE UNKNOWN for window 1 (empirical): GPU-consumes-NIC-written-payload ordering
under OUR fabric + CUDA-graph replay. Neither candidate's evidence covers our switch,
our driver stack, or graph-resident replay end-to-end.

WINDOW-1 PROBE SPEC (do not run outside an approved window):
- Name: sircl-gpu-ordering-probe. Nodes gx10-1+gx10-2, binaries already compiled in
  glm52-collab:b3 (/tmp/sircl-build upstream build).
- Allocates: per node one CUDA context + spark_tp4_probe buffers (--bytes sweep
  37KB..147KB decode shapes; ~<1 GiB VRAM total incl context) + host arenas.
- Gate: MemAvailable >= 8 GiB host AND free VRAM >= 4 GiB on both nodes, checked
  immediately before launch; abort otherwise (fail-closed).
- Proves: (a) all-reduce CORRECTNESS with gpu-roundtrip verification when payload is
  consumed by GPU kernels from the mapped arena across the switch (spark_transport_probe
  --memory cuda-mapped --gpu-producer --gpu-verifier --gpu-roundtrip, server+client);
  (b) spark_tp4_probe two-node eager AR latency + correctness at decode shapes vs NCCL
  baselines already measured; pass/fail = verifier true + p50 recorded.
- Duration: <15 min including sweeps.

## Queued LATER tasks (named, do not start tonight)
1. B3 cross-reference: veloGB10 src/dflash2/ and src/dspark/ as independent references
   for semantics-diffing our capture hook and acceptance logic.
2. Ornith-return task: deep read of veloGB10 GDN/recurrent-state-under-TP +
   unified-memory management (kv_cache.rs, memory.rs) as first credible non-vLLM
   treatment of Ornith's architecture class.
