# SIRCL -> switched-fabric port notes

Upstream: FujitsuPolycom/sparkring spark_transport/, Apache-2.0. Vendored verbatim under ../upstream/. Our fabric: switched dual-rail RoCEv2 (both rails NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0, different /24s); his: switchless direct-cable cycle.

Key facts from source study (file:line refs against upstream/ copies):
1. NO rdma_cm anywhere: raw ibverbs (ibv_open_device->PD->MR->CQ->RC QP, manual INIT/RTR/RTS), QPN/rkey/GID exchanged over TCP control channel (control_channel.*). Our old ag4_proto rdma_cm event-8 stall has no analog here.
2. GID index defaults to 3 already (tp4_session.hpp Tp4AllreduceOptions gid0/gid1=3; consumed verbs_endpoint.cpp local_info ibv_query_gid + RTR sgid_index). MUST VERIFY gid 3 is RoCEv2 IPv4-based GID on our ConnectX-class NICs.
3. TOS/traffic_class NOT SET anywhere - add ah_attr.grh.traffic_class in RTR block (verbs_endpoint.cpp :157-179) matching production NCCL marking. Single-line change, must validate against switch DSCP/PFC map.
4. Two-perfect-matchings scheduling: make_tp4_round_plan src/tp4_schedule.cpp:7-24 (round0 peer=rank^1 device0, round1 peer=rank^3 device1). VALID ON SWITCHED FABRIC as-is (every pair reachable) - keep algorithm v1, only remap which HCA serves which round via options device0/device1. True one-shot tree/pair AR = new kernels later (GpuTp4TensorWorker ctor binds two buffer layouts, gpu_tp4_tensor.hpp:19-27).
5. Protocol slot/credit math (tp4_allreduce_protocol.hpp) reusable untouched.
6. One QP per peer, never both rails to same peer (striped mode assumes two distinct peers). Rail aggregation = future work. Expect <=200G/direction per collective initially.
7. Failure contract: fatal_async_failure aborts process (tp4_session.cpp:319-323) - no in-process fallback once enqueued.
8. Affinity contract: graph sessions require pinned distinct submit/progress CPUs verified via pthread_getaffinity_np (tp4_session.cpp:341-394); GB10 has 10 cores - plan vs NCCL/DCP progress threads.
9. Control channel binds INADDR_ANY IPv4 literals only (control_channel.cpp:74-77,123-125).
10. Hardcoded 5s verbs completion timeout (verbs_endpoint.cpp:278-280); retry_cnt=7.
11. Build: cmake>=3.24 C++17, CUDAToolkit + libibverbs only, -DCMAKE_CUDA_ARCHITECTURES=121 for GB10 sm_121, no arch guards in code, aarch64 yield pauses host-side only. Micro-test targets: spark_tp4_probe (eager, args --rank --peer0 IP --peer1 IP --device0 HCA --device1 HCA --gid0 --gid1 --control-port0/1 --bytes --warmup --iterations) and spark_tp4_graph_q1_probe (graph replay, --wire-schedule sequential|dual_port_striped --allreduce-protocol --graph-kernel --elements-per-row 6144|4096).

Port plan v1 (micro-test gate): keep XOR-matchings two-round pair-sum; parameterize device names rocep1s0f0/rocep2p1s0f0 + gids; add traffic_class line; build probes in container with CUDA 13; run spark_tp4_probe two-node (gx10-1<->gx10-2) over both rails; success = allreduce correctness pass + latency recorded at decode shapes 37KB/147KB vs NCCL numbers already measured (17.9us/42us ag; comm floor 4.8/10.1/17.5us RDMA write).
