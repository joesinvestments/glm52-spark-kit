# veloGB10 transport read — intel for D3/B4 (2026-08-25)

Source: github.com/sf-stav/veloGB10 (Apache-2.0, author Stav Katsoulis), src/net.rs +
native/net_shim.c (2,284 lines) + native/tp_doorbell.h invariants I1-I9. A from-scratch
Rust+CUDA GB10 engine with a hand-built libibverbs doorbell all-reduce. Read-only source
study; no code vendored, nothing run. Findings that bear on our SIRCL port and any custom
collective on this fleet:

## 1. Our ag4_proto blocker is root-caused: rdma_cm is BROKEN on this RoCE, full stop
His comment: "TCP control-plane handshake (rdma_cm fails on this RoCE)". He does not pin
GID/TOS to fix rdma_cm; he bypasses rdma_cm entirely and exchanges QP info over a plain
TCP handshake, then uses raw ibverbs RC QPs (RoCEv2 GID index 3, same as our fabric).
Our D3 "fix ag4_proto init with GID/TOS pinning" plan was aimed at the wrong layer.
Correct fix: drop rdma_cm, TCP-handshake the QP exchange. (SIRCL also has no rdma_cm --
consistent with this.)

## 2. CAN_FLUSH_REMOTE_WRITES = 0 on GB10: the silent-corruption trap (his invariant I5)
The GPU may NOT consume NIC-written payload directly: payload DMA need not be visible to
the GPU when the completion flag is. His fix: the CPU proxy observes the peer's inline
epoch, issues a full fence, RELEASE-stores a cpu_done flag; the GPU ACQUIRE-loads that
and only then reads the buffer. He also documents a real 4-node wedge (his "world=4 §4.1
wedge": a retired 8-byte inline commit sitting in flush limbo indefinitely, rank frozen
mid-reduce) that forced deadline-bounded waits everywhere.
ACTION FOR B4: before SIRCL goes anywhere near an engine, verify how SIRCL handles this
exact invariant on the receive path. Our two-node link test validated correctness at
host-memory level; it did NOT test GPU-consumes-NIC-written-payload ordering. If SIRCL
(built for its author's topology) assumes remote-write flushability, that is a silent
data-corruption bug on our port, the worst failure class we know (rejection sampling and
gates will NOT catch corrupted activations reliably).

## 3. Design rules worth adopting wholesale in any collective we build or port
- I6: poll loops are plain load + backoff, NEVER atomic RMW -- RMW ping-pongs cache-line
  ownership on the C2C fabric that weights, NIC, and CPU all share (a hot-path tax).
- I7: every flag alone on a 64-byte line, segregated by writer (GPU/NIC/CPU); MR
  registered WITHOUT relaxed ordering.
- Unsignaled RDMA_WRITE chains with an inline length tag, signaled every S<=R: plain
  WRITE consumes no receiver WQE, which structurally eliminates the RNR-NAK class he
  describes as "ms-scale bimodal stalls when barriers cluster." (Worth remembering when
  we see bimodal collective latency under NCCL.)
- Capture hygiene: graph-captured kernels take ONLY a ctx pointer and derive
  epoch/slot on-device; any host-precomputed value freezes the protocol at capture time.
  SIRCL claims graph-resident replay; our port must honor the same rule.
- Every spin has a deadline and a cooperative abort status word (never __trap); on
  expiry both ranks no-op through the stream and the host does a FULL counter re-init on
  both nodes, never partial recovery (his I8/I9).
- Bonus for later: he runs an MTP-under-TP lockstep agreement channel (ranks exchange
  (step, accept_count, hash) per step over the same doorbell) because rank drafted-token
  desync pairs mismatched epochs permanently. Any future MTP-over-custom-collective work
  here needs the same guard.

## Standing
Watch-listed in both fleet-kit model profiles (fleet upstream picks up pushes). His TP=4
is newer than his TP=2 ("quad campaign target, DEFAULT OFF") -- treat his 4-node numbers
as younger evidence than his pair numbers. Nothing vendored; if we ever adopt code, it is
Apache-2.0 and the attribution row goes into THIRD_PARTY_NOTICES.md first.
