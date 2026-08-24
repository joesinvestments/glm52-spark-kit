# Fresh-eyes brief: the vLLM 0.27 concurrent-load wedge

Everyone chasing this is running an agent of some kind. This is the exact prompt I hand mine
when I want a genuinely independent read on the problem rather than one more pass over the
same ground. Paste it into whatever you're running. If your agent finds something new, I want
to hear about it in the issue thread linked at the bottom.

The brief is deliberately split into two tiers of evidence: what's validated and what turned
out to be contaminated by a broken test harness earlier in the investigation. That split
matters more than any single finding in it, and it stays even though it makes an earlier phase
of this work look bad, because a fresh agent reasoning from the contaminated tier would waste
your reproduction time on already-dead ends.

---

```
MISSION: Find what we're missing. A GB10/GLM-5.2/vLLM-0.27 cluster wedges under
concurrent load through a mechanism we cannot identify after a full day of systematic
elimination. Every obvious hypothesis has been tested and killed. We need hypotheses
we have not thought of, or a reframe of the problem that shows what we've been
missing. Read everything below before proposing anything, we have almost certainly
already tried your first three ideas.

═══════════════════════════════════════════════════════════════════
1. HARDWARE / FABRIC
═══════════════════════════════════════════════════════════════════
4x NVIDIA DGX Spark (GB10, compute capability 12.1 / sm_121a, aarch64, CUDA 13),
121GB unified memory each. Connected via a MikroTik CRS804-4DDQ 200G RoCE switch,
dual-rail RoCEv2 per node (two separate PCIe functions: rocep1s0f0 + roceP2p1s0f0,
each on a different /24, both required or NCCL silently drops to single-rail or TCP).
MTU 9000 on both rails, verified identical across all 4 nodes. RoCEv2 GID index is
NOT stable across node reboots on this hardware (confirmed the hard way, twice), the
serving launcher resolves it dynamically every boot and refuses to launch on
disagreement across nodes/rails.

Model served: TP=4 across all 4 nodes, `--nnodes 4 --node-rank N`, mp executor
(no Ray).

═══════════════════════════════════════════════════════════════════
2. THE MODEL
═══════════════════════════════════════════════════════════════════
GLM-5.2, 744B MoE, registry class `GlmMoeDsaForCausalLM` (DeepSeek-V2-architecture
family, served through the deepseek model classes upstream). Weights: QuantTrio
Int4-Int8Mix (compressed-tensors, w4a16 experts + w8a16 linears). Checkpoint declares
`num_nextn_predict_layers=1`, `index_topk=2048`, and an `indexer_types` list that is
mostly "shared" with periodic "full" layers, the recipe carries a matching 78-char
`index_topk_pattern` hf-override (`FFFSSS...`) derived from it; without this override
the untrained indexer layers top-k through garbage past ~2K tokens.

═══════════════════════════════════════════════════════════════════
3. THE STACK UNDER TEST (vLLM 0.27)
═══════════════════════════════════════════════════════════════════
vLLM v0.27.0, tag commit `4bdc8a788d2e2ce9165d552b3d4d8b72604626bf`. Both the
official prebuilt image (`vllm/vllm-openai:v0.27.0-aarch64`) and a from-source build
(via the eugr/spark-vllm-docker harness, `--apply-vllm-pr`, patch files applied
directly to the checked-out tag) have been tested.

Three bugs were required to get this booting at all on sm_121 (all confirmed with
hard evidence, stack traces / exit codes, not timing judgments):

  a) Official image's ENTRYPOINT is the vllm CLI itself; must pass `serve ...` args
     or override the entrypoint, not `vllm serve ...`.
  b) v0.27.0's default DeepGEMM pin drops the sm_12x branches; the DSA indexer's
     paged-MQA path asserts `Unsupported architecture`. Fixed by repinning DeepGEMM
     to commit `2fd67329` (community fix validated for DeepSeek-V4 in
     vllm-project/vllm#51758, applies equally to GLM-5.2 since it hits the same
     indexer code path).
  c) `FlashInferMLASparseSM120Impl` (the new SM120 sparse-MLA backend) never sets
     `masked_mha_available`, an attribute `vllm/model_executor/layers/attention/
     mla_attention.py`'s dispatcher reads unconditionally (~line 780) →
     AttributeError, engine dies at startup, crash-loops under any restart policy.
     Filed as vllm-project/vllm#51920. A maintainer confirmed the correct fix is
     PR #51395 (correcting an inherited `supports_dense_mha_prefill=True` capability
     declaration the SM120 subclass should not have), not the attribute itself.

With all three fixed, the engine boots and the backend selection log shows:
`Using FLASHINFER_MLA_SPARSE_SM120 attention backend` for decode,
`Using FLASH_ATTN MLA prefill backend` for prefill. First single-stream completions
return in ~0.3-1.3s. This is, as far as we can tell, the first GLM-5.2 deployment
anyone has gotten running on this backend on real hardware.

Serving config varied across tests (see §5), baseline:
`--kv-cache-dtype fp8_ds_mla --max-model-len 200000 --max-num-seqs 6
--max-num-batched-tokens 8192 --gpu-memory-utilization 0.91
--speculative-config '{"method":"mtp","num_speculative_tokens":2,
"quantization":"compressed-tensors","draft_sample_method":"probabilistic"}'`

═══════════════════════════════════════════════════════════════════
4. THE BUG (vllm-project/vllm#51921)
═══════════════════════════════════════════════════════════════════
Under concurrent load (a storm of ~6 simultaneous requests, sometimes combined with
a long cold prefill), the engine goes permanently silent. HTTP stays alive
`/v1/models` and `/metrics` keep returning 200 OK indefinitely, but
`/v1/chat/completions` requests never complete, and `generation_tokens_total` in the
Prometheus metrics freezes. All 4 rank containers stay running; every worker's last
log line is a normal post-warmup entry (graph capture finished, JIT monitor
activated), no exception, no exit, no NCCL error printed by any rank.

The head's EngineCore repeats, every 60s, forever:
    [shm_broadcast.py:802] No available shared memory broadcast block found in
    60 seconds. This typically happens when some processes are hanging or doing
    some time-consuming work (e.g. compilation, weight/kv cache quantization).

DIAGNOSTIC ALREADY DONE, which side is starved:
`vllm/distributed/device_communicators/shm_broadcast.py` logs this exact message
from two call sites: `acquire_write` (~line 682, the WRITER waiting on readers) and
`acquire_read` (~line 803, the READER waiting on a writer). We confirmed by reading
the file inside our own image that line 802/803 is inside `acquire_read` (def at
line 762), never 682. Every wedge we've captured fires ONLY the reader-side
message, from the head's EngineCore process. This means the head is blocked
*reading* a shared-memory block that no worker is writing. The polling/notification
mechanism itself was read and appears sound (bounded 5s re-check interval, would
self-heal from a single dropped ZMQ ping), this looks like a genuine producer-side
stall (a worker stuck somewhere), not a lost-wakeup/transport bug.

═══════════════════════════════════════════════════════════════════
5. FULL EXPERIMENT LEDGER, READ THIS CAREFULLY, TWO TIERS
═══════════════════════════════════════════════════════════════════
All tests use a load trigger: a storm of 6 concurrent ~1200-token-prompt / 150-token
-output requests, then one ~20-27K-token cold prefill, then (in earlier tests) an
idle drain, then a probe. Verdict rule in the VALIDATED harness: a request timeout is
never itself a verdict, after any timeout the harness attempts a small fresh
completion twice with a generous window; only two consecutive failures to serve
count as WEDGED. Boots are readiness-gated (repeated small completions attempted for
up to 15 min before a cell is considered "up") so cudagraph warmup can never be
scored as a hang.

--- TIER A: VALIDATED RESULTS (trust these) ---
All of the following used the harness described above, after it was proven able to
both (a) correctly flag a known-wedged config and (b) correctly pass a healthy
freshly-booted engine (self-tested both directions before trusting any verdict):

  - Baseline config as in §3, MTP on: WEDGES (dies in the storm phase; does not
    require reaching the drain).
  - IDENTICAL image/config with `--speculative-config` REMOVED ENTIRELY (no MTP
    at all): STILL WEDGES. Same phase, same signature.
  - Upstream PR #51538 (proposed fix for a similar-sounding issue, #51593 below),
    SOURCE-BUILT with BOTH commits, `47f6574` (Python: clamps padded MTP slots'
    negative context lengths in the DSA indexer) and `db39e67` (CUDA: hardens the
    top-k kernels, `cooperative_topk.cuh` / `persistent_topk.cuh`, against ANY
    out-of-range row length, not just the specific negative-length case the Python
    half catches). NOTE: PR #51538 does NOT apply cleanly to the v0.27.0 tag (it
    targets newer main, conflicts in unrelated files) and its commits are not
    reachable from `refs/pull/51538/head` after a rebase (cherry-pick by SHA fails
    with "bad revision"), we applied the two `.patch` files directly. Build
    verified via 7 pre-registered checks against the resulting image (not the build
    log): clamp expression + comment present in installed `indexer.py`, guard string
    present in `cooperative_topk.cuh`, compiled extension present, version reports
    0.27.x, SM120 capability declaration touched. 7/7 passed.
      -> Full pair, MTP ON: WEDGES.
      -> Full pair, MTP REMOVED (same image): WEDGES.
  - `--max-num-batched-tokens` 8192 -> 2048 (matching our other, more stable stack's
    value): WEDGES.
  - `--gpu-memory-utilization` 0.91 -> 0.85 (more headroom; nodes were observed at
    ~0GB available memory during a wedge, RSS of the vllm process itself was only
    ~1GB, so the pressure if any is in unified GPU allocations, not host RSS):
    WEDGES.

  CONCLUSION FROM TIER A: this is not the MTP-drain mechanism described in
  vllm-project/vllm#51593 (DeepSeek-V4-Flash MTP hangs after batch drains to 3
  requests, due to padded MTP slots producing negative context lengths that
  deadlock the sparse top-k kernel), removing speculative decoding entirely does
  not prevent our wedge, and the proposed fix for that mechanism (source-built, both
  halves, verified present) does not prevent it either. It is also not simply an
  activation-buffer-size or memory-headroom problem. We do not know what it is.

--- TIER B: CONTAMINATED / UNRESOLVED, DO NOT TRUST, NEEDS RE-RUN ---
Before the harness above was built, an EARLIER version judged a wedge purely on a
raw 180-second request timeout. Every single cell in that run "wedged" at exactly
180.0 seconds past the trigger start, which is the timeout firing, not evidence of
four different engines dying identically. This was caught, all four verdicts were
PUBLICLY RETRACTED, and the harness was rebuilt into the Tier-A version above. The
following variables were only ever tested under the BROKEN harness and their results
are UNKNOWN, not "eliminated":
  - CUDA graph mode: full (control) / --enforce-eager (no graphs) / PIECEWISE / NONE
    / VLLM_USE_BREAKABLE_CUDAGRAPH=1
  - MTP depth: k=1 / k=3 / k=4 (k=2 is the control, itself only validated in Tier A
    with MTP on/off, not at other depths)
  - draft_sample_method: greedy vs probabilistic
  - --max-num-seqs 3 vs 6
  - --async-scheduling on vs off
  - forcing --attention-backend TRITON_MLA instead of auto-selected
    FLASHINFER_MLA_SPARSE_SM120
This entire tier is a priority re-test with the Tier-A-grade harness. Do not assume
any of them are ruled out. In particular, forcing TRITON_MLA (avoiding the brand-new
SM120 sparse-decode path entirely) is an important clean test we have real evidence
for but not yet with a trustworthy predicate.

═══════════════════════════════════════════════════════════════════
6. CROSS-STACK / CROSS-CLUSTER CORROBORATION
═══════════════════════════════════════════════════════════════════
a) Our OTHER, older, production stack (a heavily patched vLLM fork, commit e232d26,
   with a community Triton sparse-MLA kernel overlay and MTP-specific patches
   completely different code from the 0.27 native path above) has ALSO wedged for
   real, under real overnight heavy concurrent agentic traffic (not a synthetic
   storm), within under an hour, with the SAME silent signature (HTTP alive,
   completions dead, shm_broadcast reader starvation in the logs). This happened
   BEFORE the 0.27 investigation even began and is what prompted it. So: two
   completely different vLLM trees/forks, on the same hardware, exhibit the same
   silent-wedge signature. That argues for a hardware/fabric/driver-level
   commonality rather than a bug in either codebase specifically, though it could
   equally be that both trees happen to share the same vulnerable upstream sparse-
   MLA/indexer lineage.

b) An independent 4-node GB10 cluster (different operator, different vLLM tree
   `ab666069` lineage, MTP k=5, not our k=2) reproduced a SILENT wedge (identical
   signature: dozens of consecutive `GET /v1/models`/`GET /metrics` all 200 OK, zero
   errors, while completions were dead) with their OWN watchdog, bounded to have
   occurred inside an 11-minute window AFTER a successful completion, with NO
   traffic in between, i.e. it happened at or near idle, on hardware/software we
   do not control, using a periodic real-completion keepalive that demonstrably did
   NOT prevent it. That operator separately could NOT reproduce a distinct
   "30-35s stall after an idle gap" phenomenon we see on our other stack (6 clean
   TTFT trials up to 10 minutes idle, flat ~0.22s, properly controlled with
   randomized gap order and zero-idle controls), suggesting THAT specific stall is
   NOT universal, but the hard silent wedge may be.

c) UNEXPLORED BUT HIGH-SUSPICION: on this exact same 4-node GB10 / MikroTik RoCEv2
   fabric, a COMPLETELY DIFFERENT model (DeepSeek-V4-Flash, different vLLM image
   entirely) exhibited its own wedge class that WAS fully root-caused: NCCL
   2.30.x's RoCEv2 net_ib transport resiliency path (`net_ib/p2p.cc`, the
   NCCL_NET_IB_REQ_UNUSED branch, ~line 793) was found tolerating repeated
   `ncclIbCompletionEventProcess: Receiver got a completion for a CTS but retrieved
   an 'unused' request` events under sustained real traffic, successful
   IBV_WC_RDMA_WRITE completions landing on request slots already retired to
   UNUSED, desyncing collective completion tracking fleet-wide (documented: 381
   such completions, 123 distinct QPs, 2107 distinct comms in one capture), the
   collective never completes, all ranks block, engine freezes, HTTP stays alive
   (identical alive-but-dead signature to the GLM wedges above). This was filed
   upstream as NVIDIA/nccl#2334. A partial mitigation
   (NCCL_IB_QPS_PER_CONNECTION=1 + NCCL_CROSS_NIC=0, shrinking the completion/
   request matching surface) was staged for that model but trades ~6x throughput
   for the mitigation on that stack, so its efficacy at preventing wedges vs. its
   throughput cost has not been cleanly isolated yet.
   THIS EXACT NCCL/FABRIC MECHANISM HAS NOT BEEN TESTED AGAINST THE GLM/0.27 WEDGE.
   Given (a) and (b) above, a fabric-level RDMA completion/request desync under
   TP=4 sustained concurrent collectives is a live, low-effort-to-test hypothesis
   that would explain why the symptom recurs across unrelated vLLM trees, unrelated
   operators/clusters, and is insensitive to every serving-layer variable we've
   changed (MTP on/off, buffer size, memory headroom, the specific upstream patch).
   A worker "stuck" from the shm_broadcast reader's point of view is
   indistinguishable, from vLLM's perspective, between "stuck in a CUDA kernel" and
   "stuck waiting on an NCCL collective that will never complete due to a fabric-
   level desync", both present as a silently-dead rank.

═══════════════════════════════════════════════════════════════════
7. WHAT WE WANT FROM YOU
═══════════════════════════════════════════════════════════════════
We have a live 4-node cluster that reproduces this in 15-25 minutes per
configuration and can run essentially any test you propose. We are NOT asking you
to guess blindly, we want:

  1. RANKED HYPOTHESES we have not already covered in §5/§6, each with the
     mechanism stated (not just "try X") and a specific, falsifiable prediction of
     what we'd observe if it's right vs. wrong.
  2. Whether the NCCL/fabric hypothesis in §6c deserves priority, and if so the
     EXACT diagnostic to run to confirm/deny it independent of throughput cost
     e.g. specific NCCL_DEBUG/NCCL_DEBUG_SUBSYS combinations, an `ibv_devinfo`/
     `rdma resource show` capture at the moment of a wedge, whether
     NCCL_IB_QPS_PER_CONNECTION=1 alone (no CROSS_NIC change) can be tested as a
     narrower, cheaper A/B than the full mitigation pair.
  3. Concrete IN-THE-ACT instrumentation for the next reproduction: exact `py-spy
     dump` / `gdb`/`cuda-gdb` attach commands to get a Python+CUDA stack trace from
     a worker process while it is wedged (containers stay alive, so this should be
     possible via `docker exec`); whether `nsys profile` or `NCCL_DEBUG=INFO
     NCCL_DEBUG_SUBSYS=COLL,NET` left running continuously would capture the moment
     of the stall without prohibitive overhead at C=6.
  4. Any known GB10/sm_121a/aarch64/CUDA-13-specific footguns with: the brand-new
     FlashInfer SM120 sparse-MLA decode kernels (`FLASHINFER_MLA_SPARSE_SM120`),
     Marlin mixed-precision GEMM kernels under sustained concurrency, Triton JIT
     recompilation/caching races under concurrent load, or DCP/TP collective
     ordering, that you are aware of from other reports, even if not GLM-specific.
  5. A smarter experimental design than ours to cleanly separate: backend-specific
     (FLASHINFER_MLA_SPARSE_SM120) vs. model-specific (GLM's DSA sparse indexer /
     MoE routing) vs. fabric-specific (NCCL/RoCEv2) vs. host-specific (GB10 unified
     memory pressure) causes, ideally something that gives more signal per
     15-25-minute reproduction cycle than what we're currently running.

Rules of engagement: state your confidence and the mechanism, not just a knob to
flip. If you cite a known vLLM/FlashInfer/NCCL issue, we will go verify it against
primary sources ourselves before acting on it, point us at it, don't assume we'll
take your summary on faith, we won't. If you don't have a genuinely new idea, say so
plainly rather than restating something from §5.

═══════════════════════════════════════════════════════════════════
8. REFERENCES
═══════════════════════════════════════════════════════════════════
- Our write-up + raw JSONL for every cell above: github.com/joesinvestments/
  GLM-5.2-QuantTrio-TP4-DCP2-4x-DGX-Spark (see the v027/ directory, including
  screen027-INVALID-timeout-predicate.jsonl for the retracted Tier-B run and the
  harness source itself)
- vllm-project/vllm#51920 (masked_mha_available crash, fixed by us / PR #51395)
- vllm-project/vllm#51921 (this wedge, our filing, full A/B history in comments)
- vllm-project/vllm#51593 (the MTP-drain mechanism we've now shown is NOT this bug)
- vllm-project/vllm#51538 (proposed fix for #51593, tested full pair, did not help)
- vllm-project/vllm#51758 (DeepGEMM 2fd67329 pin, validated clean on 2-node TP=2
  note we are 4-node TP=4, which may itself be the relevant variable no one else
  has tested at this width)
- NVIDIA/nccl#2334 (the unrelated-model NCCL RoCEv2 wedge on this same hardware/
  fabric, root-caused, partial mitigation staged)
```

If your agent turns up something we haven't tried, open an issue or drop it on #51921
directly. The cluster is free between our own tests and reproduces this fast.
