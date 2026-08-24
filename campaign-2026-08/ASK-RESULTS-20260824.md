# Ask results 2026-08-24 (executing session)

Answers per claude_to_agent_glm_next_phase_20260824.md. File paths and numbers
per reporting rails.

## Ask 1: BTX serving path vs production pin - NEW-TREE-ONLY, D1/D2 COUPLED

Verified from source trees via GitHub git-trees API:
- Tree at production pin 334a2d75d166becea0aa640b402d521ea0a290eb: 298 blob
  paths total. Keyword sweep for btx / mixed_trellis / trellis / rate_table /
  btx_schema / btx_synth / trellis_codebooks: **zero hits**. Directory census:
  attention 64, cute 7, distributed 6, gemm 7, integration 13, moe 22,
  quantization 2.
- Tree at new-b12x HEAD 36bce2c15: contains btx_schema.py, btx.py,
  mixed_trellis.py, btx_synth.py, trellis_codebooks.py, route_pack.py and the
  full BTX container machinery (files read directly this session; the
  btx_writer.py driver in tailquant/ runs against them).

Conclusion: **new-tree-only. D1 and D2 are coupled.** Sequence becomes
D1-front (profiler window) in parallel with D2-minimal image build, converging
at the BTX serving boot.

## Ask 8: how SparkRing serves EXL3 - custom private runtime, not upstream trellis

Source: docs/GLM52_35BPW_FIXED_MTP4_PROFILE.md + recipes/glm52-exl3-r7-3.5bpw.json
in FujitsuPolycom/sparkring.

Mechanism: his own EXL3 integration built into a private runtime image
(runtime/exl3-r7/{build-image.sh,Containerfile,pins.json} - build scripts are
published, the resulting image is NOT). Online quantization "EXL3 K6,
target-only scope". It is neither upstream b12x trellis/BTX machinery nor a
plain exllamav3 integration - it is an exl3-r7 runtime port of his own.

Corollaries:
- Ask 1's coupling answer STANDS (his stack does not prove BTX on old pin).
- His profile nevertheless proves EXL-class weights serve fast on GB10 TP4/
  DCP4: C1 19-22 stable to 128K, C4 45-48, C8 62-68, prefill 635-694, on
  nvfp4_ds_mla KV (368B records, FP8 RoPE) with FULL_AND_PIECEWISE graphs,
  mnbt 4096, seqs 16 - nearly our production geometry.
- His SIRCL covers TP all-reduce + vocabulary families only; DCP and indexer
  transports are stock NCCL even in his profile.

## Ask 10: SIRCL as D3 endgame - paper study verdict

Sources: docs/SIRCL.md, docs/ARCHITECTURE.md, spark_transport/ tree.

(a) What transfers to our SWITCHED fabric:
    - RDMA sessions, registered arenas over unified memory, device-published
      command rings, CUDA-graph-resident replay with zero host work: all
      topology-independent. This is exactly the design ag4_proto sketched.
    - Shape-admission discipline with per-shape NCCL fallback: transfers and
      is the right incremental-adoption boundary.
(b) What does NOT transfer: the two-perfect-matchings decomposition is a
    property of his switchless direct-cable cycle. On our switched fabric all
    pairs are directly reachable, so the ring scheduling simplifies rather
    than ports (a one-shot tree/pair AR replaces the two-phase matching).
(c) Adapt vs build: ADAPT SIRCL. It is shipped Apache-2.0 with its transport
    core separated from topology scheduling; finishing ag4_proto means
    rebuilding sessions/arenas/rings from scratch plus solving the rdma_cm
    GID/TOS issue SIRCL already handles via explicit device mapping. The
    nccl/ integration subtree shows the vLLM adapter seam.

NCCL-fallback boundary: admission is per collective shape ("a collective
shape not admitted to the native path must use the NCCL fallback"), so
adoption can start with the qualified TP all-reduce family only, DCP/indexer
stay stock (as in his GLM profile).

D3 reclassification: from "finish ag4_proto" to "port SIRCL transport core to
switched fabric, pin GID index 3, qualify the TP-AR family first." Effort
estimate drops accordingly; go/no-go still gated on his measured TP-family win
size vs NCCL on OUR switched fabric (his fabric is switchless-direct).

## Ask 3: drafter-finetune readiness inventory

1. Drafter weights: /Volumes/Storage/weights-staging/GLM-5.2-speculator.dspark-quanttrio-ft/weights/
   (7.1 GB: config.json, model.safetensors). Provenance header: base_model
   RedHatAI/GLM-5.2-speculator.dspark, finetuned on QuantTrio Int4-Int8Mix
   target quant, 5 layers, block 8, markov rank 256 (dspark-training/
   PROVENANCE.md). Target pairing CONFIRMED by provenance, not assumed.
2. Rig: dspark-training/{dspark-ddp.sh, dspark_finetune.py}. DDP via torchrun,
   --nproc-per-node 1 per node, MASTER_ADDR env. Data format: cap-*.pt files
   {aux [T, 30720] bf16, input_ids [T], positions [T]}; loss = CE on K true
   next tokens (Markov bias) + BCE confidence head. Last successful training
   run: the -ft checkpoint itself (bird's pipeline).
3. Capture collector: EXISTS as patches dspark-ring 0012/0013
   (VLLM_DSPARK_CAPTURE_DIR / _EVERY env-gated; v2 records raw per-layer aux
   hiddens). Applies to the DSpark-ring speculator path only - current
   production serves the MTP drafter, so collection requires a DSpark-ring
   serving boot (proven bootable Aug 17, config C/D lineage).
   Old corpus /var/tmp/glm-legacy/hf/dspark-capture/natural_clean
   (493 files / 129K tokens) is PURGED from gx10-1 - verified missing.
   ~600K captures needed => capture run is a window item, not storage-limited.
4. Natural-traffic constraint acknowledged: probe text excluded from capture
   corpus per established lesson.

## Ask 7 part 1: k=4 -> k=2 history

Reconstructed from launcher backups on gx10-1
(~/glm-legacy-stack/*.bak-preflashmla-20260813T*):

- Aug 13 18:05-18:26: FLASHMLA switchover batch created BOTH variants side by
  side: launch_gx10_k4.sh (num_speculative_tokens":4) and launch_gx10_k2.sh
  (num_speculative_tokens":2), plus nospec/capfit/da256/fp8peak variants.
- Aug 15-18 campaign standardized on k=2 as THE production baseline:
  RECOMMENDATION.md documents the dense-ladder rule with k=2 ([3,6,9,12]);
  SESSION-2026-08-17 ran every experiment against "production V1 MTP k=2"
  reference rows; HANDOFF (Aug 18) records production as k=2 fixed.
- The cross-model audit's "legacy stack runs k=4" refers to the pre-Aug-13
  identity and is stale relative to the live launcher.

Recorded reason for standardizing k=2 (not a recorded head-to-head A/B):
the dense-capture-ladder rule plus the Aug 17 finding that adaptive/deeper
speculation (+25-30 percent tokens/step, config A) did not change per-request
decode on this runner. Tonight's re-falsification on the current build
confirms that older finding independently.

Verdict for ask 7 part 2: the fixed-k=4 window cell remains worth ONE packed-
window slot (ladder [5,10,15,20]) since no recorded A/B exists at production
shape; treat 14.7 as reason to test, not a target.
