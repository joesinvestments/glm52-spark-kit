# GLM-5.2 (QuantTrio Int4-Int8): TP=4 on 4x DGX Spark, tuned for a real agent workload

This is my production GLM-5.2 serving config on a 4-node NVIDIA DGX Spark (GB10, sm_121a)
cluster over 200G RoCEv2, and the measured tuning window that produced it.

**The premise, up front:** I didn't tune this for benchmark screenshots. I built and tuned it
completely around my own agent's (Hermes) coding workloads, thousands of real requests a day,
and every decision below was measured at that traffic shape. As far as I know I'm the only
person building around that specific premise, and it changes the answers. A config tuned at
`max_num_seqs=1` on hand-picked prompts and a config tuned under a live agent are different
animals.

## 2026-08-17 update: the 72-hour session, three new repos, and where the project goes next

Production is now TP=4 + DCP=4 (V1 runner, MTP k=2, B12X sparse MLA, nvfp4_ds_mla KV) on the QuantTrio
Int4-Int8Mix unpruned body; the DCP4 pool is the KV warehouse the real workload wants, and that trumps
raw speed here (`glm52-spark-kit/docs/RECOMMENDATION.md`). Numbers this morning, same probes: 37.7 tok/s at
four concurrent, 11.6 per request, 605-626 prefill, gate passing.

What the weekend produced, in three repos:

- [glm52-spark-kit](https://github.com/joesinvestments/glm52-spark-kit): the source-form runtime kit. The b12x sparse indexer taught to run at DCP>1
  with a DCP-sharded scratch (proven live at DCP4: +3 to +8% at 32K, slower at 316K, so documented and not
  adopted); a fused Triton writer for the NVFP4 MLA KV record with a fail-closed gate; bird's DSpark ring
  drafting ported onto 0.27 and running at DCP=4 (first DSpark at DCP>1 on this stack) with a training
  capture hook, plus the honest acceptance matrix showing it does not beat MTP k=2 yet; the platform image
  on the AEON v0.27.1 base; every launcher; the session log with every result at the config it was measured.
- [glm52-aeon-crossnode-graphs](https://github.com/joesinvestments/glm52-aeon-crossnode-graphs): the hand-back to AEON. His v0.27.1 notes say cross-node CUDA graphs are
  broken; on four Sparks at TP=4 + DCP=4 on his image they captured, passed the gate, and served at parity.
  Also: nvidia's full-NVFP4 GLM-5.2 unpruned boots on his image with the CUTLASS FP4 MoE kernel, first
  time shown on four Sparks, and the memory arithmetic that says it is not the decode path (107 GB per
  rank, 4.5 bits per weight against int4's 4.1, on bandwidth-bound hardware).
- [spark-fleet-guard](https://github.com/joesinvestments/spark-fleet-guard): the failsafes written after two power cycles in two days: SBSA hardware
  watchdog, persistent rails, single-flight guarded launcher with evidence capture, NCCL RAS wedge capture.

Where it goes next, and it is open to anyone with Sparks: a decode-step profile of this model at C1 and
C4 (we run a ~50 ms step where the bandwidth bound says ~15, and nobody has published the breakdown), then
the top slice of that profile; a drafter finetuned on captures from this exact stack; and PRs upstream so
this stops being one fleet's overlays.

## 2026-08-16 update: the concurrency ceiling was a cudagraph bug

Raising `--max-num-seqs` from 4 to 16 -- while leaving the cudagraph capture
ladder alone -- moved aggregate throughput hard and unlocked a concurrency level
that had been killing the engine outright:

| | before | after |
|---|---|---|
| prose C4 | 44.97 tok/s | **50.34** (+12.7%) |
| prose C8 | 51.94 | **71.82** (+38%) |
| prose C16 | not serviceable | **99.84** |

The blocker was never the hardware or DCP. `cudagraph_capture_sizes` must be a
**dense ladder of multiples of `1 + num_speculative_tokens`**; a gap makes some
batch pad, padding rows carry `decode_len = 0`, and the non-uniform batch hits a
broken branch in the sparse MLA indexer that assumes DCP-sharded and global
block-table widths match. Scaling capture sizes up alongside `max-num-seqs` --
the natural move -- creates exactly those gaps. Full mechanism and repro matrix:
[`v027/concurrency`](v027/concurrency/).

Two more findings from the same sweep:

- **[`v027/prefix-cache`](v027/prefix-cache/)** -- cold prefill is pinned at
  ~580 tok/s and three levers failed to move it, but it barely matters: the
  prefix cache is worth **~100x** on turn 2+ (175 s -> 1.7 s at 100K context).
  Measured envelope: 16 sessions at 16K, or 6 at 100K, all resident. Tune the KV
  pool, not prefill.
- **[`v027/fabric`](v027/fabric/)** -- do **not** switch `--all2all-backend` to
  DeepEP on a 1-GPU-per-node RoCE fleet: 58-85% slower, correctness intact, which
  makes it easy to miss.

## Earlier champion: 44.6 tok/s single-stream decode, unlocked by one env flag

The kernel investigation the earlier findings called for paid off the same day, and the
answer rewrote the whole drafter picture. The quantized-draft slowness was never the
quantized compute itself. It was Marlin's default small-batch reduce path (`atomic-add off`, fp32
global reduction), at its worst at exactly the draft's tiny-M GEMM shapes, amplified by the
draft's 4 sequential forwards per step. The fix is one environment variable:

```
VLLM_MARLIN_USE_ATOMIC_ADD=1
```

Full 2×2, single-stream cold decode (repeats in `window-data/kernel-exp*.json`):

| draft config | no flag | with flag |
|---|---|---|
| bf16 (unquantized) | 21.1 tok/s @ 38.5% | 19.4 @ 32.9% (no effect) |
| quantized probabilistic | 6.5 @ 61.4% | **44.6 @ 56.8%** (repeats 42.4 / 46.9) |

The flag's entire effect lives in the quantized-draft path. So the drafter advice comes full
circle: **quantize the draft (`quantization:"compressed-tensors"`) AND set the flag**, either
alone is a regression from the other pairing, which is why half the community measures the
"fix" as a win and I measured it as a loss yesterday. Both were right on their own half of
the matrix. Outputs verified coherent and non-empty on repeated runs.

Production config: the challenger image + quantized probabilistic k=4 draft + the flag.
Two-day progression on the same weights and fleet, every step one named change:
**5.7 → 14.7 → 21.1 → 44.6 tok/s (7.8×)**.

### Reproduce the 44.6 exactly

Everything needed is in this repo. The formula:

1. **Stack:** build `legacy-stack/Dockerfile` (base: eugr/spark-vllm-docker `build-and-copy.sh
   --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8`), then `Dockerfile.v2` + `Dockerfile.v2b`
   on top (the draft-quant packed-mapping patch, both included).
2. **Launch:** `legacy-stack/launch_gx10.sh` via `resolve_gid_and_launch.sh`, TP=4 across the
   4 nodes, k=4 MTP draft with `"quantization":"compressed-tensors"` and
   `"draft_sample_method":"probabilistic"`, and the flag that makes it all work:
   `VLLM_MARLIN_USE_ATOMIC_ADD=1`. Edit the fabric values for your cluster; the GID resolver
   handles the rest.
3. **Measure:** `window-data/probe_battery.py`, the C=1 segment sends cold, cache-busted
   1,000-token prompts (unique random-token corpus per run, so the prefix cache cannot help)
   asking for a one-sentence summary, 300 max_tokens, thinking off, and computes decode tok/s
   from the server's own counters (`generation_tokens / decode_seconds` delta), not client
   chunk counting. Draft acceptance on this task runs 56-61%.

Run those three steps and you get 42-47 tok/s (my repeats: 44.6, 42.4, 46.9). Acceptance,
and therefore decode rate, scales with how predictable your output content is, the probe
JSONs in `window-data/` cover other content classes if you want the full picture. Every
number in this README comes off the server counters with the battery in this repo; check
the work.

## Production has moved: the capacity build (2026-08-11)

The 44.6 config above remains the peak-throughput trophy and ships in
legacy-stack/launch_gx10_fp8peak.sh. Production now runs a different point on the curve,
chosen by measuring my agent's actual workload: **nvfp4_ds_mla KV cache, 315,968 context,
MTP k=2** (legacy-stack/launch_gx10.sh). All numbers below are single-variable cells from
the validity-enforced battery, raw JSONL in window-data/.

| metric | fp8 peak config | production (nvfp4/316K/k2) |
|---|---|---|
| context window | 200,000 | **315,968 (+58%)** |
| prose/sustained decode (the agentic class) | 19 to 21 | 19.3 (parity) |
| deep-context decode, 30K in | ~21 | **24.8 @ 77% accept** |
| deep cold prefill | 447 to 546 | **667 tok/s** |
| summary-class peak decode | 44.6 | 20.6 (the one sacrifice) |
| C=12 storm | clean, 88-96s | clean, 82s |

Three findings from the window:

1. **The k=2 vs k=4 ablation the adaptive-k repos never published**: prose is a dead tie,
   but at 30K depth k=2 wins by 18% with 77% acceptance. Verify cost ramps with depth;
   shallow drafts win where long sessions live. If your workload is deep agentic work,
   run k=2 and stop paying for tail positions the verifier rejects.
2. **The alignment law**: the community MTP-overhang patch (mine included) leaves a seam
   between two block-table width computations. At max_model_len values not divisible by 64
   the seam opens and the engine dies under concurrent load, first request, every time
   (316,000 crashes, 315,968 is rock solid). Align your context length or move to vLLM
   0.27, whose universal block-table alignment removes the seam properly.
3. **nvfp4 KV pricing on this stack**: the format halves the high-acceptance decode peak,
   holds parity on prose, and speeds deep prefill ~30%. It is a capacity-and-depth trade,
   not a free win and not a loss: know which regime pays your bills before you adopt it.

### Known defect: vLLM 0.27 fails under concurrent load on this hardware

Five 0.27 configurations were tested against an identical trigger: a C=6 storm of 1.2K-token
prompts, then a 27K-token cold prefill, then drain, then a probe. All five died in cycle one.

| 0.27 configuration | result |
|---|---|
| prebuilt image + #51538 Python half (`47f6574`), MTP on | wedged |
| **source build, BOTH #51538 commits** (incl. the CUDA top-k guard `db39e67`), MTP on | wedged |
| same source build, **speculative decoding removed entirely** | wedged |
| same source build, activation buffer `max-num-batched-tokens` 8192 to 2048 | wedged |
| same source build, `gpu-memory-utilization` 0.91 to 0.85 | wedged |

The legacy stack in this repo survives a **heavier** storm (C=12) repeatedly, measured with the
same battery. So on this hardware the variable is the vLLM version, not speculative decoding,
not activation buffer size, and not memory headroom.

**Correction.** An earlier version of this section attributed the failure to MTP on sparse MLA,
following the upstream diagnosis in vllm-project/vllm#51593. Removing speculative decoding did
not prevent the failure, so that attribution was wrong and is withdrawn. The upstream fix
#51538 was also tested in full, source-built with both commits, and did not resolve it; that
negative result is reported on vllm-project/vllm#51921.

**How these verdicts are judged.** A request timeout is treated as a signal, never a verdict.
After any timeout the harness asks the engine to serve a small fresh request, twice, with a
generous timeout; only a failure to serve counts as a wedge. Boots are readiness-gated, so
cudagraph warmup is never mistaken for a hang. Both rules exist because earlier runs in this
repo were judged by timeout alone and produced four retracted verdicts, kept in
`v027/screen027-INVALID-timeout-predicate.jsonl`.

**What to run instead.** The legacy configuration documented above. vLLM 0.27 is not
production-viable on a 4-node GB10 cluster for this model until the concurrency failure is
resolved upstream. Everything needed to retest when it is: `v027/`.

**Update.** The `0.91 to 0.85` memory row above was a real result but a weak one, the system
stayed chronically thin either way. A full follow-up campaign found the actual `kv-cache`
floor, got the fleet to genuinely healthy headroom (4-8 GiB free, verified via
`/proc/meminfo`, not inferred), and reproduced the wedge anyway at production settings. Along
the way: all four cudagraph modes wedge, not just eager; a second, different, reproducible bug
was found and confirmed as a separate artifact of the memory-diagnostic config (not this one);
both available attention and MoE backend alternatives for this hardware were checked against
source and closed by live rejection, not left untried; and a stuck fleet was captured live with
per-rank Python stacks showing all four ranks blocked in different native kernels, not the same
one. Full writeup: **[`v027/MEMORY-AND-KERNEL-FINDINGS.md`](v027/MEMORY-AND-KERNEL-FINDINGS.md)**.

### Concurrency: the decode-aware scheduler mod, and why its shipped default is wrong

My storm number above (C=12) proves the config survives concurrent load. It does not prove
it serves well under it, and those are different bars. ThinkCode (issue #2) ran my own
battery on his 4-node fleet with the `decode-aware-scheduler` mod enabled and measured
**6.3 tok/s per-stream at C=12 against my 1.5**, same metric, same harness, server-counter
delta on both sides.

Two things that matter more than the headline:

- **This repo shipped that mod unbuilt.** `legacy-stack/Dockerfile` copied all three mods
  and built two. Anyone building from here had the mod's flags rejected. Fixed; it now
  builds, and stays inert until you pass `DECODE_AWARE=1`.
- **The shipped `DPTB=256` is only right at storm concurrency.** His C=4 sweep: DPTB=2048
  gives +11% per-stream over the mod being off with aggregate, wall and TTFT essentially
  unchanged, while DPTB=256 raises TTFT 71% (31.7s to 54.1s) to buy 9% per-stream. The
  mechanism is plain once stated: DPTB caps prefill tokens per step while any decode is
  active, so a 4,300-token prompt takes ~17 steps at 256 and ~3 at 2048. The mod's own
  README suggests 1024; the launcher ships 256.

So the finding is about the mod, not the number: enable it, and tune DPTB to your
concurrency rather than inheriting the default. A 4x win at C=12 and a net loss at C=4 came
from the same setting. Measurements and the DPTB table are ThinkCode's, on his hardware.

Methodology note worth stealing from that exchange: his C=4 table reports a client-side
median while the battery reports a server-counter figure. Those are different metrics and
do not belong on the same axis. Same class of error as comparing a peak content class
against a sustained one, which is the trap this whole README exists to document.

Credit where due: the nvfp4 port is tonyd2wild's work building on danielwoz's E2M1 kernel;
BTankut's 380K pinned-pool recipe and his deep-prefill numbers pushed this measurement up
my list; drowzeys built a parallel implementation independently. See NOTICE.

## The path there: rebuilding on the legacy stack (21.1 tok/s before the flag)

After the v4 window below, I rebuilt the *other* GLM stack, the no-DCP "legacy" lineage
(eugr/spark-vllm-docker base at vLLM `ab666069` + the ciprianveg/CosmicRaisins Triton
sparse-MLA mods + the indexer MTP-overhang patch), and probed it head-to-head against v4
with the identical battery. It won decisively and is now my production config:

| config | C=1 decode tok/s | cold prefill tok/s | accept | verdict |
|---|---|---|---|---|
| v4 (DCP2 stack, tuned, see below) | 14.7 | 282–345 | 72% | superseded |
| **legacy challenger v1** | **21.1** | 282 storm / ~900 C=1 | 38.5% | **champion** |
| + quantized probabilistic draft | 6.5 | 375 | 61.4% | rejected |
| + quantized greedy draft | 7.6 | 351 | 44.1% | rejected |

**Why I'm calling these the best numbers we've seen on this hardware for this workload:**
every published 4×-Spark GLM figure I've checked is either measured at `max_num_seqs=1` on
hand-picked high-acceptance content, counts SSE chunks (undercounts by the acceptance
factor), or measures warm cache and calls it prefill. These are cold, cache-busted,
server-counter-verified numbers at a serving config (12-seq class), cross-checked against
18 hours of live agent traffic, and the progression is measured, not vibes: 5.7 tok/s at
bring-up → 14.7 after the context-sizing window → 21.1 on the rebuilt stack. Same weights,
same fleet, ~3.7× in two days, every step attributable to one named change.

### The drafter tradeoff as first measured (kept for the record; the 2x2 above completes it)

The community's standard advice for GLM/DeepSeek MTP on these boxes, mine included, until
today, is that a draft config without `quantization:"compressed-tensors"` silently loads a
degraded drafter, and adding it is a pure win (acceptance jumps ~38% → ~61%; I reproduced
exactly that). What nobody measured: on this tree/hardware the properly-quantized w8a16
draft block runs **~3× slower per step**. Net decode: 21.1 tok/s "broken" vs 6.5–7.6
"fixed". I isolated the variables (greedy vs probabilistic, quantized vs not, see
`window-data/challenger*.json`): the cost is the quantized draft *compute*, not the
sampler. **The naive bf16 drafter wins.** If you've applied the popular fix, measure your
end-to-end decode, you may be paying 3× step time for acceptance you can't cash.

Open kernel question with a big prize: if the w8a16 draft path can be made fast
(Marlin/atomic-add config?), 61% acceptance at ~120 ms steps arithmetic says ~30 tok/s.

### Also corrected along the way

The reference repo's README says the DCP branch includes the July head-padding natively
it does not (I checked the fork source at the pin *and* the branch tip; padding is still
64). The +36–45% prefill numbers were measured on the legacy stack. If you're on the DCP
stack waiting for that prefill by rebuilding, you're rebuilding the wrong tree, the padding
(and those prefill numbers) live in the legacy lineage only. That correction is what
redirected this campaign, and the ~900 tok/s C=1 cold prefill above is that win, realized.

Launch machinery for the champion is in `legacy-stack/`, including
`resolve_gid_and_launch.sh`, which resolves the RoCEv2 GID index dynamically at every boot
and refuses to launch on disagreement. My GID moved 3→4 *during this campaign*; if yours is
hardcoded, your next reboot is a dice roll.

## The v4 config (DCP2 stack, kept in service for >200K sessions)

- **Stack:** local-inference-lab/vllm @e232d26 + PR#72 + draft-quant packed mapping + b12x,
  plus two patches of mine (below). Weights: QuantTrio Int4-Int8Mix, unpruned, 256 experts.
- **Parallelism:** TP=4, `--distributed-executor-backend mp`, **DCP2** (decode context
  parallel), interleave 1, attention `B12X_MLA_SPARSE`, moe `flashinfer_cutlass`.
- **Spec decode:** adaptive MTP ladder **k=2/4/5** (CosmicRaisins controller), probabilistic
  draft sampling, `quantization: compressed-tensors` on the draft (if you skip that field your
  drafter silently loads garbage and acceptance craters). NOTE the stack-dependence: on THIS
  DCP tree the quantized draft is fast and correct (72% accept at these speeds); on the
  legacy ab666069 tree it is a 3× step-time loss, see the champion section. Measure on
  YOUR tree; neither answer transfers.
- **Shape:** `--max-model-len 131072 --max-num-seqs 12 --max-num-batched-tokens 2048`,
  gmu 0.90, kv `fp8_ds_mla`, cudagraph FULL_AND_PIECEWISE capture 72.
- **Fabric:** dual-rail RoCEv2, NCCL 2.30.4, RoCE GID index resolved **dynamically at every
  boot** and verified identical across all nodes, GID indexes are NOT stable across boots
  and hardcoding one cost me a 10-hour outage on another model.

Full recipe: [`recipe/glm-dcp2-v4-speed128k-adaptive.yaml`](recipe/glm-dcp2-v4-speed128k-adaptive.yaml)

## Measured results (window 2026-08-10, all cells probed identically)

Probes: C=12 storm of cold 1.2K-token prompts (my real burst shape), segmented C=1 cold
decode, deep cold prefill. Cold means cache-busted, if your "prefill" number is over ~1000
tok/s on this hardware you are measuring your prefix cache, not your prefill.

| Cell | Config | C=1 decode tok/s | cold prefill tok/s | accept | verdict |
|---|---|---|---|---|---|
| 0 | 262K ctx / mnbt 1024 | 5.7 | 282 | 65.6% | stable baseline |
| 1 | + upstream July Triton kernel set |, |, |, | **crashed fleet on deep prefill** |
| 1b | + head-pad 64→16 only |, |, |, | **wedged under C=12 storm** |
| 2 | 262K ctx / mnbt 2048 |, |, |, | **crashed on deep prefill (indexer law)** |
| **3b** | **131K ctx / mnbt 2048** | **14.7 (+158%)** | **345 (+23%)** | **72.2%** | **window winner (since superseded, above)** |

### The two findings worth stealing

**1. The indexer law.** The DSA sparse-indexer scratch scales with `max_model_len ×
max_num_batched_tokens`. On GB10 at TP=4/DCP2 the survivable product is ≈ **262144 × 1024**.
Exceed it and the engine dies on the first deep cold prefill, not at boot, not on short
prompts, so you'll ship it and find out later. All the working community configs I've checked
respect this product without saying so; mine crashed twice proving it.

**2. Context you don't use costs decode speed you do use.** Past moderate depth GLM's decode
step is dominated by the sparse indexer, and the indexer working set follows the context
window. Halving `max-model-len` 262K→131K (with mnbt rescaled along the constant product)
took single-stream decode from 5.7 to **14.7 tok/s** on identical hardware, weights and
drafter. My live traffic's biggest prompt is ~2K tokens; I was paying a 2.6× decode tax for
window I never touched. Size the window to the workload you actually have, and revert the day
your workload changes, for me that's one launcher run back to the 262K recipe.

### Negative results (so you don't burn the day I burned)

- **The July head-padding prefill kernels (+36-45% upstream) do not patch onto an older tree.**
  The full 10-file set crashes on deep prefill and the padding change alone wedges under
  concurrency: the b12x glue that accepts 16-head alignment postdates my image snapshot.
  If you want that win, rebuild the image from the current upstream tree. There is no shortcut;
  I tried both.
- **mnbt above the indexer product = delayed crash**, see above.
- Adaptive-k works, but note nobody publishing it (including upstream) has shipped an
  adaptive-vs-fixed ablation at concurrency. Mine holds k=5 under real traffic with a smooth
  per-position decay (85/72/58/49/42%), which is what a healthy drafter looks like. If your
  positions collapse (60/28/18…), your draft weights didn't map, fix that before touching k.

## My patches (in `patches/`)

- **`fix-indexer-mtp-overhang.py`**, the indexer's expanded block-table has no headroom
  for MTP draft spill when `max_model_len % (block_size × cp) == 0`; at ≥3 concurrent
  requests the engine crashes. My exact production shape. Community-reported first; this is
  the anchored patch for the e232d26 tree.
- **`launcher-rank-verification`**, my launcher refuses to declare the fleet SERVING unless
  all 4 ranks are verified running. The head node happily answers `/v1/models` with a dead
  worker, and the first collective then kills the engine with an NCCL retry storm. A
  container name-conflict race handed me a 3/4-rank "healthy" fleet exactly once, which is
  once more than acceptable.

## The NCCL wedge, and what the standard mitigations actually cost

This stack inherits the frozen-collective failure mode I documented on DeepSeek
(NVIDIA/nccl issue 2334): under sustained mixed traffic at max concurrency, hours into
serving, an all-gather stops completing and the engine freezes while HTTP stays alive.
Never triggered by probes or benchmarks, only by real traffic. It hit this GLM stack on
night one (NCCL 2.30.4 via LD_PRELOAD, so it is not version-specific; the issue thread
has the full multi-model evidence).

Here is the data nobody publishes: what the standard mitigations cost at this workload
shape, measured on the identical config and probe:

| NCCL config | C=1 decode tok/s | wedge exposure |
|---|---|---|
| CROSS_NIC=1, QPS default (the 44.6 config) | 44.6 | wedged night one |
| CROSS_NIC=0 + IB_QPS_PER_CONNECTION=1 | measurement pending idle window | mitigated |
| CROSS_NIC=0 alone | measurement pending idle window | partial |

The two pending rows were first measured while live production traffic shared the engine,
which is not a valid C=1 number; they will be re-measured in a declared idle window. The
probe battery in window-data/ now enforces this itself: it refuses to start against a
non-idle server and discards any segment where foreign requests complete mid-measurement.

Both flags strangle the tiny-message collectives this decode path lives on. My verdict:
the mitigation costs more than the disease. I serve the full-speed config and treat the
wedge as an availability event: an external watchdog detects the frozen-engine signature
(prefill AND decode stalled while HTTP answers, which no HTTP health check will ever
catch), pages my phone, and auto-relaunches. A wedge costs ~15 minutes; the flags cost
2.5-6.6x forever. Make your own call, but make it with these numbers.

Two hard-won rules for whatever watchdog you run: verify its recovery path actually
reproduces the SERVING config (dry-run the recovery launcher and diff it against the live
container; mine silently pointed at a retired stack through one real outage), and prove
the check can FAIL before you trust it passing.

## Operational notes that matter more than the config

- **Streaming clients only.** A non-streaming client with a timeout turns every timeout into
  a zombie request that decodes to full budget while the client retries. Aggregate throughput
  moved ~3× on my other model the day the agent switched to streaming.
- **GLM's chat template deletes prior-turn reasoning.** That rewrites history every turn and
  invalidates the server prefix cache mid-conversation. For multi-turn agent sessions send
  `{"chat_template_kwargs":{"clear_thinking":false}}`, my workload is heavily
  prefix-cache-dependent and this is a real TTFT lever.
- **Thinking eats max_tokens.** GLM will happily spend your entire output budget on reasoning
  and return an empty answer with HTTP 200. Budget ~2× your expected answer, or disable
  thinking on latency-sensitive lanes (`enable_thinking:false`).
- Drop page caches before launch: 405GB of weights through the page cache will trip vLLM's
  free-memory guard at gmu 0.90 on unified memory.

## Credit where due

The stack rides on CosmicRaisins' glm-5.2-gb10 work (fork pin, DCP2 recipes, adaptive-MTP
controller). The indexer-overhang bug and several env caps come out of the tonyd2wild /
0xdfi / drowzeys 4×-Spark lineage, I verified everything against my own fleet before
adopting, and their READMEs are worth your time. Errors here are mine.
