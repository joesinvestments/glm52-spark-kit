# Window K closeout + campaign handover to GLM-5.3 (2026-08-26)

The execution lane's session ended mid-window (API budget), after the k6 arm launched but
before its replay measurement. This closeout records the end state from live verification
and closes the 5.2 campaign; GLM-5.3-Flash intake begins.

## Verified end state (checked live from outside the dead session)
- Production: k6-trim arm serving (k=6, ladder [7,14,28,42,56,84,112], pool 615,050 tok
  after the pre-authorized trim), endpoint healthy, ~1h uptime at check.
- Noise floor LANDED: run 2 = 1288/1319 (97.65%) vs baseline 1284 (97.35%), McNemar exact
  p=0.454 NOT significant, floor = 16 discordant items (~±1.2pp). The GSM8K acceptance
  gate is complete and permanent for this checkpoint (transfers across spec-depth changes;
  a depth-change accuracy shift beyond floor = bug alarm, not a trade).
- Incumbent bar (W3-1, replay set, 200/200): hybrid k4+l12 at 3.341 tokens/step,
  60.99 tok/s aggregate on agentic-class replay at production concurrency.
- k6 live-traffic counters after ~1h: 39.0% draft acceptance -> ~3.34 tokens/step.
  LABELED: live-mix derived, not the replay protocol; directionally decisive, not
  print-grade.

## Verdict
k6 TIES the k4 hybrid on tokens/step while paying ~35% of the KV pool for its ladder.
Per the pre-agreed tie rule (a tie at higher depth spends memory and stability risk for
nothing): k=4 is the right production depth. The formal step (boot k4-a3, replay M1,
keep) was interrupted; k6 remains serving safely in the interim, decision recorded here.
THE CAMPAIGN'S ACTUAL SHIPPED GAIN: the k-depth investigation moved production from the
k2-era identity to k4-class serving at ~61 tok/s aggregate on agentic replay -- the first
structural speed change of the campaign, initially by accident (win1 restore defect),
then validated on purpose.

## What carries to GLM-5.3-Flash (intake assessment, same day as release)
GLM-5.3-Flash (zai-org, MIT, 320B-A18B, natively multimodal, 1M ctx) is NOT a weight
swap: new architecture class `Glm5NextForConditionalGeneration`, hybrid LINEAR+SPARSE
attention, mHC hyper-connections, IndexPool (4:1 indexer-key pooling), 45 layers,
62 BF16 shards (~640GB -- needs W8A8 (~320GB, fits TP=4) or int4-class quant (~165GB,
fits TP=2) before it serves on this fleet). vLLM support = OPEN PR #53906 (filed
release day, unmerged) -- the unmerged-PR smoke-test gate rule applies. SGLang path
likely ahead of vLLM (Z.ai's own serving is SGLang-derived).
Lessons that transfer: ALL fleet/ops discipline (boot rules, preflight, identity-diff,
window methodology, noise-floor gates); the ENTIRE sparse-indexer lineage (index_topk,
indexer heads are 5.2-family concepts); and -- notable convergence -- the linear-attention
half transfers from the ORNITH campaign (recurrent-state serving, mamba-class prefix
caching, hybrid spec-decode machinery). The two prior campaigns' architectures merged
into this one model. Per-model non-transfer: every 5.2-specific number, k-depth verdicts,
quant configs -- re-derive, never inherit (ledger rule).
