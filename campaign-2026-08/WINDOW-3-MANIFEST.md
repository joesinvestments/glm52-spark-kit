# WINDOW-3 MANIFEST — k-depth deployment (k=6 primary / k=4 fallback / k=2 rollback)

Status: DRAFT for Joe's approval. **GO is gated on P0 completion**: the noise-floor JSON
must exist and the chain must have exited cleanly before W3-1's first takedown — verify,
do not assume. This is a DEPLOYMENT with a gate: if an arm wins it STAYS in production.

Estimated total wall: **~7h** (three arms × measure, two extra boots, restores).

## Decision rule (fixed before any boot — no post-hoc reading)

Metric pair, per arm, measured identically:
  M1 (primary): tokens/step AND aggregate tok/s on the REPLAYED AGENTIC/CODE prompt set
       (the content class production actually serves), sampled from server spec-decode
       counters over the replay window.
  M2 (secondary): one hour of ORGANIC traffic counter deltas (same counters, live mix).
Keep/revert: arm keeps production iff M1 tokens/step >= k2-reference M1 tokens/step
(like-for-like, same replay set, same sampler) AND M1 aggregate tok/s does not regress
beyond tonight's noise floor. M2 corroborates; M1 decides. Prose regression is EXPECTED
and tolerated (0xSero's own honest caveat) — the criterion is the production class only.

Reference caveat, recorded: the circulating **2.57 tokens/step** figure comes from the
CURRENT boot whose traffic is GSM8K-contaminated (math content inflates acceptance).
It is context, not the bar. W3-1 re-measures k=2 on the replay set before any change;
that number is the actual bar.

## Arms (pre-staged before window opens)

- k6: `num_speculative_tokens: 6`, ladder = dense multiples of 7: [7,14,21,28,35,42,49,
  56,63,70,77,84,91,98,105,112] (covers dominant C16 => 112-wide steps).
  CONSERVATIVE KV PROVISION (A2 lesson): pre-approved trim if the KV-delta gate fails —
  [7,14,28,42,56,84,112] (still dense-multiples family); anything further = treat as
  boot failure and fall through.
  CAPTURE-STABILITY RISK, explicit: multi-size FULL-graph ladders at untested depths are
  the k5-instability family (root cause #4). Forensics armed (py-spy sidecar per D4
  method), earlyoom/watchdog mask verified pre-boot, ONE retry per the abort ladder,
  then fall to k=4.
- k4: EXACT window-1 A3 configuration (k=4, ladder [5,10,15,20], boot-proven). Staged as
  a launcher file swap (`launch_gx10.sh.a3` variant), not a rebuild.
- k2 rollback: pristine launcher (current identity).

## Cells

W3-0 Pre-flight + G-rails - 15m
  - P0-COMPLETE GATE: accuracy_baseline JSON + noise-floor JSON both present, chain
    exited. If not: window does not open.
  - mem/disk/daemons/orphans gates; sentinel disabled check; launch-lock control plan.
  - Build replay set: 200 agentic/code-class prompts staged locally to gx10-1 probes dir
    (source: fleet-kit agentic battery content class; fixed file, sha256 noted).

W3-1 k2 criterion capture (on CURRENT boot, before any takedown) - 1h30m
  - Replay M1: stream the 200-prompt set at production concurrency; sample spec-decode
    counters (drafts/accepted blocks + tokens) start/end -> tokens/step, tok/s.
  - Organic M2: 60-min counter-delta window on live traffic.
  - These two numbers ARE the bar. Journal + push before touching containers.

W3-2 k6 arm - 2h (boot 25m + replay 20m + organic 60m + slack)
  - Takedown; k6 launcher; boot.
  - KV-DELTA GATE at capture completion (before any measurement counts): record
    Available-KV GiB/rank + pool tokens; apply conservative-trim ladder if outside
    budget; re-capture once; still outside = treat per rung below.
  - Capture-stability watch during graph capture (forensics sidecar live).
  - Boot/capture FAILS TWICE (including trim-ladder retry) -> skip to W3-4 with k4.
  - On healthy boot: warmup volley discarded, replay M1, organic M2.

W3-3 k6 verdict - 10m
  - Apply decision rule vs W3-1 bar. WIN -> k6 stays; jump to W3-6 hand-back cells.
  - REGRESS -> k4 via launcher swap (pre-staged), continue.

W3-4 k4 arm - 2h (boot 25m + replay 20m + organic 60m + slack)
  - Same structure: KV-delta recorded, capture-stability watched (A3 proved this exact
    config boots and captures), warmup discarded, replay M1, organic M2.

W3-5 k4 verdict - 10m
  - WIN -> k4 stays; REGRESS -> restore k2 identity (pristine launcher).
  - Book closes with all three arms' production-shape data regardless of outcome.

W3-6 Hand-back - 40m
  - Final boot of whichever identity won (or k2 rollback), correctness gate, warmup
    batch discarded, endpoint verified.
  - Resume P0-equivalent monitoring; journal + push everything.

## Passengers (between boots only, never alongside serving)

P-a. Lane-2 oracle run (Joe approves as rider): CPU-only container, uniform-K2 planning
     confirmation + oracle_result.json capture. Gates per run_uniform_k2_oracle.sh.
P-b. SIRCL AR sweep: ONLY if the dual-port handshake fix lands before W3-2; otherwise
     deferred without prejudice. Staged script already in repo.

## G-rail addition (Joe, post-window-1): POST-RESTORE IDENTITY DIFF

End-state discipline now REQUIRES, after every restore/final boot in this window (and all
future windows): diff the launcher's INTENDED argv against `docker inspect .Args` of the
live head container - the verify_recovery_fidelity.sh pattern (fleet-kit lib/, read-only,
mechanical argv diff). Window 1's hybrid defect would have been caught by exactly this
check at hand-back; it is now mandatory before "restored and gated" may be claimed.
Runbook: `~/Desktop/GX10-FLEET-KIT/lib/verify_recovery_fidelity.sh --strict`
(adapt HOST/profile per identity), or the inspect-diff one-liner; divergence = end state
NOT reached, fix + reboot + re-diff.

## Failure rungs (binding order)

1. k6 boot/capture fails twice (incl. trimmed-ladder retry) -> k4 arm.
2. k6 boots but regresses on the criterion -> k4 arm.
3. k4 also regresses -> restore k2 identity; verdict recorded; book closed with
   production-shape data for all three depths.
Any node unreachable (power-cycle class): immediate abort, stopped-clean end state on
reachable nodes, INCIDENT file, stop. Production-dark time is bounded by the same
restore-or-stop-clean guarantee as window 1.
