# Screening vLLM 0.27 for the drain/concurrency deadlock

Live ledger. This file is regenerated as each cell finishes, including the cells that
disprove my own hypotheses. Raw JSONL: `screen027.jsonl`. Harness: `wedge_trigger.py`
(C=6 storm, then a 27K-token cold prefill, then drain, then probe, 3 cycles) and
`screen_027.sh` (boots each cell unattended, one variable changed from the control).

**Control:** v0.27.0 + DeepGEMM `2fd67329` + #51920 workaround + #51538's `47f6574`,
GLM-5.2 TP=4, FLASHINFER_MLA_SPARSE_SM120, fp8_ds_mla, ctx 200000, seqs 6, MTP k=2.
The control survives light single-turn traffic across idle gaps and dies under real load.

| cell | variable | verdict | RAS at wedge |
|---|---|---|---|
| CONTROL | cudagraph_mode FULL | WEDGED | clean |
| cg_piecewise | cudagraph_mode PIECEWISE | WEDGED | clean |
| cg_none | cudagraph_mode NONE | WEDGED | mismatched (3 ranks op 1356, 1 rank op 1355) |
| eager | `--enforce-eager` | WEDGED | clean |

All four cudagraph modes wedge with the identical storm-phase-timeout signature. Cudagraph mode
does not gate the failure. Full detail, plus the memory campaign, two closed backend-override
levers, and a live stuck-kernel capture with per-rank Python stacks:
**[`MEMORY-AND-KERNEL-FINDINGS.md`](MEMORY-AND-KERNEL-FINDINGS.md)**.

## Findings so far

Cells still to run are listed in the table as they land. Nothing here is filtered:
a hypothesis that dies gets published the same as one that survives.
