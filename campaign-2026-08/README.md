# glm52-campaign

Campaign record + custom tooling for GLM-5.2 serving on 4x DGX Spark GB10.
Start with CAMPAIGN-STATE.md - it centralizes everything: state, fixes,
measurements, findings, roadmap.

Layout:
  CAMPAIGN-STATE.md   master document (state, findings, roadmap)
  JOURNAL.md          chronological execution log
  PHASE3-PLAN.md      original plan doc (superseded in parts by JOURNAL)
  patches/            numbered vLLM overlay patches (001-004), apply-tested
  images/             Dockerfile + build script + sidecar regen
  benchmarks/probe.py protocol-complete battery harness
  tailquant/          frequency-aware mixed-precision expert pipeline
                      (split.py planner, btx_writer.py container writer,
                       probe, design doc)
  comm/               RDMA decode-shape floor bench
  o14-diff/           machine-generated diff vs partner's published overlays
  results/            battery JSONLs (production baseline x2, k5 falsification)
