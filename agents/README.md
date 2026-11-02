# agents/ — PHASE 2, deferred (placeholder only)

Per `docs/instructions.md` §16: monitoring, diagnosis,
Genie-analysis, and feature-engineering agents are **deferred to phase 2**.
No agent code is built now — this folder is a placeholder so the eventual
agent layer has a home without disturbing the v1 layout.

Target design (from `docs/instructions.md` §16, not built):
- **Monitoring** — drift/threshold-breach/DQ/override-rate detection over
  `backtest_metrics`, `dq_audit`, `overrides`, `output`.
- **Diagnosis** — root-causes a failing/odd series (data gap, new item,
  missing event, model mismatch) and recommends an action.
- **Genie Analysis** — natural-language analytics over the forecast/
  decomposition tables for DS & leaders.
- **Feature-Engineering** — profiles sources and proposes new features for
  DS approval.

All agents are read/advisory by default; destructive or model-promoting
actions require DS approval. Do not build against this folder until v1
accuracy + explainability acceptance criteria (`docs/instructions.md` §21)
are met.
