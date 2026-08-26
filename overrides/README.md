# overrides/ — PHASE 2, deferred (placeholder only)

Per `docs/instructions.md` §15 and CLAUDE.md lock #11: the planner-override
feedback loop is **deferred to phase 2**. No ingest/feedback code is built
now. The `overrides` and `config_override_source` table contracts
already exist (`data/ddl/tables.sql`, `config/ddl/config_tables.sql`)
so nothing blocks adding this loop later.

Target design (from `docs/instructions.md` §15, not built):
1. **Ingest** — union a configurable list of silver-layer override tables
   (source TBD; a second, more granular table may arrive later) into
   `overrides`, normalized to one schema. Adding a new source table is
   a `config_override_source` row, not a code change.
2. **Measure** — override rate per category/market vs the 5-10% target,
   surfaced in Power BI.
3. **Feed back** — in the weekly retrain: flag high-override series for the
   diagnosis agent; treat overrides as a candidate bias-correction signal
   within `model_uplift` bounds, applied only human-in-the-loop (DS approval).
4. Override-derived adjustments are themselves decomposed as their own
   component, so explainability holds.

Do not build against this folder until v1 accuracy + explainability
acceptance criteria (`docs/instructions.md` §21) are met.
