-- =============================================================================
-- config/ddl/migrate_v2_mase_bias.sql
--
-- Migration: add MASE-based champion selection columns to backtest_metrics
-- and champion_index (v2 — training/champion.py now selects on mase_7d).
--
-- SAFE TO RE-RUN: ADD COLUMN IF NOT EXISTS is idempotent on Delta tables.
--
-- Context:
--   Champion selection was switched from wmape_7d to mase_7d (scale-free,
--   defined for zero-volume series). WMAPE is retained as the planner-facing
--   gate metric for lag-1/lag-2 evaluation — thresholds unchanged.
--   See training/backtest.py::backtest_metrics() for computation details.
--
-- Parameterize ${catalog} — swap in your Free Edition workspace's catalog name.
-- =============================================================================

-- 1. backtest_metrics
--    New columns stored alongside wmape / passed (the unchanged gate columns).
-- ---------------------------------------------------------------------------
ALTER TABLE ${catalog}.gold.backtest_metrics
    ADD COLUMN IF NOT EXISTS bias_pct   DOUBLE  COMMENT 'Volume-weighted directional error fraction: (sum_yhat - sum_actual) / sum_actual. Positive = over-forecast. Supplementary diagnostic — does NOT affect the gate.';

ALTER TABLE ${catalog}.gold.backtest_metrics
    ADD COLUMN IF NOT EXISTS mase_7d    DOUBLE  COMMENT 'Volume-weighted MASE across series in this grain group: MAE / lag-7 naïve MAE. Values < 1 beat the naïve benchmark. Supplementary diagnostic — does NOT affect the gate.';

-- 2. champion_index
--    mase_7d is the new selection key (audit trail: why this model was chosen).
--    wmape_7d, bias_pct, naive_mae_7d are carried through for diagnostics.
-- ---------------------------------------------------------------------------
ALTER TABLE ${catalog}.gold.champion_index
    ADD COLUMN IF NOT EXISTS mase_7d      DOUBLE  COMMENT 'MASE vs lag-7 naïve at champion-selection time. Champion selected by lowest mase_7d (training/champion.py v2). Stored for auditability.';

ALTER TABLE ${catalog}.gold.champion_index
    ADD COLUMN IF NOT EXISTS bias_pct     DOUBLE  COMMENT 'Directional error fraction at champion-selection time: (sum_yhat - sum_actual) / sum_actual across backtest folds. Supplementary.';

ALTER TABLE ${catalog}.gold.champion_index
    ADD COLUMN IF NOT EXISTS naive_mae_7d DOUBLE  COMMENT 'Lag-7 naïve MAE at champion-selection time (MASE denominator). Stored so downstream code can recompute MASE at different grains without re-running backtest.';

-- ---------------------------------------------------------------------------
-- Verification: confirm new columns are present and nullable (should show
-- the four new columns with nulls on all pre-migration rows).
-- ---------------------------------------------------------------------------
SELECT
    'backtest_metrics'            AS tbl,
    COUNT(*)                           AS total_rows,
    COUNT(bias_pct)                    AS rows_with_bias_pct,
    COUNT(mase_7d)                     AS rows_with_mase_7d
FROM ${catalog}.gold.backtest_metrics

UNION ALL

SELECT
    'champion_index'              AS tbl,
    COUNT(*)                           AS total_rows,
    COUNT(mase_7d)                     AS rows_with_mase_7d,
    COUNT(naive_mae_7d)                AS rows_with_naive_mae_7d
FROM ${catalog}.gold.champion_index;
