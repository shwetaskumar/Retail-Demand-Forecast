-- =============================================================================
-- config/ddl/migrate_v3_skill_gate.sql
--
-- Migration: add the relative ("skill") gate columns to backtest_metrics and
-- champion_index (v3 — training/evaluation.py gates on skill vs the trailing
-- 28-day-mean baseline when control.forecast_config sets
-- category_skill_thresholds for a category).
--
-- SAFE TO RE-RUN: ADD COLUMN IF NOT EXISTS is idempotent on Delta tables.
-- Only needed for tables created before v3 — fresh deploys get these columns
-- on first write (utils/io.py::merge_upsert bootstrap).
--
-- Parameterize ${catalog} — swap in your Free Edition workspace's catalog name.
-- =============================================================================

-- 1. backtest_metrics
-- ---------------------------------------------------------------------------
ALTER TABLE ${catalog}.gold.backtest_metrics
    ADD COLUMN IF NOT EXISTS baseline_wmape DOUBLE  COMMENT 'Volume-weighted WMAPE of the trailing-28-day-mean baseline on the same backtest folds, over the series in this grain group.';

ALTER TABLE ${catalog}.gold.backtest_metrics
    ADD COLUMN IF NOT EXISTS skill          DOUBLE  COMMENT '1 - wmape / baseline_wmape: fraction by which the champions beat the naive baseline. Gate metric when min_skill is set.';

ALTER TABLE ${catalog}.gold.backtest_metrics
    ADD COLUMN IF NOT EXISTS min_skill      DOUBLE  COMMENT 'Configured min_skill_vs_baseline for this category (NULL = absolute accuracy_threshold gate applied).';

-- 2. champion_index
-- ---------------------------------------------------------------------------
ALTER TABLE ${catalog}.gold.champion_index
    ADD COLUMN IF NOT EXISTS baseline_wmape_7d DOUBLE  COMMENT 'Trailing-28-day-mean baseline WMAPE on this series backtest folds at champion-selection time (relative gate yardstick).';

-- ---------------------------------------------------------------------------
-- Verification: new columns present; NULL on all pre-migration rows.
-- ---------------------------------------------------------------------------
SELECT 'backtest_metrics' AS tbl, COUNT(*) AS total_rows, COUNT(skill) AS rows_with_new_column
FROM ${catalog}.gold.backtest_metrics
UNION ALL
SELECT 'champion_index'   AS tbl, COUNT(*) AS total_rows, COUNT(baseline_wmape_7d) AS rows_with_new_column
FROM ${catalog}.gold.champion_index;
