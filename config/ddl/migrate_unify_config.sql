-- =============================================================================
-- config/ddl/migrate_unify_config.sql
--
-- Migration: carry forward any existing rows in config_market/_category/
-- _model/_feature (the old 4-table layout) into the new single config
-- EAV table (config/ddl/config_tables.sql). Safe to re-run — INSERT
-- targets are computed fresh each time and MERGEd (not appended) via the
-- natural key, so re-running just re-syncs rather than duplicating.
--
-- Does NOT drop the 4 old tables — that's a deliberate, separate, manual step
-- you run once you've confirmed config/loader.py::load_config() reads
-- correctly from control.forecast_config against the new table (e.g. via
-- tests/test_config_loader.py locally, then a real read in your workspace).
--
-- RUN IN DATABRICKS, after config_tables.sql has created control.forecast_config.
-- If you're setting up a brand-new environment with no data in the old 4
-- tables, skip this file entirely and just run config/seed_defaults.py
-- against the new config table.
--
-- Parameterize ${catalog} — swap in your Free Edition workspace's catalog name.
-- =============================================================================

MERGE INTO ${catalog}.control.forecast_config AS target
USING (
  -- config_market -> param_group='market', one row per column
  SELECT market_id, '' AS store_id, '' AS sku_id, '' AS category, '' AS model_type, '' AS feature_name,
         'market' AS param_group, 'active' AS param_name, CAST(active AS STRING) AS param_value
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'horizon_days', CAST(horizon_days AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'eval_horizon_days', CAST(eval_horizon_days AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'history_start', CAST(history_start AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'incremental_read_days', CAST(incremental_read_days AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'incremental_write_days', CAST(incremental_write_days AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'backtest_folds', CAST(backtest_folds AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'val_weeks', CAST(val_weeks AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'hpt_trigger', CAST(hpt_trigger AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'hpt_trials', CAST(hpt_trials AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'uplift_enabled', CAST(uplift_enabled AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'uplift_learner', CAST(uplift_learner AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'uplift_clip_frac', CAST(uplift_clip_frac AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'reconstruction_tol_pct', CAST(reconstruction_tol_pct AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'run_budget_usd', CAST(run_budget_usd AS STRING)
  FROM ${catalog}.gold.config_market
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'max_champion_bias', CAST(max_champion_bias AS STRING)
  FROM ${catalog}.gold.config_market WHERE max_champion_bias IS NOT NULL
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'forecast_cap_mult', CAST(forecast_cap_mult AS STRING)
  FROM ${catalog}.gold.config_market WHERE forecast_cap_mult IS NOT NULL
  UNION ALL
  SELECT market_id, '', '', '', '', '', 'market', 'cap_trailing_days', CAST(cap_trailing_days AS STRING)
  FROM ${catalog}.gold.config_market WHERE cap_trailing_days IS NOT NULL

  UNION ALL

  -- config_category -> param_group='category', one row per column, keyed by category dimension
  SELECT market_id, '', '', category, '', '', 'category', 'accuracy_threshold', CAST(accuracy_threshold AS STRING)
  FROM ${catalog}.gold.config_category
  UNION ALL
  SELECT market_id, '', '', category, '', '', 'category', 'gate_grains', gate_grains
  FROM ${catalog}.gold.config_category
  UNION ALL
  SELECT market_id, '', '', category, '', '', 'category', 'series_grain', series_grain
  FROM ${catalog}.gold.config_category
  UNION ALL
  SELECT market_id, '', '', category, '', '', 'category', 'drift_tolerance', CAST(drift_tolerance AS STRING)
  FROM ${catalog}.gold.config_category WHERE drift_tolerance IS NOT NULL

  UNION ALL

  -- config_model -> param_group='model', one row per column, keyed by model_type dimension
  SELECT market_id, '', '', '', model_type, '', 'model', 'enabled', CAST(enabled AS STRING)
  FROM ${catalog}.gold.config_model
  UNION ALL
  SELECT market_id, '', '', '', model_type, '', 'model', 'param_space', param_space
  FROM ${catalog}.gold.config_model

  UNION ALL

  -- config_feature -> param_group='feature', one row per column, keyed by feature_name dimension
  SELECT market_id, '', '', '', '', feature_name, 'feature', 'enabled', CAST(enabled AS STRING)
  FROM ${catalog}.gold.config_feature
  UNION ALL
  SELECT market_id, '', '', '', '', feature_name, 'feature', 'feature_class', feature_class
  FROM ${catalog}.gold.config_feature
  UNION ALL
  SELECT market_id, '', '', '', '', feature_name, 'feature', 'params', params
  FROM ${catalog}.gold.config_feature WHERE params IS NOT NULL
) AS source
ON  target.market_id   = source.market_id
AND target.store_id    = source.store_id
AND target.sku_id      = source.sku_id
AND target.category    = source.category
AND target.model_type  = source.model_type
AND target.feature_name = source.feature_name
AND target.param_group  = source.param_group
AND target.param_name   = source.param_name
WHEN MATCHED THEN UPDATE SET target.param_value = source.param_value, target.updated_ts = current_timestamp()
WHEN NOT MATCHED THEN INSERT (
  market_id, store_id, sku_id, category, model_type, feature_name, param_group, param_name, param_value, updated_ts
) VALUES (
  source.market_id, source.store_id, source.sku_id, source.category, source.model_type, source.feature_name,
  source.param_group, source.param_name, source.param_value, current_timestamp()
);

-- ---------------------------------------------------------------------------
-- Verification: row counts should roughly match (old tables' row count x
-- their column count) vs the new table's row count for that param_group.
-- ---------------------------------------------------------------------------
SELECT param_group, COUNT(*) AS n_rows, COUNT(DISTINCT market_id) AS n_markets
FROM ${catalog}.control.forecast_config
GROUP BY param_group
ORDER BY param_group;
