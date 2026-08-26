-- ============================================================================
-- migrate_v3_champion_params.sql — Add champion_params to champion_index
--
-- Adds a lightweight JSON column storing the HPT-tuned hyperparameters that won
-- champion selection for each series. Distinct from the existing `params` column
-- which may include full model state (serialized artifacts, forecasts, etc.).
--
-- Example value: {"n_estimators":250, "learning_rate":0.08, "recency_decay":0.005}
--
-- Written by training/champion.py::write_champion_index. If the upstream engine
-- provides an `hpt_params` column (pure Optuna trial dict), that's used;
-- otherwise falls back to `params` (which for most models post-artifact-
-- extraction is already just the hyperparameters).
--
-- Safe to re-run: ADD COLUMNS is a no-op if column already exists (checked in
-- the migration notebook 99_migrate_config_v2.py cell 6 via Python guard).
--
-- Parameterize ${catalog} — swap in your Free Edition workspace's catalog name.
-- ============================================================================

ALTER TABLE ${catalog}.gold.champion_index
  ADD COLUMNS (
    champion_params STRING COMMENT 'JSON: tuned hyperparameters that won HPT for this series'
  );

-- Verification query:
-- SELECT snapshot_date, market_id, model_type, champion_params
-- FROM ${catalog}.gold.champion_index
-- WHERE champion_params IS NOT NULL
-- LIMIT 10;
