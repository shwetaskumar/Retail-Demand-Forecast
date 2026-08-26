-- ============================================================================
-- tables.sql — gold output tables (managed Delta, Liquid Clustering).
-- The Databricks App is the CRUD front-end — jobs read these via config/loader.py.
-- Adding a market = INSERT rows here (through the app), never a new YAML file.
-- Parameterize ${catalog}. RUN IN DATABRICKS.
-- ============================================================================

CREATE TABLE IF NOT EXISTS ${catalog}.gold.config_market (
  market_id               STRING NOT NULL,
  active                  BOOLEAN,
  horizon_days            INT,     -- 140
  eval_horizon_days       INT,     -- 7 (WMAPE gate)
  history_start           DATE,
  incremental_read_days   INT,     -- 400
  incremental_write_days  INT,     -- 365
  backtest_folds          INT,     -- 5
  val_weeks               INT,     -- 10
  hpt_trigger             STRING,  -- 'threshold_miss'
  hpt_trials              INT,     -- 25
  uplift_enabled          BOOLEAN,
  uplift_learner          STRING,  -- 'lgbm_global' | 'chronos'
  uplift_clip_frac        DOUBLE,  -- 0.25
  reconstruction_tol_pct  DOUBLE,  -- 0.005
  run_budget_usd          DOUBLE,  -- 100
  max_champion_bias       DOUBLE,  -- 0.5  : champion selection deprioritises |holdout bias| above this
  forecast_cap_mult       DOUBLE,  -- 2.5  : inference clips daily yhat to this × trailing max daily demand
  cap_trailing_days       INT,     -- 365  : trailing window for the forecast-cap ceiling
  updated_by STRING, updated_ts TIMESTAMP DEFAULT current_timestamp()
) USING DELTA CLUSTER BY (market_id)
COMMENT 'Per-market run settings (app-managed)'
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');

CREATE TABLE IF NOT EXISTS ${catalog}.gold.config_category (
  market_id STRING NOT NULL, category STRING NOT NULL,
  accuracy_threshold DOUBLE,                 -- 1 - WMAPE (e.g. A=0.85)
  gate_grains STRING COMMENT 'JSON: [category_cohort, market_sku, market_store_sku]',
  series_grain STRING,                       -- market_store_sku
  drift_tolerance    DOUBLE  COMMENT 'Phase 1 stability: max fractional WMAPE degradation vs previous champion before re-evaluation is triggered (e.g. 0.05 = 5%). Also requires wmape_now <= 1-accuracy_threshold (absolute floor) — previously bad champions never get a free pass.',
  updated_ts TIMESTAMP DEFAULT current_timestamp()
) USING DELTA CLUSTER BY (market_id, category)
COMMENT 'Per-market x category thresholds, gating grains, and drift tolerance'
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');

CREATE TABLE IF NOT EXISTS ${catalog}.gold.config_model (
  market_id STRING NOT NULL, model_type STRING NOT NULL,
  enabled    BOOLEAN  COMMENT 'toggle on/off per market via the App; evaluation order is autonomous (evaluation_cost on the wrapper class, not user-configured)',
  param_space STRING  COMMENT 'JSON hyperparameter space for HPT',
  updated_ts TIMESTAMP DEFAULT current_timestamp()
) USING DELTA CLUSTER BY (market_id, model_type)
COMMENT 'Per-market candidate models + HPT search space. priority removed: engine orders autonomously by model evaluation_cost.'
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');

CREATE TABLE IF NOT EXISTS ${catalog}.gold.config_feature (
  market_id STRING NOT NULL, feature_name STRING NOT NULL,
  enabled BOOLEAN,
  feature_class STRING COMMENT 'known_at_inference | historical_only',
  params STRING COMMENT 'JSON: lags/windows/±N etc.',
  updated_ts TIMESTAMP DEFAULT current_timestamp()
) USING DELTA CLUSTER BY (market_id, feature_name)
COMMENT 'Per-market feature toggles & params'
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');

CREATE TABLE IF NOT EXISTS ${catalog}.gold.config_override_source (
  market_id STRING NOT NULL, source_table STRING NOT NULL,
  override_level STRING, enabled BOOLEAN,
  updated_ts TIMESTAMP DEFAULT current_timestamp()
) USING DELTA CLUSTER BY (market_id)
COMMENT 'PHASE 2: silver override tables per market'
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');
