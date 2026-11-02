-- ============================================================================
-- config_tables.sql — Compact hierarchical config table (v2)
--
-- Replaces the v1 EAV table (270+ rows per market) with a compact JSON-packed
-- design (~12 rows per market). Supports cascading resolution for per-combo
-- tuning: market → category → sku → store (most specific wins).
--
-- Key: (market_id, category, sku_id, store_id, param_name)
--   NULL in category/sku_id/store_id means "applies to all at that level".
--
-- Cascading resolution (config/loader.py::resolve_series_config):
--   1. (market, category, sku_id, store_id)  — exact combo
--   2. (market, category, sku_id, NULL)      — sku across all stores
--   3. (market, category, NULL, NULL)        — category-level
--   4. (market, NULL, NULL, NULL)            — market default (always exists)
--   First non-NULL match wins.
--
-- Merge/replace rules:
--   market_settings → MERGE (override row provides only changed keys)
--   models, features_*, param_space_* → REPLACE (override row replaces entirely)
--
-- param_name vocabulary (fixed, small set):
--   models              — enabled models + cost_tier rank as JSON object
--   param_space_<model> — HPT search space per model (one row each)
--   features_historical — JSON list of feature names only available in training
--   features_known      — JSON list of feature names known at inference time
--   market_settings     — all scalar market settings as one JSON object
--   category_thresholds — accuracy gates per category as JSON object
--   category_skill_thresholds — optional relative gate per category as JSON object
--                     ({category: min_skill_vs_baseline}; see training/evaluation.py)
--
-- Parameterize ${catalog}. RUN IN DATABRICKS.
-- ============================================================================

CREATE TABLE IF NOT EXISTS ${catalog}.control.forecast_config (
  market_id     STRING NOT NULL  COMMENT 'Market code (US, DE, etc.)',
  category      STRING           COMMENT 'NULL = applies to all categories (A/B/C/D/SEASONAL)',
  sku_id        STRING           COMMENT 'NULL = applies to all skus',
  store_id      STRING           COMMENT 'NULL = applies to all stores',
  param_name    STRING NOT NULL  COMMENT 'What is being configured (models, param_space_lgbm, features_historical, features_known, market_settings, category_thresholds)',
  param_value   STRING NOT NULL  COMMENT 'JSON payload (object, array, or scalar as JSON string)',
  updated_by    STRING,
  updated_ts    TIMESTAMP DEFAULT current_timestamp()
) USING DELTA CLUSTER BY (market_id)
COMMENT 'Compact hierarchical forecast config (v2). ~12 rows per market at market level; additional rows only for per-category/sku/store overrides. Natural key for MERGE: (market_id, COALESCE(category,__), COALESCE(sku_id,__), COALESCE(store_id,__), param_name).'
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');
