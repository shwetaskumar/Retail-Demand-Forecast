-- ============================================================================
-- scoring_failures.sql — per-series JOB 3 scoring failure diagnostics.
-- Durable record of why a series produced no forecast this run (no history to
-- refit, from_serialized() error, or any exception during fit/predict/decompose
-- — see inference/batch_infer.py::_fail_row / write_scoring_failures in
-- inference/dq.py). Queryable via plain SQL instead of hunting through
-- executor stderr logs, which are lost once a job cluster terminates or scales
-- down (no cluster_log_conf configured for this project's clusters).
-- Parameterize ${catalog}. RUN IN DATABRICKS.
-- ============================================================================

CREATE TABLE IF NOT EXISTS ${catalog}.gold.scoring_failures (
  snapshot_date  DATE   NOT NULL,
  run_id         STRING NOT NULL,
  market_id      STRING NOT NULL,
  category       STRING,
  series_key     STRING NOT NULL,
  store_id       STRING,
  sku_id         STRING,
  model_type     STRING,
  fail_stage     STRING COMMENT 'no_horizon_or_history | artifact_load_failed_no_history | scoring_exception',
  fail_reason    STRING COMMENT 'human-readable detail, e.g. the caught exception message',
  created_ts     TIMESTAMP DEFAULT current_timestamp()
) USING DELTA CLUSTER BY (market_id, snapshot_date)
COMMENT 'Per-series JOB 3 scoring failure diagnostics — durable, queryable replacement for hunting through ephemeral executor logs'
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');
