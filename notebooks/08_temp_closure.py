# Databricks notebook source
# DBTITLE 1,08_temp_closure header
# MAGIC %md
# MAGIC ## 08_temp_closure — Post-Model Transform: Temporary Closure Events
# MAGIC Zeros out forecasts for sku-stores with active temporary closure events
# MAGIC (TEMP_CLOSURE, PARTIAL_CLOSURE, TEMP_BREAK_IN_SALES, TEMP_OUT_OF_STOCK).
# MAGIC
# MAGIC Runs LAST in the post-model chain (after `06_infer_final`) so the final
# MAGIC output reflects closures regardless of any prior adjustments.
# MAGIC
# MAGIC Logic:
# MAGIC 1. Read events table for active closure events overlapping the forecast horizon.
# MAGIC 2. Join against the output table to find forecasts that fall within event windows.
# MAGIC 3. MERGE those rows with yhat=0 (matched-only, no insert).
# MAGIC
# MAGIC **Dual-table write:** zeros out BOTH `output.yhat` and ALL `decomposition.contribution`
# MAGIC values for affected rows (consistent: if yhat=0, components must also = 0).
# MAGIC
# MAGIC Inputs: `output` (post inference), `decomposition`, `gold.events`.
# MAGIC Outputs: `output` (MERGE matched-only), `decomposition` (MERGE matched-only).
# MAGIC Params: market_id, env, snapshot_date.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("market_id", "CA")
dbutils.widgets.text("env", "retail_forecast")
dbutils.widgets.text("snapshot_date", "")
dbutils.widgets.text("filter_sku", "")
dbutils.widgets.text("filter_store", "")

# COMMAND ----------

# DBTITLE 1,Read widgets
market_id     = dbutils.widgets.get("market_id")
env           = dbutils.widgets.get("env")
snapshot_date = dbutils.widgets.get("snapshot_date") or None

def _parse_filter(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]

filter_sku   = _parse_filter(dbutils.widgets.get("filter_sku"))
filter_store = _parse_filter(dbutils.widgets.get("filter_store"))

# COMMAND ----------

# DBTITLE 1,sys.path — repo root importable
import os
import sys

# Databricks Free Edition serverless compute can stat/list Workspace Files but
# cannot open() them for content (Input/output error from
# SharedUCWorkspaceLocalFileSystem — a Free Edition environment limitation).
# Shared modules are synced to a UC Volume instead of imported straight from
# the bundle-deployed Workspace path.
_REPO_ROOT = f"/Volumes/{env}/default/pylib"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# COMMAND ----------

# DBTITLE 1,Imports
from utils.dates import resolve_snapshot_date
from utils.io import read_table
from utils.logging import get_logger
from utils.spark import get_or_create_spark

# COMMAND ----------

# DBTITLE 1,Spark session, config
catalog = env
spark   = get_or_create_spark(f"job3-temp-closure-{market_id}")
logger  = get_logger(__name__, market_id=market_id, env=env)

as_of     = resolve_snapshot_date(snapshot_date)
as_of_str = as_of.isoformat()
safe_market_id = market_id.replace("'", "''")

_ETL_PROCESS = "TEMP CLOSURE"

logger.info(f"TEMP_CLOSURE starting market_id={market_id} snapshot_date={as_of_str}")

# COMMAND ----------

# DBTITLE 1,Step 1 — Identify forecasts overlapping temp closure events
# Closure event subtypes that should zero out forecasts
_CLOSURE_SUBTYPES = (
    "'TEMP_CLOSURE', 'PARTIAL_CLOSURE', 'TEMP_BREAK_IN_SALES', 'TEMP_OUT_OF_STOCK'"
)

# Resolve latest champion_version per (store_id, sku_id) for this snapshot.
# Only the latest version should be zeroed — older versions are historical.
latest_version_query = f"""
SELECT store_id, sku_id, MAX(champion_version) AS latest_champion_version
FROM {catalog}.forecast.output
WHERE snapshot_date = DATE('{as_of_str}')
  AND upper(market_id) = upper('{safe_market_id}')
GROUP BY store_id, sku_id
"""
spark.sql(latest_version_query).createOrReplaceTempView("_latest_versions")

temp_closure_query = f"""
WITH vw_events AS (
    SELECT DISTINCT
           store_id,
           sku_id,
           start_date,
           end_date
    FROM {catalog}.gold.events
    WHERE upper(market_id) = upper('{safe_market_id}')
      AND event_subtype IN ({_CLOSURE_SUBTYPES})
      AND sku_id IS NOT NULL
      AND tactic is not null
      -- Only events that overlap with the forecast horizon
      AND end_date >= DATE('{as_of_str}')
)
SELECT a.snapshot_date,
       a.run_id,
       a.market_id,
       a.category,
       a.store_id,
       a.sku_id,
       a.date_id,
       a.horizon_day,
       -- Zero out forecasts during closure events
       CAST(0 AS DOUBLE) AS yhat,
       CAST(0 AS DOUBLE) AS yhat_lower,
       CAST(0 AS DOUBLE) AS yhat_upper,
       a.reconstruction_error,
       a.model_type,
       a.champion_version
FROM {catalog}.forecast.output a
JOIN vw_events b
    ON lower(a.sku_id) = lower(b.sku_id)
    AND a.store_id = b.store_id
    AND a.date_id BETWEEN b.start_date AND b.end_date
JOIN _latest_versions lv
    ON a.store_id = lv.store_id
    AND lower(a.sku_id) = lower(lv.sku_id)
    AND a.champion_version = lv.latest_champion_version
WHERE a.snapshot_date = DATE('{as_of_str}')
  AND upper(a.market_id) = upper('{safe_market_id}')
"""

closure_df = spark.sql(temp_closure_query)

# Apply optional scope filters
if filter_sku:
    closure_df = closure_df.filter(closure_df.sku_id.isin(filter_sku))
if filter_store:
    closure_df = closure_df.filter(closure_df.store_id.isin(filter_store))

# Deduplicate: overlapping events for the same sku-store-date produce
# identical output rows — collapse before MERGE to avoid multiple-match error.
closure_df = closure_df.dropDuplicates()

n_zeroed = closure_df.count()
logger.info(f"TEMP_CLOSURE: {n_zeroed} forecast rows to zero out")

# COMMAND ----------

# DBTITLE 1,Step 2 — Merge zeroed forecasts into output (matched-only update)
if n_zeroed > 0:
    closure_df.createOrReplaceTempView("_temp_closure_stg")

    # Output table: set yhat/yhat_lower/yhat_upper to 0 + update watermarks
    spark.sql(f"""
        MERGE INTO {catalog}.forecast.output t
        USING _temp_closure_stg s
        ON t.snapshot_date = s.snapshot_date
           AND t.market_id = s.market_id
           AND t.category = s.category
           AND t.store_id = s.store_id
           AND lower(t.sku_id) = lower(s.sku_id)
           AND t.date_id = s.date_id
           AND t.champion_version = s.champion_version
        WHEN MATCHED THEN UPDATE SET
            t.yhat = 0,
            t.yhat_lower = 0,
            t.yhat_upper = 0,
            t.updated_date = current_timestamp(),
            t.updated_by = CONCAT(t.updated_by,' {_ETL_PROCESS}')
    """)
    logger.info(f"TEMP_CLOSURE: zeroed {n_zeroed} output rows")
else:
    logger.info("TEMP_CLOSURE: no active closure events overlapping forecast — skipping")

# COMMAND ----------

# DBTITLE 1,Step 3 — Zero out decomposition contributions for closure events
if n_zeroed > 0:
    # Decomposition table: set all component contributions to 0 for affected sku-store-dates
    spark.sql(f"""
        MERGE INTO {catalog}.forecast.decomposition t
        USING (
            SELECT DISTINCT snapshot_date, market_id, store_id, sku_id, date_id, champion_version
            FROM _temp_closure_stg
        ) s
        ON t.snapshot_date = s.snapshot_date
           AND t.market_id = s.market_id
           AND t.store_id = s.store_id
           AND lower(t.sku_id) = lower(s.sku_id)
           AND t.date_id = s.date_id
           AND t.champion_version = s.champion_version
        WHEN MATCHED THEN UPDATE SET
            t.contribution = 0,
            t.updated_date = current_timestamp(),
            t.updated_by = CONCAT(t.updated_by,' {_ETL_PROCESS}')
    """)
    logger.info(f"TEMP_CLOSURE: zeroed decomposition contributions for affected rows")

logger.info(f"TEMP_CLOSURE complete for market_id={market_id}")
