# Databricks notebook source
# MAGIC %md
# MAGIC # 09_outbound.py — Publish Decomposition to the downstream feed
# MAGIC
# MAGIC Publishes the **latest champion_version decomposition** for the current snapshot
# MAGIC to `forecast.downstream_feed` for downstream consumption (e.g. a replenishment or
# MAGIC order-management system — out of scope for this build; this notebook is where the
# MAGIC published forecast hands off).
# MAGIC
# MAGIC **Filters applied:**
# MAGIC 1. Latest champion_version per (store_id, sku_id, date_id, component_type, component_name)
# MAGIC 2. Active sku-store pairs only
# MAGIC 3. Date-bounded: `date_id >= active_from AND date_id <= LEAST(snapshot + forecast_horizon, active_to)`
# MAGIC
# MAGIC **Write method:** Full overwrite (INSERT OVERWRITE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets & Parameters

# COMMAND ----------

import logging
from datetime import date

dbutils.widgets.text("env", "retail_forecast", "Environment (catalog)")
dbutils.widgets.text("snapshot_date", "", "Snapshot date (YYYY-MM-DD, blank=today)")
dbutils.widgets.text("market_id", "CA", "Market ID")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve Parameters

# COMMAND ----------

catalog = dbutils.widgets.get("env")
market_id = dbutils.widgets.get("market_id")
snapshot_date_raw = dbutils.widgets.get("snapshot_date").strip()

# Resolve snapshot_date
if snapshot_date_raw:
    as_of_str = snapshot_date_raw
else:
    as_of_str = str(date.today())

safe_market_id = market_id.replace("'", "''")

# Constants
_ETL_PROCESS = "OUTBOUND"
_TARGET_TABLE = f"{catalog}.forecast.downstream_feed"

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"OUTBOUND: catalog={catalog}, market_id={market_id}, snapshot_date={as_of_str}")
logger.info(f"OUTBOUND: target={_TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Read forecast_horizon from config

# COMMAND ----------

# forecast_config uses the compact v2 schema (config/seed_defaults.py,
# config/ddl/config_tables.sql): scalar market settings are packed into ONE
# JSON blob under param_name='market_settings', not a separate row per
# setting — there is no standalone 'horizon_days' row to read directly.
import json as _json

_market_settings_rows = spark.sql(f"""
    SELECT param_value
    FROM {catalog}.control.forecast_config
    WHERE upper(market_id) = upper('{safe_market_id}')
      AND param_name = 'market_settings'
""").collect()
if not _market_settings_rows:
    raise ValueError(
        f"No market_settings config row found for market_id={safe_market_id} "
        f"in {catalog}.control.forecast_config — run config/seed_defaults.py first."
    )
forecast_horizon = int(_json.loads(_market_settings_rows[0].param_value)["horizon_days"])
logger.info(f"OUTBOUND: forecast_horizon={forecast_horizon} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Build active sku-store filter from source reference tables

# COMMAND ----------

active_sku_store_query = f"""
SELECT DISTINCT
    a.store_id,
    upper(a.sku_id) AS sku_id,
    a.active_from,
    a.active_to
FROM {catalog}.source.sku_store a
JOIN {catalog}.source.sku b
    ON lower(a.sku_id) = lower(b.sku_id)
JOIN {catalog}.source.store c
    ON a.store_id = c.store_id
WHERE upper(a.market_id) = upper('{safe_market_id}')
  AND a.active_from <> '1970-01-01'
"""

spark.sql(active_sku_store_query).createOrReplaceTempView("_active_sku_stores")
n_active = spark.sql("SELECT COUNT(*) AS cnt FROM _active_sku_stores").collect()[0].cnt
logger.info(f"OUTBOUND: {n_active} active sku-store pairs from source reference")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Extract latest champion_version decomposition, filtered and date-bounded

# COMMAND ----------

outbound_query = f"""
WITH latest_decomp AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY snapshot_date, market_id, category,
                            store_id, sku_id, date_id,
                            component_type, component_name
               ORDER BY champion_version DESC
           ) AS _rn
    FROM {catalog}.forecast.decomposition
    WHERE snapshot_date = DATE('{as_of_str}')
      AND upper(market_id) = upper('{safe_market_id}')
)
SELECT
    d.snapshot_date,
    d.champion_version,
    d.run_id,
    d.market_id,
    d.category,
    d.store_id,
    d.sku_id,
    d.date_id,
    d.component_type,
    d.component_name,
    d.contribution_raw,
    d.contribution,
    d.model_type,
    d.created_date,
    d.created_by,
    d.updated_date,
    d.updated_by
FROM latest_decomp d
JOIN _active_sku_stores s
    ON d.store_id = s.store_id
    AND lower(d.sku_id) = lower(s.sku_id)
WHERE d._rn = 1
  AND d.date_id >= s.active_from
  AND d.date_id <= LEAST(
      date_add(d.snapshot_date, {forecast_horizon}),
      s.active_to
  )
"""

outbound_df = spark.sql(outbound_query)
n_rows = outbound_df.count()
logger.info(f"OUTBOUND: {n_rows:,} decomposition rows to publish")

if n_rows == 0:
    logger.warning("OUTBOUND: 0 rows to publish — check snapshot_date and data availability")
    dbutils.notebook.exit("NO_DATA")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Write to forecast.downstream_feed (full overwrite)

# COMMAND ----------

outbound_df.write.mode("overwrite").saveAsTable(_TARGET_TABLE)

logger.info(f"OUTBOUND: successfully published {n_rows:,} rows to {_TARGET_TABLE}")
logger.info(f"OUTBOUND complete for market_id={market_id}, snapshot_date={as_of_str}")
