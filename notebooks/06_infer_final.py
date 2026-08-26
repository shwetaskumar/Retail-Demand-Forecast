# Databricks notebook source
# MAGIC %md
# MAGIC ## 06_infer_final — inference job final re-scoring task
# MAGIC Runs daily, after `05_hpt_retune`, and self-gates the same way: it checks
# MAGIC `champion_index` for any series whose champion HPT actually improved
# MAGIC today (`version >= 2`) and exits immediately if there are none. Re-scores
# MAGIC only those series — the initial `03_score` run stands as final for
# MAGIC everyone else.
# MAGIC
# MAGIC Inputs: `champion_index`, `sku_store_features`.
# MAGIC Outputs: `output`, `decomposition`, `scoring_failures` (only for HPT-updated series).
# MAGIC Params: market_id, env, snapshot_date.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("market_id", "US")
dbutils.widgets.text("env", "")
dbutils.widgets.text("snapshot_date", "")

# ── Optional debug / test scope filters ──────────────────────────────────────
dbutils.widgets.text("filter_sku",      "")
dbutils.widgets.text("filter_store",  "")
dbutils.widgets.text("filter_category", "")

# COMMAND ----------

display(dbutils.widgets.getAll())

# COMMAND ----------

# DBTITLE 1,Install pinned deps (interactive runs only — job clusters already have these)
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Read widgets
market_id = dbutils.widgets.get("market_id")
env           = dbutils.widgets.get("env")
snapshot_date = dbutils.widgets.get("snapshot_date") or None


def _parse_filter(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


filter_sku      = _parse_filter(dbutils.widgets.get("filter_sku"))
filter_store    = _parse_filter(dbutils.widgets.get("filter_store"))
filter_category = _parse_filter(dbutils.widgets.get("filter_category"))

# COMMAND ----------

# DBTITLE 1,sys.path — repo root importable
import os
import sys

# Ask the runtime for THIS notebook's own workspace path, then go up one level
# (out of notebooks/) to the repo root. Portable across workspaces/usernames and
# regardless of whether this is deployed via `databricks bundle deploy` or linked
# as a Databricks Git folder (Repos) backed by Azure DevOps/GitHub — no hardcoded
# path. __file__ is not available inside a notebook cell, which is why this asks
# the runtime instead.
_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_REPO_ROOT = "/Workspace" + os.path.dirname(os.path.dirname(_notebook_path))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# COMMAND ----------

# DBTITLE 1,Imports
import uuid

from utils.dates import resolve_snapshot_date
from utils.io import read_table
from utils.logging import get_logger
from utils.spark import get_or_create_spark
from config.loader import load_config
from inference.batch_infer import score, write_outputs
from inference.dq import write_scoring_failures
from inference.future_spine import build_future_spine

# COMMAND ----------

# DBTITLE 1,Spark session, config
catalog     = env
spark       = get_or_create_spark(f"job3-infer-final-{market_id}")
logger      = get_logger(__name__, market_id=market_id, env=env)
as_of       = resolve_snapshot_date(snapshot_date)
as_of_str   = as_of.isoformat()
config      = load_config(spark, catalog, market_id)
safe_market_id = market_id.replace("'", "''")

logger.info(f"INFER_FINAL starting market_id={market_id} snapshot_date={as_of_str}")

# COMMAND ----------

# DBTITLE 1,Step 1 — find HPT-updated series
# version = 2 means the champion was updated by HPT — version 1 was set by the
# initial training/fit run.
# Use latest version per series — handles multiple HPT runs on the same day
updated_series_df = spark.sql(f"""
    WITH ranked AS (
        SELECT series_key, store_id, sku_id, version,
               ROW_NUMBER() OVER (
                   PARTITION BY series_key ORDER BY version DESC
               ) AS rn
        FROM {catalog}.gold.champion_index
        WHERE market_id = '{safe_market_id}'
          AND snapshot_date <= DATE('{as_of_str}')
          AND version >= 2
    )
    SELECT series_key, store_id, sku_id
    FROM ranked WHERE rn = 1
""")

# Apply scope filters (same as 03_score)
if filter_sku:
    updated_series_df = updated_series_df.filter(updated_series_df.sku_id.isin(filter_sku))
if filter_store:
    updated_series_df = updated_series_df.filter(updated_series_df.store_id.isin(filter_store))
n_updated = updated_series_df.count()
n_updated = updated_series_df.count()

_no_updates = n_updated == 0
if _no_updates:
    logger.info("INFER_FINAL: no HPT-updated series — exiting (initial inference is final)")
else:
    logger.info(f"INFER_FINAL: {n_updated} HPT-updated series — re-running inference")

# COMMAND ----------

# DBTITLE 1,Step 2 — load features + future spine
if not _no_updates:
    run_id = str(uuid.uuid4())

    features_df = read_table(spark, catalog, "gold", "sku_store_features")
    updated_series_df.createOrReplaceTempView("_updated_series")
    features_df.createOrReplaceTempView("_all_features")

    snapshot_features_df = spark.sql(f"""
        SELECT f.*
        FROM _all_features f
        JOIN _updated_series u USING (store_id, sku_id)
        WHERE f.snapshot_date = DATE('{as_of_str}')
    """)

    future_spine_df = build_future_spine(spark, catalog, market_id, as_of, config.market_config.horizon_days)
    future_spine_for_updated = future_spine_df.join(
        updated_series_df.select("store_id", "sku_id"),
        on=["store_id", "sku_id"], how="inner"
    )

# COMMAND ----------

# DBTITLE 1,Step 3 — score + write outputs
if not _no_updates:
    output_df, decomp_df, failed_df = score(spark, catalog, snapshot_features_df, future_spine_for_updated,
                                             config, as_of_str, run_id)
    write_outputs(spark, output_df, decomp_df, catalog)
    write_scoring_failures(spark, failed_df, catalog)

    logger.info(f"INFER_FINAL complete: refreshed {n_updated} series for market_id={market_id}")
