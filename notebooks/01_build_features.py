# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC ## 01_build_features — JOB 1 DATA
# MAGIC Builds `{catalog}.gold.sku_store_features` for `market` as of
# MAGIC `snapshot_date`. Source tables (sales/sku_store/sku/store/events/date)
# MAGIC are READ-ONLY; this notebook writes ONLY the new parallel feature table.
# MAGIC Independent of training/inference — runnable standalone for any
# MAGIC `snapshot_date` (replay/backfill).
# MAGIC
# MAGIC Each pipeline step is its own cell so a failure points directly at the
# MAGIC step that broke, and any step can be re-run individually against the
# MAGIC DataFrames already sitting in this notebook's Python state — no need to
# MAGIC re-run the whole job to debug one stage. Shared/tested feature logic lives
# MAGIC in `data/features/*.py` and `utils/*.py`; this notebook only sequences it.
# MAGIC
# MAGIC Inputs: source tables (read-only, per feature_engineering_spec.md).
# MAGIC Outputs: `{catalog}.gold.sku_store_features`.
# MAGIC Params: market_id, env, snapshot_date.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("market_id", "")
dbutils.widgets.text("env", "")
dbutils.widgets.text("snapshot_date", "")
dbutils.widgets.dropdown("disable_halo_cannibalization", "False", ["False", "True"])

# COMMAND ----------

display(dbutils.widgets.getAll())

# COMMAND ----------

# DBTITLE 1,Install pinned deps (interactive runs only)
import subprocess as _sp, os as _os
try:
    _ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _is_job = _ctx.tags().get("jobId").isDefined()
except Exception:
    _is_job = False
if not _is_job:
    _sp.check_call(["pip", "install", "-q", "-r",
                    _os.path.join(_os.path.dirname(_os.getcwd()), "requirements.txt")])

# COMMAND ----------

# DBTITLE 1,Restart Python (interactive only)
# Only restart when running interactively — avoids breaking dbutils.notebook.run() context
if not _is_job:
    dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Read widgets
market_id                    = dbutils.widgets.get("market_id")
env                          = dbutils.widgets.get("env")
snapshot_date                = dbutils.widgets.get("snapshot_date") or None
disable_halo_cannibalization = dbutils.widgets.get("disable_halo_cannibalization") == "True"

# COMMAND ----------

# DBTITLE 1,sys.path — repo root importable
import importlib
import os
import shutil
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

# Clear stale .pyc bytecode caches so every import reads from the .py source files.
# Mirrors the same pattern in notebooks/00_create_tables.py — necessary because the
# cluster may have cached a previous bytecode version across dbutils.library.restartPython()
# calls or across repeated dbutils.notebook.run() invocations on the same cluster.
# Walk ALL sub-packages (including data/features/) — not just top-level dirs.
for _root, _dirs, _ in os.walk(_REPO_ROOT):
    if "__pycache__" in _dirs:
        _cache = os.path.join(_root, "__pycache__")
        shutil.rmtree(_cache)
        _dirs.remove("__pycache__")  # don't descend into removed dir
importlib.invalidate_caches()

# COMMAND ----------

# DBTITLE 1,Imports
import datetime as dt

from pyspark import StorageLevel

from utils.dates import read_window, resolve_snapshot_date, write_window
from utils.logging import get_logger
from utils.spark import get_or_create_spark
from config.loader import load_config
from data.features.active_window import build_active_window
from data.features.category_classification import add_category, compute_sku_category
from data.features.category_features import build_category_features
from data.features.event_features import build_holiday_features, build_promo_features
from data.features.gap_fill import gap_fill_zeros
from data.features.halo_cannibalization_features import (
    build_historical_partner_demand_features,
    build_known_at_inference_features,
    discover_partner_relationships,
    write_partner_map,
)
from data.features.lifecycle import (
    compute_is_active,
    compute_item_lifecycle,
    compute_store_lifecycle,
)
from data.features.sales_base import build_sales_base
from data.features.stat_features import add_statistical_features
from data.features.temporal_features import add_temporal_features
from data.features.writer import write_features

# Maximum lookback needed by stat_features before write_start:
# lag_365 (365) + rolling_28 warm-up (28) = 393 days.
_STAT_WARMUP_DAYS = 393

# COMMAND ----------

# DBTITLE 1,Spark session, config, read/write windows
catalog = env
spark   = get_or_create_spark(f"job1-data-build-features-{market_id}")
# Set DBFS checkpoint dir here, before any df.checkpoint() call.
# DBFS-backed checkpoints survive executor preemption (Azure spot) unlike
# localCheckpoint, which stores data on executor-local disk and becomes
# permanently unreadable if that executor is terminated mid-run.
spark.sparkContext.setCheckpointDir("dbfs:/tmp/fcst-engine/checkpoints")
logger  = get_logger(__name__, market_id=market_id, env=env)

as_of = resolve_snapshot_date(snapshot_date)
config = load_config(spark, catalog, market_id)
read_start, read_end = read_window(as_of, config.market_config.incremental_read_days)
write_start, write_end = write_window(as_of, config.market_config.incremental_write_days)
logger.info(
    f"Building features market_id={market_id} snapshot_date={as_of} "
    f"read_window=[{read_start},{read_end}] write_window=[{write_start},{write_end}]"
)

# COMMAND ----------

# DBTITLE 1,Step 1 — sales base (target)
# Persisted (DISK_ONLY) because it's consumed 4x downstream:
# store_lifecycle, item_lifecycle, gap_fill, category.
sales_base_df = build_sales_base(spark, catalog, market_id, read_start, read_end)
sales_base_df.persist(StorageLevel.DISK_ONLY)
sales_base_df.count()

# COMMAND ----------

# DBTITLE 1,Step 2 — active window bounds from sku_store
active_window_df = build_active_window(spark, catalog, market_id, sales_base_df)

# COMMAND ----------

# DBTITLE 1,Step 3 — lifecycle flags from actual POS sales
# is_new_store: store has < 365 distinct sales days in read window.
# is_new_item:  item has < 365 distinct sales days at market level in read window.
# Both computed from raw sales_base_df BEFORE gap_fill so zero-fills don't inflate
# counts. item_lifecycle_df is reused below by compute_sku_category.
store_lifecycle_df = compute_store_lifecycle(spark, sales_base_df)
item_lifecycle_df = compute_item_lifecycle(spark, catalog, market_id)

# COMMAND ----------

# DBTITLE 1,Step 4 — dense daily spine + isActive
# isActive = 1 ONLY IF window_active=1 AND is_new_store=0 AND is_new_item=0.
# snapshot_date plays no role here; read_end = as_of already constrains the window.
#
# checkpoint(eager=True) cuts the expensive explode(sequence()) lineage so
# downstream stages (gap_fill, stat_features, event joins) don't re-derive the
# ~218M-row explosion on every action boundary. Uses DBFS (set above) rather
# than localCheckpoint so the data survives executor preemption — without this,
# Steps 5-10 can run for 10-30 min and if any executor holding a local
# checkpoint block is terminated the entire pipeline fails irrecoverably.
is_active_df = compute_is_active(
    spark, active_window_df, store_lifecycle_df, item_lifecycle_df, read_start, read_end
).checkpoint(eager=True)

# COMMAND ----------

# DBTITLE 1,Step 5 — gap fill (true zeros only where isActive=1)
gap_filled_df = gap_fill_zeros(spark, is_active_df, sales_base_df)

# COMMAND ----------

# DBTITLE 1,Step 6 — category classification
# A/B/C/D/SEASONAL: static volume-based ABC/D/SEASONAL rule, per user direction
# 2026-07-01 (see data/features/category_classification.py).
# Computed here (before unpersist) because it reads sales_base_df.
category_df = compute_sku_category(spark, sales_base_df, item_lifecycle_df)

# sales_base_df no longer needed — release persisted storage.
sales_base_df.unpersist()

# COMMAND ----------

# DBTITLE 1,Step 7 — narrow to stat-feature window
# Only keep rows needed for stat feature correctness: write_start minus the
# maximum lookback (365-day lag + 28-day rolling warm-up = 393 days) through
# read_end. This avoids computing expensive window functions on ancient rows
# that will be discarded by the write filter.
# When read_window ≈ write_window + warmup (daily incremental), the filter is a
# no-op; it saves significantly on large backfill windows (read_days >> 400).
stat_floor = max(read_start, write_start - dt.timedelta(days=_STAT_WARMUP_DAYS))
narrowed_df = gap_filled_df.filter(f"date_id >= '{stat_floor.isoformat()}'")

# COMMAND ----------

# DBTITLE 1,Step 8 — statistical + temporal + event features
stat_df = add_statistical_features(spark, narrowed_df)
temporal_df = add_temporal_features(spark, catalog, stat_df)
promo_df = build_promo_features(spark, catalog, market_id, temporal_df)
holiday_df = build_holiday_features(spark, catalog, market_id, promo_df)
# Disruption features removed — closure/stockout/sales-break features were reducing accuracy.
event_df = holiday_df

# COMMAND ----------

# DBTITLE 1,Step 9 — attach category
# Attach category BEFORE any category-scoped feature step. This also fixes a
# pre-existing ordering bug: build_category_features()'s SQL groups/joins on
# `category`, which previously did not exist on its input frame until AFTER it
# ran (add_category was called last) — that would fail at runtime on Databricks
# with "cannot resolve 'category'" the first time this job actually executed there.
with_category_df = add_category(spark, event_df, category_df)

# COMMAND ----------

# DBTITLE 1,Step 10 — halo / cannibalization (data-driven, no manual mapping)
# feature_engineering_spec.md L48: "sub-events (LTO/bundle cannibalization)".
# discover_partner_relationships derives each sku's halo (complement) and
# cannibalization (substitute) partners purely from realized demand co-movement.
# Persisted separately (halo_cannib_partners) so JOB 3's future spine can
# reconstruct the KNOWN_AT_INFERENCE columns without re-reading source tables.
if disable_halo_cannibalization:
    logger.warning(
        f"Skipping halo/cannibalization features for this run: market_id={market_id} snapshot_date={as_of}"
    )
    partner_map_df = spark.sql("""
        SELECT CAST(NULL AS STRING) AS sku_id,
               CAST(NULL AS STRING) AS partner_sku_id,
               CAST(NULL AS DOUBLE) AS corr,
               CAST(NULL AS INT)    AS n_overlap_days,
               CAST(NULL AS STRING) AS relation_type
        WHERE 1 = 0
    """)
    write_partner_map(spark, partner_map_df, catalog, market_id, as_of)
    with_category_df.createOrReplaceTempView("_hc_disabled_input")
    halo_df = spark.sql("""
        SELECT t.*,
               CAST(0.0 AS FLOAT) AS halo_partner_qty_7d_roll,
               CAST(0.0 AS FLOAT) AS cannib_partner_qty_7d_roll,
               CAST(0.0 AS FLOAT) AS halo_partner_corr_strength,
               CAST(0.0 AS FLOAT) AS cannib_partner_corr_strength,
               CAST(0 AS INT)     AS halo_partner_promo_active_flag,
               CAST(0 AS INT)     AS halo_partner_promo_count,
               CAST(0 AS INT)     AS cannib_partner_promo_active_flag,
               CAST(0 AS INT)     AS cannib_partner_promo_count
        FROM _hc_disabled_input t
    """)
else:
    partner_map_df = discover_partner_relationships(spark, with_category_df)
    write_partner_map(spark, partner_map_df, catalog, market_id, as_of)
    known_halo_df = build_known_at_inference_features(
        spark, catalog, market_id, with_category_df, partner_map_df
    ).localCheckpoint()
    halo_df = build_historical_partner_demand_features(spark, known_halo_df, partner_map_df)

# COMMAND ----------

# DBTITLE 1,Step 11 — category-context features
# Cut the lazy lineage from Steps 5-10 before the two read-intensive operations
# below. build_category_features scans _cat_features_input three times and
# write_features triggers a MERGE that re-reads the full chain. Without this
# checkpoint Spark stacks thousands of stages, overwhelming the driver scheduler
# and causing a DRIVER_ERROR OOM crash on the full PL 750-day window.
#
# Use reliable checkpoint (writes to DBFS) instead of localCheckpoint.
# localCheckpoint stores data on executor-local disk: on an autoscaling cluster
# the autoscaler can reclaim the executor that holds a partition before Spark
# reads it back, producing CHECKPOINT_RDD_BLOCK_ID_NOT_FOUND with 461 failures
# on the same block — irrecoverable because the lineage has been truncated.
halo_df = halo_df.checkpoint(eager=True)

full_df = build_category_features(spark, halo_df)

# COMMAND ----------

# DBTITLE 1,Step 12 — write (idempotent MERGE, incremental write window only)
# Materialise before the MERGE so the planner sees a clean checkpoint scan
# instead of the full build_category_features chain.
# Reliable checkpoint (same dir set in Step 11); same executor-loss risk applies.
full_df = full_df.checkpoint(eager=True)

write_features(
    spark,
    full_df,
    catalog,
    as_of,
    config.market_config.incremental_write_days,
    market_id=market_id,
)
logger.info(f"JOB 1 DATA complete for market_id={market_id} snapshot_date={as_of}")