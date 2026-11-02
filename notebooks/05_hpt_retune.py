# Databricks notebook source
# MAGIC %md
# MAGIC ## 05_hpt_retune — inference job conditional HPT task
# MAGIC Runs daily, after `04_evaluate`, but self-gates internally: it queries
# MAGIC `breach_flag` for today (falling back to the training-time
# MAGIC `backtest_metrics` gate) and exits immediately (near-zero cost) unless
# MAGIC a series has been in sustained breach for the full 2-week window — see
# MAGIC `04_evaluate`. On a normal day this is a one-query no-op; Optuna only runs
# MAGIC for the specific breached series, never the whole market
# MAGIC (docs/instructions.md §12: HPT never in the daily critical path).
# MAGIC
# MAGIC Inputs: `breach_flag`, `sku_store_features`, `champion_index`.
# MAGIC Outputs: `champion_index` (only for series HPT actually improved).
# MAGIC Params: market_id, env, snapshot_date, train_start_date.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("market_id", "CA")
dbutils.widgets.text("env", "")
dbutils.widgets.text("snapshot_date", "")
dbutils.widgets.text("train_start_date", "")

# ── Optional debug / test scope filters ──────────────────────────────────────
dbutils.widgets.text("filter_sku",      "")
dbutils.widgets.text("filter_store",  "")
dbutils.widgets.text("filter_category", "")

# COMMAND ----------

# DBTITLE 1,TEMPORARY: skip HPT for this backfill run
# TEMPORARY KILL SWITCH — remove after backfill completes
dbutils.notebook.exit("SKIPPED: HPT disabled for backfill speed")

# COMMAND ----------

display(dbutils.widgets.getAll())

# COMMAND ----------

# DBTITLE 1,Dependencies
# Job-triggered runs get prophet/statsforecast/lightgbm/etc. from the job
# definition's environment_key: ml_env (see databricks.yml) — no install needed
# here. A previous version of this cell tried to detect interactive-vs-job via
# dbutils.notebook...getContext().tags(), but that API is not whitelisted on
# Databricks Free Edition serverless (Py4JSecurityException), so the guard
# always evaluated as "not a job" and attempted a pip install against a
# Workspace-relative path — which itself fails on Free Edition (Workspace
# Files can't be open()'d for content — see the sys.path note below) and was
# destabilizing the kernel for the next command. For interactive use, install
# requirements.txt manually from the workspace's own Volume-synced copy.

# COMMAND ----------

# DBTITLE 1,Read widgets
market_id        = dbutils.widgets.get("market_id")
env              = dbutils.widgets.get("env")
snapshot_date    = dbutils.widgets.get("snapshot_date") or None
train_start_date = dbutils.widgets.get("train_start_date") or None

# COMMAND ----------

# DBTITLE 1,sys.path — repo root importable
import importlib
import os
import shutil
import sys

# Databricks Free Edition serverless compute can stat/list Workspace Files but
# cannot open() them for content (java.io.FileNotFoundException wrapped as
# "Input/output error" from SharedUCWorkspaceLocalFileSystem — a Free Edition
# environment limitation, confirmed empirically, not specific to this repo).
# Shared modules are synced to a UC Volume as the source of truth, but importing
# directly off the Volume triggers Databricks' per-import autoreload-discoverability
# hook once per module — each one a network round-trip against Volume-backed
# storage, enough to blow the platform's 80s cell-setup budget once import
# count grows. One bulk copy to local ephemeral disk avoids the per-import
# overhead entirely.
_VOLUME_ROOT = f"/Volumes/{env}/default/pylib"
_REPO_ROOT = "/tmp/retail_forecast_pylib"
if os.path.isdir(_REPO_ROOT):
    shutil.rmtree(_REPO_ROOT)
shutil.copytree(_VOLUME_ROOT, _REPO_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
importlib.invalidate_caches()

# COMMAND ----------

# DBTITLE 1,Imports
from utils.dates import resolve_snapshot_date
from utils.io import read_table
from utils.logging import get_logger
from utils.spark import get_or_create_spark
from config.loader import load_config
from training.champion import write_champion_index
from training.engine import tag_series_key
from training.hpt import run_conditional_hpt, select_improved_retunes

# COMMAND ----------

# DBTITLE 1,Spark session, config
catalog     = env
spark       = get_or_create_spark(f"job2-hpt-{market_id}")

# sys.path only affects the DRIVER's own Python process — Spark executors run
# in separate processes/containers that never see it (confirmed empirically:
# an applyInPandas UDF referencing training.* raised ModuleNotFoundError until
# this fix). Zip the local copy and ship it to every executor via Spark
# Connect's artifact API — the serverless-native equivalent of sc.addPyFile().
import zipfile
_PYLIB_ZIP = "/tmp/retail_forecast_pylib.zip"
if os.path.exists(_PYLIB_ZIP):
    os.remove(_PYLIB_ZIP)
with zipfile.ZipFile(_PYLIB_ZIP, "w", zipfile.ZIP_DEFLATED) as _zf:
    for _root, _dirs, _files in os.walk(_REPO_ROOT):
        _dirs[:] = [d for d in _dirs if d != "__pycache__"]
        for _f in _files:
            if _f.endswith(".pyc"):
                continue
            _full = os.path.join(_root, _f)
            _zf.write(_full, os.path.relpath(_full, _REPO_ROOT))
spark.addArtifacts(_PYLIB_ZIP, pyfile=True)

logger      = get_logger(__name__, market_id=market_id, env=env)
as_of       = resolve_snapshot_date(snapshot_date)
as_of_str   = as_of.isoformat()
config      = load_config(spark, catalog, market_id)
safe_market_id = market_id.replace("'", "''")

logger.info(f"HPT_RETUNE starting market_id={market_id} snapshot_date={as_of_str}")

# COMMAND ----------

# DBTITLE 1,Step 1 — load breached series
failing_df = spark.sql(f"""
    SELECT series_key,
           SPLIT(series_key, '[|]')[2] AS store_id,
           SPLIT(series_key, '[|]')[3] AS sku_id
    FROM {catalog}.gold.breach_flag
    WHERE market_id = '{safe_market_id}'
      AND eval_date = DATE('{as_of_str}')
""")
n_failing = failing_df.count()

if n_failing == 0:
    # Fallback: training-time backtest gate
    failing_df = spark.sql(f"""
        SELECT entity_key                    AS series_key,
               SPLIT(entity_key, '[|]')[2]  AS store_id,
               SPLIT(entity_key, '[|]')[3]  AS sku_id
        FROM {catalog}.gold.backtest_metrics
        WHERE market_id = '{safe_market_id}'
          AND snapshot_date = DATE('{as_of_str}')
          AND grain         = 'market_store_sku'
          AND passed        = false
    """)
    n_failing = failing_df.count()

_no_failing = n_failing == 0
if _no_failing:
    logger.info("HPT_RETUNE: no breached/failing series — exiting")
else:
    logger.info(f"HPT_RETUNE: {n_failing} failing series — starting Optuna retuning")

# COMMAND ----------

# DBTITLE 1,Step 2 — reconstruct training features for failing series
if not _no_failing:
    train_start     = resolve_snapshot_date(train_start_date) if train_start_date else config.market_config.history_start
    train_start_str = train_start.isoformat() if hasattr(train_start, "isoformat") else str(train_start)

    features_df = read_table(spark, catalog, "gold", "sku_store_features")
    features_df.createOrReplaceTempView("_all_features")
    training_features_df = spark.sql(f"""
        WITH eligible AS (
            SELECT * FROM _all_features
            WHERE snapshot_date <= DATE('{as_of_str}')
              AND date_id BETWEEN DATE('{train_start_str}') AND DATE('{as_of_str}')
        ),
        latest_per_day AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY store_id, sku_id, date_id
                ORDER BY snapshot_date DESC
            ) AS rn FROM eligible
        )
        SELECT * EXCEPT (rn) FROM latest_per_day WHERE rn = 1
    """)

    # Filter to failing series only
    failing_df.createOrReplaceTempView("_failing_series")
    failing_features_df = training_features_df.join(
        failing_df.select("store_id", "sku_id"),
        on=["store_id", "sku_id"], how="inner"
    )

# COMMAND ----------

# DBTITLE 1,Step 3 — load existing champions as HPT baseline
if not _no_failing:
    # Use the latest version per series (handles re-training for same snapshot)
    champions_pdf = spark.sql(f"""
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY series_key
                ORDER BY snapshot_date DESC, version DESC
            ) AS rn
            FROM {catalog}.gold.champion_index
            WHERE market_id = '{safe_market_id}'
              AND snapshot_date <= DATE('{as_of_str}')
        )
        SELECT * EXCEPT (rn) FROM ranked WHERE rn = 1
    """).toPandas()

# COMMAND ----------

# DBTITLE 1,Step 4 — run conditional HPT (Optuna, failing series only)
if not _no_failing:
    failing_features_tagged = tag_series_key(spark, failing_features_df, market_id, "_hpt_input")
    gated_pdf = failing_df.toPandas()
    gated_pdf["grain"]          = "market_store_sku"
    gated_pdf["entity_key"]     = gated_pdf["series_key"]
    gated_pdf["market_id"]      = market_id
    gated_pdf["category"]       = gated_pdf["series_key"].str.split("|").str[1]
    gated_pdf["passed"]     = False

    hpt_metrics_df  = run_conditional_hpt(spark, failing_features_tagged, gated_pdf, champions_pdf, config, as_of_str)
    hpt_metrics_pdf = hpt_metrics_df.toPandas()
    improved_pdf    = select_improved_retunes(hpt_metrics_pdf, champions_pdf)

# COMMAND ----------

# DBTITLE 1,Diagnostic — HPT status breakdown
import importlib, training.hpt as _hpt_mod
importlib.reload(_hpt_mod)
from training.hpt import run_conditional_hpt, select_improved_retunes
print(f"champions_pdf rows  : {len(champions_pdf)}")
print(f"gated_pdf rows      : {len(gated_pdf)}")
print(f"hpt_metrics_pdf rows: {len(hpt_metrics_pdf)}")
print(f"improved_pdf rows   : {len(improved_pdf)}")
if not hpt_metrics_pdf.empty:
    print("\nstatus breakdown:")
    print(hpt_metrics_pdf['status'].value_counts().to_string())
    ok = hpt_metrics_pdf[(hpt_metrics_pdf['status']=='ok') & hpt_metrics_pdf['passed_decomposition']]
    if not ok.empty:
        orig = champions_pdf.set_index('series_key')['wmape_7d']
        ok = ok.copy()
        ok['orig_wmape'] = ok['series_key'].map(orig)
        ok['improved'] = ok['wmape_7d'] < ok['orig_wmape']
        print(f"\nok+passed rows: {len(ok)} | improved: {ok['improved'].sum()}")
        print(ok[['series_key','wmape_7d','orig_wmape','improved']].head(10).to_string())

# COMMAND ----------

# DBTITLE 1,Step 5 — write improved champions
if not _no_failing:
    if not improved_pdf.empty:
        logger.info(f"HPT_RETUNE: {len(improved_pdf)} series improved — updating champions")
        write_champion_index(spark, spark.createDataFrame(improved_pdf), catalog, as_of_str)
    else:
        logger.info("HPT_RETUNE: HPT found no improvements over default-param champions")

    logger.info(f"HPT_RETUNE complete market_id={market_id}")
