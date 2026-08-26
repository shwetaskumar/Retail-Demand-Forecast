# Databricks notebook source
# MAGIC %md
# MAGIC ## 02_train — JOB 2 TRAINING
# MAGIC Trains candidates, selects champions, gates at 3 grains, optionally runs
# MAGIC conditional HPT, and registers champions to MLflow.
# MAGIC
# MAGIC Pipeline (one cell per step, so a failure — e.g. a bad champion write, a
# MAGIC gate SQL error, an Optuna crash — points straight at the cell that broke,
# MAGIC and any step can be re-run individually against the DataFrames already in
# MAGIC this notebook's state):
# MAGIC   1. Load feature history `[train_start_date, snapshot_date]` (latest snapshot per date_id)
# MAGIC   2. `run_engine_layered()` — config-driven cascade, cost-tier Spark passes (`training/engine_layered.py`)
# MAGIC   3. `select_champions_spark()` — lowest MASE among valid decomposing candidates
# MAGIC   4. `write_champion_index()` — MERGE into `champion_index`
# MAGIC   5. `gate()` — volume-weighted WMAPE at 3 grains vs category thresholds
# MAGIC   6. `write_backtest_metrics()` — MERGE into `backtest_metrics`
# MAGIC   7. Conditional HPT (only if `run_hpt=True`) — re-tunes series/cohorts that failed the gate
# MAGIC   8. MLflow registration — one pyfunc per market×category
# MAGIC
# MAGIC HPT is NOT run by default. Set the `run_hpt` widget to `True` to trigger
# MAGIC conditional Optuna re-tuning for series/cohorts that failed the gate. This
# MAGIC keeps the weekly critical path fast; HPT is an explicit remediation step,
# MAGIC not automatic overhead on every run.
# MAGIC
# MAGIC Inputs: `{catalog}.gold.sku_store_features` @ nearest snapshot_date <= as_of
# MAGIC (exact snapshot used if available; nearest earlier snapshot used otherwise — see Step 1).
# MAGIC Outputs: `champion_index`, `backtest_metrics`, artifacts, category pyfunc.
# MAGIC Params: market_id, env, snapshot_date, train_start_date, run_hpt, use_batch_engine.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("market_id",      "US")
dbutils.widgets.text("env",             "")
dbutils.widgets.text("snapshot_date",   "")
dbutils.widgets.text("train_start_date","")
dbutils.widgets.text("run_hpt",         "False")
dbutils.widgets.text("use_batch_engine","False")

# ── Optional debug / test scope filters ──────────────────────────────────────
# Leave ALL empty to train the full market (production default).
# Each accepts a single value or a comma-separated list.
# Tip: combine freely — e.g. filter_sku=M-FRIES, filter_category=A trains
#      only category A fries series, not every category A sku.
dbutils.widgets.text("filter_sku",      "")   # e.g. "M-GR-BT-SAUCE" or "M-FRIES,M-COLA"
dbutils.widgets.text("filter_store",  "")   # e.g. "REST001" or "REST001,REST002"
dbutils.widgets.text("filter_category", "")   # e.g. "A" or "A,B"

# COMMAND ----------

# display(dbutils.widgets.getAll())  # disabled — unnecessary action in job context

# COMMAND ----------

# DBTITLE 1,Install pinned deps (interactive runs only — job clusters already have these)
# Only install when running interactively — job clusters already have libraries
# pre-installed via the job definition. Saves 3-5 min per notebook.run() call.
import subprocess as _sp, os as _os
try:
    _ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _is_job = _ctx.tags().get("jobId").isDefined()
except Exception:
    _is_job = False
if not _is_job:
    _sp.check_call(["pip", "install", "-q", "-r", _os.path.join(_os.path.dirname(_os.getcwd()), "requirements.txt")])
    dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Restart Python (interactive only)
# Restart merged into pip-install cell above (only runs in interactive context)

# COMMAND ----------

# DBTITLE 1,Read widgets
market_id        = dbutils.widgets.get("market_id")
env              = dbutils.widgets.get("env")
snapshot_date    = dbutils.widgets.get("snapshot_date") or None
train_start_date = dbutils.widgets.get("train_start_date") or None
run_hpt          = dbutils.widgets.get("run_hpt").strip().lower() == "true"
use_batch_engine = dbutils.widgets.get("use_batch_engine").strip().lower() == "true"


def _parse_filter(raw: str) -> list[str]:
    """Split a comma-separated widget value into a clean list; empty string → []."""
    return [v.strip() for v in raw.split(",") if v.strip()]


filter_sku      = _parse_filter(dbutils.widgets.get("filter_sku"))
filter_store    = _parse_filter(dbutils.widgets.get("filter_store"))
filter_category = _parse_filter(dbutils.widgets.get("filter_category"))
_scoped_run     = bool(filter_sku or filter_store or filter_category)

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
import time as _t
import uuid

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window as W

from utils.dates import resolve_snapshot_date
from utils.logging import get_logger
from utils.spark import get_or_create_spark
from config.loader import load_config
from training.champion import select_champions_spark, write_champion_index
from training.engine import run_engine_batch
from training.engine_layered import run_engine_layered
from training.evaluation import gate, write_backtest_metrics

# COMMAND ----------

# DBTITLE 1,Spark session, config, run_id, timer helper
catalog = env
spark   = get_or_create_spark(f"job2-training-{market_id}")
logger  = get_logger(__name__, market_id=market_id, env=env)

_job_t0 = _t.perf_counter()
_phase_t = _job_t0


def _log_timer(label: str) -> None:
    global _phase_t
    elapsed = _t.perf_counter() - _phase_t
    logger.info(f"[TIMER] {label:<40} {elapsed:>7.1f}s  ({elapsed/60:.1f}min)")
    _phase_t = _t.perf_counter()


as_of            = resolve_snapshot_date(snapshot_date)
as_of_str        = as_of.isoformat()
run_id           = str(uuid.uuid4())
config           = load_config(spark, catalog, market_id)
train_start      = resolve_snapshot_date(train_start_date) if train_start_date else config.market_config.history_start
train_start_str  = train_start.isoformat() if hasattr(train_start, "isoformat") else str(train_start)
_log_timer("01_spark_and_config_load")

logger.info(
    f"JOB 2 TRAINING starting market_id={market_id} snapshot_date={as_of_str} "
    f"train_start={train_start_str} run_hpt={run_hpt} run_id={run_id}"
)

# COMMAND ----------

# DBTITLE 1,Step 1 — load pre-materialised training features
# sku_store_features is built by JOB 1 DATA (01_build_features).
# Each snapshot covers only the last incremental_write_days (= 30) days,
# so no single snapshot contains the full training history.
#
# Read strategy: for each (store, sku, date_id) in the training window
# take the feature row from the MOST RECENT snapshot built on or before
# as_of. Point-in-time correct — no future feature leakage — and assembles
# the full multi-year history from multiple incremental snapshots naturally.
training_feat_table = f"{catalog}.gold.sku_store_features"

# Scope filters: push sku/store into the inner query so the ROW_NUMBER
# window only runs over the requested series — critical for fast dev/test
# runs (avoids computing partitions for all 110K series when testing 3).
# category filter is applied after Step 2 where the final category is assigned.
def _sql_in(col: str, vals: list[str]) -> str:
    """Build a safe SQL IN clause for string literals."""
    escaped = ", ".join("'" + v.replace("'", "''") + "'" for v in vals)
    return f"AND {col} IN ({escaped})"

_sku_clause   = _sql_in("sku_id",   filter_sku)   if filter_sku   else ""
_store_clause = _sql_in("store_id", filter_store) if filter_store else ""

training_features_df = spark.sql(f"""
    SELECT * EXCEPT (snapshot_date, _rn)
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY market_id, store_id, sku_id, date_id
                   ORDER BY snapshot_date DESC
               ) AS _rn
        FROM {training_feat_table}
        WHERE market_id = '{market_id}'
          AND snapshot_date     <= DATE('{as_of_str}')
          AND date_id BETWEEN DATE('{train_start_str}') AND DATE('{as_of_str}')
          {_sku_clause}
          {_store_clause}
    )
    WHERE _rn = 1
""")

if training_features_df.isEmpty():
    raise RuntimeError(
        f"No training features found for market_id={market_id} "
        f"date_id BETWEEN {train_start_str} AND {as_of_str}. "
        f"Run JOB 1 DATA (01_build_features) first."
    )

logger.info(
    f"Loaded training features: market_id={market_id} "
    f"date_id=[{train_start_str}, {as_of_str}] "
    f"(latest snapshot per date, all snapshots <= {as_of_str})"
)

# COMMAND ----------

# DBTITLE 1,Step 2 — deduplicate category (latest classification wins)
# Volume-based ABC classification can reclassify a sku over time (e.g. B in
# 2024 -> A in 2025). We assign the CURRENT category to ALL history so each
# physical product gets exactly ONE series_key and ONE champion.
_cat_w = W.partitionBy("store_id", "sku_id").orderBy(F.col("date_id").desc())
training_features_df = (
    training_features_df
    .withColumn("_latest_category", F.first("category").over(_cat_w))
    .drop("category")
    .withColumnRenamed("_latest_category", "category")
)

# training_set: table name forwarded to MLflow registry for dataset lineage
training_set = training_feat_table

# ── category filter (applied here, after Step 2 reassigns category) ──────────
if filter_category:
    training_features_df = training_features_df.filter(
        F.col("category").isin(filter_category)
    )

# ── scope warning ─────────────────────────────────────────────────────────────
if _scoped_run:
    _n_series = training_features_df.select("store_id", "sku_id").distinct().count()
    logger.warning(
        "\n" + "=" * 70 + "\n"
        "  SCOPED RUN — training on a SUBSET of the market.\n"
        f"  filter_sku      = {filter_sku or '(all)'}\n"
        f"  filter_store    = {filter_store or '(all)'}\n"
        f"  filter_category = {filter_category or '(all)'}\n"
        f"  Series in scope : {_n_series:,}\n"
        "  Champions will be written to champion_index for these\n"
        "  series only. All other series keep their existing champions.\n"
        "  To train the full market leave all filter_* widgets empty.\n"
        + "=" * 70
    )

logger.info(
    f"TRAINING ARGS\n"
    f"  market_id         = {market_id}\n"
    f"  env/catalog       = {catalog}\n"
    f"  snapshot_date     = {as_of_str}\n"
    f"  train_start_date  = {train_start_str}\n"
    f"  features_table    = {training_feat_table}\n"
    f"  run_hpt           = {run_hpt}\n"
    f"  use_batch_engine  = {use_batch_engine}\n"
    f"  run_id            = {run_id}\n"
    f"  scoped_run        = {_scoped_run}"
)

# COMMAND ----------

# DBTITLE 1,Step 3 — (DISABLED) prev champions not used by engine_layered
# DISABLED: prev_champions collected 1.7 GB to driver but run_engine_layered()
# never actually reads the parameter. Skipping saves ~5 min per training window.
# Re-enable if/when Phase-1 stability check is implemented in the engine.
prev_champions: dict = {}
logger.info("Step 3 skipped — prev_champions not used by engine (dead code).")

# COMMAND ----------

# DBTITLE 1,Step 4 — run engine (cascade or batch)
if use_batch_engine:
    logger.info("ENGINE mode: batch (run_engine_batch) — ETS+Ridge vectorised per store, cold-start only")
    candidate_metrics_df = run_engine_batch(spark, training_features_df, config, as_of_str).cache()
else:
    logger.info("ENGINE mode: layered (run_engine_layered) — config-driven cascade, "
                "models from config_model grouped by evaluation_cost")
    candidate_metrics_df = run_engine_layered(
        spark, training_features_df, config, as_of_str, prev_champions=prev_champions
    ).cache()

# Materialize once (the .cache() above ensures this is a single pass)
total_candidates = candidate_metrics_df.count()
logger.info(f"Engine produced {total_candidates} candidate rows total.")

# COMMAND ----------

# DBTITLE 1,Step 5 — champion selection
champions_df = select_champions_spark(
    spark, candidate_metrics_df, categories=config.categories, max_bias=config.market_config.max_champion_bias
)

# COMMAND ----------

# DBTITLE 1,Step 6 — write champions
write_champion_index(spark, champions_df, catalog, as_of_str, training_run_id=run_id)

# COMMAND ----------

# DBTITLE 1,Step 7 — gate + backtest metrics
# gate() only needs metric columns — exclude the heavy `params` JSON (16KB avg)
# to avoid collecting 1.7 GB to the driver.
_gate_cols = [c for c in ["market_id", "category", "sku_id", "series_key",
              "model_type", "wmape_7d", "volume", "bias_pct", "mase_7d"] if c in champions_df.columns]
champions_pdf = champions_df.select(*_gate_cols).toPandas()
gated_pdf     = gate(champions_pdf, config.categories)
write_backtest_metrics(spark, gated_pdf, catalog, run_id, as_of_str)

n_failed = int((gated_pdf["passed"] == False).sum())  # noqa: E712
logger.info(
    f"Gate complete: {len(gated_pdf)} rows, {n_failed} failing. "
    f"Re-run with run_hpt=True to trigger conditional tuning for failing series."
    if n_failed else
    f"Gate complete: all {len(gated_pdf)} rows passed."
)

# COMMAND ----------

# DBTITLE 1,Step 8 — conditional HPT (explicit remediation only, run_hpt widget)
if run_hpt:
    from training.hpt import run_conditional_hpt, select_improved_retunes
    logger.info("HPT requested: tuning failing series/cohorts...")
    hpt_metrics_df  = run_conditional_hpt(spark, training_features_df, gated_pdf, champions_pdf, config, as_of_str)
    hpt_metrics_pdf = hpt_metrics_df.toPandas()
    improved_pdf    = select_improved_retunes(hpt_metrics_pdf, champions_pdf)

    if not improved_pdf.empty:
        logger.info(f"HPT improved {len(improved_pdf)} series; updating champions")
        write_champion_index(spark, spark.createDataFrame(improved_pdf), catalog, as_of_str, training_run_id=run_id)
        champions_pdf = pd.concat(
            [champions_pdf[~champions_pdf["series_key"].isin(improved_pdf["series_key"])], improved_pdf],
            ignore_index=True,
        )
        gated_pdf = gate(champions_pdf, config.categories)
        write_backtest_metrics(spark, gated_pdf, catalog, run_id, as_of_str)
    else:
        logger.info("HPT found no improvements over default-param champions.")
elif n_failed > 0:
    logger.info(f"{n_failed} rows failed gate — set run_hpt=True to trigger tuning.")

# COMMAND ----------

# DBTITLE 1,Step 9 — MLflow registration + run_id linkage
from training.mlflow_registry import register_champions_to_mlflow
from training.champion import update_mlflow_run_ids

logger.info("Registering champions to MLflow UC Model Registry...")
try:
    # Only select columns needed for MLflow metrics (NOT params — 16KB avg, 1.7GB total)
    final_champions_df = spark.sql(f"""
        SELECT market_id, category, series_key, store_id, sku_id,
               model_type, wmape_7d, mase_7d, bias_pct, passed_gate, version, volume
        FROM {catalog}.gold.champion_index
        WHERE market_id = '{market_id.replace("'", "''")}'
          AND snapshot_date = DATE('{as_of_str}')
    """)
    run_ids = register_champions_to_mlflow(
        spark, catalog, market_id, as_of_str, final_champions_df, training_features_df,
        training_set=training_set, repo_root=_REPO_ROOT,
    )
    logger.info(f"MLflow registration complete: {len(run_ids)} categories registered")

    # Link run_ids back to champion_index so inference can verify provenance
    n_updated = update_mlflow_run_ids(spark, catalog, market_id, as_of_str, run_ids)
    logger.info(f"Linked {n_updated} categories' mlflow_run_id to champion_index")
except Exception as e:
    logger.error(f"MLflow registration failed (non-fatal): {e}")

logger.info(f"JOB 2 TRAINING complete market_id={market_id} snapshot_date={as_of_str} run_id={run_id}")
