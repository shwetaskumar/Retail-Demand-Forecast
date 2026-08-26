# Databricks notebook source
# DBTITLE 1,03_score header
# MAGIC %md
# MAGIC ## 03_score — JOB 3 INFERENCE
# MAGIC Champions + feature table -> 140-day future spine -> predict + additive
# MAGIC decompose -> `output`, `decomposition` -> DQ -> `dq_audit` ->
# MAGIC Power BI. Scoring only, no training. Independent of JOB 1/JOB 2 (reads
# MAGIC whatever champions `champion_index` holds).
# MAGIC
# MAGIC One cell per step so a failure (e.g. a missing future-spine table, a
# MAGIC per-series scoring exception) is immediately traceable to the step that
# MAGIC broke and re-runnable in isolation.
# MAGIC
# MAGIC Inputs: `champion_index` + `sku_store_features` @ snapshot_date.
# MAGIC Outputs: `output`, `decomposition`, `scoring_failures`, `dq_audit`.
# MAGIC Params: market_id, env, snapshot_date.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("market_id", "US")
dbutils.widgets.text("env", "")
dbutils.widgets.text("snapshot_date", "")

# ── Optional debug / test scope filters ──────────────────────────────────────
# Leave ALL empty to score the full market (production default).
# Each accepts a single value or a comma-separated list.
dbutils.widgets.text("filter_sku",      "")   # e.g. "M-FR-FRIES-LRG" or "M-FRIES,M-COLA"
dbutils.widgets.text("filter_store",  "")   # e.g. "11018" or "11018,11019"
dbutils.widgets.text("filter_category", "")   # e.g. "A" or "A,B"

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
import datetime
import time as _time
import uuid

from pyspark.sql.utils import AnalysisException as _AnalysisException

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
catalog = env
spark   = get_or_create_spark(f"job3-inference-{market_id}")
logger  = get_logger(__name__, market_id=market_id, env=env)

as_of     = resolve_snapshot_date(snapshot_date)
as_of_str = as_of.isoformat()
config    = load_config(spark, catalog, market_id)
safe_market_id = market_id.replace("'", "''")

logger.info(f"JOB 3 INFERENCE starting market_id={market_id} snapshot_date={as_of_str}")

# COMMAND ----------

# DBTITLE 1,Step 1 — latest feature snapshot as of as_of
# Historical context for models. Use the most-recent snapshot <= inference date
# — in production the data job will have built an exact snapshot; in backfill
# we fall back to the closest available snapshot (e.g. 2025-12-31 for all Jan
# 2026 inference days).
#
# Resolve the right snapshot_date first with a targeted agg (market_id
# filter avoids a full scan across all markets and all snapshots before filtering).
_latest_snap = spark.sql(f"""
    SELECT MAX(snapshot_date)
    FROM {catalog}.gold.sku_store_features
    WHERE snapshot_date <= DATE('{as_of_str}')
      AND market_id = '{safe_market_id}'
""").collect()[0][0]

if _latest_snap is None:
    raise RuntimeError(
        f"sku_store_features has no snapshot <= {as_of_str} for market_id={market_id}. "
        f"Run JOB 1 DATA first to populate it."
    )

logger.info(f"Using feature snapshot_date={_latest_snap} for inference as_of={as_of_str}")

snapshot_features_df = (
    spark.table(f"{catalog}.gold.sku_store_features")
    .filter(f"snapshot_date = DATE('{_latest_snap}') AND market_id = '{safe_market_id}'")
)

# ── Apply optional scope filters (same pattern as 02_train) ──────────────────
if filter_sku:
    snapshot_features_df = snapshot_features_df.filter(
        snapshot_features_df.sku_id.isin(filter_sku)
    )
if filter_store:
    snapshot_features_df = snapshot_features_df.filter(
        snapshot_features_df.store_id.isin(filter_store)
    )
if filter_category:
    snapshot_features_df = snapshot_features_df.filter(
        snapshot_features_df.category.isin(filter_category)
    )
if _scoped_run:
    logger.info(
        f"Scoped inference: filter_sku={filter_sku}, filter_store={filter_store}, "
        f"filter_category={filter_category}"
    )

# COMMAND ----------

# DBTITLE 1,Step 2 — 140-day future spine
# Read the pre-built table (data job); fall back to on-the-fly construction if
# the table doesn't exist or has no row for this date.
_use_table = False
try:
    future_spine_df = spark.sql(f"""
        SELECT * FROM {catalog}.gold.future_spine
        WHERE market_id = '{safe_market_id}' AND snapshot_date = DATE('{as_of_str}')
    """)
    # Cheap emptiness check — take(1) short-circuits after one row. Avoid
    # .rdd.isEmpty(), which forces a full DataFrame->RDD conversion + job.
    _use_table = len(future_spine_df.take(1)) > 0
except _AnalysisException:
    _use_table = False

if not _use_table:
    logger.warning(
        "future_spine not available for snapshot_date=%s — building on the fly "
        "(table missing or data job hasn't run yet for this date)", as_of_str
    )
    future_spine_df = build_future_spine(
        spark=spark, catalog=catalog, market_id=market_id,
        snapshot_date=as_of_str, horizon_days=config.market_config.horizon_days,
    )

logger.info(
    f"Future spine ready: horizon={config.market_config.horizon_days} days "
    f"({as_of + datetime.timedelta(1)} to "
    f"{as_of + datetime.timedelta(config.market_config.horizon_days)})"
)

# COMMAND ----------

# DBTITLE 1,Step 3 — score all champion series (fit + predict + decompose)
run_id = str(uuid.uuid4())
_t0 = _time.perf_counter()
logger.info(f"SCORING start — run_id={run_id}")

output_df, decomp_df, failed_df = score(
    spark=spark,
    catalog=catalog,
    snapshot_features_df=snapshot_features_df,
    future_spine_df=future_spine_df,
    config=config,
    snapshot_date=as_of_str,
    run_id=run_id,
)

_t_score = _time.perf_counter() - _t0
n_output = output_df.count()
n_decomp = decomp_df.count()
n_failed = failed_df.count()
logger.info(
    f"SCORING done in {_t_score:.1f}s ({_t_score/60:.1f}min) — "
    f"{n_output:,} output rows, {n_decomp:,} decomp rows, {n_failed:,} series failed"
)

# COMMAND ----------

# DBTITLE 1,Step 4 — write outputs
_t1 = _time.perf_counter()
write_outputs(spark, output_df, decomp_df, catalog)
write_scoring_failures(spark, failed_df, catalog)
_t_write = _time.perf_counter() - _t1
logger.info(
    f"WRITE done in {_t_write:.1f}s — MERGE into output + decomposition"
    f"{' + scoring_failures (' + str(n_failed) + ' rows)' if n_failed else ''}"
)

# Release the persisted scored result + champion cache now that both the
# count() logs and the MERGE have consumed them.
spark.catalog.clearCache()

_total = _time.perf_counter() - _t0
logger.info(
    f"JOB 3 INFERENCE complete market_id={market_id} snapshot_date={as_of_str} "
    f"run_id={run_id} total={_total:.1f}s ({_total/60:.1f}min)"
)


# COMMAND ----------

# DBTITLE 1,Step 5 — Backcast: In-sample evaluation (2 years)
# MAGIC %md
# MAGIC ### Backcast — In-Sample Fit (2 Years)
# MAGIC Reconstruct each champion model from its stored artifact and predict on
# MAGIC the **training history** (2 years before snapshot). Compare fitted values
# MAGIC to actuals → in-sample WMAPE. This is NOT a validation/test set — it's
# MAGIC the data the model already saw. Useful for:
# MAGIC - Detecting underfitting (high in-sample error = model can't capture pattern)
# MAGIC - Verifying decomposition adds up (sum of components ≈ yhat)
# MAGIC - Baseline for out-of-sample degradation (forward WMAPE should be close to backcast)

# COMMAND ----------

# DBTITLE 1,Backcast scoring
# ═══════════════════════════════════════════════════════════════════════════════
# BACKCAST — predict on 2 years of history using the champion model artifact.
# Mirrors the forward scoring path: reconstruct model → predict → compare.
# ═══════════════════════════════════════════════════════════════════════════════
import json as _json
import numpy as _np
import pandas as _pd
from pyspark.sql import functions as _F
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, DoubleType, IntegerType
)

from training.models.registry import build as _registry_build, get_registry as _get_registry
from training.models.uplift import UpliftModel as _UpliftModel

BACKCAST_YEARS = 2
_backcast_start = (as_of - datetime.timedelta(days=365 * BACKCAST_YEARS)).isoformat()
_backcast_end = as_of_str

logger.info(
    f"BACKCAST starting: {_backcast_start} → {_backcast_end} "
    f"({BACKCAST_YEARS} years of in-sample evaluation)"
)

# ── 1. Load champion metadata (same champions as the forward scoring) ────────
_champ_w = _F.row_number().over(
    Window.partitionBy("series_key")
    .orderBy(_F.col("snapshot_date").desc(), _F.col("version").desc())
)
_champions_bc = (
    spark.table(f"{catalog}.gold.champion_index")
    .filter(
        f"market_id = '{safe_market_id}' "
        f"AND snapshot_date <= DATE('{as_of_str}')"
    )
    .withColumn("_rn", _champ_w)
    .filter("_rn = 1")
    .drop("_rn")
    .select("series_key", "store_id", "sku_id", "category",
            "model_type", "params", "version")
)

# ── 2. Load historical features for the backcast window ──────────────────────
_features_bc = (
    snapshot_features_df
    .filter(
        (_F.col("date_id") >= _backcast_start) &
        (_F.col("date_id") <= _backcast_end) &
        _F.col("qty").isNotNull() &
        (_F.col("qty") > 0)
    )
)

# ── 3. Join features with champion metadata ──────────────────────────────────
_combined_bc = (
    _features_bc
    .join(
        _champions_bc.select("store_id", "sku_id", "model_type", "params", "category"),
        on=["store_id", "sku_id"],
        how="inner",
    )
)
_n_bc_series = _combined_bc.select("store_id", "sku_id").distinct().count()
_n_bc_rows = _combined_bc.count()
logger.info(f"BACKCAST: {_n_bc_series:,} series, {_n_bc_rows:,} feature rows to score")

# ── 4. Per-series backcast UDF (predict + decompose) ────────────────────────
# Returns TWO record types in a single DataFrame:
#   record_type='output'  → one row per date (actual vs backcast)
#   record_type='decomp'  → one row per component per date (full decomposition)
_BACKCAST_SCHEMA = StructType([
    StructField("record_type", StringType()),       # 'output' or 'decomp'
    StructField("store_id", StringType()),
    StructField("sku_id", StringType()),
    StructField("category", StringType()),
    StructField("model_type", StringType()),
    StructField("date_id", DateType()),
    StructField("actual_qty", DoubleType()),        # only for 'output' rows
    StructField("backcast_qty", DoubleType()),      # only for 'output' rows
    StructField("component_type", StringType()),    # only for 'decomp' rows
    StructField("component_name", StringType()),    # only for 'decomp' rows
    StructField("contribution", DoubleType()),      # only for 'decomp' rows
])


def _backcast_one_series(pdf: "_pd.DataFrame") -> "_pd.DataFrame":
    """Reconstruct champion model, predict AND decompose on historical features."""
    _cols = [f.name for f in _BACKCAST_SCHEMA.fields]
    if pdf.empty:
        return _pd.DataFrame(columns=_cols)

    store_id = pdf["store_id"].iloc[0]
    sku_id = pdf["sku_id"].iloc[0]
    category = str(pdf["category"].iloc[0] or "")
    model_type = str(pdf["model_type"].iloc[0] or "")
    params_raw = pdf["params"].dropna().iloc[0] if pdf["params"].notna().any() else None
    params = _json.loads(params_raw) if isinstance(params_raw, str) and params_raw else {}

    try:
        # Only backcast if a serialized artifact exists (no refit — that would cheat)
        has_serialized = bool(
            params.get("model_artifact") or params.get("ets_forecast_values")
        )
        if not has_serialized:
            return _pd.DataFrame(columns=_cols)

        # Reconstruct model
        if model_type.endswith("+uplift"):
            base_type = model_type.replace("+uplift", "")
            base_cls = _get_registry()[base_type]
            model = _UpliftModel(base_cls())
        else:
            model = _registry_build(model_type)

        model = type(model).from_serialized(params)

        # Sort by date, predict, and decompose
        pdf = pdf.sort_values("date_id").reset_index(drop=True)
        yhat = model.predict(pdf)
        yhat = _np.clip(yhat, 0.0, None)

        # Apply forecast cap if stored
        cap = params.get("forecast_cap")
        if cap not in (None, 0):
            yhat = _np.clip(yhat, 0.0, float(cap))

        # ── Output rows (actual vs predicted) ────────────────────────────────
        n = len(pdf)
        output_rows = _pd.DataFrame({
            "record_type": "output",
            "store_id": store_id,
            "sku_id": sku_id,
            "category": category,
            "model_type": model_type,
            "date_id": pdf["date_id"],
            "actual_qty": pdf["qty"].astype(float),
            "backcast_qty": yhat.astype(float),
            "component_type": None,
            "component_name": None,
            "contribution": _np.nan,
        })

        # ── Decomposition rows ───────────────────────────────────────────────
        decomp_rows = []
        try:
            decomp = model.decompose(pdf)

            # Standard components (each is a pd.Series aligned to pdf index)
            _std_components = [
                ("trend", "trend", decomp.trend),
                ("seasonality_weekly", "seasonality_weekly", decomp.seasonality_weekly),
                ("seasonality_yearly", "seasonality_yearly", decomp.seasonality_yearly),
                ("temporal_features_effect", "temporal_features_effect", decomp.temporal_features_effect),
                ("model_uplift", "model_uplift", decomp.model_uplift),
            ]
            for comp_type, comp_name, series in _std_components:
                if series is not None and not series.empty:
                    for i in range(n):
                        val = float(series.iloc[i])
                        if val != 0.0:
                            decomp_rows.append({
                                "record_type": "decomp",
                                "store_id": store_id,
                                "sku_id": sku_id,
                                "category": category,
                                "model_type": model_type,
                                "date_id": pdf["date_id"].iloc[i],
                                "actual_qty": _np.nan,
                                "backcast_qty": _np.nan,
                                "component_type": comp_type,
                                "component_name": comp_name,
                                "contribution": val,
                            })

            # Dict components (holiday_lift, promo_lift, cross_sku_lift)
            _dict_components = [
                ("holiday_lift", decomp.holiday_lift),
                ("promo_lift", decomp.promo_lift),
            ]
            if hasattr(decomp, "cross_sku_lift"):
                _dict_components.append(("cross_sku_lift", decomp.cross_sku_lift))

            for comp_type, comp_dict in _dict_components:
                if comp_dict:
                    for feat_name, series in comp_dict.items():
                        if series is not None and not series.empty:
                            for i in range(n):
                                val = float(series.iloc[i])
                                if val != 0.0:
                                    decomp_rows.append({
                                        "record_type": "decomp",
                                        "store_id": store_id,
                                        "sku_id": sku_id,
                                        "category": category,
                                        "model_type": model_type,
                                        "date_id": pdf["date_id"].iloc[i],
                                        "actual_qty": _np.nan,
                                        "backcast_qty": _np.nan,
                                        "component_type": comp_type,
                                        "component_name": feat_name,
                                        "contribution": val,
                                    })
        except Exception:
            pass  # decompose failed — still return output rows

        if decomp_rows:
            all_rows = _pd.concat([output_rows, _pd.DataFrame(decomp_rows)], ignore_index=True)
        else:
            all_rows = output_rows
        return all_rows

    except Exception:
        return _pd.DataFrame(columns=_cols)


# ── 5. Run backcast ──────────────────────────────────────────────────────────
_t_bc0 = _time.perf_counter()
_backcast_raw = (
    _combined_bc
    .groupBy("store_id", "sku_id")
    .applyInPandas(_backcast_one_series, schema=_BACKCAST_SCHEMA)
)
_backcast_raw.cache()
_n_total = _backcast_raw.count()
_t_bc = _time.perf_counter() - _t_bc0

# Split into output (actual vs predicted) and decomposition
_backcast_df = _backcast_raw.filter(_F.col("record_type") == "output")
_backcast_decomp_df = _backcast_raw.filter(_F.col("record_type") == "decomp")

_n_output = _backcast_df.count()
_n_decomp = _backcast_decomp_df.count()
logger.info(
    f"BACKCAST scored {_n_output:,} output rows + {_n_decomp:,} decomp rows "
    f"in {_t_bc:.1f}s ({_t_bc/60:.1f}min)"
)

# ── 5b. Write backcast decomposition to table ────────────────────────────────
# Writes to the SAME decomposition table — backcast rows are identifiable by
# date_id < snapshot_date (forward forecast has date_id > snapshot_date).
# Uses run_id prefixed with 'backcast-' for easy filtering.
_bc_run_id = f"backcast-{run_id}"
_backcast_decomp_out = (
    _backcast_decomp_df
    .select(
        _F.lit(as_of_str).cast("date").alias("snapshot_date"),
        _F.lit(_bc_run_id).alias("run_id"),
        _F.lit(market_id).alias("market_id"),
        "store_id", "sku_id", "date_id",
        "component_type", "component_name",
        _F.col("contribution").alias("contribution_raw"),
        "contribution",
        _F.current_timestamp().alias("created_ts"),
        "model_type", "category",
        _F.lit(None).cast("int").alias("champion_version"),
        _F.current_timestamp().alias("created_date"),
        _F.lit("backcast").alias("created_by"),
        _F.current_timestamp().alias("updated_date"),
        _F.lit("backcast").alias("updated_by"),
    )
)
_backcast_decomp_out.write.format("delta").mode("append").saveAsTable(
    f"{catalog}.forecast.decomposition"
)
logger.info(f"BACKCAST decomposition written: {_n_decomp:,} rows (run_id={_bc_run_id})")

# ── Cleanup ──────────────────────────────────────────────────────────────────
_backcast_raw.unpersist()
logger.info("BACKCAST complete.")


# COMMAND ----------

# DBTITLE 1,Step 6 — DQ audit
# TODO(Step 9): inference.dq.run_dq_checks / write_dq_audit
