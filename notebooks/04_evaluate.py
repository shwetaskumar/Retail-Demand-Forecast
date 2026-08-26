# Databricks notebook source
# MAGIC %md
# MAGIC ## 04_evaluate — inference job EVALUATE task
# MAGIC Runs daily after `03_score`. Compares the lag-7 forecast against actual
# MAGIC POS sales at market x category x store x sku granularity, writes
# MAGIC `eval_daily`, and flags `breach_flag` when a series has been
# MAGIC above its category's WMAPE threshold using a two-tier breach gate:
# MAGIC - **Tier 1 (immediate):** WMAPE > threshold × 1.5 → clearly broken model, HPT same day.
# MAGIC - **Tier 2 (persistent):** above threshold+tolerance for 7 consecutive days → sustained drift.
# MAGIC Additional guardrails prevent wasteful repeat retuning:
# MAGIC - **Cooldown:** after HPT runs, suppress re-triggering for 7 days.
# MAGIC - **Once per window:** if HPT already ran since the current lag-date forecast snapshot went live, do not trigger again in that same window.
# MAGIC This task always runs; `05_hpt_retune` (next in the chain) self-gates on
# MAGIC whatever this one finds.
# MAGIC
# MAGIC WMAPE thresholds are read from `config` via `load_config()` (CategoryConfig.
# MAGIC accuracy_threshold). The `wmape_threshold` widget is kept as a fallback for
# MAGIC categories not returned by load_config (e.g. new markets with incomplete config).
# MAGIC
# MAGIC Inputs: `output` (lag-7 snapshot) + `source.sales` (actuals).
# MAGIC Outputs: `eval_daily`, `breach_flag`.
# MAGIC Params: market_id, env, snapshot_date, wmape_threshold.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("market_id", "US")
dbutils.widgets.text("env", "")
dbutils.widgets.text("snapshot_date", "")
dbutils.widgets.text("wmape_threshold", "0.5")

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
market_id   = dbutils.widgets.get("market_id")
env             = dbutils.widgets.get("env")
snapshot_date   = dbutils.widgets.get("snapshot_date") or None
wmape_threshold = float(dbutils.widgets.get("wmape_threshold"))


def _parse_filter(raw: str) -> list[str]:
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

from pyspark.sql import functions as F

from utils.dates import resolve_snapshot_date
from utils.io import merge_upsert
from utils.logging import get_logger
from utils.spark import get_or_create_spark

# _EVAL_LAG_DAYS: forecasts are compared against actuals 7 days later (a
# "lag-7 forecast", per this notebook's docstring above).
# _BREACH_WINDOW_DAYS: a series must be above threshold on ALL of the last 14
# (daily) eval dates to be flagged — these are two independent concepts and
# must NOT share one constant (a prior version of this logic conflated them,
# which silently turned the lag-7 comparison into a lag-14 one).
_EVAL_LAG_DAYS      = 7    # forecast horizon being evaluated
_BREACH_WINDOW_DAYS = 7    # consecutive days above threshold -> trigger HPT (1 week)
_SEVERITY_MULT      = 1.5  # immediate HPT if wmape > threshold × this (clearly broken model)
_HPT_COOLDOWN_DAYS  = 7    # after HPT, give the new champion time to prove itself before re-triggering


def _set_task_value(spark, key: str, value: int) -> None:
    """Set a Databricks Jobs task output value (safe no-op outside a job context)."""
    try:
        from pyspark.dbutils import DBUtils
        dbutils_ = DBUtils(spark)
        dbutils_.jobs.taskValues.set(key=key, value=value)
    except Exception:
        pass

# COMMAND ----------

# DBTITLE 1,Spark session
catalog = env

spark   = get_or_create_spark(f"job3-evaluate-{market_id}")
logger  = get_logger(__name__, market_id=market_id, env=env)

eval_date     = resolve_snapshot_date(snapshot_date)
eval_date_str = eval_date.isoformat()
lag_date      = eval_date - datetime.timedelta(days=_EVAL_LAG_DAYS)
lag_date_str  = lag_date.isoformat()
safe_market_id = market_id.replace("'", "''")

logger.info(
    f"EVALUATE market_id={market_id} eval_date={eval_date_str} "
    f"comparing forecasts from {lag_date_str} against actuals"
)

# COMMAND ----------

# DBTITLE 1,Step 1 — forecasts from lag_date snapshot
# Evaluate the full 7-day horizon of the lag_date snapshot
fcst_df = spark.sql(f"""
    SELECT
        DATE('{eval_date_str}') AS eval_date,
        snapshot_date,
        market_id, category, store_id, sku_id,
        CONCAT_WS('|', market_id, category, store_id, sku_id) AS series_key,
        date_id,
        DATEDIFF(date_id, snapshot_date) AS horizon_day,
        yhat AS forecast_qty
    FROM {catalog}.forecast.output
    WHERE market_id = '{safe_market_id}'
      AND snapshot_date  = DATE('{lag_date_str}')
      AND date_id BETWEEN DATE('{lag_date_str}') AND DATE('{eval_date_str}')
""")

_no_forecasts = fcst_df.rdd.isEmpty()
if _no_forecasts:
    logger.warning(
        f"EVALUATE no forecasts found for snapshot_date={lag_date_str}; "
        "skipping evaluation — breach_count=0"
    )
    _set_task_value(spark, "breach_count", 0)

# COMMAND ----------

# DBTITLE 1,Step 2 — actual POS for the 7-day horizon
if not _no_forecasts:
    actuals_df = spark.sql(f"""
        SELECT
            date_id,
            store_id,
            sku_id,
            SUM(units_sold + units_promotional + units_shrinkage) AS actual_qty
        FROM {catalog}.source.sales
        WHERE market_id = '{safe_market_id}'
          AND date_id BETWEEN DATE('{lag_date_str}') AND DATE('{eval_date_str}')
        GROUP BY date_id, store_id, sku_id
    """)

# COMMAND ----------

# DBTITLE 1,Step 3 — join + aggregate WMAPE
if not _no_forecasts:
    # Per-category WMAPE thresholds + drift tolerance from config; widget value is fallback
    from config.loader import load_config as _load_config
    _cats = _load_config(spark, catalog, market_id).categories
    config_df = spark.createDataFrame([{
        "market_id":          c.market_id,
        "category":           c.category,
        "effective_threshold": round(1.0 - c.accuracy_threshold, 6),
        "drift_tolerance":    c.drift_tolerance,
    } for c in _cats])
    # Join forecasts with actuals per day across the 7-day horizon
    daily_df = (
        fcst_df.join(actuals_df, on=["store_id", "sku_id", "date_id"], how="left")
        .withColumn("actual_qty", F.coalesce(F.col("actual_qty"), F.lit(0.0)))
        .withColumn("abs_error", F.abs(F.col("forecast_qty") - F.col("actual_qty")))
    )
    # WMAPE = sum(|error|) / sum(|actual|) aggregated over the full horizon per series
    joined_df = (
        daily_df
        .groupBy("eval_date", "snapshot_date", "market_id", "category", "series_key",
                 "store_id", "sku_id")
        .agg(
            (F.sum("abs_error") / F.greatest(F.sum("actual_qty"), F.lit(1.0))).alias("wmape"),
        )
        .join(config_df, on=["market_id", "category"], how="left")
        .withColumn(
            "effective_threshold",
            F.coalesce(F.col("effective_threshold"), F.lit(wmape_threshold))
        )
        .withColumn(
            "above_threshold",
            F.col("wmape") > F.col("effective_threshold") * (F.lit(1.0) + F.coalesce(F.col("drift_tolerance"), F.lit(0.0)))
        )
        .drop("drift_tolerance")
    )

# COMMAND ----------

# DBTITLE 1,Step 4 — write eval_daily
if not _no_forecasts:
    merge_upsert(
        spark, joined_df, catalog, "gold", "eval_daily",
        merge_keys=["eval_date", "series_key"],
    )
    n_evals = joined_df.count()
    n_above = joined_df.filter(F.col("above_threshold")).count()
    logger.info(
        f"EVALUATE written {n_evals:,} rows to eval_daily "
        f"({n_above:,} above per-category threshold)"
    )

# COMMAND ----------

# DBTITLE 1,Step 5 — breach check
# Two-tier breach logic with re-tune suppression:
# Tier 1 (IMMEDIATE): today's WMAPE > threshold × _SEVERITY_MULT → clearly broken, HPT now
# Tier 2 (PERSISTENT): above_threshold on ALL of last _BREACH_WINDOW_DAYS → sustained drift
# Suppression 1 (COOLDOWN): if HPT already ran in the last _HPT_COOLDOWN_DAYS, skip re-triggering
# Suppression 2 (ONCE PER WINDOW): if HPT already ran since this forecast snapshot went live, skip it
if not _no_forecasts:
    # --- Tier 1: severe breaches (today only) ---
    severe_df = (
        joined_df
        .filter(
            F.col("wmape") > F.col("effective_threshold") * F.lit(_SEVERITY_MULT)
        )
        .select("series_key")
    )
    n_severe = severe_df.count()

    # --- Tier 2: persistent breaches (rolling window) ---
    breach_window_start = (eval_date - datetime.timedelta(days=_BREACH_WINDOW_DAYS - 1)).isoformat()
    persistent_df = spark.sql(f"""
        WITH recent AS (
            SELECT series_key,
                   COUNT(*)                     AS eval_days,
                   SUM(CASE WHEN above_threshold THEN 1 ELSE 0 END) AS days_above
            FROM {catalog}.gold.eval_daily
            WHERE market_id = '{safe_market_id}'
              AND eval_date BETWEEN DATE('{breach_window_start}') AND DATE('{eval_date_str}')
            GROUP BY series_key
        )
        SELECT series_key
        FROM recent
        WHERE eval_days >= {_BREACH_WINDOW_DAYS}
          AND days_above >= {_BREACH_WINDOW_DAYS}
    """)
    n_persistent = persistent_df.count()

    # Candidate breaches before suppression
    breach_candidates_df = severe_df.union(persistent_df).distinct()
    n_candidates = breach_candidates_df.count()

    # --- Suppression 1: cooldown after HPT ---
    cooldown_start = (eval_date - datetime.timedelta(days=_HPT_COOLDOWN_DAYS - 1)).isoformat()
    cooldown_df = spark.sql(f"""
        SELECT DISTINCT series_key
        FROM {catalog}.gold.champion_index
        WHERE market_id = '{safe_market_id}'
          AND version >= 2
          AND snapshot_date BETWEEN DATE('{cooldown_start}') AND DATE('{eval_date_str}')
    """)

    # --- Suppression 2: only one HPT per forecast window ---
    # If this series has already been HPT-updated since the lag_date forecast snapshot
    # went live, do not trigger again during the same inference/eval window.
    window_df = spark.sql(f"""
        SELECT DISTINCT series_key
        FROM {catalog}.gold.champion_index
        WHERE market_id = '{safe_market_id}'
          AND version >= 2
          AND snapshot_date BETWEEN DATE('{lag_date_str}') AND DATE('{eval_date_str}')
    """)

    suppressed_df = cooldown_df.union(window_df).distinct()
    n_suppressed = suppressed_df.count()

    breach_df = breach_candidates_df.join(suppressed_df, on=["series_key"], how="left_anti")
    n_breached = breach_df.count()

    if n_breached > 0:
        # Persist breach list for hpt_retune to consume
        (
            breach_df
            .withColumn("eval_date", F.lit(eval_date_str).cast("date"))
            .withColumn("market_id", F.lit(market_id))
            .write.format("delta").mode("overwrite")
            .option("replaceWhere", f"market_id = '{safe_market_id}' AND eval_date = DATE('{eval_date_str}')")
            .option("mergeSchema", "true")
            .saveAsTable(f"{catalog}.gold.breach_flag")
        )
        logger.warning(
            f"EVALUATE breach detected: {n_breached:,} series "
            f"({n_severe:,} severe >{_SEVERITY_MULT:.1f}× threshold, "
            f"{n_persistent:,} persistent >{_BREACH_WINDOW_DAYS}d, "
            f"{n_candidates - n_breached:,} suppressed by cooldown/window guardrails) — HPT will be triggered"
        )
    else:
        logger.info(
            f"EVALUATE no breach detected across {n_evals:,} series "
            f"({n_candidates:,} candidates, {n_candidates:,} suppressed by cooldown/window guardrails)"
            if n_candidates > 0 and n_suppressed > 0 else
            f"EVALUATE no breach detected across {n_evals:,} series"
        )

    _set_task_value(spark, "breach_count", n_breached)
    logger.info(f"EVALUATE complete eval_date={eval_date_str} breach_count={n_breached}")
