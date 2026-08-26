# Databricks notebook source
# MAGIC %md
# MAGIC ## 07_full_backfill — full historical simulation of the production pipeline
# MAGIC Manual validation harness only — NOT wired into `databricks.yml` (no job,
# MAGIC no schedule). Processes a series of monthly training snapshots, running
# MAGIC the complete daily inference + evaluation + HPT loop for each window:
# MAGIC
# MAGIC   For each training date (e.g. Jan 1, Feb 1, ... Jul 1):
# MAGIC     1. `02_train` (use_batch_engine=True) -> train ETS+Ridge cold-start
# MAGIC        champions, gate, register MLflow pyfuncs
# MAGIC     2. For each day in `[training_date, next_training_date)`:
# MAGIC        a. `03_score`       -> 140-day forecast snapshot
# MAGIC        b. `04_evaluate`    -> lag-7 WMAPE vs POS actuals -> `eval_daily`
# MAGIC                              (only once >= 7 days of eval history exist).
# MAGIC                              Uses per-category WMAPE thresholds from
# MAGIC                              `config_category` (A=15%, SEASONAL=20%,
# MAGIC                              B=25%, C=35%, D=40%) — NOT a global threshold.
# MAGIC        c. check breach     -> if 14 consecutive days (2 weeks) above category
# MAGIC           threshold (`04_evaluate`'s `_BREACH_WINDOW_DAYS` — the real,
# MAGIC           authoritative window):
# MAGIC              - `05_hpt_retune`  -> retune breached series
# MAGIC              - `06_infer_final` -> re-score improved series for this day
# MAGIC
# MAGIC Every step below is a separate Databricks notebook invoked via
# MAGIC `dbutils.notebook.run(...)`, not an imported Python function — each call
# MAGIC gets its own job-run page in the Databricks UI, so a failure on e.g. day 42
# MAGIC of window 3 is immediately visible as that specific run (with its own logs
# MAGIC and stack trace), rather than being buried in one monolithic driver log.
# MAGIC
# MAGIC Total expected runtime: ~8-10 hrs on `fcst_data_cluster` (8-16 D8ds_v5 workers).
# MAGIC
# MAGIC Params: market_id, env, training_dates (comma-separated ISO dates),
# MAGIC start_date (resume support), train_start_date, wmape_threshold (fallback
# MAGIC only), inference_only (skip training; training already done),
# MAGIC max_infer_date (cap the daily loop, useful for testing one window).

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("market_id", "US")
dbutils.widgets.text("env", "")
dbutils.widgets.text(
    "training_dates",
    "2026-01-01,2026-02-01,2026-03-01,2026-04-01,2026-05-01,2026-06-01,2026-07-01",
)
dbutils.widgets.text("start_date", "")
dbutils.widgets.text("train_start_date", "2023-01-01")
dbutils.widgets.text("wmape_threshold", "0.5")
dbutils.widgets.dropdown("inference_only", "False", ["False", "True"])
dbutils.widgets.text("max_infer_date", "")
dbutils.widgets.text("infer_start_date", "")
dbutils.widgets.text("train_from_date", "")
dbutils.widgets.text("infer_step_days", "1")
dbutils.widgets.text("filter_category", "")  # optional: A/B/C/D/SEASONAL — train+infer only this category

# COMMAND ----------

display(dbutils.widgets.getAll())

# COMMAND ----------

# DBTITLE 1,Install pinned deps (interactive runs only — job clusters already have these)
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Read widgets
market_id         = dbutils.widgets.get("market_id")
env               = dbutils.widgets.get("env")
training_dates    = [d.strip() for d in dbutils.widgets.get("training_dates").split(",")]
start_date        = dbutils.widgets.get("start_date") or None
train_start_date  = dbutils.widgets.get("train_start_date")
wmape_threshold   = float(dbutils.widgets.get("wmape_threshold"))
inference_only    = dbutils.widgets.get("inference_only") == "True"
max_infer_date    = dbutils.widgets.get("max_infer_date") or None
infer_start_date  = dbutils.widgets.get("infer_start_date") or None
train_from_date   = dbutils.widgets.get("train_from_date") or None
infer_step        = int(dbutils.widgets.get("infer_step_days") or "1")
filter_category   = dbutils.widgets.get("filter_category") or None

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

# DBTITLE 1,Imports + helpers
import datetime as dt

from utils.logging import get_logger
from utils.spark import get_or_create_spark

_EVAL_LAG_DAYS = 7    # matches 04_evaluate's _EVAL_LAG_DAYS: eval needs >= 7 days since training
_NOTEBOOK_TIMEOUT_SECONDS = 21600  # 6h ceiling per child run — matches fcst_training_job's own timeout


def _date_range(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)

# COMMAND ----------

# DBTITLE 1,Spark session
catalog = env
spark   = get_or_create_spark(f"job-full-backfill-{market_id}")
logger  = get_logger(__name__, market_id=market_id, env=env)

train_dates = list(training_dates)
today       = dt.date.today()

# Optional resume: skip training windows before start_date
if start_date:
    skip_before = dt.date.fromisoformat(start_date)
    train_dates = [d for d in train_dates if dt.date.fromisoformat(d) >= skip_before]

logger.info(
    f"FULL_BACKFILL start market_id={market_id} env={env} "
    f"training_dates={train_dates} "
    f"wmape_threshold={wmape_threshold} (fallback; per-category thresholds from config) "
    f"inference_only={inference_only}"
)

# COMMAND ----------

# DBTITLE 1,Main loop — per training window: train, then daily infer/eval/HPT
for idx, train_str in enumerate(train_dates):
    train_date = dt.date.fromisoformat(train_str)

    # Inference window: [train_date, next_train_date - 1] or [train_date, today]
    if idx + 1 < len(train_dates):
        window_end = dt.date.fromisoformat(train_dates[idx + 1]) - dt.timedelta(days=1)
    else:
        window_end = today
    window_end = min(window_end, today)
    if max_infer_date:
        window_end = min(window_end, dt.date.fromisoformat(max_infer_date))

    logger.info(
        f"== WINDOW {idx+1}/{len(train_dates)}: "
        f"train={train_str}  infer=[{train_str} -> {window_end}] =="
    )

    # -- Step 1: monthly training (skip when inference_only=True or before train_from_date)
    _skip_training = inference_only or (
        train_from_date and train_date < dt.date.fromisoformat(train_from_date)
    )
    if _skip_training:
        logger.info(
            f"  [TRAIN] SKIPPED (inference_only={inference_only}, train_from_date={train_from_date}) -- "
            f"assuming champions already exist for {train_str}"
        )
    else:
        logger.info(f"  [TRAIN] 02_train snapshot_date={train_str}")
        try:
            dbutils.notebook.run(
                "02_train",
                _NOTEBOOK_TIMEOUT_SECONDS,
                {
                    "market_id": market_id, "env": env,
                    "snapshot_date": train_str,
                    "train_start_date": train_start_date,
                    "run_hpt": "False",           # this notebook's own loop handles HPT, per-day
                    "use_batch_engine": "False",  # cold-start ETS+Ridge, matches monthly cadence
                    "filter_category": filter_category or "",  # scoped training
                },
            )
        except Exception as exc:
            logger.error(f"  [TRAIN] FAILED: {exc} -- skipping window {train_str}")
            continue

    # -- Step 2: daily inference + evaluation loop ------------------------------
    hpt_done_on: set = set()   # days where HPT already ran

    infer_days = list(_date_range(train_date, window_end))
    if infer_start_date:
        _skip_before = dt.date.fromisoformat(infer_start_date)
        infer_days = [d for d in infer_days if d >= _skip_before]
    infer_days = infer_days[::infer_step]  # weekly when infer_step=7
    n_infer_ok = 0
    for day_idx, infer_date in enumerate(infer_days):
        infer_str = infer_date.isoformat()
        label = f"  [{day_idx+1}/{len(infer_days)}]"

        # -- 2a. Inference ------------------------------------------------------
        logger.info(f"{label} [INFER] snapshot_date={infer_str}")
        try:
            dbutils.notebook.run(
                "03_score", _NOTEBOOK_TIMEOUT_SECONDS,
                {"market_id": market_id, "env": env, "snapshot_date": infer_str},
            )
            n_infer_ok += 1
        except Exception as exc:
            logger.warning(f"{label} [INFER] FAILED: {exc} -- skipping day")
            continue

        # -- 2b. Evaluate (once enough lag history exists) ----------------------
        days_since_train = (infer_date - train_date).days
        if days_since_train < _EVAL_LAG_DAYS:
            logger.info(
                f"{label} [EVAL] skipped -- only {days_since_train}d since training "
                f"(need {_EVAL_LAG_DAYS}d)"
            )
            continue

        logger.info(f"{label} [EVAL] snapshot_date={infer_str}")
        breach_count = 0
        try:
            dbutils.notebook.run(
                "04_evaluate", _NOTEBOOK_TIMEOUT_SECONDS,
                {
                    "market_id": market_id, "env": env,
                    "snapshot_date": infer_str,
                    "wmape_threshold": str(wmape_threshold),
                },
            )
            # Read breach_count from breach_flag (written by 04_evaluate)
            safe_market_id = market_id.replace("'", "''")
            if spark.catalog.tableExists(f"{catalog}.gold.breach_flag"):
                bf = spark.sql(f"""
                    SELECT COUNT(*) AS n FROM {catalog}.gold.breach_flag
                    WHERE market_id = '{safe_market_id}' AND eval_date = DATE('{infer_str}')
                """).collect()[0]["n"]
                breach_count = int(bf)
        except Exception as exc:
            logger.warning(f"{label} [EVAL] FAILED: {exc} -- skipping eval")

        if breach_count == 0 or infer_date in hpt_done_on:
            continue

        # -- 2c. HPT for breached series ----------------------------------------
        logger.info(
            f"{label} [HPT] breach_count={breach_count} -> "
            f"retuning breached series for snapshot_date={infer_str}"
        )
        try:
            dbutils.notebook.run(
                "05_hpt_retune", _NOTEBOOK_TIMEOUT_SECONDS,
                {
                    "market_id": market_id, "env": env,
                    "snapshot_date": infer_str,
                    "train_start_date": train_start_date,
                },
            )
            hpt_done_on.add(infer_date)
        except Exception as exc:
            logger.warning(f"{label} [HPT] FAILED: {exc}")
            continue

        # -- 2d. Re-inference for HPT-improved series ---------------------------
        logger.info(f"{label} [INFER_FINAL] re-scoring HPT-updated series")
        try:
            dbutils.notebook.run(
                "06_infer_final", _NOTEBOOK_TIMEOUT_SECONDS,
                {"market_id": market_id, "env": env, "snapshot_date": infer_str},
            )
        except Exception as exc:
            logger.warning(f"{label} [INFER_FINAL] FAILED: {exc}")

    logger.info(
        f"== WINDOW {idx+1} COMPLETE: "
        f"train={train_str} total={len(infer_days)} days  ok={n_infer_ok} =="
    )

logger.info(f"FULL_BACKFILL COMPLETE market_id={market_id} -- all {len(train_dates)} windows processed")
