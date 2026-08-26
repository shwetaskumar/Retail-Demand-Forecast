# Databricks notebook source
# MAGIC %md
# MAGIC ## 01b_backfill_features — on-demand historical feature backfill
# MAGIC Steps `01_build_features` across historical snapshot_dates so
# MAGIC `sku_store_features` has continuous coverage from `train_start_date`
# MAGIC onward, without reprocessing the entire history on every daily run. Each
# MAGIC snapshot_date computes features for `[snapshot_date - read_days,
# MAGIC snapshot_date]` and persists only `[snapshot_date - write_days,
# MAGIC snapshot_date]` — so to cover a long range this notebook steps through
# MAGIC multiple snapshot_dates spaced `step_days` apart.
# MAGIC
# MAGIC On-demand only — not on a schedule (see `databricks.yml`'s
# MAGIC `fcst_backfill_job`, which has no `schedule:` block).
# MAGIC
# MAGIC This notebook invokes `01_build_features` via `dbutils.notebook.run` once
# MAGIC per snapshot_date rather than importing a shared loop function — each
# MAGIC child run gets its own job-run page in the Databricks UI, so a failure on
# MAGIC snapshot N is immediately visible as that specific run, without needing to
# MAGIC scroll through one giant driver log.
# MAGIC
# MAGIC Inputs: source tables (read-only). Outputs: `{catalog}.gold.sku_store_features`.
# MAGIC Params: market_id, env, train_start_date, end_date, step_days.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("market_id", "US")
dbutils.widgets.text("env", "")
dbutils.widgets.text("train_start_date", "2023-01-01")
dbutils.widgets.text("end_date", "")
dbutils.widgets.text("step_days", "")

# COMMAND ----------

display(dbutils.widgets.getAll())

# COMMAND ----------

# DBTITLE 1,Install pinned deps (interactive runs only — job clusters already have these)
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Read widgets
market_id        = dbutils.widgets.get("market_id")
env              = dbutils.widgets.get("env")
train_start_date = dbutils.widgets.get("train_start_date")
end_date         = dbutils.widgets.get("end_date") or None
_step_days_raw   = dbutils.widgets.get("step_days")
step_days        = int(_step_days_raw) if _step_days_raw else None

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
import datetime as dt

from utils.dates import resolve_snapshot_date
from utils.logging import get_logger
from utils.spark import get_or_create_spark
from config.loader import load_config

# COMMAND ----------

# DBTITLE 1,Spark session, config
spark   = get_or_create_spark(f"job1-backfill-{market_id}")
logger  = get_logger(__name__, market_id=market_id, env=env)
catalog = env
config  = load_config(spark, catalog, market_id)

# COMMAND ----------

# DBTITLE 1,Step 1 — build the list of snapshot_dates to step through
write_days  = step_days or config.market_config.incremental_write_days
train_start = resolve_snapshot_date(train_start_date)
as_of       = resolve_snapshot_date(end_date)

# Build list of snapshot_dates stepping forward so that each write window
# [snap - write_days, snap] tiles the range [train_start, as_of] without gaps.
# First snap = train_start + write_days so its write window starts exactly at train_start.
snapshot_dates: list[dt.date] = []
snap = train_start + dt.timedelta(days=write_days)
while snap < as_of:
    snapshot_dates.append(snap)
    snap += dt.timedelta(days=write_days)
# Always include as_of as the final snapshot so today's data is covered.
if not snapshot_dates or snapshot_dates[-1] != as_of:
    snapshot_dates.append(as_of)

total = len(snapshot_dates)
logger.info(
    f"BACKFILL starting: market_id={market_id} train_start={train_start} "
    f"end={as_of} step_days={write_days} total_snapshots={total}"
)

# COMMAND ----------

# DBTITLE 1,Step 2 — run 01_build_features once per snapshot_date
_NOTEBOOK_TIMEOUT_SECONDS = 7200  # matches fcst_data_job/build_features's own timeout in databricks.yml

_failed: list[str] = []
for i, snap_date in enumerate(snapshot_dates, 1):
    snap_str = snap_date.isoformat()
    logger.info(f"Backfill snapshot {i}/{total}: snapshot_date={snap_str}")
    try:
        dbutils.notebook.run(
            "01_build_features",
            _NOTEBOOK_TIMEOUT_SECONDS,
            {"market_id": market_id, "env": env, "snapshot_date": snap_str},
        )
        logger.info(f"  [{i}/{total}] {snap_str} OK")
    except Exception as exc:
        logger.error(f"  [{i}/{total}] {snap_str} FAILED: {exc} -- continuing with remaining snapshots")
        _failed.append(snap_str)

if _failed:
    raise RuntimeError(
        f"Backfill completed with {len(_failed)}/{total} snapshot failures: {_failed}\n"
        "Fix the error above, then re-run with train_start_date set to the first failed date."
    )
logger.info(f"BACKFILL complete: {total} snapshots processed for market_id={market_id}")
