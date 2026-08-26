# Databricks notebook source
# MAGIC %md
# MAGIC ## 00b_generate_synthetic_source
# MAGIC On-demand: generates a synthetic multi-store x multi-sku demand dataset
# MAGIC (`data/synthetic/generator.py`) and writes it as the `source.*` / `gold.events`
# MAGIC / `gold.calendar_date` tables JOB 1 (`01_build_features.py`) reads read-only.
# MAGIC
# MAGIC Stands in for a real POS/ERP source feed — run this once before the first
# MAGIC `01_build_features.py` run on a fresh Free Edition workspace, or any time you
# MAGIC want to regenerate the demo data with different scale/seed. Not part of the
# MAGIC daily/weekly job schedule (`databricks.yml`) — on-demand only, same family as
# MAGIC `01b_backfill_features.py`.
# MAGIC
# MAGIC Inputs: none (self-contained synthetic generator).
# MAGIC Outputs: `{catalog}.source.sales`, `.sku`, `.store`, `.sku_store`,
# MAGIC `{catalog}.gold.events`, `{catalog}.gold.calendar_date` (full overwrite each run).
# MAGIC Params: env, market_id, n_stores, n_skus, n_days, seed.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("env", "")
dbutils.widgets.text("market_id", "US")
dbutils.widgets.text("n_stores", "30")
dbutils.widgets.text("n_skus", "300")
dbutils.widgets.text("n_days", "900")
dbutils.widgets.text("seed", "42")

# COMMAND ----------

# DBTITLE 1,Install pinned deps (interactive runs only)
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,sys.path — repo root importable
import os
import sys

_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_REPO_ROOT = "/Workspace" + os.path.dirname(os.path.dirname(_notebook_path))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# COMMAND ----------

# DBTITLE 1,Read widgets
import datetime as _dt

catalog   = dbutils.widgets.get("env")
market_id = dbutils.widgets.get("market_id")
n_stores  = int(dbutils.widgets.get("n_stores"))
n_skus    = int(dbutils.widgets.get("n_skus"))
n_days    = int(dbutils.widgets.get("n_days"))
seed      = int(dbutils.widgets.get("seed"))

# COMMAND ----------

# DBTITLE 1,Imports
from data.synthetic.generator import generate_synthetic_source
from utils.logging import get_logger
from utils.spark import get_or_create_spark

spark  = get_or_create_spark(f"job1-generate-synthetic-source-{market_id}")
logger = get_logger(__name__, market_id=market_id, env=catalog)

# COMMAND ----------

# DBTITLE 1,Step 1 — generate synthetic frames (pure pandas, no Spark)
frames = generate_synthetic_source(
    seed=seed, n_stores=n_stores, n_skus=n_skus, n_days=n_days,
    markets=[market_id], end_date=_dt.date.today() - _dt.timedelta(days=1),
)
for name, pdf in frames.items():
    logger.info(f"generated {name}: {len(pdf):,} rows")

# COMMAND ----------

# DBTITLE 1,Step 2 — write as source/gold Delta tables (full overwrite)
_TARGETS = {
    "sales":         f"{catalog}.source.sales",
    "sku":           f"{catalog}.source.sku",
    "store":         f"{catalog}.source.store",
    "sku_store":     f"{catalog}.source.sku_store",
    "events":        f"{catalog}.gold.events",
    "calendar_date": f"{catalog}.gold.calendar_date",
}

for frame_name, full_table in _TARGETS.items():
    spark.createDataFrame(frames[frame_name]).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(full_table)
    logger.info(f"wrote {full_table}")

logger.info(f"SYNTHETIC SOURCE generation complete for market_id={market_id}")
