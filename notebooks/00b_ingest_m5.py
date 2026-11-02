# Databricks notebook source
# MAGIC %md
# MAGIC ## 00b_ingest_m5
# MAGIC On-demand: loads the Kaggle M5 Forecasting Accuracy dataset (Walmart daily
# MAGIC unit sales) from a Unity Catalog Volume and writes it as the `source.*` /
# MAGIC `gold.events` / `gold.calendar_date` tables JOB 1 (`01_build_features.py`)
# MAGIC reads read-only. Transformation logic lives in `data/kaggle_m5/loader.py`
# MAGIC (pure pandas — reused as-is here, not reimplemented).
# MAGIC
# MAGIC **Getting the raw files onto the Volume (one-time, done from your own machine
# MAGIC — M5 is a Kaggle *competition*, so this needs your own Kaggle account and API
# MAGIC credentials; nothing here can download it for you):**
# MAGIC ```
# MAGIC pip install kaggle
# MAGIC # accept the competition rules at kaggle.com/competitions/m5-forecasting-accuracy first
# MAGIC # place your API token at ~/.kaggle/kaggle.json (from kaggle.com/settings)
# MAGIC kaggle competitions download -c m5-forecasting-accuracy -p m5_raw
# MAGIC unzip m5_raw/m5-forecasting-accuracy.zip -d m5_raw
# MAGIC databricks fs cp -r m5_raw dbfs:/Volumes/{catalog}/source/m5_raw
# MAGIC ```
# MAGIC (Create the `{catalog}.source` Volume first if it doesn't exist — Catalog
# MAGIC Explorer → your catalog → source schema → Create → Volume.)
# MAGIC
# MAGIC Not part of the daily/weekly cadence — run once before the first JOB 1 run,
# MAGIC or again with different `states`/`max_skus`/`trailing_days` widgets. Same
# MAGIC job family as `01b_backfill_features.py`.
# MAGIC
# MAGIC Inputs: `calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv` on
# MAGIC the Volume path given by the `volume_path` widget.
# MAGIC Outputs: `{catalog}.source.sales`, `.sku`, `.store`, `.sku_store`,
# MAGIC `{catalog}.gold.events`, `{catalog}.gold.calendar_date` (full overwrite each run).
# MAGIC Params: env, volume_path, states, max_skus, trailing_days.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("env", "")
dbutils.widgets.text("volume_path", "")
dbutils.widgets.text("states", "CA")            # comma-separated subset of CA,TX,WI
dbutils.widgets.text("max_skus", "300")          # empty = load all ~3,049 items
dbutils.widgets.text("trailing_days", "730")     # empty = all 1,941 days

# COMMAND ----------

# DBTITLE 1,sys.path — repo root importable
import os
import sys

# Databricks Free Edition serverless compute can stat/list Workspace Files but
# cannot open() them for content (java.io.FileNotFoundException wrapped as
# "Input/output error" from SharedUCWorkspaceLocalFileSystem — a Free Edition
# environment limitation, confirmed empirically, not specific to this repo).
# Shared modules are synced to a UC Volume instead of imported straight from
# the bundle-deployed Workspace path.
_REPO_ROOT = f"/Volumes/{dbutils.widgets.get('env')}/default/pylib"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# COMMAND ----------

# DBTITLE 1,Read widgets
catalog = dbutils.widgets.get("env")
volume_path = dbutils.widgets.get("volume_path")

_states_raw = dbutils.widgets.get("states").strip()
states = [s.strip() for s in _states_raw.split(",") if s.strip()] or None

_max_skus_raw = dbutils.widgets.get("max_skus").strip()
max_skus = int(_max_skus_raw) if _max_skus_raw else None

_trailing_days_raw = dbutils.widgets.get("trailing_days").strip()
trailing_days = int(_trailing_days_raw) if _trailing_days_raw else None

if not volume_path:
    raise ValueError(
        "volume_path widget is required — point it at the Volume directory holding "
        "calendar.csv / sales_train_evaluation.csv / sell_prices.csv (see this "
        "notebook's header cell for how to get them there)."
    )

# COMMAND ----------

# DBTITLE 1,Imports
from data.kaggle_m5.loader import load_m5_source
from utils.logging import get_logger
from utils.spark import get_or_create_spark

spark  = get_or_create_spark(f"job1-ingest-m5-{'-'.join(states or ['all'])}")
logger = get_logger(__name__, market_id=",".join(states or ["all"]), env=catalog)

# COMMAND ----------

# DBTITLE 1,Step 1 — load + transform M5 (pure pandas, on the driver)
# The raw files are small enough (sales_train_evaluation.csv is ~120MB) to load and
# melt on the driver with pandas — the same code path demo/run_local_demo.py uses
# locally, so there's exactly one implementation of the M5 transform, not two.
logger.info(f"loading M5 from {volume_path} (states={states}, max_skus={max_skus}, trailing_days={trailing_days})")
frames = load_m5_source(volume_path, states=states, max_skus=max_skus, trailing_days=trailing_days)
for name, pdf in frames.items():
    logger.info(f"loaded {name}: {len(pdf):,} rows")

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

logger.info(f"M5 INGESTION complete for states={states}")
