# Databricks notebook source
# MAGIC %md
# MAGIC ## 00_create_tables
# MAGIC Purpose: run the DDL for the feature/output tables (`data/ddl/tables.sql`)
# MAGIC and the config tables (`config/ddl/config_tables.sql`), then optionally
# MAGIC seed the demo market's defaults (`config/seed_defaults.py`). Wiring only — no logic here.
# MAGIC Inputs: none (DDL is idempotent `CREATE TABLE IF NOT EXISTS`).
# MAGIC Outputs: `{catalog}.gold.*` tables + `config_*` tables.
# MAGIC Params: env.

# COMMAND ----------

# DBTITLE 1,Create tables + seed demo-market defaults
import importlib
import os
import shutil
import sys

# ── widget (env drives catalog name — architecture lock 10) ─────────────────
dbutils.widgets.text("env", "")
env     = dbutils.widgets.get("env")
catalog = env  # catalog == env

# ── repo root on sys.path so utils / config / data are importable ────────────
# Databricks Free Edition serverless compute can stat/list Workspace Files but
# cannot open() them for content (java.io.FileNotFoundException wrapped as
# "Input/output error" from SharedUCWorkspaceLocalFileSystem — a Free Edition
# environment limitation, confirmed empirically, not specific to this repo).
# Shared modules are synced to a UC Volume (see notebooks/README or the deploy
# notes in docs/instructions.md) instead of imported straight from the
# bundle-deployed Workspace path.
_REPO_ROOT = f"/Volumes/{catalog}/default/pylib"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── imports (after sys.path is set) ─────────────────────────────────────────
import utils.io
import utils.spark
import config.seed_defaults
import data.features.event_features
import data.features.stat_features
import data.features.temporal_features

# Clear stale .pyc bytecode caches so reload reads from the .py source files
for _pkg in ["utils", "config", "data"]:
    _cache = os.path.join(_REPO_ROOT, _pkg, "__pycache__")
    if os.path.isdir(_cache):
        shutil.rmtree(_cache)
importlib.invalidate_caches()

# Reload to pick up any in-session file edits
for _mod in [
    utils.io, utils.spark,
    data.features.event_features, data.features.stat_features, data.features.temporal_features,
    config.seed_defaults,
]:
    importlib.reload(_mod)

from utils.spark import get_or_create_spark       # noqa: E402
from utils.io import run_ddl_file                 # noqa: E402
from config.seed_defaults import seed_default_market  # noqa: E402

spark = get_or_create_spark("00_create_tables")

# ── schemas (lock #5: source/gold/control/forecast) — idempotent ────────────
# `CREATE TABLE`/`saveAsTable` require the schema to already exist; the DDL
# files below only create tables, so the four project schemas are created here.
for _schema in ["source", "gold", "control", "forecast"]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{_schema}")
print(f"Schemas ready: {catalog}.{{source,gold,control,forecast}}")

# ── DDL — idempotent CREATE TABLE IF NOT EXISTS ──────────────────────────────
run_ddl_file(spark, f"{_REPO_ROOT}/config/ddl/config_tables.sql", catalog)
run_ddl_file(spark, f"{_REPO_ROOT}/data/ddl/tables.sql", catalog)
run_ddl_file(spark, f"{_REPO_ROOT}/data/ddl/scoring_failures.sql", catalog)
print(f"DDL complete — {catalog}.gold.* tables ready")

# ── seed demo-market bootstrap rows (idempotent merge_upsert) ───────────────
seed_default_market(spark, catalog)
print(f"Seeded demo-market defaults into {catalog}.control.forecast_config")