"""
utils/spark.py — Spark session helpers (docs/instructions.md §18).

A SparkSession is never created at import time. Every job entrypoint obtains
(or is given) a session and injects it into pure transform functions, so every
module here stays importable on a laptop with no Databricks runtime present.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def get_or_create_spark(app_name: str) -> "SparkSession":
    """Return the active SparkSession (Databricks-provided) or build one for local dev.

    `pyspark` is imported lazily so this module stays importable without it;
    never called at module import time, only from job entrypoints.
    """
    from pyspark.sql import SparkSession

    return SparkSession.builder.appName(app_name).getOrCreate()


def get_dbutils(spark: "SparkSession"):
    """Return the `dbutils` handle for the given session (Databricks-provided).

    Lazily imported; on Databricks this is the standard way to obtain a
    notebook-equivalent `dbutils` from a plain `.py` job entrypoint.
    """
    from pyspark.dbutils import DBUtils

    return DBUtils(spark)
