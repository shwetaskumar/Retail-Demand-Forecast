"""
data/features/sales_base.py — Step 1: sales base / target (feature_engineering_spec.md).

qty = SUM(units_sold + units_promotional + units_shrinkage), grouped to
(store_id, sku_id, date_id), filtered to market and to
[history_start, snapshot_date]. Reads source.sales READ-ONLY.

Implemented as a parameterized `spark.sql(...)` string (matching the SQL given
in feature_engineering_spec.md Step 1 almost verbatim) rather than the
DataFrame/Column API, so this only ever calls duck-typed methods on the
injected `spark` — no pyspark import needed here.

store_id is CAST to STRING here to match the type produced by active_window.py
(CAST(sw.store_id AS STRING)). All downstream JOINs — compute_is_active,
gap_fill_zeros, compute_store_lifecycle — depend on a consistent STRING type.
Without the cast, ANSI mode on DBR 17.3 throws an AnalysisException; permissive
mode silently relies on implicit cast which is fragile.
"""
from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def build_sales_base(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    history_start: dt.date,
    snapshot_date: dt.date,
) -> "DataFrame":
    """Read source.sales read-only and return the (store_id, sku_id, date_id, qty) base."""
    safe_market_id = market_id.replace("'", "''")
    return spark.sql(f"""
        SELECT CAST(store_id AS STRING) AS store_id,
               UPPER(sku_id) AS sku_id,
               CAST(date_id AS DATE) AS date_id,
               CAST(SUM(NVL(units_sold, 0) + NVL(units_promotional, 0) + NVL(units_shrinkage, 0)) AS FLOAT) AS qty
        FROM {catalog}.source.sales
        WHERE UPPER(market_id) = '{safe_market_id}'
          AND CAST(date_id AS DATE) BETWEEN DATE('{history_start.isoformat()}') AND DATE('{snapshot_date.isoformat()}')
        GROUP BY ALL
    """)
