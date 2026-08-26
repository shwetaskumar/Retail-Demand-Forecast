"""
data/features/active_window.py — Step 2: active window bounds.

Source: source.sku_store (the sku-store activeness window).

Three paths:
  1. Established stores: have at least one non-sentinel active_from row — use the
     real listing dates directly (sentinel rows for that store are still excluded).
  2. New stores (sentinel-only): ALL sku_store rows have active_from='1970-01-01'.
     These stores have sales but no valid listing dates in the source system.
     Their spine start (active_from) is derived from the store's first sale date;
     active_to is kept from sku_store (far-future, clipped by compute_is_active).
     This ensures new stores enter sku_store_features with is_new_store=1 and
     isActive=0.
  3. Post-delisting sales: sales that occur AFTER sku_store.active_to. These
     are captured with is_delisted=1 and isActive=0. Ensures all sales activity is
     recorded in the feature table for reconciliation and completeness, even when
     the source listing says the sku was delisted.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def _build_established_store_window(
    spark: "SparkSession", catalog: str, safe_market_id: str
) -> "DataFrame":
    """Established stores: non-sentinel active_from rows from sku_store."""
    return spark.sql(f"""
        SELECT CAST(store_id AS STRING) AS store_id,
               UPPER(sku_id) AS sku_id,
               active_from,
               active_to
        FROM {catalog}.source.sku_store
        WHERE UPPER(market_id) = '{safe_market_id}'
          AND active_from <> '1970-01-01'
    """)


def _build_new_store_window(
    spark: "SparkSession", catalog: str, safe_market_id: str
) -> "DataFrame":
    """New stores: ALL sku_store rows have sentinel active_from ('1970-01-01').

    Derives active_from from the store's first sale date (across all skus at that
    store). active_to is kept from sku_store. Skus are sourced from sku_store
    (authoritative listing) — not limited to skus with actual sales — so skus with
    zero sales at the new store still get a spine row and are gap-filled later.

    The resulting rows flow through compute_is_active where:
      - compute_store_lifecycle marks the store as is_new_store=1 (< 365 days via
        DATEDIFF from first sale)
      - isActive = 0 (because is_new_store=1)
    """
    return spark.sql(f"""
        WITH sentinel_only_stores AS (
            -- Stores where EVERY sku_store row has the sentinel active_from
            SELECT CAST(store_id AS STRING) AS store_id
            FROM {catalog}.source.sku_store
            WHERE UPPER(market_id) = '{safe_market_id}'
            GROUP BY store_id
            HAVING SUM(CASE WHEN active_from <> '1970-01-01' THEN 1 ELSE 0 END) = 0
        ),
        store_first_sale AS (
            -- First sale date per sentinel-only store
            SELECT CAST(p.store_id AS STRING) AS store_id,
                   MIN(CAST(p.date_id AS DATE)) AS first_sale_date
            FROM {catalog}.source.sales p
            JOIN sentinel_only_stores s
                ON CAST(p.store_id AS STRING) = s.store_id
            WHERE UPPER(p.market_id) = '{safe_market_id}'
            GROUP BY p.store_id
        )
        SELECT CAST(ss.store_id AS STRING) AS store_id,
               UPPER(ss.sku_id) AS sku_id,
               fs.first_sale_date AS active_from,
               ss.active_to
        FROM {catalog}.source.sku_store ss
        JOIN sentinel_only_stores s
            ON CAST(ss.store_id AS STRING) = s.store_id
        JOIN store_first_sale fs
            ON CAST(ss.store_id AS STRING) = fs.store_id
        WHERE UPPER(ss.market_id) = '{safe_market_id}'
    """)


def _build_post_delisting_window(
    spark: "SparkSession", sales_base_df: "DataFrame", active_window_df: "DataFrame"
) -> "DataFrame":
    """Capture sales that fall outside the sku_store listing window.

    These are (store_id, sku_id, date_id) tuples present in sales_base_df
    but NOT covered by any active_window_df row's [active_from, active_to] range.
    They represent post-delisting sales — skus that continue to sell at a store
    after sku_store says they were delisted.

    Returns rows with:
      - store_id, sku_id (from sales)
      - active_from = date_id, active_to = date_id (each sale day is its own window)
      - is_delisted = 1
    """
    sales_base_df.createOrReplaceTempView("_sb_for_delisting")
    active_window_df.createOrReplaceTempView("_aw_for_delisting")
    return spark.sql("""
        WITH sale_dates AS (
            SELECT DISTINCT store_id, sku_id, date_id
            FROM _sb_for_delisting
        ),
        aw_exploded AS (
            SELECT store_id, sku_id, active_from, active_to
            FROM _aw_for_delisting
        )
        SELECT p.store_id,
               p.sku_id,
               p.date_id AS active_from,
               p.date_id AS active_to,
               1 AS is_delisted
        FROM sale_dates p
        LEFT JOIN aw_exploded aw
            ON p.store_id = aw.store_id
            AND p.sku_id = aw.sku_id
            AND p.date_id BETWEEN aw.active_from AND aw.active_to
        WHERE aw.store_id IS NULL
    """)


def build_active_window(
    spark: "SparkSession", catalog: str, market_id: str,
    sales_base_df: "DataFrame | None" = None,
) -> "DataFrame":
    """Return (store_id, sku_id, active_from, active_to, is_delisted) from source.sku_store.

    Combines three paths:
      1. Established stores — non-sentinel active_from from sku_store.
      2. New stores (sentinel-only) — active_from derived from first sale date.
      3. Post-delisting sales — sales outside any [active_from, active_to] window.
         Only included when sales_base_df is provided.

    active_from/active_to provide the valid date range, consumed by compute_is_active
    to build the daily spine. is_delisted=1 for path 3 rows, 0 otherwise.
    """
    from pyspark.sql.functions import lit

    safe_market_id = market_id.replace("'", "''")

    established_df = _build_established_store_window(spark, catalog, safe_market_id)
    new_store_df = _build_new_store_window(spark, catalog, safe_market_id)

    # Add is_delisted=0 to the listed paths
    combined_df = (
        established_df.withColumn("is_delisted", lit(0))
        .unionByName(new_store_df.withColumn("is_delisted", lit(0)))
    )

    # Path 3: post-delisting sales (only when sales_base_df is available)
    if sales_base_df is not None:
        post_delisting_df = _build_post_delisting_window(spark, sales_base_df, combined_df)
        combined_df = combined_df.unionByName(post_delisting_df)

    return combined_df
