"""
data/features/lifecycle.py — Steps 2-4: window_active, lifecycle flags, isActive
(feature_engineering_spec.md).

Step 2: window_active = 1 for every row in the dense spine built from sku_store
        [active_from, active_to] ranges — by construction every spine row is within a listing window.

Step 3: is_new_store = 1 if the store has been selling for < 365 days as of the date of
        processing (CURRENT_DATE()). Uses DATEDIFF(CURRENT_DATE(), MIN(date_id)) < 365
        so the flag reflects the store's maturity at run-time, not at the historical
        snapshot_date — critical for backfill correctness (avoids incorrectly flagging
        now-mature stores as new when reprocessing old snapshots).
        is_new_item  = 1 if the item has fewer than 365 distinct positive-sales days in
        the trailing 400-day window from CURRENT_DATE(). Queries sales directly (not via
        sales_base_df) so the flag is independent of the feature-build window.
        Both are computed from actual sales transactions, NOT from sku_store listing
        dates.

Step 4: isActive = 1 ONLY IF window_active=1 AND is_new_store=0 AND is_new_item=0 AND is_delisted=0.

snapshot_date has NO role in isActive. It controls the read window (read_end = snapshot_date)
upstream. Whether an item was active on date_id is a historical fact determined solely by
active_from <= date_id <= active_to in sku_store.

compute_series_eligibility is kept at the bottom for the INFERENCE job (which series are
currently worth scoring). It is NOT part of the isActive pipeline.
"""
from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


# ── Step 3 helpers ────────────────────────────────────────────────────────────


def compute_store_lifecycle(spark: "SparkSession", sales_base_df: "DataFrame") -> "DataFrame":
    """Return (store_id, is_new_store) from actual sales history.

    Spec Step 3: is_new_store = 1 if the store has been selling for < 365 days as of
    CURRENT_DATE() (the date of processing). Uses DATEDIFF from the store's first sale
    date so the flag is stable regardless of which snapshot_date the backfill is
    building — a store that is mature TODAY is never flagged new in any historical
    snapshot. Computed from real sales transactions only — zero-filled rows must NOT be
    included, which is why this runs on the raw sales_base_df before gap_fill.

    Within a windowed sales_base_df this is still correct:
      - Store opened WITHIN the window: MIN(date_id) = true first sale → DATEDIFF accurate.
      - Store opened BEFORE the window: MIN(date_id) = window start → DATEDIFF >= window
        size (750+ days) > 365 → correctly NOT new.
    """
    sales_base_df.createOrReplaceTempView("_store_lifecycle_input")
    return spark.sql("""
        SELECT store_id,
               CASE WHEN DATEDIFF(CURRENT_DATE(), MIN(date_id)) < 365 THEN 1 ELSE 0 END AS is_new_store
        FROM _store_lifecycle_input
        GROUP BY store_id
    """)


def compute_item_lifecycle(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    lookback_days: int = 400,
) -> "DataFrame":
    """Return (sku_id, is_new_item) from actual sales history at market level.

    Spec Step 3: is_new_item = 1 if the item has fewer than 365 distinct days of
    positive sales within the trailing lookback_days (default 400) from CURRENT_DATE()
    (the date of processing). Queries sales directly so the flag always reflects the
    item's sales density as of run-time — critical for backfill correctness where
    sales_base_df is scoped to a historical window that may not contain the current
    trailing period.

    Grouped by sku_id only (market-level) and broadcast to every (store_id, sku_id)
    row for that item in compute_is_active.
    """
    safe_market_id = market_id.replace("'", "''")
    return spark.sql(f"""
        SELECT UPPER(sku_id) AS sku_id,
               CASE WHEN COUNT(DISTINCT CAST(date_id AS DATE)) < 365 THEN 1 ELSE 0 END AS is_new_item
        FROM {catalog}.source.sales
        WHERE UPPER(market_id) = '{safe_market_id}'
          AND date_id >= DATE_SUB(CURRENT_DATE(), {lookback_days})
        GROUP BY UPPER(sku_id)
    """)


# ── Step 4 ────────────────────────────────────────────────────────────────────

def compute_is_active(
    spark: "SparkSession",
    active_window_df: "DataFrame",
    store_lifecycle_df: "DataFrame",
    item_lifecycle_df: "DataFrame",
    read_start: dt.date,
    read_end: dt.date,
) -> "DataFrame":
    """Explode active_window_df into a dense daily spine; stamp window_active,
    is_new_store, is_new_item, is_delisted, and isActive per spec Steps 2-4.

    Step 2 — window_active:
        Clip each (store_id, sku_id) listing range to [read_start, read_end],
        then explode via sequence() into one row per day. window_active = 1 for all
        spine rows (every row is within an [active_from, active_to] window by construction).
        Post-delisting rows (is_delisted=1) already have active_from=active_to=date_id, so
        sequence() yields exactly one row per sale day.

    Step 3 — lifecycle flags:
        LEFT JOIN store_lifecycle_df on store_id → is_new_store.
        LEFT JOIN item_lifecycle_df  on sku_id   → is_new_item.
        COALESCE with 1 so series with zero sales history are treated as new.

    Step 4 — isActive:
        isActive = 1 ONLY IF is_new_store=0 AND is_new_item=0 AND is_delisted=0.
        (window_active is always 1 on the spine, so the AND window_active=1 condition
        is implicit for all rows returned.)
    """
    active_window_df.createOrReplaceTempView("_aw_ranges")
    store_lifecycle_df.createOrReplaceTempView("_store_lifecycle")
    item_lifecycle_df.createOrReplaceTempView("_item_lifecycle")
    return spark.sql(f"""
        WITH bounded_ranges AS (
            SELECT store_id, sku_id,
                   GREATEST(active_from, DATE('{read_start.isoformat()}')) AS range_start,
                   LEAST(active_to,   DATE('{read_end.isoformat()}'))   AS range_end,
                   COALESCE(is_delisted, 0) AS is_delisted
            FROM _aw_ranges
        ),
        exploded AS (
            SELECT DISTINCT store_id, sku_id,
                   explode(sequence(range_start, range_end, interval 1 day)) AS date_id,
                   is_delisted
            FROM bounded_ranges
            WHERE range_start <= range_end
        )
        SELECT
            e.store_id,
            e.sku_id,
            e.date_id,
            1                                                       AS window_active,
            COALESCE(sl.is_new_store, 1)                           AS is_new_store,
            COALESCE(il.is_new_item,  1)                           AS is_new_item,
            e.is_delisted,
            CASE
                WHEN COALESCE(sl.is_new_store, 1) = 0
                 AND COALESCE(il.is_new_item,  1) = 0
                 AND e.is_delisted = 0
                THEN 1 ELSE 0
            END                                                     AS isActive
        FROM exploded e
        LEFT JOIN _store_lifecycle sl ON e.store_id = sl.store_id
        LEFT JOIN _item_lifecycle  il ON e.sku_id     = il.sku_id
    """)


# ── Inference-only ────────────────────────────────────────────────────────────

def compute_series_eligibility(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    snapshot_date: dt.date,
    recency_days: int = 90,
) -> "DataFrame":
    """Return (store_id, sku_id, is_eligible=1) for series currently worth scoring.

    FOR INFERENCE USE ONLY — not used to derive isActive in the feature table.

    The inference job calls this to restrict scoring to sku-store pairs
    still actively listed within recency_days of snapshot_date (active_to >= snapshot_date
    - recency_days). Generating forecasts for long-discontinued items is wasteful and
    those series have no downstream consumer.

    is_eligible = 1 for rows in sku_store where:
      - market_id matches the market
      - active_from <> '1970-01-01'  (exclude sentinel/placeholder records)
      - active_to >= snapshot_date - recency_days
    """
    def _d(v): return v if isinstance(v, dt.date) else dt.date.fromisoformat(str(v)[:10])
    snapshot_date  = _d(snapshot_date)
    safe_market_id = market_id.replace("'", "''")
    recency_cutoff = (snapshot_date - dt.timedelta(days=recency_days)).isoformat()

    return spark.sql(f"""
        SELECT DISTINCT
            CAST(lvc.store_id AS STRING) AS store_id,
            UPPER(lvc.sku_id)            AS sku_id,
            1                            AS is_eligible
        FROM {catalog}.source.sku_store lvc
        WHERE UPPER(lvc.market_id) = '{safe_market_id}'
          AND lvc.active_from <> '1970-01-01'
          AND lvc.active_to >= DATE('{recency_cutoff}')
    """)
