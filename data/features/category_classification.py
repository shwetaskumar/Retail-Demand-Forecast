"""
data/features/category_classification.py — static ABC/D/SEASONAL category classification.

[ASSUMPTION, per user direction 2026-07-01]: no source table names a per-sku
category (A/B/C/D/SEASONAL) classification — feature_engineering_spec.md and
docs/instructions.md §6.1 don't define one, and config_category only
holds market x category *thresholds*, not a sku -> category mapping (the gap
training/engine.py originally flagged). Until planners provide a real
mapping, classify each sku_id (market-wide, across all stores) by trailing
sales volume:
  - `is_new_item=1` (data/features/lifecycle.py) -> "SEASONAL". No dedicated
    SEASONAL flag exists in source data; a new-to-market item is the closest
    available proxy the user approved.
  - else, ABC by cumulative share of total market volume (skus sorted by
    volume descending): cumulatively covering the first 80% of volume -> "A";
    next 10% -> "B"; next 10% -> "C".
  - any sku with total_qty <= SLOW_MOVER_MAX_QTY over the window -> "D" (very
    slow movers, ~1-2 units), overriding the A/B/C band.

Static and coarse by design (the user's framing): a placeholder business rule,
not a real merchandising classification. Replace with a planner-sourced
mapping (a new config_* or silver table) when available.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

SLOW_MOVER_MAX_QTY = 2.0  # TODO: source from config once a real threshold is agreed
A_CUM_SHARE = 0.80
B_CUM_SHARE = 0.90


def classify_volume_band(
    total_qty: float,
    cum_share: float,
    slow_mover_max_qty: float = SLOW_MOVER_MAX_QTY,
    a_cum_share: float = A_CUM_SHARE,
    b_cum_share: float = B_CUM_SHARE,
) -> str:
    """The A/B/C/D banding rule, mirrored exactly by compute_sku_category's SQL CASE: a sku
    with <= slow_mover_max_qty total volume over the window is 'D' regardless of its
    cumulative share; otherwise it's 'A'/'B'/'C' by cumulative share of total market volume."""
    if total_qty <= slow_mover_max_qty:
        return "D"
    if cum_share <= a_cum_share:
        return "A"
    if cum_share <= b_cum_share:
        return "B"
    return "C"


def resolve_category(is_new_item: bool, volume_band: str) -> str:
    """SEASONAL takes precedence over the volume band — see module docstring for why
    is_new_item is the proxy."""
    return "SEASONAL" if is_new_item else volume_band


def compute_sku_category(
    spark: "SparkSession", sales_base_df: "DataFrame", item_lifecycle_df: "DataFrame"
) -> "DataFrame":
    """Return (sku_id, category) — one row per sku_id, market-wide (see module
    docstring). Mirrors classify_volume_band + resolve_category exactly; keep both in sync
    if either changes."""
    sales_base_df.createOrReplaceTempView("_category_sales_base")
    item_lifecycle_df.createOrReplaceTempView("_category_item_lifecycle")
    return spark.sql(f"""
        WITH item_volume AS (
            SELECT sku_id, SUM(qty) AS total_qty
            FROM _category_sales_base
            GROUP BY sku_id
        ),
        ranked AS (
            SELECT sku_id, total_qty,
                   SUM(total_qty) OVER (ORDER BY total_qty DESC, sku_id ASC
                                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                     / SUM(total_qty) OVER () AS cum_share
            FROM item_volume
        ),
        banded AS (
            SELECT sku_id, total_qty,
                   CASE
                       WHEN total_qty <= {SLOW_MOVER_MAX_QTY} THEN 'D'
                       WHEN cum_share <= {A_CUM_SHARE} THEN 'A'
                       WHEN cum_share <= {B_CUM_SHARE} THEN 'B'
                       ELSE 'C'
                   END AS volume_category
            FROM ranked
        )
        SELECT b.sku_id,
               CASE WHEN COALESCE(il.is_new_item, 0) = 1 THEN 'SEASONAL' ELSE b.volume_category END AS category
        FROM banded b
        LEFT JOIN _category_item_lifecycle il ON b.sku_id = il.sku_id
    """)


def add_category(spark: "SparkSession", df: "DataFrame", category_df: "DataFrame") -> "DataFrame":
    """Left-join `category` (from compute_sku_category) onto `df` by sku_id."""
    df.createOrReplaceTempView("_add_category_input")
    category_df.createOrReplaceTempView("_add_category_lookup")
    return spark.sql("""
        SELECT t.*, c.category
        FROM _add_category_input t
        LEFT JOIN _add_category_lookup c ON t.sku_id = c.sku_id
    """)
