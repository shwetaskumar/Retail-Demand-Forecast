"""
data/features/category_features.py — Step 6 (halo/cannibalization group, feature_engineering_spec.md §7).

Computes demand-context features at the (store_id × date_id) and
(store_id × category × date_id) grain — NOT per sku_id — and then
joins them onto the per-series spine.  This keeps the feature table from
exploding (no item×item pair columns) while still giving models the signal
they need for halo and cannibalization effects.

Three feature families:

1. category_qty_7d_roll — rolling 7-day sum of qty for every (store, category)
   combination.  Captures within-store demand level for the item's own category.
   A second column, other_category_qty_7d_roll, is the rolling total across ALL
   OTHER categories at the same store.  Together these give LGBM and Elastic
   Net enough signal to learn "when the store is busy overall, individual items
   spike" without needing explicit item-level cross features.

2. competing_promo_count — for each (store, item, date), how many OTHER items in
   the same category at the same store have an active promo today.  A non-zero
   value indicates potential cannibalization pressure (the item itself may be
   passed over in favour of the promoted competitor).  Requires promo_active_flag
   to already be present in the input frame (must run after build_promo_features).

3. category_active_sku_count — rolling 28-day count of distinct active items in
   the same category at the same store.  Provides a "breadth of assortment"
   signal.

All features are historical_only (they look at realized qty) EXCEPT competing_
promo_count, which is known_at_inference because promo schedules are known
ahead of time (as in event_features.py, promos are reliable 2m-1yr out).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

# Exported for config/seed_defaults.py feature seeding.
HISTORICAL_ONLY_COLUMNS: list[str] = [
    "category_qty_7d_roll",
    "other_category_qty_7d_roll",
    "category_active_sku_count",
]
KNOWN_AT_INFERENCE_COLUMNS: list[str] = [
    "competing_promo_count",
]


def build_category_features(
    spark: "SparkSession",
    df: "DataFrame",
) -> "DataFrame":
    """Attach category-level context features to a per-series spine.

    Args:
        spark: Active SparkSession.
        df:    Per-series feature frame, must carry at minimum:
               store_id, sku_id, category, date_id, qty, isActive,
               promo_active_flag (produced by build_promo_features).

    Returns:
        df widened with the columns listed in HISTORICAL_ONLY_COLUMNS and
        KNOWN_AT_INFERENCE_COLUMNS.  Rows without a matching category context
        (e.g. single-item categories) receive 0 for the new columns.
    """
    df.createOrReplaceTempView("_cat_features_input")

    return spark.sql("""
        -- 1. Category-level rolling 7-day qty aggregate (HISTORICAL_ONLY)
        -- Computed at (store_id, category, date_id) grain -- no per-item explosion.
        -- Uses only rows with isActive = 1 so gap-filled zeros at inactive stores
        -- don't dilute the rolling signal.
        WITH cat_daily AS (
            SELECT store_id, category, date_id,
                   SUM(COALESCE(qty, 0)) AS cat_qty_today
            FROM _cat_features_input
            WHERE isActive = 1
            GROUP BY store_id, category, date_id
        ),
        cat_roll AS (
            SELECT store_id, category, date_id,
                   SUM(cat_qty_today) OVER (
                       PARTITION BY store_id, category
                       ORDER BY date_id
                       ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                   ) AS category_qty_7d_roll
            FROM cat_daily
        ),
        -- Rolling total across ALL categories at this store (for other_category signal)
        store_daily AS (
            SELECT store_id, date_id,
                   SUM(cat_qty_today) AS store_qty_today
            FROM cat_daily
            GROUP BY store_id, date_id
        ),
        store_roll AS (
            SELECT store_id, date_id,
                   SUM(store_qty_today) OVER (
                       PARTITION BY store_id
                       ORDER BY date_id
                       ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                   ) AS store_qty_7d_roll
            FROM store_daily
        ),

        -- 2. Competing promo count (KNOWN_AT_INFERENCE)
        -- For each spine row: count distinct OTHER items in the same category at the
        -- same store that have promo_active_flag = 1 on this date.
        -- Requires build_promo_features to have run first (promo_active_flag present).
        --
        -- REWRITE (O(N) not O(N²)): the original self-join explodes to
        -- items_per_category² rows per (store, category, date) and OOMs the
        -- driver on the full PL read window. Instead:
        --   1. Aggregate total promo-active items per (store, category, date).
        --   2. For each focal row, subtract 1 if the focal item itself has a promo
        --      (to exclude it from its own competitor count).
        -- Mathematically identical: competing = total_promos - self_promo.
        total_promo_per_bucket AS (
            SELECT store_id, category, date_id,
                   COUNT(DISTINCT sku_id) AS total_promo_items
            FROM _cat_features_input
            WHERE promo_active_flag = 1
            GROUP BY store_id, category, date_id
        ),
        competing_promo AS (
            SELECT t.store_id, t.sku_id, t.date_id,
                   GREATEST(0,
                       COALESCE(p.total_promo_items, 0)
                       - COALESCE(t.promo_active_flag, 0)
                   ) AS competing_promo_count
            FROM _cat_features_input t
            LEFT JOIN total_promo_per_bucket p
              ON  t.store_id = p.store_id
              AND t.category    = p.category
              AND t.date_id     = p.date_id
        ),

        -- 3. Active item count in category at store, trailing 28 days (HISTORICAL_ONLY)
        -- "Breadth of assortment" proxy.
        active_item_daily AS (
            SELECT store_id, category, date_id,
                   COUNT(DISTINCT sku_id) AS active_items_today
            FROM _cat_features_input
            WHERE isActive = 1
            GROUP BY store_id, category, date_id
        ),
        active_item_roll AS (
            SELECT store_id, category, date_id,
                   MAX(active_items_today) OVER (
                       PARTITION BY store_id, category
                       ORDER BY date_id
                       ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
                   ) AS category_active_sku_count
            FROM active_item_daily
        )

        SELECT t.*,
               COALESCE(cr.category_qty_7d_roll, 0)    AS category_qty_7d_roll,
               COALESCE(lr.store_qty_7d_roll, 0)
                   - COALESCE(cr.category_qty_7d_roll, 0) AS other_category_qty_7d_roll,
               COALESCE(cp.competing_promo_count, 0)   AS competing_promo_count,
               COALESCE(ar.category_active_sku_count, 0) AS category_active_sku_count
        FROM _cat_features_input t
        LEFT JOIN cat_roll cr
          ON t.store_id = cr.store_id
         AND t.category    = cr.category
         AND t.date_id     = cr.date_id
        LEFT JOIN store_roll lr
          ON t.store_id = lr.store_id
         AND t.date_id     = lr.date_id
        LEFT JOIN competing_promo cp
          ON t.store_id = cp.store_id
         AND t.sku_id     = cp.sku_id
         AND t.date_id     = cp.date_id
        LEFT JOIN active_item_roll ar
          ON t.store_id = ar.store_id
         AND t.category    = ar.category
         AND t.date_id     = ar.date_id
    """)
