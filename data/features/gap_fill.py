"""
data/features/gap_fill.py — Step 5: gap fill / true zeros (feature_engineering_spec.md).

Left-join sales onto the dense (store_id, sku_id, date_id) spine from
data/features/lifecycle.py::compute_is_active; fill qty=0 ONLY where isActive=1
and no sales row exists. Never zero-pad isActive=0 / new / SEASONAL / inactive periods.

The fill rule is a plain, row-independent function (`resolve_qty`) so it has
exactly one definition that both the SQL CASE below and tests/test_features_leakage.py
can be checked against — no per-series computation needed here (see stat_features.py
for the one place that genuinely needs per-series pandas logic).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def resolve_qty(raw_qty: float | None, is_active: bool) -> float | None:
    """The Step 5 zero-fill rule: keep a real sales value if present; else 0.0 only if
    `is_active`; else None (never zero-padded for isActive=0 / new / SEASONAL / inactive periods)."""
    if raw_qty is not None:
        return raw_qty
    return 0.0 if is_active else None


def gap_fill_zeros(spark: "SparkSession", is_active_df: "DataFrame", sales_df: "DataFrame") -> "DataFrame":
    """Left-join `sales_df` onto the dense `is_active_df` spine and apply the `resolve_qty` rule.

    Mirrors `resolve_qty` exactly: `CASE WHEN s.qty IS NOT NULL THEN s.qty
    WHEN sp.isActive = 1 THEN 0.0 ELSE NULL END`.
    """
    is_active_df.createOrReplaceTempView("_gap_fill_spine")
    sales_df.createOrReplaceTempView("_gap_fill_sales")
    return spark.sql("""
        SELECT sp.store_id, sp.sku_id, sp.date_id,
               sp.window_active, sp.is_new_store, sp.is_new_item, sp.is_delisted, sp.isActive,
               CASE
                   WHEN s.qty IS NOT NULL THEN s.qty
                   WHEN sp.isActive = 1 THEN CAST(0.0 AS FLOAT)
                   ELSE NULL
               END AS qty
        FROM _gap_fill_spine sp
        LEFT JOIN _gap_fill_sales s
          ON sp.store_id = s.store_id
         AND sp.sku_id = s.sku_id
         AND sp.date_id = s.date_id
    """)
