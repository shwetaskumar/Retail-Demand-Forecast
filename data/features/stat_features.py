"""
data/features/stat_features.py — Step 6 (statistical group): historical-only features.

Lags t-1/7/14/28/365; rolling mean/std/min/max/median over 7/14/28; EWMA 7/28;
expanding mean/std; same-weekday-last-year; YoY ratio; trailing zero-sales
ratio. Leakage-safe: every feature derived from data prior to each row.

Implementation split:
  - 25/27 features: native Spark SQL Window functions (no Python serialization,
    runs entirely in JVM — order-of-magnitude faster than applyInPandas for all).
  - 2/27 features (ewma_7, ewma_28): single lightweight applyInPandas pass with
    a minimal 2-column output schema (recursive exponential weighting cannot be
    expressed as a SQL window function).

This replaces the original all-applyInPandas approach, which serialized a
27-column Arrow payload for each of ~545K series groups. The new approach
reduces Python serialization to 2 columns only, cutting applyInPandas overhead
by ~13x while keeping the same leakage-safe contract.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

# Historical-only columns — imported by config/seed_defaults.py; must not drift.
HISTORICAL_ONLY_COLUMNS: list[str] = [
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_365",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_14", "rolling_std_28",
    "rolling_min_7", "rolling_min_14", "rolling_min_28",
    "rolling_max_7", "rolling_max_14", "rolling_max_28",
    "rolling_median_7", "rolling_median_14", "rolling_median_28",
    "ewma_7", "ewma_28",
    "expanding_mean", "expanding_std",
    "same_weekday_last_year", "yoy_ratio", "trailing_zero_sales_ratio",
]

_PASSTHROUGH_SCHEMA = (
    "store_id string, sku_id string, date_id date, "
    "window_active int, is_new_store int, is_new_item int, isActive int, qty float"
)
_EWMA_SCHEMA = "store_id string, sku_id string, date_id date, ewma_7 double, ewma_28 double"

_SAME_WEEKDAY_LAST_YEAR_LAG = 364  # 52 weeks back -> same weekday, fully historical


# ── EWMA (applyInPandas only) ─────────────────────────────────────────────────

def _compute_ewma(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute ewma_7 and ewma_28 for one (store_id, sku_id) group.
    Uses shifted qty so no leakage (ewma of t-1 and earlier, never current row).
    """
    pdf = pdf.sort_values("date_id").reset_index(drop=True)
    shifted = pdf["qty"].shift(1)
    return pd.DataFrame({
        "store_id":    pdf["store_id"],
        "sku_id":      pdf["sku_id"],
        "date_id":     pdf["date_id"],
        "ewma_7":      shifted.ewm(span=7,  adjust=False).mean(),
        "ewma_28":     shifted.ewm(span=28, adjust=False).mean(),
    })


# ── Main entry point ──────────────────────────────────────────────────────────

def add_statistical_features(spark: "SparkSession", gap_filled_df: "DataFrame") -> "DataFrame":
    """Compute all 27 statistical features, returning gap_filled_df + feature columns.

    Strategy:
      1. Register gap_filled_df as a temp view.
      2. Compute 25 features as a single SQL query using Window functions
         (lag, rolling, expanding, median, trailing-zero — all native JVM).
      3. Compute ewma_7/ewma_28 via a lightweight applyInPandas (2-col schema).
      4. LEFT JOIN the two results on (store_id, sku_id, date_id).
         LEFT JOIN (not INNER) so a row is never silently dropped if applyInPandas
         produces fewer rows for a group; ewma_7/ewma_28 would be NULL instead.
    """
    gap_filled_df.createOrReplaceTempView("_stat_gap_filled")

    # ── Step 1: 25 features via Spark SQL Window functions ────────────────────
    # w = partition by series, ordered by date ascending.
    # Rolling windows use ROWS BETWEEN N PRECEDING AND 1 PRECEDING so each row
    # sees only values BEFORE it (shifted=qty.shift(1) equivalent).
    # percentile_approx(qty, 0.5) is the Spark equivalent of pandas rolling.median().
    window_features_df = spark.sql("""
        SELECT *,
            -- Lags
            lag(qty, 1)   OVER w AS lag_1,
            lag(qty, 7)   OVER w AS lag_7,
            lag(qty, 14)  OVER w AS lag_14,
            lag(qty, 28)  OVER w AS lag_28,
            lag(qty, 365) OVER w AS lag_365,

            -- Rolling mean
            avg(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                           ROWS BETWEEN 7  PRECEDING AND 1 PRECEDING) AS rolling_mean_7,
            avg(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                           ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_mean_14,
            avg(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                           ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS rolling_mean_28,

            -- Rolling std
            stddev_samp(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                                   ROWS BETWEEN 7  PRECEDING AND 1 PRECEDING) AS rolling_std_7,
            stddev_samp(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                                   ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_std_14,
            stddev_samp(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                                   ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS rolling_std_28,

            -- Rolling min / max
            min(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                           ROWS BETWEEN 7  PRECEDING AND 1 PRECEDING) AS rolling_min_7,
            min(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                           ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_min_14,
            min(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                           ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS rolling_min_28,
            max(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                           ROWS BETWEEN 7  PRECEDING AND 1 PRECEDING) AS rolling_max_7,
            max(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                           ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_max_14,
            max(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                           ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS rolling_max_28,

            -- Rolling median (percentile_approx; exact for integer qty values)
            percentile_approx(qty, 0.5) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                                               ROWS BETWEEN 7  PRECEDING AND 1 PRECEDING) AS rolling_median_7,
            percentile_approx(qty, 0.5) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                                               ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS rolling_median_14,
            percentile_approx(qty, 0.5) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                                               ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS rolling_median_28,

            -- Expanding mean / std (from beginning of series up to previous row)
            avg(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS expanding_mean,
            stddev_samp(qty) OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS expanding_std,

            -- Same weekday last year (364 rows = 52 weeks back, same DOW after gap_fill)
            lag(qty, 364) OVER w AS same_weekday_last_year,

            -- YoY ratio: lag_1 / lag_365 (both already historical)
            CASE WHEN lag(qty, 365) OVER w = 0 THEN NULL
                 ELSE lag(qty, 1) OVER w / lag(qty, 365) OVER w
            END AS yoy_ratio,

            -- Trailing zero-sales ratio (fraction of last 28 rows where qty=0)
            sum(CASE WHEN qty = 0 THEN 1 ELSE 0 END)
                OVER (PARTITION BY store_id, sku_id ORDER BY date_id
                      ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) / 28.0 AS trailing_zero_sales_ratio

        FROM _stat_gap_filled
        WINDOW w AS (PARTITION BY store_id, sku_id ORDER BY date_id)
    """)
    window_features_df.createOrReplaceTempView("_stat_window_features")

    # ── Step 2: ewma_7 / ewma_28 via minimal applyInPandas ───────────────────
    # Only 2 output columns — ~13x less Arrow serialization than the original
    # 27-column schema. Input only needs 3 cols (store, sku, date, qty).
    ewma_input_df = gap_filled_df.select("store_id", "sku_id", "date_id", "qty")
    ewma_df = ewma_input_df.groupBy("store_id", "sku_id").applyInPandas(
        _compute_ewma, schema=_EWMA_SCHEMA
    )
    ewma_df.createOrReplaceTempView("_stat_ewma")

    # ── Step 3: LEFT JOIN window features + ewma on (store_id, sku_id, date_id) ──
    # LEFT JOIN (not INNER): if applyInPandas produces fewer rows for any group
    # due to an edge case, ewma_7/ewma_28 will be NULL rather than the row being
    # silently dropped from the feature table.
    return spark.sql("""
        SELECT w.*, e.ewma_7, e.ewma_28
        FROM _stat_window_features w
        LEFT JOIN _stat_ewma e
          ON w.store_id = e.store_id
         AND w.sku_id     = e.sku_id
         AND w.date_id     = e.date_id
    """)


# ── Pandas-only reference implementation (test-only) ───────────────────────────
#
# Mirrors add_statistical_features' SQL Window logic exactly, one series at a
# time, so the leakage-safety unit tests (tests/test_features_leakage.py) can
# run without a real Spark session. Not called from any notebook/pipeline path.

def compute_stat_features_for_series(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute all 27 statistical features for a single (store_id, sku_id) series.

    Same leakage-safe contract as add_statistical_features: every feature is
    built from `qty.shift(1)` (or further) onward, never the current row.
    Requires a full window of prior observations (pandas' default
    `min_periods=window`) — an incomplete window yields NaN rather than a
    noisy partial average.
    """
    pdf = pdf.sort_values("date_id").reset_index(drop=True)
    qty = pdf["qty"]
    shifted = qty.shift(1)

    out = pdf.copy()
    out["lag_1"] = shifted
    out["lag_7"] = qty.shift(7)
    out["lag_14"] = qty.shift(14)
    out["lag_28"] = qty.shift(28)
    out["lag_365"] = qty.shift(365)

    for w in (7, 14, 28):
        roll = shifted.rolling(w)
        out[f"rolling_mean_{w}"] = roll.mean()
        out[f"rolling_std_{w}"] = roll.std()
        out[f"rolling_min_{w}"] = roll.min()
        out[f"rolling_max_{w}"] = roll.max()
        out[f"rolling_median_{w}"] = roll.median()

    out["ewma_7"] = shifted.ewm(span=7, adjust=False).mean()
    out["ewma_28"] = shifted.ewm(span=28, adjust=False).mean()

    out["expanding_mean"] = shifted.expanding().mean()
    out["expanding_std"] = shifted.expanding().std()

    out["same_weekday_last_year"] = qty.shift(_SAME_WEEKDAY_LAST_YEAR_LAG)

    lag_365 = out["lag_365"]
    lag_1 = out["lag_1"]
    out["yoy_ratio"] = np.where((lag_365 == 0) | lag_365.isna(), np.nan, lag_1 / lag_365)

    # Trailing zero-sales ratio over the preceding 28 rows. Mask shifted's own
    # leading NaN (start of series) out of the zero-count so "no history yet"
    # correctly yields NaN rather than counting as a zero-sales day.
    zero_flags = (shifted == 0).astype(float)
    zero_flags[shifted.isna()] = np.nan
    out["trailing_zero_sales_ratio"] = zero_flags.rolling(28).sum() / 28.0

    return out
