"""
training/evaluation.py — volume-weighted 7-day WMAPE gate at 3 grains (docs/instructions.md §3, §21).

Grains: category_cohort, market_sku, market_store_sku. Each is gated
against its category's accuracy_threshold from config_category. Writes
backtest_metrics with per-level pass/fail.

`gate` takes per-series CHAMPION metrics (training/champion.py's
select_champions/select_champions_spark output — one row per series, with
`wmape_7d` and `volume`, the last backtest fold's total |qty|) and aggregates
up to market_sku and category_cohort by volume-weighted averaging:
`grain_wmape = sum(wmape_i * volume_i) / sum(volume_i)`. This is not an
approximation — since `wmape_i = sum(abs_error_i) / volume_i` by definition,
weighting each series' WMAPE by its own volume and averaging is algebraically
identical to `sum(all abs_error) / sum(all volume)` computed directly. That
identity only holds starting from *volume*-weighted per-series WMAPE, which is
exactly what training/engine.py already produces — hence why `volume` had to
be added there in this step.

Supplementary metrics (docs/instructions.md §3):
  - WMAPE: primary gate metric (unchanged). Planners see this for lag-1 and lag-2.
  - bias_pct: (sum_yhat - sum_actual) / sum_actual, volume-weighted across series.
              Directional error — a model can have low WMAPE but systematic over/under.
  - mase_7d: MAE / naive_mae_7d; scale-free, handles zero-volume series (MASE < 1
              beats the 7-day naïve benchmark). Requires `naive_mae_7d` in champions_pdf.
              Gracefully omitted if the engine hasn't produced naive_mae_7d yet.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from utils.io import merge_upsert

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from config.loader import CategoryConfig

_SCHEMA = "gold"
_TABLE = "backtest_metrics"
_HORIZON_DAYS = 7  # this gate is the 7-day champion-selection metric (docs/instructions.md §3);
# 14/28/140-day WMAPE is secondary/monitoring-only reporting, not implemented here.


def wmape(actual: pd.Series, predicted: pd.Series) -> float:
    """Weekly-aggregated WMAPE: sum(|A_week - F_week|) / sum(A_week).

    Aggregates daily actuals and forecasts into 7-day weeks FIRST, then
    computes absolute error on the weekly totals. This matches the business
    definition: intra-week over/under-forecasts cancel if the weekly total
    is correct.

    For a single 7-day holdout (the standard backtest fold), this simplifies
    to |sum(actual) - sum(predicted)| / sum(actual).

    For multi-week horizons (14, 28, 140 days), chunks into consecutive
    7-day periods and aggregates: sum_weeks(|A_w - F_w|) / sum_weeks(A_w).

    NaN if actual has zero total volume (undefined WMAPE).
    """
    import numpy as np

    a = actual.values.astype(float)
    f = predicted.values.astype(float)
    n = len(a)

    if n == 0:
        return float("nan")

    # Chunk into 7-day weeks (last chunk may be partial)
    n_full_weeks = n // 7
    remainder = n % 7

    weekly_actual = []
    weekly_predicted = []
    for w in range(n_full_weeks):
        weekly_actual.append(a[w * 7:(w + 1) * 7].sum())
        weekly_predicted.append(f[w * 7:(w + 1) * 7].sum())
    if remainder > 0:
        weekly_actual.append(a[n_full_weeks * 7:].sum())
        weekly_predicted.append(f[n_full_weeks * 7:].sum())

    wa = np.array(weekly_actual)
    wf = np.array(weekly_predicted)

    denom = wa.sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(wa - wf).sum() / denom)


def bias_pct(actual: pd.Series, predicted: pd.Series) -> float:
    """Directional error: (sum_yhat - sum_actual) / sum_actual.

    Positive = over-forecast (waste risk); negative = under-forecast (stockout risk).
    NaN if sum_actual == 0.
    """
    denom = actual.sum()
    if denom == 0:
        return float("nan")
    return float((predicted.sum() - actual.sum()) / denom)


def mase(actual: pd.Series, predicted: pd.Series, naive_mae: float) -> float:
    """Mean Absolute Scaled Error: MAE / naive_mae.

    Scale-free and defined for zero-volume series (unlike WMAPE). Uses the
    7-day seasonal naïve as the benchmark (same weekday, prior week).
    Values < 1.0 beat the naïve benchmark; values > 1.0 are worse than naïve.
    NaN if naive_mae == 0 (constant series — naïve is perfect).
    """
    if naive_mae == 0 or pd.isna(naive_mae):
        return float("nan")
    n = len(actual)
    if n == 0:
        return float("nan")
    mae_val = float((actual - predicted).abs().mean())
    return mae_val / naive_mae


def _volume_weighted_wmape(group: pd.DataFrame) -> float:
    total_volume = group["volume"].sum()
    if total_volume == 0:
        return float("nan")
    return float((group["wmape_7d"] * group["volume"]).sum() / total_volume)


def _volume_weighted_bias(group: pd.DataFrame) -> float:
    """Volume-weighted bias_pct across series.

    bias_pct is stored as a fraction (not pct) in the per-series row.
    Aggregate: weight each series' bias by its own volume so high-volume series
    dominate the aggregate signal (same weighting as WMAPE aggregate).
    Returns NaN if the column is absent (graceful — older engine rows won't have it).
    """
    if "bias_pct" not in group.columns:
        return float("nan")
    valid = group[group["bias_pct"].notna()]
    total_volume = valid["volume"].sum()
    if total_volume == 0:
        return float("nan")
    return float((valid["bias_pct"] * valid["volume"]).sum() / total_volume)


def _volume_weighted_mase(group: pd.DataFrame) -> float:
    """Volume-weighted MASE across series.

    Only series with a valid naive_mae_7d contribute. Gracefully returns NaN if
    the column is absent (older engine rows that pre-date naive baseline tracking).
    """
    if "mase_7d" not in group.columns:
        return float("nan")
    valid = group[group["mase_7d"].notna()]
    total_volume = valid["volume"].sum()
    if total_volume == 0:
        return float("nan")
    return float((valid["mase_7d"] * valid["volume"]).sum() / total_volume)


def _grain_rows(
    champions_pdf: pd.DataFrame,
    grain: str,
    group_cols: list[str],
    threshold_by_category: dict,
) -> list[dict]:
    rows = []
    for keys, group in champions_pdf.groupby(group_cols):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_map = dict(zip(group_cols, key_values))
        market = key_map["market_id"]
        category = key_map["category"]
        entity_key = key_map[group_cols[-1]]

        wmape_value = _volume_weighted_wmape(group)
        volume = float(group["volume"].sum())
        threshold = threshold_by_category.get((market, category))

        # ── GATE: WMAPE only — planners read this for lag-1 and lag-2 ──────────
        passed = None if threshold is None or pd.isna(wmape_value) else bool((1 - wmape_value) >= threshold)

        # ── Supplementary diagnostics (no gate role) ───────────────────────────
        bias_value = _volume_weighted_bias(group)
        mase_value = _volume_weighted_mase(group)

        rows.append(
            {
                "market_id": market,
                "category": category,
                "grain": grain,
                "entity_key": entity_key,
                "horizon": _HORIZON_DAYS,
                "wmape": wmape_value,
                "volume": volume,
                "threshold": threshold,
                "passed": passed,
                "bias_pct": bias_value,      # supplementary — NOT a gate column
                "mase_7d": mase_value,        # supplementary — NOT a gate column
            }
        )
    return rows


def gate(champions_pdf: pd.DataFrame, categories: list[CategoryConfig]) -> pd.DataFrame:
    """Compute volume-weighted 7-day WMAPE at all three grains (category_cohort, market_sku,
    market_store_sku) from per-series champion metrics, and compare each to its
    category's accuracy_threshold.

    `champions_pdf` must carry: market_id, category, sku_id, series_key, wmape_7d, volume.
    Optional supplementary columns (produced by newer engine versions):
      - bias_pct: per-series directional error fraction
      - mase_7d: per-series MASE vs 7-day naïve benchmark
    These are stored in backtest_metrics but do NOT affect gate pass/fail.
    WMAPE gate thresholds are the production accuracy contract — planners check
    lag-1 and lag-2 against these thresholds.
    """
    if champions_pdf.empty:
        return pd.DataFrame(
            columns=[
                "market_id", "category", "grain", "entity_key", "horizon",
                "wmape", "volume", "threshold", "passed", "bias_pct", "mase_7d",
            ]
        )

    threshold_by_category = {(c.market_id, c.category): c.accuracy_threshold for c in categories}

    rows: list[dict] = []
    rows += _grain_rows(champions_pdf, "market_store_sku", ["market_id", "category", "series_key"], threshold_by_category)
    rows += _grain_rows(champions_pdf, "market_sku", ["market_id", "category", "sku_id"], threshold_by_category)
    rows += _grain_rows(champions_pdf, "category_cohort", ["market_id", "category"], threshold_by_category)
    return pd.DataFrame(rows)


def write_backtest_metrics(
    spark: "SparkSession",
    gated_pdf: pd.DataFrame,
    catalog: str,
    run_id: str,
    snapshot_date: str,
) -> None:
    """MERGE gated metrics into `{catalog}.gold.backtest_metrics`, keyed on
    (snapshot_date, run_id, grain, entity_key).

    Columns written: wmape (gate metric), bias_pct and mase_7d (supplementary).
    The backtest_metrics DDL must include FLOAT columns bias_pct and mase_7d
    (nullable) — run config/ddl/backtest_metrics_alter.sql once to add them if
    upgrading an existing table (see config/ddl/).
    """
    if gated_pdf.empty:
        return
    to_write = gated_pdf.assign(snapshot_date=snapshot_date, run_id=run_id)
    columns = [
        "snapshot_date", "run_id", "market_id", "category", "grain", "entity_key",
        "horizon", "wmape", "volume", "threshold", "passed",
        "bias_pct", "mase_7d",   # supplementary — nullable on older rows
    ]
    # Only include supplementary columns if present (forward-compatible with old engine output).
    available = [c for c in columns if c in to_write.columns]
    spark_df = spark.createDataFrame(to_write[available])
    merge_upsert(spark, spark_df, catalog, _SCHEMA, _TABLE, merge_keys=["snapshot_date", "run_id", "market_id", "category", "grain", "entity_key"])
