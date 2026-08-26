"""
tests/test_features_leakage.py — leakage-safety unit tests (docs/instructions.md §17).

Exercises the pure pandas logic directly: data/features/stat_features.py's
per-series computation (the one place with real sequential lag/rolling/EWMA/
expanding logic) and data/features/gap_fill.py's zero-fill rule. No Spark/mock
needed — both are plain functions over pandas/values.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.features.gap_fill import resolve_qty
from data.features.stat_features import HISTORICAL_ONLY_COLUMNS, compute_stat_features_for_series


def _series_pdf(n_days: int, qty_fn=lambda i: float(i), start_date: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "store_id": ["S1"] * n_days,
            "sku_id": ["ITEM_A"] * n_days,
            "date_id": dates,
            "window_active": [1] * n_days,
            "is_new_store": [0] * n_days,
            "is_new_item": [0] * n_days,
            "isActive": [1] * n_days,
            "qty": [qty_fn(i) for i in range(n_days)],
        }
    )


def test_output_columns_match_the_declared_historical_only_contract():
    pdf = _series_pdf(40)
    out = compute_stat_features_for_series(pdf)
    assert set(HISTORICAL_ONLY_COLUMNS).issubset(out.columns)


def test_lag_1_is_strictly_the_previous_row():
    pdf = _series_pdf(10)  # qty = 0, 1, 2, ..., 9
    out = compute_stat_features_for_series(pdf)
    assert pd.isna(out.loc[0, "lag_1"])
    for i in range(1, 10):
        assert out.loc[i, "lag_1"] == pdf.loc[i - 1, "qty"]


def test_rolling_mean_7_excludes_the_current_row():
    pdf = _series_pdf(15)  # qty = 0, 1, 2, ..., 14
    out = compute_stat_features_for_series(pdf)
    # rolling_mean_7 at row i (i>=7) = mean(qty[i-7 : i]) = i - 4 for this ramp
    for i in range(7, 15):
        assert out.loc[i, "rolling_mean_7"] == pytest.approx(i - 4)
    # not enough history before row 7 -> NaN
    for i in range(0, 7):
        assert pd.isna(out.loc[i, "rolling_mean_7"])


def test_same_weekday_last_year_is_exactly_364_days_back():
    pdf = _series_pdf(400)
    out = compute_stat_features_for_series(pdf)
    assert pd.isna(out.loc[363, "same_weekday_last_year"])
    assert out.loc[364, "same_weekday_last_year"] == pdf.loc[0, "qty"]
    assert out.loc[400 - 1, "same_weekday_last_year"] == pdf.loc[400 - 1 - 364, "qty"]


def test_yoy_ratio_handles_zero_denominator_without_raising():
    pdf = _series_pdf(400, qty_fn=lambda i: 0.0 if i == 0 else float(i))
    out = compute_stat_features_for_series(pdf)  # must not raise ZeroDivisionError/warn-as-error
    # row 364: lag_1 = qty[363], lag_365 = qty[0] = 0.0 -> ratio must be NaN, not inf
    assert pd.isna(out.loc[364, "yoy_ratio"])


def test_trailing_zero_sales_ratio_excludes_current_row():
    # 28 leading zeros, then a run of non-zero values
    pdf = _series_pdf(35, qty_fn=lambda i: 0.0 if i < 28 else 5.0)
    out = compute_stat_features_for_series(pdf)
    # at row 28 (first non-zero, qty=5.0), the trailing 28 prior rows (0..27) are all zero,
    # so the ratio must be 1.0 even though the CURRENT row itself is non-zero.
    assert out.loc[28, "trailing_zero_sales_ratio"] == pytest.approx(1.0)


def test_perturbing_a_future_row_never_changes_past_feature_values():
    """Generic leakage invariant: no feature for row i may depend on row i or any row after it."""
    pdf = _series_pdf(60)
    baseline = compute_stat_features_for_series(pdf.copy())

    perturbed_pdf = pdf.copy()
    last_idx = perturbed_pdf.index[-1]
    perturbed_pdf.loc[last_idx, "qty"] = 999_999.0
    perturbed = compute_stat_features_for_series(perturbed_pdf)

    feature_cols = HISTORICAL_ONLY_COLUMNS
    before_last = baseline.iloc[:-1][feature_cols]
    after_last = perturbed.iloc[:-1][feature_cols]
    pd.testing.assert_frame_equal(before_last, after_last)


def test_perturbing_a_middle_row_never_changes_earlier_feature_values():
    pdf = _series_pdf(60)
    baseline = compute_stat_features_for_series(pdf.copy())

    perturbed_pdf = pdf.copy()
    mid_idx = 30
    perturbed_pdf.loc[mid_idx, "qty"] = 999_999.0
    perturbed = compute_stat_features_for_series(perturbed_pdf)

    feature_cols = HISTORICAL_ONLY_COLUMNS
    before_mid = baseline.iloc[:mid_idx][feature_cols]
    after_mid = perturbed.iloc[:mid_idx][feature_cols]
    pd.testing.assert_frame_equal(before_mid, after_mid)


# --- gap_fill.resolve_qty: the "never zero-pad inactive periods" business rule ---


def test_resolve_qty_keeps_real_sales_value_regardless_of_isActive():
    assert resolve_qty(5.0, is_active=True) == 5.0
    assert resolve_qty(5.0, is_active=False) == 5.0


def test_resolve_qty_zero_fills_only_when_active():
    assert resolve_qty(None, is_active=True) == 0.0


def test_resolve_qty_never_zero_pads_inactive_new_or_seasonal_periods():
    assert resolve_qty(None, is_active=False) is None
