"""
tests/test_wrmsse.py — M5 WRMSSE spec conformance (training/wrmsse.py).

Pure pandas/numpy, no Spark. These pin the two spec details that are easy to get
wrong and that silently shift the headline score if wrong (see the module docstring):
the lag-1 (not lag-7) naive denominator, and the active-window start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.wrmsse import LEVELS, compute_wrmsse, naive_scale, rmsse


class TestNaiveScale:
    def test_is_lag1_not_lag7(self):
        # Strict alternation: every lag-1 step is +-1 so RMS of diffs is exactly 1.0.
        # A lag-7 denominator would see zero differences and blow up instead.
        y = np.array([1.0, 2.0] * 20)
        assert naive_scale(y) == pytest.approx(1.0)

    def test_starts_at_first_nonzero_observation(self):
        """Leading structural zeros (item not yet stocked) must not enter the denominator.

        Counting them shrinks the naive error and inflates every RMSSE built on it.
        """
        active = [5.0, 7.0, 6.0, 9.0]
        with_leading_zeros = np.array([0.0] * 50 + active)
        assert naive_scale(with_leading_zeros) == pytest.approx(naive_scale(np.array(active)))

    def test_interior_zeros_are_kept(self):
        # Zeros AFTER the first sale are genuine no-demand days and stay in the sample.
        y = np.array([4.0, 0.0, 4.0, 0.0])
        assert naive_scale(y) == pytest.approx(4.0)

    def test_never_sold_series_is_undefined(self):
        assert np.isnan(naive_scale(np.zeros(30)))

    def test_single_active_point_is_undefined(self):
        assert np.isnan(naive_scale(np.array([0.0, 0.0, 3.0])))


class TestRmsse:
    def test_perfect_forecast_is_zero(self):
        a = np.array([3.0, 4.0, 5.0])
        assert rmsse(a, a.copy(), scale=2.0) == pytest.approx(0.0)

    def test_matches_manual_formula(self):
        a = np.array([2.0, 4.0])
        f = np.array([4.0, 4.0])   # errors: -2, 0 -> RMS = sqrt(4/2) = sqrt(2)
        assert rmsse(a, f, scale=2.0) == pytest.approx(np.sqrt(2.0) / 2.0)

    def test_naive_grade_model_scores_about_one(self):
        """The property that makes WRMSSE interpretable: a forecast whose RMS error
        equals the in-sample naive error scores ~1.0."""
        rng = np.random.default_rng(0)
        train = np.cumsum(rng.normal(0, 1, 400)) + 50
        scale = naive_scale(np.abs(train))
        actual = np.array([10.0] * 28)
        forecast = actual + rng.normal(0, scale, 28)
        assert 0.6 < rmsse(actual, forecast, scale) < 1.6

    def test_undefined_scale_propagates(self):
        assert np.isnan(rmsse(np.array([1.0]), np.array([1.0]), scale=float("nan")))


def _frames(n_train=120, h=28, n_items=3, n_stores=2, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n_train + h, freq="D")
    rows = []
    for i in range(n_items):
        for s in range(n_stores):
            lvl = 5.0 + 3 * i
            qty = rng.poisson(lvl, n_train + h).astype(float)
            for d, q in zip(dates, qty):
                rows.append({
                    "item_id": f"ITEM_{i}", "store_id": f"CA_{s+1}",
                    "state_id": "CA", "cat_id": "FOODS", "dept_id": f"FOODS_{i%2+1}",
                    "date_id": d, "qty": q,
                })
    df = pd.DataFrame(rows)
    split = dates[n_train]
    train = df[df["date_id"] < split].copy()
    actual = df[df["date_id"] >= split].copy()
    price = df[["item_id", "store_id", "date_id"]].copy()
    price["sell_price"] = 2.0
    return train, actual, price


class TestComputeWrmsse:
    def test_perfect_forecast_scores_zero(self):
        train, actual, price = _frames()
        fc = actual.rename(columns={"qty": "yhat"})
        out = compute_wrmsse(train, actual, fc, price)
        assert out["wrmsse"] == pytest.approx(0.0, abs=1e-9)

    def test_worse_forecast_scores_higher(self):
        train, actual, price = _frames()
        good = actual.rename(columns={"qty": "yhat"}).copy()
        bad = good.copy()
        bad["yhat"] = bad["yhat"] + 5.0
        assert compute_wrmsse(train, actual, bad, price)["wrmsse"] > \
               compute_wrmsse(train, actual, good, price)["wrmsse"]

    def test_reports_every_level_and_flags_degenerate_ones(self):
        train, actual, price = _frames()
        out = compute_wrmsse(train, actual, actual.rename(columns={"qty": "yhat"}), price)
        assert [lv["level"] for lv in out["levels"]] == [n for n, _ in LEVELS]
        by_name = {lv["level"]: lv for lv in out["levels"]}
        # Single-state, single-category fixture: these collapse to one group and are
        # flagged, so a subset run can be quoted honestly rather than as a leaderboard score.
        assert by_name["total"]["degenerate"] is True
        assert by_name["state"]["degenerate"] is True
        assert by_name["item_store"]["degenerate"] is False
        assert by_name["item_store"]["n_series"] == 6

    def test_weights_are_revenue_not_units(self):
        """Doubling one item's price must change the score: M5 weights are revenue-based,
        so a unit-weighted implementation would be insensitive to this."""
        train, actual, price = _frames()
        fc = actual.rename(columns={"qty": "yhat"}).copy()
        fc.loc[fc["item_id"] == "ITEM_0", "yhat"] += 4.0   # concentrate error on ITEM_0
        base = compute_wrmsse(train, actual, fc, price)["wrmsse"]

        pricier = price.copy()
        pricier.loc[pricier["item_id"] == "ITEM_0", "sell_price"] = 20.0
        assert compute_wrmsse(train, actual, fc, pricier)["wrmsse"] > base
