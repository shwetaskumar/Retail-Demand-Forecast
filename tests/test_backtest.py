"""
tests/test_backtest.py — rolling-origin fold logic unit tests (Step 6, docs/instructions.md §11).

Pure pandas — no Spark, no real forecasting libraries. Uses a small
FakeForecastModel test double so backtest_model's averaging/wiring is
verified in isolation from Prophet/statsforecast/lgbm (already covered by
tests/test_models.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.backtest import BASELINE_WINDOW_DAYS, backtest_metrics, backtest_model, baseline_wmape, rolling_origin_folds
from training.models.base import Decomposition, ForecastModel


class _FakeForecastModel(ForecastModel):
    """Reads `qty` straight from whatever frame it's given, offset by `bias` — lets tests make
    backtest_model's error deterministic without fitting a real model."""

    model_type = "fake"

    def __init__(self) -> None:
        self._bias = 0.0

    def fit(self, series_pdf: pd.DataFrame, known_future_pdf: pd.DataFrame, params: dict) -> "_FakeForecastModel":
        self._bias = float(params.get("bias", 0.0))
        return self

    def predict(self, horizon_pdf: pd.DataFrame) -> pd.Series:
        return horizon_pdf["qty"].astype(float) + self._bias

    def decompose(self, horizon_pdf: pd.DataFrame) -> Decomposition:
        yhat = self.predict(horizon_pdf)
        zeros = pd.Series(0.0, index=horizon_pdf.index)
        return Decomposition(
            trend=yhat,
            seasonality_weekly=zeros,
            seasonality_yearly=zeros,
            holiday_lift={},
            promo_lift={},
            temporal_features_effect=zeros,
            yhat=yhat,
        )

    def fitted_values(self) -> pd.Series:
        raise NotImplementedError


def _series_pdf(n_days: int, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "store_id": "S1",
            "sku_id": "ITEM_A",
            "date_id": dates,
            "qty": np.arange(1, n_days + 1, dtype=float),
            "isActive": 1,
        }
    )


class TestRollingOriginFolds:
    def test_folds_are_point_in_time(self):
        folds = rolling_origin_folds(_series_pdf(150), n_folds=5, val_weeks=10, eval_horizon_days=7)
        assert len(folds) >= 1
        for train_pdf, holdout_pdf in folds:
            assert train_pdf["date_id"].max() < holdout_pdf["date_id"].min()
            assert len(holdout_pdf) == 7
            assert (holdout_pdf["date_id"].max() - holdout_pdf["date_id"].min()).days == 6

    def test_at_most_n_folds(self):
        folds = rolling_origin_folds(_series_pdf(150), n_folds=5, val_weeks=10, eval_horizon_days=7)
        assert len(folds) <= 5

    def test_fold_origins_increase_monotonically(self):
        folds = rolling_origin_folds(_series_pdf(150), n_folds=5, val_weeks=10, eval_horizon_days=7)
        origins = [train_pdf["date_id"].max() for train_pdf, _ in folds]
        assert origins == sorted(origins)
        assert len(set(origins)) == len(origins)  # strictly increasing, no duplicate origins

    def test_empty_series_returns_no_folds(self):
        empty = pd.DataFrame(columns=["store_id", "sku_id", "date_id", "qty", "isActive"])
        assert rolling_origin_folds(empty, n_folds=5, val_weeks=10, eval_horizon_days=7) == []

    def test_too_short_series_returns_no_folds(self):
        folds = rolling_origin_folds(_series_pdf(5), n_folds=5, val_weeks=10, eval_horizon_days=7)
        assert folds == []

    def test_short_series_still_yields_at_least_one_fold(self):
        # 20 days of history, far short of val_weeks=10 (70 days) — origin clipping must still
        # produce the one fold that fits (last eval_horizon_days as holdout).
        folds = rolling_origin_folds(_series_pdf(20), n_folds=5, val_weeks=10, eval_horizon_days=7)
        assert len(folds) >= 1


class TestBacktestModel:
    def test_perfect_predictor_gives_zero_wmape(self):
        score = backtest_model(_FakeForecastModel(), _series_pdf(150), n_folds=5, val_weeks=10, eval_horizon_days=7, params={})
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_biased_predictor_gives_positive_wmape(self):
        score = backtest_model(
            _FakeForecastModel(), _series_pdf(150), n_folds=5, val_weeks=10, eval_horizon_days=7, params={"bias": 5.0}
        )
        assert score > 0.0
        assert np.isfinite(score)

    def test_nan_when_no_usable_folds(self):
        score = backtest_model(_FakeForecastModel(), _series_pdf(5), n_folds=5, val_weeks=10, eval_horizon_days=7, params={})
        assert np.isnan(score)

    def test_mutates_model_to_last_fold_state(self):
        """training/engine.py relies on this: after backtest_model returns, the model instance
        must reflect a fit on the LAST fold's training window (so decompose() can reuse it)."""
        model = _FakeForecastModel()
        backtest_model(model, _series_pdf(150), n_folds=5, val_weeks=10, eval_horizon_days=7, params={"bias": 2.0})
        assert model._bias == 2.0


class TestBaselineWmape:
    def test_flat_trailing_mean_forecast(self):
        train = pd.DataFrame({"qty": [0.0] * 100 + [2.0] * BASELINE_WINDOW_DAYS})
        holdout = pd.DataFrame({"qty": [4.0] * 7})
        # baseline forecasts 2/day -> weekly 14 vs actual 28 -> 50%
        assert baseline_wmape(train, holdout) == pytest.approx(0.5)

    def test_zero_volume_holdout_is_nan(self):
        train = pd.DataFrame({"qty": [1.0] * 40})
        assert np.isnan(baseline_wmape(train, pd.DataFrame({"qty": [0.0] * 7})))

    def test_backtest_metrics_reports_baseline(self):
        metrics = backtest_metrics(_FakeForecastModel(), _series_pdf(120), 3, 6, 7, {})
        assert "baseline_wmape_7d" in metrics
        assert metrics["baseline_wmape_7d"] == metrics["baseline_wmape_7d"]  # finite, not NaN

