"""
tests/test_hpt.py — conditional HPT unit tests (Step 8, docs/instructions.md §12).

`needs_tuning` (the "selection" BUILD_PROMPTS.md Step 8 asks to unit-test) is
pure pandas. `tune_series` uses the real, already-installed Optuna (no mock —
proving the search actually converges) against a fake, deterministic
ForecastModel (same trick as tests/test_backtest.py). `tune_one_series` is
exercised directly on pandas with a monkeypatched candidate registry, mirroring
tests/test_engine.py's approach for training/engine.py::fit_one_series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import engine
from training.hpt import needs_tuning, select_improved_retunes, tune_one_series, tune_series
from training.models.base import Decomposition, ForecastModel


# --- shared fake model (same shape as tests/test_backtest.py's / tests/test_engine.py's) ---


class _FakeForecastModel(ForecastModel):
    model_type = "fake"

    def __init__(self) -> None:
        self._bias = 0.0

    def fit(self, series_pdf, known_future_pdf, params):
        self._bias = float(params.get("bias", 0.0))
        return self

    def predict(self, horizon_pdf):
        return horizon_pdf["qty"].astype(float) + self._bias

    def decompose(self, horizon_pdf):
        yhat = self.predict(horizon_pdf)
        zeros = pd.Series(0.0, index=horizon_pdf.index)
        return Decomposition(
            trend=yhat, seasonality_weekly=zeros, seasonality_yearly=zeros,
            holiday_lift={}, promo_lift={}, temporal_features_effect=zeros, yhat=yhat,
        )

    def fitted_values(self):
        raise NotImplementedError


class _AlwaysFailingModel(ForecastModel):
    model_type = "always_fails"

    def fit(self, series_pdf, known_future_pdf, params):
        raise RuntimeError("synthetic HPT trial failure")

    def predict(self, horizon_pdf):
        raise NotImplementedError

    def decompose(self, horizon_pdf):
        raise NotImplementedError


def _series_pdf(n_days: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {"store_id": "S1", "sku_id": "ITEM_A", "date_id": dates, "qty": np.arange(1, n_days + 1, dtype=float), "isActive": 1}
    )


# --- needs_tuning ---


def _gate_row(market, category, grain, entity_key, passed, wmape=0.5, volume=100.0, threshold=0.85):
    return {
        "market_id": market, "category": category, "grain": grain, "entity_key": entity_key,
        "horizon": 7, "wmape": wmape, "volume": volume, "threshold": threshold, "passed": passed,
    }


def _champion_row(series_key, market, category, sku_id, model_type="fake", wmape_7d=0.5, volume=100.0):
    return {
        "series_key": series_key, "market_id": market, "category": category, "sku_id": sku_id,
        "store_id": "S1", "model_type": model_type, "params": "{}", "wmape_7d": wmape_7d, "volume": volume,
    }


class TestNeedsTuning:
    def test_series_grain_failure_triggers_tuning(self):
        gated = pd.DataFrame(
            [
                _gate_row("US", "A", "market_store_sku", "US|A|S1|ITEM1", passed=False),
                _gate_row("US", "A", "market_store_sku", "US|A|S2|ITEM2", passed=True),
                _gate_row("US", "A", "market_sku", "ITEM1", passed=True),
                _gate_row("US", "A", "market_sku", "ITEM2", passed=True),
                _gate_row("US", "A", "category_cohort", "A", passed=True),
            ]
        )
        champions = pd.DataFrame(
            [
                _champion_row("US|A|S1|ITEM1", "US", "A", "ITEM1"),
                _champion_row("US|A|S2|ITEM2", "US", "A", "ITEM2"),
            ]
        )
        result = needs_tuning(gated, champions)
        assert set(result["series_key"]) == {"US|A|S1|ITEM1"}

    def test_market_sku_grain_failure_triggers_tuning_for_every_series_of_that_sku(self):
        gated = pd.DataFrame(
            [
                _gate_row("US", "A", "market_store_sku", "US|A|S1|ITEM1", passed=True),
                _gate_row("US", "A", "market_store_sku", "US|A|S2|ITEM1", passed=True),
                _gate_row("US", "A", "market_sku", "ITEM1", passed=False),  # aggregate fails
                _gate_row("US", "A", "category_cohort", "A", passed=True),
            ]
        )
        champions = pd.DataFrame(
            [
                _champion_row("US|A|S1|ITEM1", "US", "A", "ITEM1"),
                _champion_row("US|A|S2|ITEM1", "US", "A", "ITEM1"),
            ]
        )
        result = needs_tuning(gated, champions)
        # both series pass individually, but the sku-level aggregate failed -> both tuned
        assert set(result["series_key"]) == {"US|A|S1|ITEM1", "US|A|S2|ITEM1"}

    def test_category_cohort_failure_alone_does_not_trigger_tuning(self):
        """Deliberately narrowed: a category_cohort-only miss no longer sweeps in
        individually-passing series (see needs_tuning's docstring — full HPT is
        n_trials x backtest_folds fits/series, too costly to spend chasing a
        cohort aggregate that's already close, e.g. 80% vs an 85% target)."""
        gated = pd.DataFrame(
            [
                _gate_row("US", "A", "market_store_sku", "US|A|S1|ITEM1", passed=True),
                _gate_row("US", "A", "market_store_sku", "US|A|S1|ITEM2", passed=True),
                _gate_row("US", "A", "market_sku", "ITEM1", passed=True),
                _gate_row("US", "A", "market_sku", "ITEM2", passed=True),
                _gate_row("US", "A", "category_cohort", "A", passed=False),  # cohort aggregate fails
            ]
        )
        champions = pd.DataFrame(
            [
                _champion_row("US|A|S1|ITEM1", "US", "A", "ITEM1"),
                _champion_row("US|A|S1|ITEM2", "US", "A", "ITEM2"),
            ]
        )
        assert needs_tuning(gated, champions).empty

    def test_no_tuning_when_everything_passes(self):
        gated = pd.DataFrame(
            [
                _gate_row("US", "A", "market_store_sku", "US|A|S1|ITEM1", passed=True),
                _gate_row("US", "A", "market_sku", "ITEM1", passed=True),
                _gate_row("US", "A", "category_cohort", "A", passed=True),
            ]
        )
        champions = pd.DataFrame([_champion_row("US|A|S1|ITEM1", "US", "A", "ITEM1")])
        assert needs_tuning(gated, champions).empty

    def test_null_passed_does_not_trigger_tuning(self):
        gated = pd.DataFrame(
            [
                _gate_row("US", "A", "market_store_sku", "US|A|S1|ITEM1", passed=None),
                _gate_row("US", "A", "market_sku", "ITEM1", passed=None),
                _gate_row("US", "A", "category_cohort", "A", passed=None),
            ]
        )
        champions = pd.DataFrame([_champion_row("US|A|S1|ITEM1", "US", "A", "ITEM1")])
        assert needs_tuning(gated, champions).empty

    def test_empty_inputs_return_empty_frame(self):
        assert needs_tuning(pd.DataFrame(), pd.DataFrame()).empty


# --- tune_series (real Optuna) ---


class TestTuneSeries:
    def test_finds_a_near_optimal_bias(self):
        series_pdf = _series_pdf()
        param_space = {"bias": {"type": "int", "low": -5, "high": 5}}
        best_params = tune_series(
            _FakeForecastModel, series_pdf, param_space, n_trials=25, n_folds=5, val_weeks=10, eval_horizon_days=7
        )
        assert "bias" in best_params
        # re-run backtest with the chosen params and confirm it's genuinely good (true optimum is bias=0)
        from training.backtest import backtest_model

        score = backtest_model(_FakeForecastModel(), series_pdf, 5, 10, 7, best_params)
        assert score < 0.05

    def test_empty_param_space_returns_empty_dict(self):
        result = tune_series(_FakeForecastModel, _series_pdf(), {}, n_trials=10, n_folds=5, val_weeks=10, eval_horizon_days=7)
        assert result == {}

    def test_all_trials_failing_returns_empty_dict_without_crashing(self):
        param_space = {"bias": {"type": "int", "low": -5, "high": 5}}
        result = tune_series(
            _AlwaysFailingModel, _series_pdf(), param_space, n_trials=5, n_folds=5, val_weeks=10, eval_horizon_days=7
        )
        assert result == {}

    def test_params_transform_is_applied_before_backtesting(self):
        """A transform that always maps to bias=0 (the true optimum) regardless of what Optuna
        suggests must yield a near-zero score even with a param_space that's otherwise way off."""
        param_space = {"bias": {"type": "int", "low": 100, "high": 200}}  # deliberately bad range
        best_params = tune_series(
            _FakeForecastModel,
            _series_pdf(),
            param_space,
            n_trials=5,
            n_folds=5,
            val_weeks=10,
            eval_horizon_days=7,
            params_transform=lambda raw: {"bias": 0.0},
        )
        from training.backtest import backtest_model

        score = backtest_model(_FakeForecastModel(), _series_pdf(), 5, 10, 7, {"bias": 0.0})
        assert score == pytest.approx(0.0, abs=1e-9)
        assert "bias" in best_params  # the raw (pre-transform) suggestion is still returned


# --- select_improved_retunes ---


class TestSelectImprovedRetunes:
    def test_keeps_only_ok_passed_and_strictly_improved_rows(self):
        original = pd.DataFrame(
            [
                _champion_row("S1", "US", "A", "ITEM1", wmape_7d=0.30),
                _champion_row("S2", "US", "A", "ITEM2", wmape_7d=0.20),
            ]
        )
        retuned = pd.DataFrame(
            [
                {"series_key": "S1", "wmape_7d": 0.10, "status": "ok", "passed_decomposition": True},  # improved
                {"series_key": "S2", "wmape_7d": 0.25, "status": "ok", "passed_decomposition": True},  # worse
            ]
        )
        result = select_improved_retunes(retuned, original)
        assert set(result["series_key"]) == {"S1"}

    def test_error_status_is_never_kept_even_if_wmape_looks_better(self):
        original = pd.DataFrame([_champion_row("S1", "US", "A", "ITEM1", wmape_7d=0.30)])
        retuned = pd.DataFrame([{"series_key": "S1", "wmape_7d": 0.01, "status": "error:boom", "passed_decomposition": False}])
        assert select_improved_retunes(retuned, original).empty

    def test_failed_decomposition_is_never_kept(self):
        original = pd.DataFrame([_champion_row("S1", "US", "A", "ITEM1", wmape_7d=0.30)])
        retuned = pd.DataFrame([{"series_key": "S1", "wmape_7d": 0.01, "status": "ok", "passed_decomposition": False}])
        assert select_improved_retunes(retuned, original).empty

    def test_series_missing_from_original_is_excluded(self):
        original = pd.DataFrame([_champion_row("S1", "US", "A", "ITEM1", wmape_7d=0.30)])
        retuned = pd.DataFrame([{"series_key": "S2", "wmape_7d": 0.01, "status": "ok", "passed_decomposition": True}])
        assert select_improved_retunes(retuned, original).empty

    def test_empty_retuned_returns_empty(self):
        original = pd.DataFrame([_champion_row("S1", "US", "A", "ITEM1", wmape_7d=0.30)])
        assert select_improved_retunes(pd.DataFrame(), original).empty


# --- tune_one_series orchestration ---


@pytest.fixture(autouse=True)
def _register_fake_models(monkeypatch):
    monkeypatch.setitem(engine.STAGE1_MODEL_REGISTRY, "fake", _FakeForecastModel)


def _forecast_config_with_fake(uplift_enabled: bool = False):
    from config.loader import CategoryConfig, FeatureConfig, ForecastConfig, MarketConfig, ModelConfig

    market_cfg = MarketConfig(
        market_id="US", active=True, horizon_days=140, eval_horizon_days=7, history_start="2022-01-01",
        incremental_read_days=400, incremental_write_days=365, backtest_folds=5, val_weeks=10,
        hpt_trigger="threshold_miss", hpt_trials=10, uplift_enabled=uplift_enabled, uplift_learner="lgbm_global",
        uplift_clip_frac=0.25, reconstruction_tol_pct=0.005, run_budget_usd=100.0,
    )
    return ForecastConfig(
        market_config=market_cfg,
        categories=[CategoryConfig(
            market_id="US", category="A", accuracy_threshold=0.85, gate_grains=[],
            series_grain="market_store_sku", drift_tolerance=0.05,
        )],
        models=[ModelConfig(market_id="US", model_type="fake", enabled=True, param_space={"bias": {"type": "int", "low": -3, "high": 3}})],
        features=[FeatureConfig(market_id="US", feature_name="lag_1", enabled=True, feature_class="historical_only", params={})],
    )


class TestTuneOneSeries:
    def test_retunes_and_reevaluates_with_engine_result_shape(self):
        config = _forecast_config_with_fake()
        pdf = _series_pdf().assign(
            series_key="US|A|S1|ITEM_A", market_id="US", category="A", champion_model_type="fake"
        )
        result = tune_one_series(pdf, config, "2024-06-01")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["model_type"] == "fake"
        assert row["status"] == "ok"
        assert set(result.columns) == set(engine.OUTPUT_SCHEMA_COLUMNS)

    def test_missing_param_space_short_circuits(self):
        from config.loader import ForecastConfig, ModelConfig

        config = _forecast_config_with_fake()
        config = ForecastConfig(
            market_config=config.market_config,
            categories=config.categories,
            models=[ModelConfig(market_id="US", model_type="fake", enabled=True, param_space={})],
            features=config.features,
        )
        pdf = _series_pdf().assign(series_key="US|A|S1|ITEM_A", market_id="US", category="A", champion_model_type="fake")
        result = tune_one_series(pdf, config, "2024-06-01")
        assert result.iloc[0]["status"] == "no_tunable_param_space"

    def test_unknown_champion_model_type_short_circuits(self):
        config = _forecast_config_with_fake()
        pdf = _series_pdf().assign(series_key="US|A|S1|ITEM_A", market_id="US", category="A", champion_model_type="nonexistent")
        result = tune_one_series(pdf, config, "2024-06-01")
        assert result.iloc[0]["status"] == "no_tunable_param_space"
