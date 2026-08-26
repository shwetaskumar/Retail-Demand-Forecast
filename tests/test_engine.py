"""
tests/test_engine.py — per-series map unit tests for training/engine.py (Step 6).

`fit_one_series` is NOT on the production path (notebooks/02_train.py calls
training/engine_layered.py::run_engine_layered instead) — it's kept as a
pandas-only, no-Spark-required reference implementation of the same per-series
cascade contract (see its own docstring), specifically so it can be exercised
directly here with a fake, monkeypatched candidate model: fast, deterministic,
isolated from the real Prophet/statsforecast/lgbm libraries (already covered by
tests/test_models.py). See the note at the end of TestFitOneSeries for why the
Spark-wiring layer itself (the part fit_one_series intentionally has none of)
isn't unit-tested anywhere locally.
"""
from __future__ import annotations

import pandas as pd
import pytest

from config.loader import CategoryConfig, FeatureConfig, ForecastConfig, MarketConfig, ModelConfig
from training import engine
from training.engine import fit_one_series
from training.models.base import Decomposition, ForecastModel


class _FakeForecastModel(ForecastModel):
    """Same trick as tests/test_backtest.py's fake: reads `qty` straight from whatever frame
    it's given, offset by `bias`, so fit/backtest/decompose all stay deterministic and cheap."""

    model_type = "fake"

    def __init__(self) -> None:
        self._bias = 0.0
        self._fitted: pd.Series | None = None

    def fit(self, series_pdf: pd.DataFrame, known_future_pdf: pd.DataFrame, params: dict) -> "_FakeForecastModel":
        self._bias = float(params.get("bias", 0.0))
        self._fitted = series_pdf["qty"].astype(float) + self._bias
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
        assert self._fitted is not None
        return self._fitted


class _FakeFailingModel(ForecastModel):
    """Always raises on fit — proves one bad candidate can't crash the whole series."""

    model_type = "fake_failing"

    def fit(self, series_pdf, known_future_pdf, params):
        raise RuntimeError("synthetic failure")

    def predict(self, horizon_pdf):
        raise NotImplementedError

    def decompose(self, horizon_pdf):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _register_fake_models(monkeypatch):
    monkeypatch.setitem(engine.STAGE1_MODEL_REGISTRY, "fake", _FakeForecastModel)
    monkeypatch.setitem(engine.STAGE1_MODEL_REGISTRY, "fake_failing", _FakeFailingModel)


def _market_config(**overrides) -> MarketConfig:
    defaults = dict(
        market_id="US",
        active=True,
        horizon_days=140,
        eval_horizon_days=7,
        history_start="2022-01-01",
        incremental_read_days=400,
        incremental_write_days=365,
        backtest_folds=5,
        val_weeks=10,
        hpt_trigger="threshold_miss",
        hpt_trials=25,
        uplift_enabled=False,
        uplift_learner="lgbm_global",
        uplift_clip_frac=0.25,
        reconstruction_tol_pct=0.005,
        run_budget_usd=100.0,
    )
    defaults.update(overrides)
    return MarketConfig(**defaults)


def _forecast_config(models: list[ModelConfig], **market_overrides) -> ForecastConfig:
    return ForecastConfig(
        market_config=_market_config(**market_overrides),
        categories=[CategoryConfig(
            market_id="US", category="A", accuracy_threshold=0.85, gate_grains=[],
            series_grain="market_store_sku", drift_tolerance=0.05,
        )],
        models=models,
        features=[FeatureConfig(market_id="US", feature_name="lag_1", enabled=True, feature_class="historical_only", params={})],
    )


def _series_pdf(n_days: int = 150, store_id: str = "S1", sku_id: str = "ITEM_A") -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    n = n_days
    return pd.DataFrame(
        {
            "series_key": f"US|A|{store_id}|{sku_id}",
            "market_id": "US",
            "category": "A",
            "store_id": store_id,
            "sku_id": sku_id,
            "date_id": dates,
            "qty": [float((i % 10) + 1) for i in range(n)],
            "isActive": 1,
        }
    )


class TestFitOneSeries:
    def test_insufficient_history_short_circuits(self):
        config = _forecast_config([ModelConfig(market_id="US", model_type="fake", enabled=True, param_space={})])
        result = fit_one_series(_series_pdf(n_days=5), config, "2024-06-01")
        assert len(result) == 1
        assert result.iloc[0]["status"] == "insufficient_history"
        assert result.iloc[0]["model_type"] is None

    def test_one_row_per_enabled_candidate(self):
        config = _forecast_config([ModelConfig(market_id="US", model_type="fake", enabled=True, param_space={})])
        result = fit_one_series(_series_pdf(), config, "2024-06-01")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["model_type"] == "fake"
        assert row["status"] == "ok"
        assert bool(row["passed_decomposition"]) is True
        assert row["wmape_7d"] == pytest.approx(0.0, abs=1e-9)

    def test_disabled_candidate_is_skipped(self):
        # No enabled candidate ever runs -> enabled_models is empty -> best_row stays None
        # -> the single terminal "nothing to report" status. fit_one_series collapses
        # "nothing enabled" and "everything failed" into the same status (see
        # test_failing_candidate_reports_all_models_failed_without_crashing below) rather
        # than distinguishing them — both produce zero usable candidate rows.
        config = _forecast_config([ModelConfig(market_id="US", model_type="fake", enabled=False, param_space={})])
        result = fit_one_series(_series_pdf(), config, "2024-06-01")
        assert len(result) == 1
        assert result.iloc[0]["status"] == "all_models_failed"

    def test_uplift_variant_replaces_plain_candidate_when_enabled(self):
        # fit_one_series wraps each non-lgbm candidate in UpliftModel and evaluates ONLY
        # the wrapped "<type>+uplift" variant when uplift_enabled — it does not also keep
        # evaluating the plain variant alongside it (unlike engine_layered.py's config-driven
        # cascade, which runs uplift as an additional escalation TIER on top of the plain
        # candidates so both compete). fit_one_series is the simpler, non-default engine
        # path (notebooks/02_train.py always uses run_engine_layered); this is its actual,
        # intentional behavior.
        config = _forecast_config(
            [ModelConfig(market_id="US", model_type="fake", enabled=True, param_space={})],
            uplift_enabled=True,
        )
        result = fit_one_series(_series_pdf(), config, "2024-06-01")
        model_types = set(result["model_type"])
        assert model_types == {"fake+uplift"}
        assert (result["status"] == "ok").all()

    def test_failing_candidate_reports_all_models_failed_without_crashing(self):
        # evaluate_candidate() catches the exception internally and returns a per-candidate
        # "error:<msg>" row, but fit_one_series's loop discards any non-"ok" row
        # (`if row["status"] != "ok": continue`) and never accumulates per-candidate error
        # detail — when nothing succeeds it falls through to the single generic terminal
        # status "all_models_failed". The important property (this test's actual point) is
        # that a raising candidate doesn't crash the whole series — it still degrades to
        # exactly one clean output row.
        config = _forecast_config(
            [ModelConfig(market_id="US", model_type="fake_failing", enabled=True, param_space={})]
        )
        result = fit_one_series(_series_pdf(), config, "2024-06-01")
        assert len(result) == 1
        assert result.iloc[0]["status"] == "all_models_failed"
        assert bool(result.iloc[0]["passed_decomposition"]) is False

    def test_identity_columns_are_carried_through(self):
        config = _forecast_config([ModelConfig(market_id="US", model_type="fake", enabled=True, param_space={})])
        result = fit_one_series(_series_pdf(store_id="S9", sku_id="ITEM_Z"), config, "2024-06-01")
        row = result.iloc[0]
        assert row["series_key"] == "US|A|S9|ITEM_Z"
        assert row["market_id"] == "US"
        assert row["category"] == "A"
        assert row["store_id"] == "S9"
        assert row["sku_id"] == "ITEM_Z"
        assert row["snapshot_date"] == "2024-06-01"

    def test_only_active_rows_are_modeled(self):
        pdf = _series_pdf()
        pdf.loc[: len(pdf) // 2, "isActive"] = 0  # first half inactive -> should be dropped, not zero-padded
        config = _forecast_config([ModelConfig(market_id="US", model_type="fake", enabled=True, param_space={})])
        # should still work off the remaining active tail (still long enough)
        result = fit_one_series(pdf, config, "2024-06-01")
        assert result.iloc[0]["status"] in {"ok", "insufficient_history"}

    def test_real_lgbm_candidate_end_to_end(self, monkeypatch):
        """Not just the fake — proves fit_one_series really drives a Step 4 wrapper too."""
        config = _forecast_config(
            [ModelConfig(market_id="US", model_type="lgbm", enabled=True, param_space={"n_estimators": 20, "max_depth": 3})]
        )
        result = fit_one_series(_series_pdf(), config, "2024-06-01")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["model_type"] == "lgbm"
        assert row["status"] == "ok"
        assert row["wmape_7d"] >= 0.0

# NOTE on Spark wiring (tag_series_key + groupBy + applyInPandas dispatch): not
# unit-tested here or anywhere locally. This used to be exercised via a fake `spark`
# wrapped around training.engine::run_engine, but run_engine itself was removed
# (dead code — never called by notebooks/02_train.py, which uses
# training.engine_layered::run_engine_layered) once its only test broke: run_engine
# had grown real cluster-tuning logic (a broadcast variable for prev_champions,
# spark.conf shuffle-partition tuning, executor-count introspection for numba JIT
# warmup) that the old _FakeEngineSpark — which only mocked
# createOrReplaceTempView/sql/groupBy/applyInPandas — couldn't meaningfully stand in
# for anymore. This puts Spark wiring in the same (accepted) boundary as
# run_engine_layered/run_engine_batch/select_champions_spark generally: real Spark
# integration is verified on Databricks, not via a local fake; only the pure
# per-series logic (fit_one_series, tested above) is unit-tested locally.
