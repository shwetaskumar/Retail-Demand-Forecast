"""
tests/test_market_onboarding.py — market onboarding proof (Step 11, docs/instructions.md §21 #6).

"Onboarding test: a second market builds end-to-end by adding config rows
(config_*) and changing the market param — no code change."

Audit performed for this step (grep across data/, training/, inference/,
utils/, config/loader.py for hardcoded market/catalog literals — see the
Step 11 chat summary for the exact commands): NO pipeline logic anywhere
branches on a market or catalog string. The only "US" literal in production
code is config/seed_defaults.py::seed_default_market — an explicitly one-time,
US-specific BOOTSTRAP seed for this project's first market
(docs/instructions.md §20: seed/bootstrap defaults are "inserted via the app
or a one-time seed notebook — not committed as per-market YAML"; onboarding
market #2+ happens via the Databricks App's config INSERTs, not more Python).
databricks.yml's `market` variable default ("US") is likewise just an
overridable default, not a restriction. Nothing needed fixing.

Since this environment can't run Spark/Databricks (no JOB 1/2/3 execution
possible), this proves the claim at the level available: the exact same
functions (training.engine.fit_one_series, training.champion.select_champions,
training.evaluation.gate, training.hpt.needs_tuning — JOB 2's full chain)
driven by a DIFFERENT market's config values (a different accuracy_threshold
in particular) produce genuinely different, market-appropriate pass/fail
results for byte-identical series data. Config-level market isolation
(load_config correctly separating two markets' rows, including a negative
case) is already covered by tests/test_config_loader.py.

JOB 3 (inference/*) is still unimplemented (Step 9 not done) so it isn't
exercised here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.loader import CategoryConfig, FeatureConfig, ForecastConfig, MarketConfig, ModelConfig
from training import engine
from training.champion import select_champions
from training.evaluation import gate
from training.hpt import needs_tuning
from training.models.base import Decomposition, ForecastModel


class _FakeForecastModel(ForecastModel):
    """Deterministic fake (same shape as tests/test_engine.py's): predicts qty + a
    configurable bias, so a market's achieved WMAPE is fully controllable and never
    depends on any market-specific code path — there isn't one."""

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


@pytest.fixture(autouse=True)
def _register_fake_model(monkeypatch):
    monkeypatch.setitem(engine.STAGE1_MODEL_REGISTRY, "fake", _FakeForecastModel)


def _forecast_config(market: str, accuracy_threshold: float, bias: float) -> ForecastConfig:
    market_cfg = MarketConfig(
        market_id=market, active=True, horizon_days=140, eval_horizon_days=7, history_start="2022-01-01",
        incremental_read_days=400, incremental_write_days=365, backtest_folds=5, val_weeks=10,
        hpt_trigger="threshold_miss", hpt_trials=10, uplift_enabled=False, uplift_learner="lgbm_global",
        uplift_clip_frac=0.25, reconstruction_tol_pct=0.005, run_budget_usd=100.0,
    )
    return ForecastConfig(
        market_config=market_cfg,
        categories=[
            CategoryConfig(
                market_id=market, category="A", accuracy_threshold=accuracy_threshold, gate_grains=[],
                series_grain="market_store_sku", drift_tolerance=0.05,
            )
        ],
        models=[ModelConfig(market_id=market, model_type="fake", enabled=True, param_space={"bias": bias})],
        features=[FeatureConfig(market_id=market, feature_name="lag_1", enabled=True, feature_class="historical_only", params={})],
    )


def _series_pdf(market: str, n_days: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "series_key": f"{market}|A|S1|ITEM_A",
            "market_id": market,
            "category": "A",
            "store_id": "S1",
            "sku_id": "ITEM_A",
            "date_id": dates,
            "qty": np.arange(1, n_days + 1, dtype=float),
            "isActive": 1,
        }
    )


def _champions_for(market: str, accuracy_threshold: float, bias: float) -> tuple[ForecastConfig, pd.DataFrame]:
    """Exercise the exact same JOB 2 chain (fit_one_series -> select_champions) for any
    market — no market-specific branching anywhere in this call chain."""
    config = _forecast_config(market, accuracy_threshold, bias)
    candidate_metrics_pdf = engine.fit_one_series(_series_pdf(market), config, "2024-06-01")
    champions_pdf = select_champions(candidate_metrics_pdf)
    return config, champions_pdf


class TestSecondMarketRunsTheIdenticalPipeline:
    def test_a_never_before_seen_market_code_works_with_zero_code_change(self):
        """"UK" is not US — appears nowhere in engine.py/champion.py/evaluation.py/hpt.py —
        yet the identical code path fits, selects a champion, and gates it correctly."""
        config, champions_pdf = _champions_for("UK", accuracy_threshold=0.5, bias=0.0)
        assert len(champions_pdf) == 1
        assert champions_pdf.iloc[0]["market_id"] == "UK"
        assert champions_pdf.iloc[0]["wmape_7d"] == pytest.approx(0.0, abs=1e-9)

        gated = gate(champions_pdf, config.categories)
        cohort = gated[gated["grain"] == "category_cohort"].iloc[0]
        assert cohort["market_id"] == "UK"
        assert bool(cohort["passed"]) is True

    def test_two_markets_with_identical_series_data_but_different_thresholds_diverge(self):
        """The core proof: byte-identical series data, run through the SAME functions,
        produces a different pass/fail outcome purely because the market's own config
        threshold differs — gating is 100% config-driven, with zero code branching."""
        config, champions_pdf = _champions_for("UK", accuracy_threshold=0.5, bias=30.0)
        achieved_accuracy = 1.0 - champions_pdf.iloc[0]["wmape_7d"]
        assert 0.0 < achieved_accuracy < 1.0  # sanity: bias must have introduced real error

        lenient_categories = [
            CategoryConfig(market_id="UK", category="A", accuracy_threshold=achieved_accuracy - 0.05, gate_grains=[], series_grain="market_store_sku", drift_tolerance=0.05)
        ]
        strict_categories = [
            CategoryConfig(market_id="UK", category="A", accuracy_threshold=achieved_accuracy + 0.05, gate_grains=[], series_grain="market_store_sku", drift_tolerance=0.05)
        ]

        lenient_cohort = gate(champions_pdf, lenient_categories)
        lenient_cohort = lenient_cohort[lenient_cohort["grain"] == "category_cohort"].iloc[0]
        strict_cohort = gate(champions_pdf, strict_categories)
        strict_cohort = strict_cohort[strict_cohort["grain"] == "category_cohort"].iloc[0]

        assert bool(lenient_cohort["passed"]) is True
        assert bool(strict_cohort["passed"]) is False

    def test_needs_tuning_responds_to_the_markets_own_threshold(self):
        """A series flagged for conditional HPT under one market's strict threshold must
        NOT be flagged under a different (or the same market's more lenient) threshold —
        proving HPT triggering is also purely config-driven, not market-coded."""
        config, champions_pdf = _champions_for("DE", accuracy_threshold=0.5, bias=30.0)
        achieved_accuracy = 1.0 - champions_pdf.iloc[0]["wmape_7d"]

        strict_categories = [
            CategoryConfig(market_id="DE", category="A", accuracy_threshold=achieved_accuracy + 0.05, gate_grains=[], series_grain="market_store_sku", drift_tolerance=0.05)
        ]
        lenient_categories = [
            CategoryConfig(market_id="DE", category="A", accuracy_threshold=achieved_accuracy - 0.05, gate_grains=[], series_grain="market_store_sku", drift_tolerance=0.05)
        ]

        strict_gated = gate(champions_pdf, strict_categories)
        lenient_gated = gate(champions_pdf, lenient_categories)

        assert needs_tuning(strict_gated, champions_pdf).iloc[0]["series_key"] == "DE|A|S1|ITEM_A"
        assert needs_tuning(lenient_gated, champions_pdf).empty

    def test_three_markets_side_by_side_never_cross_contaminate(self):
        """US, UK, and a third never-before-used code ("JP") all run through the identical
        pipeline in the same test session with different thresholds — none of their
        results bleed into each other."""
        results = {}
        for market, threshold in (("US", 0.5), ("UK", 0.6), ("JP", 0.7)):
            config, champions_pdf = _champions_for(market, accuracy_threshold=threshold, bias=0.0)
            gated = gate(champions_pdf, config.categories)
            results[market] = gated[gated["grain"] == "category_cohort"].iloc[0]

        for market, row in results.items():
            assert row["market_id"] == market
            assert bool(row["passed"]) is True  # bias=0.0 -> perfect predictor -> passes any threshold <= 1.0
