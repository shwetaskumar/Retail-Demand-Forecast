"""
tests/test_config_loader.py — unit tests for config/loader.py.

Exercises `load_config` against a fake `spark` whose `.table(...).filter(...).toPandas()`
chain is backed by a single in-memory pandas frame (the compact config table) — no
real Spark/Delta/pyspark import required.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from config.loader import ForecastConfig, load_config, load_overrides, resolve_series_config


class _FakeConfigTable:
    """Stands in for a Spark DataFrame: only `.filter(...)` and `.toPandas()` are used by the loader."""

    def __init__(self, pdf: pd.DataFrame):
        self._pdf = pdf

    def filter(self, expr: str) -> "_FakeConfigTable":
        # loader always calls this as: "market_id = '<value>'"
        _, raw_value = expr.split("=", 1)
        value = raw_value.strip().strip("'").replace("''", "'")
        return _FakeConfigTable(self._pdf[self._pdf["market_id"] == value])

    def toPandas(self) -> pd.DataFrame:
        return self._pdf.reset_index(drop=True)


class FakeConfigSpark:
    """Stands in for `spark`: `.table("catalog.control.forecast_config")` returns the fake table."""

    def __init__(self, pdf: pd.DataFrame):
        self._table = _FakeConfigTable(pdf)

    def table(self, full_name: str) -> _FakeConfigTable:
        assert full_name.endswith(".forecast_config"), f"loader must read forecast_config, got {full_name!r}"
        return self._table


def _row(market_id, param_name, param_value, *, category=None, sku_id=None, store_id=None) -> dict:
    return {
        "market_id": market_id, "category": category, "sku_id": sku_id, "store_id": store_id,
        "param_name": param_name, "param_value": param_value,
    }


_US_MARKET_SETTINGS = json.dumps({
    "active": True, "horizon_days": 140, "eval_horizon_days": 7, "history_start": "2022-01-01",
    "uplift_clip_frac": 0.25, "reconstruction_tol_pct": 0.005,
})


def _us_config_pdf() -> pd.DataFrame:
    rows = [
        _row("US", "market_settings", _US_MARKET_SETTINGS),
        _row("US", "models", json.dumps({"prophet": 3, "lgbm": 2})),
        _row("US", "param_space_prophet", json.dumps({})),
        _row("US", "param_space_lgbm", json.dumps({"num_leaves": [16, 32]})),
        _row("US", "features_historical", json.dumps(["lag_7"])),
        _row("US", "features_known", json.dumps(["dow"])),
        _row("US", "category_thresholds", json.dumps({"A": 0.85, "B": 0.95})),
        # a different market's row must never leak into a US load
        _row("DE", "market_settings", json.dumps({
            "active": True, "horizon_days": 140, "eval_horizon_days": 7, "history_start": "2022-01-01",
        })),
        _row("DE", "category_thresholds", json.dumps({"A": 0.80})),
        # a series-level override row (should never surface via load_config, only via load_overrides):
        # this sku/store combo skips lgbm entirely and drops its cost tier.
        _row("US", "models", json.dumps({"prophet": 3}), store_id="S1", sku_id="ITEM_A"),
    ]
    return pd.DataFrame(rows)


def test_load_config_returns_typed_forecast_config():
    spark = FakeConfigSpark(_us_config_pdf())

    config = load_config(spark, catalog="retail_forecast", market_id="US")

    assert isinstance(config, ForecastConfig)
    assert config.market_config.market_id == "US"
    assert config.market_config.horizon_days == 140
    assert config.market_config.eval_horizon_days == 7
    assert config.market_config.uplift_clip_frac == 0.25
    assert config.market_config.reconstruction_tol_pct == 0.005


def test_load_config_filters_by_market():
    spark = FakeConfigSpark(_us_config_pdf())

    config = load_config(spark, catalog="retail_forecast", market_id="US")

    # the DE category row must not leak into a US load
    assert {c.category for c in config.categories} == {"A", "B"}


def test_load_config_parses_json_columns():
    spark = FakeConfigSpark(_us_config_pdf())

    config = load_config(spark, catalog="retail_forecast", market_id="US")

    lgbm = next(m for m in config.models if m.model_type == "lgbm")
    assert lgbm.param_space == {"num_leaves": [16, 32]}

    prophet = next(m for m in config.models if m.model_type == "prophet")
    assert prophet.param_space == {}


def test_load_config_tolerates_missing_defaults():
    spark = FakeConfigSpark(_us_config_pdf())

    config = load_config(spark, catalog="retail_forecast", market_id="US")

    # not present in _US_MARKET_SETTINGS -> falls back to the dataclass/JSON default
    assert config.market_config.backtest_folds == 3
    assert config.market_config.val_weeks == 4


def test_load_config_excludes_series_level_override_rows():
    """A series-level override row (store_id/sku_id set) must never leak into
    the market-wide ModelConfig/FeatureConfig lists — only load_overrides() surfaces it."""
    spark = FakeConfigSpark(_us_config_pdf())

    config = load_config(spark, catalog="retail_forecast", market_id="US")

    lgbm = next(m for m in config.models if m.model_type == "lgbm")
    assert lgbm.enabled is True  # market-wide row, not the series override


def test_load_config_reads_optional_skill_thresholds():
    """category_skill_thresholds switches named categories to the relative gate; the
    rest keep min_skill_vs_baseline=None (absolute gate)."""
    pdf = pd.concat([
        _us_config_pdf(),
        pd.DataFrame([_row("US", "category_skill_thresholds", json.dumps({"A": 0.05}))]),
    ], ignore_index=True)

    config = load_config(FakeConfigSpark(pdf), catalog="retail_forecast", market_id="US")

    by_cat = {c.category: c for c in config.categories}
    assert by_cat["A"].min_skill_vs_baseline == 0.05
    assert by_cat["B"].min_skill_vs_baseline is None


def test_load_config_without_skill_thresholds_keeps_absolute_gate():
    config = load_config(FakeConfigSpark(_us_config_pdf()), catalog="retail_forecast", market_id="US")
    assert all(c.min_skill_vs_baseline is None for c in config.categories)


def test_load_config_raises_when_market_missing():
    spark = FakeConfigSpark(_us_config_pdf())

    with pytest.raises(ValueError, match="No market-level config rows"):
        load_config(spark, catalog="retail_forecast", market_id="FR")


def test_load_overrides_returns_only_series_scoped_rows():
    spark = FakeConfigSpark(_us_config_pdf())

    overrides = load_overrides(spark, catalog="retail_forecast", market_id="US")

    assert len(overrides) == 1
    row = overrides.iloc[0]
    assert row["store_id"] == "S1"
    assert row["sku_id"] == "ITEM_A"
    assert row["param_name"] == "models"
    assert json.loads(row["param_value"]) == {"prophet": 3}


def test_load_overrides_filters_by_market():
    spark = FakeConfigSpark(_us_config_pdf())

    overrides = load_overrides(spark, catalog="retail_forecast", market_id="DE")

    assert overrides.empty


def test_resolve_series_config_applies_series_override():
    spark = FakeConfigSpark(_us_config_pdf())
    base_config = load_config(spark, catalog="retail_forecast", market_id="US")
    overrides = load_overrides(spark, catalog="retail_forecast", market_id="US")

    # base config has both prophet + lgbm; the S1/ITEM_A override replaces the
    # models list entirely with just prophet (REPLACE semantics, not MERGE).
    assert {m.model_type for m in base_config.models} == {"prophet", "lgbm"}

    resolved = resolve_series_config(base_config, overrides, category="A", sku_id="ITEM_A", store_id="S1")

    assert {m.model_type for m in resolved.models} == {"prophet"}


def test_resolve_series_config_returns_base_when_no_override_matches():
    spark = FakeConfigSpark(_us_config_pdf())
    base_config = load_config(spark, catalog="retail_forecast", market_id="US")
    overrides = load_overrides(spark, catalog="retail_forecast", market_id="US")

    resolved = resolve_series_config(base_config, overrides, category="B", sku_id="OTHER_SKU", store_id="S2")

    assert resolved == base_config
