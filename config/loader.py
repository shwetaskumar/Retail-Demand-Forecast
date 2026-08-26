"""
config/loader.py — reads the compact hierarchical config table and returns
the same typed config objects every training/data/inference module consumes.

Config table schema:
  (market_id, category, sku_id, store_id, param_name, param_value)
  ~12 rows per market at market level; additional rows only when a specific
  category/sku/store needs different settings.

Cascading resolution (resolve_series_config):
  1. (market, category, sku_id, store_id)      — exact combo
  2. (market, category, sku_id, NULL)          — sku across all stores
  3. (market, category, NULL, NULL)            — category-level
  4. (market, NULL, NULL, NULL)                — market default
  First non-NULL match wins.

Merge vs Replace:
  market_settings → MERGE (override provides only changed keys; rest inherited)
  models, features_*, param_space_* → REPLACE (override replaces entirely)

`load_config` only calls `.table(...)`/`.filter(...)`/`.toPandas()` on the injected
`spark` object (duck-typed) — it never imports pyspark itself, so it is exercised
in tests with a plain fake/mock standing in for `spark` (see tests/test_config_loader.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
_CONFIG_SCHEMA = "control"
_CONFIG_TABLE = "forecast_config"



@dataclass(frozen=True)
class MarketConfig:
    """Typed run settings for one market — config rows where param_group='market'."""

    market_id: str
    active: bool
    horizon_days: int
    eval_horizon_days: int
    history_start: str
    incremental_read_days: int
    incremental_write_days: int
    backtest_folds: int
    val_weeks: int
    hpt_trigger: str
    hpt_trials: int
    uplift_enabled: bool
    uplift_learner: str
    uplift_clip_frac: float
    reconstruction_tol_pct: float
    run_budget_usd: float
    # Selection / guardrail knobs (defaulted so older config rows still load):
    max_champion_bias: float = 0.5   # champion selection deprioritises |holdout bias| above this
    forecast_cap_mult: float = 2.5   # inference clips daily yhat to this × trailing max daily demand
    cap_trailing_days: int = 365     # trailing window for the forecast-cap ceiling


@dataclass(frozen=True)
class CategoryConfig:
    """Typed per-category thresholds/gate grains — config rows where
    param_group='category', grouped by the category dimension (§3, §20)."""

    market_id: str
    category: str
    accuracy_threshold: float
    gate_grains: list[str]
    series_grain: str
    drift_tolerance: float  # Phase 1: max fractional WMAPE degradation vs previous champion


@dataclass(frozen=True)
class ModelConfig:
    """Typed candidate-model config — config rows where param_group='model',
    grouped by the model_type dimension (§10.2, §20).

    priority removed: evaluation order is autonomous, derived from
    ForecastModel.evaluation_cost on each wrapper class (training/models/registry.py).
    """

    market_id: str
    model_type: str
    enabled: bool
    param_space: dict[str, Any]
    cost_tier: int | None = None  # evaluation cost bucket (from config table)


@dataclass(frozen=True)
class FeatureConfig:
    """Typed feature toggle/params — config rows where param_group='feature',
    grouped by the feature_name dimension (feature_engineering_spec.md, §20)."""

    market_id: str
    feature_name: str
    enabled: bool
    feature_class: str  # "known_at_inference" | "historical_only"
    params: dict[str, Any]


@dataclass(frozen=True)
class ForecastConfig:
    """Full typed config bundle for one market, as returned by `load_config`."""

    market_config: MarketConfig
    categories: list[CategoryConfig]
    models: list[ModelConfig]
    features: list[FeatureConfig]


def _read_config(spark: "SparkSession", catalog: str, market_id: str) -> pd.DataFrame:
    """Row-filtered read of the config table as pandas.

    Config is tiny (~12 rows per market + sparse overrides), so filtering
    server-side then `toPandas()` is safe here.
    """
    safe_market_id = market_id.replace("'", "''")
    table_df = spark.table(f"{catalog}.{_CONFIG_SCHEMA}.{_CONFIG_TABLE}").filter(f"market_id = '{safe_market_id}'")
    return table_df.toPandas()


def _market_level_rows(pdf: pd.DataFrame) -> dict[str, str]:
    """Extract market-level rows (category IS NULL, sku_id IS NULL, store_id IS NULL)
    and return {param_name: param_value} dict.
    """
    mask = pdf["category"].isna() & pdf["sku_id"].isna() & pdf["store_id"].isna()
    rows = pdf[mask]
    return dict(zip(rows["param_name"], rows["param_value"]))


def _override_rows(pdf: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that are NOT market-level (i.e., at least one of
    category/sku_id/store_id is non-NULL). These are per-combo overrides.
    """
    mask = pdf["category"].notna() | pdf["sku_id"].notna() | pdf["store_id"].notna()
    return pdf[mask]


def _parse_json(value: Any, default: Any) -> Any:
    """Parse a JSON-string config value; tolerate None/empty and already-parsed values."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPACT CONFIG FORMAT — ~12 JSON-packed rows per market
# ═══════════════════════════════════════════════════════════════════════════════

def _load_config_compact(full_pdf: pd.DataFrame, market_id: str) -> ForecastConfig:
    """Parse the compact config format (JSON-packed, ~12 rows per market)."""
    params = _market_level_rows(full_pdf)
    if not params:
        raise ValueError(f"No market-level config rows for market_id={market_id!r}")

    # ── market_settings (single JSON object with all scalars) ──
    ms = _parse_json(params.get("market_settings"), {})
    market_config = MarketConfig(
        market_id=market_id,
        active=ms.get("active", True),
        horizon_days=int(ms["horizon_days"]),
        eval_horizon_days=int(ms["eval_horizon_days"]),
        history_start=str(ms["history_start"]),
        incremental_read_days=int(ms.get("incremental_read_days", 750)),
        incremental_write_days=int(ms.get("incremental_write_days", 365)),
        backtest_folds=int(ms.get("backtest_folds", 3)),
        val_weeks=int(ms.get("val_weeks", 4)),
        hpt_trigger=str(ms.get("hpt_trigger", "threshold_miss")),
        hpt_trials=int(ms.get("hpt_trials", 10)),
        uplift_enabled=bool(ms.get("uplift_enabled", False)),
        uplift_learner=str(ms.get("uplift_learner", "lgbm_global")),
        uplift_clip_frac=float(ms.get("uplift_clip_frac", 0.25)),
        reconstruction_tol_pct=float(ms.get("reconstruction_tol_pct", 0.005)),
        run_budget_usd=float(ms.get("run_budget_usd", 100.0)),
        max_champion_bias=float(ms.get("max_champion_bias", 0.5)),
        forecast_cap_mult=float(ms.get("forecast_cap_mult", 2.5)),
        cap_trailing_days=int(ms.get("cap_trailing_days", 365)),
    )

    # ── models (JSON object: {model_type: cost_tier}) ──
    models_dict = _parse_json(params.get("models"), {})
    models = []
    for model_type, cost_tier in models_dict.items():
        ps_key = f"param_space_{model_type}"
        param_space = _parse_json(params.get(ps_key), {})
        models.append(ModelConfig(
            market_id=market_id,
            model_type=model_type,
            enabled=True,
            param_space=param_space,
            cost_tier=int(cost_tier),
        ))

    # ── features (two JSON lists: historical + known) ──
    hist_features = _parse_json(params.get("features_historical"), [])
    known_features = _parse_json(params.get("features_known"), [])
    features = []
    for fn in hist_features:
        features.append(FeatureConfig(
            market_id=market_id, feature_name=fn,
            enabled=True, feature_class="historical_only", params={},
        ))
    for fn in known_features:
        features.append(FeatureConfig(
            market_id=market_id, feature_name=fn,
            enabled=True, feature_class="known_at_inference", params={},
        ))

    # ── category thresholds (JSON object: {category: threshold}) ──
    thresholds = _parse_json(params.get("category_thresholds"), {})
    categories = [
        CategoryConfig(
            market_id=market_id,
            category=cat,
            accuracy_threshold=float(thresh),
            gate_grains=["category_cohort", "market_sku", "market_store_sku"],
            series_grain="market_store_sku",
            drift_tolerance=0.05,
        )
        for cat, thresh in thresholds.items()
    ]

    return ForecastConfig(market_config=market_config, categories=categories, models=models, features=features)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — auto-detects v1/v2, cascading resolution
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(spark: "SparkSession", catalog: str, market_id: str) -> ForecastConfig:
    """Read config into a typed ForecastConfig (~12 JSON-packed rows per market).

    Every job (data/training/inference) calls this once at startup.
    """
    full_pdf = _read_config(spark, catalog, market_id)
    return _load_config_compact(full_pdf, market_id)


def load_overrides(spark: "SparkSession", catalog: str, market_id: str) -> pd.DataFrame:
    """Load all override rows (non-market-level) for a market.

    Returns DataFrame with: category, sku_id, store_id, param_name, param_value.
    For most markets this is empty. Call once at job startup; pass to
    resolve_series_config() per series.
    """
    full_pdf = _read_config(spark, catalog, market_id)
    return _override_rows(full_pdf)


def resolve_series_config(
    base_config: ForecastConfig,
    overrides_pdf: pd.DataFrame,
    category: str,
    sku_id: str,
    store_id: str,
) -> ForecastConfig:
    """Cascading resolution for a specific (category, sku_id, store_id) combo.

    Resolution order (first match wins per param_name):
      1. (category, sku_id, store_id) — exact combo
      2. (category, sku_id, NULL)         — item across all stores
      3. (category, NULL, NULL)            — category-level
      4. (NULL, NULL, NULL)                — market default (base_config)

    For most series, no override rows exist → returns base_config unchanged.
    Merge rule: market_settings → MERGE keys; others → REPLACE entirely.
    """
    if overrides_pdf.empty:
        return base_config

    def _specificity(row: pd.Series) -> int:
        """Score: higher = more specific. -1 = doesn't apply to this combo."""
        score = 0
        r_cat = row.get("category")
        r_item = row.get("sku_id")
        r_loc = row.get("store_id")
        if pd.notna(r_cat):
            if r_cat != category:
                return -1
            score += 1
        if pd.notna(r_item):
            if r_item != sku_id:
                return -1
            score += 2
        if pd.notna(r_loc):
            if r_loc != store_id:
                return -1
            score += 4
        return score

    scored = overrides_pdf.copy()
    scored["_score"] = scored.apply(_specificity, axis=1)
    valid = scored[scored["_score"] > 0].sort_values("_score", ascending=False)
    if valid.empty:
        return base_config

    # Take most-specific row per param_name
    resolved: dict[str, str] = {}
    for _, row in valid.iterrows():
        pn = row["param_name"]
        if pn not in resolved:
            resolved[pn] = row["param_value"]

    # Apply overrides
    import copy
    cfg = copy.deepcopy(base_config)

    if "market_settings" in resolved:
        override_ms = _parse_json(resolved["market_settings"], {})
        for k, v in override_ms.items():
            if hasattr(cfg.market_config, k):
                expected_type = type(getattr(cfg.market_config, k))
                object.__setattr__(cfg.market_config, k, expected_type(v))

    if "models" in resolved:
        models_dict = _parse_json(resolved["models"], {})
        new_models = []
        for mt, ct in models_dict.items():
            ps_key = f"param_space_{mt}"
            ps = _parse_json(resolved.get(ps_key), {})
            if not ps:
                base_m = next((m for m in base_config.models if m.model_type == mt), None)
                ps = base_m.param_space if base_m else {}
            new_models.append(ModelConfig(
                market_id=cfg.market_config.market_id,
                model_type=mt, enabled=True, param_space=ps, cost_tier=int(ct),
            ))
        cfg = ForecastConfig(
            market_config=cfg.market_config, categories=cfg.categories,
            models=new_models, features=cfg.features,
        )

    for pn, pv in resolved.items():
        if pn.startswith("param_space_") and "models" not in resolved:
            mt = pn[len("param_space_"):]
            new_space = _parse_json(pv, {})
            for m in cfg.models:
                if m.model_type == mt:
                    object.__setattr__(m, "param_space", new_space)

    if "features_historical" in resolved or "features_known" in resolved:
        h_list = _parse_json(resolved["features_historical"], []) if "features_historical" in resolved else [f.feature_name for f in cfg.features if f.feature_class == "historical_only"]
        k_list = _parse_json(resolved["features_known"], []) if "features_known" in resolved else [f.feature_name for f in cfg.features if f.feature_class == "known_at_inference"]
        new_features = [
            FeatureConfig(market_id=cfg.market_config.market_id, feature_name=fn, enabled=True, feature_class="historical_only", params={})
            for fn in h_list
        ] + [
            FeatureConfig(market_id=cfg.market_config.market_id, feature_name=fn, enabled=True, feature_class="known_at_inference", params={})
            for fn in k_list
        ]
        cfg = ForecastConfig(
            market_config=cfg.market_config, categories=cfg.categories,
            models=cfg.models, features=new_features,
        )

    return cfg
