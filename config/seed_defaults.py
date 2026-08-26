"""
config/seed_defaults.py — bootstrap inserts for the compact v2 config table.

V2 Schema:
  market_id, category, sku_id, store_id, param_name, param_value,
  updated_by, updated_ts

  Natural key: (market_id, category, sku_id, store_id, param_name)
  NULL in category/sku_id/store_id = market-wide default.

Produces ~12 compact JSON-packed rows per market (vs 270+ legacy EAV rows):
  - models:               {model_type: cost_tier} for enabled models
  - param_space_<model>:  HPT search space per enabled model
  - features_historical:  JSON list of enabled historical-only feature names
  - features_known:       JSON list of enabled known-at-inference feature names
  - market_settings:      all scalar market settings as one JSON object
  - category_thresholds:  {category: accuracy_threshold}

Idempotent — routed through utils.io.merge_upsert on the natural key.
References feature modules for feature name lists (source of truth).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

_CONFIG_SCHEMA = "control"
_CONFIG_TABLE = "forecast_config"

_MERGE_KEYS = ["market_id", "category", "sku_id", "store_id", "param_name"]


def _v2_row(
    market_id: str,
    param_name: str,
    param_value: object,
    *,
    category: str | None = None,
    sku_id: str | None = None,
    store_id: str | None = None,
) -> dict:
    """Create one compact v2 config row.

    param_value is JSON-serialized (dicts/lists → JSON string, scalars → str).
    NULL in category/sku_id/store_id = market-wide default.
    """
    if isinstance(param_value, (dict, list)):
        pv_str = json.dumps(param_value, separators=(",", ":"))
    else:
        pv_str = str(param_value)
    return {
        "market_id": market_id,
        "category": category,
        "sku_id": sku_id,
        "store_id": store_id,
        "param_name": param_name,
        "param_value": pv_str,
    }


# ══════════════════════════════════════════════════════════════════════════════════
# MARKET DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════════

# Demo-market category thresholds = 1 - WMAPE (docs/instructions.md §3, §20).
DEFAULT_CATEGORY_THRESHOLDS: dict[str, float] = {
    "A": 0.85,
    "B": 0.75,
    "C": 0.65,
    "D": 0.60,
    "SEASONAL": 0.80,
}

# Demo-market settings (docs/instructions.md §23 key assumptions).
DEFAULT_MARKET_SETTINGS: dict[str, object] = {
    "active": True,
    "horizon_days": 140,
    "eval_horizon_days": 7,
    "history_start": "2022-01-01",
    "incremental_read_days": 750,
    "incremental_write_days": 365,
    "backtest_folds": 5,
    "val_weeks": 10,
    "hpt_trigger": "threshold_miss",
    "hpt_trials": 10,
    "uplift_enabled": True,
    "uplift_learner": "lgbm_global",
    "uplift_clip_frac": 0.25,
    "reconstruction_tol_pct": 0.005,
    "run_budget_usd": 100.0,
    "max_champion_bias": 0.5,
    "forecast_cap_mult": 2.5,
    "cap_trailing_days": 365,
}

# Models disabled by default — absent from the models dict.
_DEFAULT_DISABLED_MODELS: set[str] = {"elastic_net"}


# ══════════════════════════════════════════════════════════════════════════════════
# HPT SEARCH SPACES
# ══════════════════════════════════════════════════════════════════════════════════
# Schema: {"name": {"type": "int"|"float"|"categorical", "low"/"high"(+"step"/"log") | "choices"}}.
# Empty {} → HPT is a no-op for that model_type.

LGBM_PARAM_SPACE: dict = {
    "n_estimators":      {"type": "int",   "low": 50,    "high": 300,  "step": 25},
    "max_depth":         {"type": "int",   "low": 3,     "high": 8},
    "num_leaves":        {"type": "int",   "low": 7,     "high": 63},
    "learning_rate":     {"type": "float", "low": 0.01,  "high": 0.3,  "log": True},
    "min_child_samples": {"type": "int",   "low": 5,     "high": 50},
}
ELASTIC_NET_PARAM_SPACE: dict = {
    "alpha":    {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
    "l1_ratio": {"type": "float", "low": 0.0,    "high": 1.0},
}
PROPHET_PARAM_SPACE: dict = {
    "changepoint_prior_scale":  {"type": "float", "low": 0.001, "high": 0.5,  "log": True},
    "seasonality_prior_scale":  {"type": "float", "low": 0.01,  "high": 10.0, "log": True},
    "changepoint_range":        {"type": "float", "low": 0.8,   "high": 0.95},
    "n_changepoints":           {"type": "int",   "low": 10,    "high": 35},
}

# Model cost tiers (lower = cheaper = preferred on tie).
_DEFAULT_MODEL_COST_TIERS: dict[str, int] = {
    "ets_event": 1,
    "auto_arima": 1,
    "statsforecast": 1,
    "croston": 0,
    "elastic_net": 1,
    "lgbm": 2,
    "prophet": 3,
}

_MODEL_PARAM_SPACES: dict[str, dict] = {
    "ets_event": {},
    "auto_arima": {},
    "statsforecast": {},
    "croston": {},
    "elastic_net": ELASTIC_NET_PARAM_SPACE,
    "lgbm": LGBM_PARAM_SPACE,
    "prophet": PROPHET_PARAM_SPACE,
}


# ══════════════════════════════════════════════════════════════════════════════════
# V2 COMPACT ROW BUILDERS
# ══════════════════════════════════════════════════════════════════════════════════

def _get_feature_lists() -> tuple[list[str], list[str]]:
    """Import feature name lists from their owning modules (source of truth).

    Returns (historical_only, known_at_inference) sorted lists.
    """
    from data.features.category_features import HISTORICAL_ONLY_COLUMNS as CAT_HIST
    from data.features.category_features import KNOWN_AT_INFERENCE_COLUMNS as CAT_KNOWN_AT_INFERENCE
    from data.features.event_features import KNOWN_AT_INFERENCE_COLUMNS as EVENT_KNOWN_AT_INFERENCE
    from data.features.halo_cannibalization_features import HISTORICAL_ONLY_COLUMNS as HALO_HIST
    from data.features.halo_cannibalization_features import (
        KNOWN_AT_INFERENCE_COLUMNS as HALO_KNOWN_AT_INFERENCE,
    )
    from data.features.stat_features import HISTORICAL_ONLY_COLUMNS
    from data.features.temporal_features import KNOWN_AT_INFERENCE_COLUMNS as TEMPORAL_KNOWN_AT_INFERENCE

    known_at_inference = sorted(set(
        [*TEMPORAL_KNOWN_AT_INFERENCE, *EVENT_KNOWN_AT_INFERENCE,
         *CAT_KNOWN_AT_INFERENCE, *HALO_KNOWN_AT_INFERENCE]
    ))
    historical_only = sorted(set([*HISTORICAL_ONLY_COLUMNS, *CAT_HIST, *HALO_HIST]))
    return historical_only, known_at_inference


def build_market_rows(
    market_id: str,
    *,
    market_settings: dict[str, object],
    category_thresholds: dict[str, float],
    model_cost_tiers: dict[str, int],
    model_param_spaces: dict[str, dict],
    disabled_models: set[str] | None = None,
) -> list[dict]:
    """Build the ~12 compact v2 config rows for one market.

    Args:
        market_id: Market code (e.g. 'US', 'DE').
        market_settings: Scalar market settings dict.
        category_thresholds: {category: accuracy_threshold}.
        model_cost_tiers: {model_type: cost_tier} for ALL models.
        model_param_spaces: {model_type: param_space_dict} for ALL models.
        disabled_models: Models to EXCLUDE from the seeded config.

    Returns:
        List of row dicts ready for spark.createDataFrame().
    """
    disabled = disabled_models or set()
    rows: list[dict] = []

    # 1. models: {model_type: cost_tier} — only enabled models
    models_dict = {
        mt: ct for mt, ct in model_cost_tiers.items()
        if mt not in disabled
    }
    rows.append(_v2_row(market_id, "models", models_dict))

    # 2. param_space_<model> — one row per enabled model
    for model_type in models_dict:
        ps = model_param_spaces.get(model_type, {})
        rows.append(_v2_row(market_id, f"param_space_{model_type}", ps))

    # 3. features — two JSON lists
    hist_features, known_features = _get_feature_lists()
    rows.append(_v2_row(market_id, "features_historical", hist_features))
    rows.append(_v2_row(market_id, "features_known", known_features))

    # 4. market_settings — single JSON object
    rows.append(_v2_row(market_id, "market_settings", market_settings))

    # 5. category_thresholds — single JSON object
    rows.append(_v2_row(market_id, "category_thresholds", category_thresholds))

    return rows


# ══════════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════════

def seed_default_market(spark: "SparkSession", catalog: str) -> None:
    """Idempotently insert the demo-market ("US") bootstrap rows into v2 config (~12 rows).

    Run once per environment (or re-run — merge_upsert makes it idempotent).
    """
    from utils.io import merge_upsert

    all_rows = build_market_rows(
        "US",
        market_settings=DEFAULT_MARKET_SETTINGS,
        category_thresholds=DEFAULT_CATEGORY_THRESHOLDS,
        model_cost_tiers=_DEFAULT_MODEL_COST_TIERS,
        model_param_spaces=_MODEL_PARAM_SPACES,
        disabled_models=_DEFAULT_DISABLED_MODELS,
    )
    merge_upsert(
        spark,
        spark.createDataFrame(all_rows),
        catalog,
        _CONFIG_SCHEMA,
        _CONFIG_TABLE,
        merge_keys=_MERGE_KEYS,
    )


def seed_market_defaults(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    *,
    market_settings: dict[str, object],
    category_thresholds: dict[str, float],
    model_cost_tiers: dict[str, int] | None = None,
    model_param_spaces: dict[str, dict] | None = None,
    disabled_models: set[str] | None = None,
) -> None:
    """Idempotently insert bootstrap rows for ANY market into v2 config.

    Defaults to the demo market's model/param_space definitions when not provided.
    Adding a market = call this once with market-specific settings.
    """
    from utils.io import merge_upsert

    all_rows = build_market_rows(
        market_id,
        market_settings=market_settings,
        category_thresholds=category_thresholds,
        model_cost_tiers=model_cost_tiers or _DEFAULT_MODEL_COST_TIERS,
        model_param_spaces=model_param_spaces or _MODEL_PARAM_SPACES,
        disabled_models=disabled_models,
    )
    merge_upsert(
        spark,
        spark.createDataFrame(all_rows),
        catalog,
        _CONFIG_SCHEMA,
        _CONFIG_TABLE,
        merge_keys=_MERGE_KEYS,
    )
