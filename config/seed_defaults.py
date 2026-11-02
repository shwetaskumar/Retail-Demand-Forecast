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
  - category_skill_thresholds: {category: min_skill_vs_baseline} (relative gate)

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

# Explicit schema for spark.createDataFrame(all_rows) — category/sku_id/store_id
# are NULL on every market-default row (lock #6: cascading resolution), which
# Spark Connect's dict-based schema inference can't type on its own
# (PySparkValueError: CANNOT_DETERMINE_TYPE).
_ROW_SCHEMA = "market_id STRING, category STRING, sku_id STRING, store_id STRING, param_name STRING, param_value STRING"


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
#
# History worth keeping, because it explains why these moved twice:
#   - 2026-09-05 these were LOOSENED (A 0.85->0.80, B 0.75->0.70, D 0.60->0.55,
#     SEASONAL 0.80->0.70) to sit just above the then-measured WMAPE of A 22.7% /
#     B 34.5% / C 36.6%, with a comment asserting the inherited QSR values were
#     simply wrong for M5 grocery data.
#   - That measurement was invalid. champion_index.volume — the weight in every
#     volume-weighted gate aggregate — was being computed over two different windows
#     by the cascade and the HPT-retune path (see training/engine.py's volume
#     assignment and its 2026-09-10 fix). Re-scored on one consistent weight, real
#     performance is A 13.9% / B 21.7% / C 25.2%: every inherited threshold would
#     have passed all along, and the loosening was never warranted.
#
# Restored 2026-09-10 to the inherited values for A/B/D/SEASONAL. C is set to 0.72
# rather than its inherited 0.65 because 0.65 (<=35% WMAPE) leaves ~10pp of slack
# against measured 25.2% — a gate nothing can fail teaches nothing. At these values
# A/B/C clear by 1.1pp / 3.3pp / 2.8pp: passing, but only just, which is the point.
# The gate is also the HPT trigger (training/hpt.py::needs_tuning routes gate-failing
# series to Optuna), so a threshold with real tension routes more of the tail into
# retuning instead of letting it coast.
DEFAULT_CATEGORY_THRESHOLDS: dict[str, float] = {
    "A": 0.85,
    "B": 0.75,
    "C": 0.72,
    "D": 0.60,
    "SEASONAL": 0.80,
}

# Relative ("skill") gate, the default pass/fail rule (training/evaluation.py): a grain
# passes iff its WMAPE beats the trailing-28-day-mean baseline's WMAPE on the same backtest
# folds by at least this fraction. With these set, DEFAULT_CATEGORY_THRESHOLDS above stay
# in config as informational targets only.
#
# Why relative: on a local Kaggle M5 sample (CA, 150 store×sku series) the median
# category-A series sells ~13 units/week, where Poisson sampling noise alone puts the
# best achievable weekly WMAPE near 18% — an 85% absolute accuracy target is unreachable
# at that grain no matter the model, so it measures the data, not the forecaster. "Beats
# the naive baseline by X%" is achievable, comparable across categories, and fails
# exactly when a model adds nothing.
#
# Values fixed BEFORE the README demo run (not fit to it), from two disjoint 150-series
# diagnostic samples where the auto_arima+ets_event+lgbm ensemble beat the baseline by
# ~5-7% per category: 3% demands real, repeatable skill with margin for sampling noise.
# D (long tail, mostly zeros) only has to not lose to the baseline.
DEFAULT_CATEGORY_SKILL_THRESHOLDS: dict[str, float] = {
    "A": 0.03,
    "B": 0.03,
    "C": 0.03,
    "D": 0.0,
    "SEASONAL": 0.03,
}

# Demo-market settings (docs/instructions.md §23 key assumptions).
# history_start = 2014-01-01: the demo dataset is Kaggle M5 (static historical
# data, sales_train_evaluation.csv spans 2014-05-24 through 2016-05-22 —
# verified from the ingested data itself, not the competition docs). A modern
# "2022-01-01" default (this repo's original assumption for a live production
# source) would exclude the entire dataset.
DEFAULT_MARKET_SETTINGS: dict[str, object] = {
    "active": True,
    "horizon_days": 140,
    "eval_horizon_days": 7,
    "history_start": "2014-01-01",
    "incremental_read_days": 750,
    "incremental_write_days": 365,
    "backtest_folds": 5,
    "val_weeks": 10,
    "hpt_trigger": "threshold_miss",
    "hpt_trials": 30,
    "uplift_enabled": True,
    "uplift_learner": "lgbm_global",
    "uplift_clip_frac": 0.25,
    "reconstruction_tol_pct": 0.005,
    "run_budget_usd": 100.0,
    "max_champion_bias": 0.5,
    "forecast_cap_mult": 2.5,
    "cap_trailing_days": 365,
    # See training/engine_layered.py::_survivors — a cost-0 (lgbm) candidate
    # must beat 85% of its category's gate_wmape, not just clear it outright,
    # to skip comparison against the classical models in later cascade tiers.
    "escalation_margin": 0.85,
}

# Models disabled by default — absent from the models dict.
#
# prophet disabled 2026-09-06: diagnosed via a direct evaluate_candidate(ProphetModel(), ...)
# call against 10 of the hardest still-failing M5 series (Databricks ml_env, prophet==1.1.6) —
# every single fit raised "'Prophet' object has no attribute 'stan_backend'" before producing
# a forecast. This is the cmdstanpy/Stan backend never getting built on Databricks serverless's
# ephemeral, declarative-only environment spec (pip-installing prophet does not itself compile
# the Stan model; that normally happens once via `cmdstanpy.install_cmdstan()`, which has no
# durable place to run on compute that respins per job). This is also why prophet had exactly
# 0 champion wins across two full training runs despite being tier 4/5 (last resort) in the
# cascade — it was never a "loses on merit" case, every attempt silently errored and
# evaluate_candidate's try/except (training/engine.py) just returned status="error:...",
# which _survivors treats as no valid candidate and the series kept whatever it already had
# from an earlier tier. Re-enable only after confirming a working Stan backend in ml_env.
#
# ets_event: tried disabling 2026-09-07 on the theory that its bad average WMAPE meant it
# was actively harmful rather than just weak, then re-enabled the same day after the run
# appeared to get worse. BOTH readings were measurement noise — that whole experiment was
# run against the broken champion_index.volume weight (fixed 2026-09-10, see
# training/engine.py). Re-scored on a consistent weight, with and without ets_event are
# 16.0% overall and 13.8% vs 13.9% category A: the removal was NEUTRAL, not harmful.
# It stays enabled because there is no measured reason to remove it, not because removing
# it was shown to hurt.
#
# The durable lesson is about method, not this model: a model's average WMAPE is computed
# over the population that SELECTED it, and champion selection hands the hardest series to
# whichever model is least-bad for them. So a high average is evidence about the population,
# not about the model, and the only way to judge a removal is to measure what happens to
# that population without it — on a metric you have verified means what you think it means.
#
# 2026-11-02 — single models disabled as standalone candidates; `ensemble` (which still
# fits auto_arima + ets_event + lgbm internally) is the only default candidate. Measured with
# demo/compare_selection_rules.py on Kaggle M5 (CA, 150 store×sku series, 5 target weeks =
# 750 out-of-sample series-weeks, each rule choosing from the 3 prior weeks only):
#     always ensemble                  37.8% WMAPE, +1.4% vs 28-day-mean baseline (90% CI -3.2..+6.2)
#     ensemble unless a single model   39.4-40.0%,  -2.8% to -4.2%  (margins 10/20/30%)
#       beats it in backtest by a margin
#     best candidate per series        41.1%,       -7.1%           (90% CI -13.7..0.0)
# Picking the backtest winner per series on a few noisy weeks selects whichever model got
# lucky, so per-series switching made forecasts measurably worse than not switching at all.
# Re-enable singles (delete them from this set) to bring per-series selection back — the
# engine, cascade and champion code are unchanged and still support it.
_DEFAULT_DISABLED_MODELS: set[str] = {
    "elastic_net", "prophet", "lgbm", "auto_arima", "statsforecast", "croston", "ets_event",
}


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
    "recency_decay":     {"type": "float", "low": 0.0,   "high": 0.02},
    # Added 2026-09-07: lgbm carries the largest champion population (53% of series) and
    # the largest absolute count of still-gate-failing series (143 of 251) — its WMAPE
    # spread (p10=3.5%, median=16.6%, p90=39.6% on the M5 run) is the classic signature of
    # an under-regularized tree overfitting on a few hundred noisy per-series rows. These
    # three are standard LightGBM overfitting counterweights; training/models/lgbm.py:141-142
    # already safely passes through any valid LGBMRegressor kwarg (filtered against
    # lgb.LGBMRegressor().get_params()), so no wrapper code change is needed — HPT simply
    # never had these dimensions to explore.
    "reg_alpha":         {"type": "float", "low": 0.001, "high": 10.0, "log": True},
    "reg_lambda":        {"type": "float", "low": 0.001, "high": 10.0, "log": True},
    "colsample_bytree":  {"type": "float", "low": 0.5,   "high": 1.0},
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

# Added 2026-09-06: these four model_types previously had param_space={}, which makes
# tune_series() a silent no-op (training/hpt.py:137-138) — HPT never touched their real,
# fittable knobs (season_length, recency truncation, croston's smoothing constants), so
# ~40% of series (everything that isn't lgbm) never got tuned at all, just re-fit at
# hardcoded defaults every run. Ranges below are centered on each wrapper's own default
# (season_length=7, trend_window_days=90, croston alpha=0.1, ridge_alpha=1.0 — see
# training/models/{auto_arima,statsforecast,croston,ets_event}.py) with recency_decay
# capped at 0.02: training/models/base.py::apply_recency_truncation shows decay=0.02
# already truncates a 730-row history down to roughly the trailing 4 months, and higher
# values just saturate at the _RECENCY_MIN_ROWS=60 floor without exploring anything new.
AUTO_ARIMA_PARAM_SPACE: dict = {
    "season_length": {"type": "int",   "low": 5,   "high": 10},
    "recency_decay": {"type": "float", "low": 0.0, "high": 0.02},
}
STATSFORECAST_PARAM_SPACE: dict = {
    "season_length":     {"type": "int",   "low": 5,   "high": 10},
    "trend_window_days": {"type": "int",   "low": 30,  "high": 180, "step": 15},
    "recency_decay":     {"type": "float", "low": 0.0, "high": 0.02},
}
CROSTON_PARAM_SPACE: dict = {
    "alpha_z":     {"type": "float", "low": 0.02,  "high": 0.5},
    "alpha_p":     {"type": "float", "low": 0.02,  "high": 0.5},
    "ridge_alpha": {"type": "float", "low": 0.01,  "high": 10.0, "log": True},
    "recency_decay": {"type": "float", "low": 0.0, "high": 0.02},
}
ETS_EVENT_PARAM_SPACE: dict = {
    "season_length": {"type": "int",   "low": 5,   "high": 10},
    "recency_decay": {"type": "float", "low": 0.0, "high": 0.02},
}

# Model cost tiers (lower = cheaper = evaluated first in the cascade). This is the
# config-table-driven source of truth: training/engine_layered.py::run_engine_layered
# reads config_model.cost_tier (seeded from this dict) in preference to each model
# class's own `evaluation_cost` default — re-ordering the cascade is a config change
# here (or a control.forecast_config row edit via the app), never a code change.
#
# Cascade: lgbm(0, evaluated first, alone — nothing else shares cost 0/1 so
# engine_layered.py's fast-tier merge doesn't group it with anything)
# -> auto_arima/statsforecast(2, merged tier)
# -> croston/elastic_net(3, merged tier) -> prophet(4, last resort).
#
# statsforecast moved 3->2 on 2026-09-07: measured weighted WMAPE (17.8%) is
# consistently the best or second-best of the classical models, clearly stronger than
# auto_arima (24.8%) on this run — pairing them means every survivor that reaches tier 2
# gets BOTH fit and champion-selected between directly, instead of auto_arima getting an
# uncontested first look and statsforecast only seeing series auto_arima already failed.
_DEFAULT_MODEL_COST_TIERS: dict[str, int] = {
    "lgbm": 0,
    "auto_arima": 2,
    "statsforecast": 2,
    "croston": 3,
    "ets_event": 3,
    "elastic_net": 3,
    "prophet": 4,
    # training/models/ensemble.py — refits auto_arima+ets_event+lgbm, so it is the last
    # tier the cascade reaches, only for series no single model gets past the gate.
    "ensemble": 4,
}

_MODEL_PARAM_SPACES: dict[str, dict] = {
    "ets_event": ETS_EVENT_PARAM_SPACE,
    "auto_arima": AUTO_ARIMA_PARAM_SPACE,
    "statsforecast": STATSFORECAST_PARAM_SPACE,
    "croston": CROSTON_PARAM_SPACE,
    "elastic_net": ELASTIC_NET_PARAM_SPACE,
    "lgbm": LGBM_PARAM_SPACE,
    "prophet": PROPHET_PARAM_SPACE,
    "ensemble": {},  # members are tuned individually; nothing to search at the ensemble level
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
    category_skill_thresholds: dict[str, float] | None = None,
) -> list[dict]:
    """Build the ~12 compact v2 config rows for one market.

    Args:
        market_id: Market code (e.g. 'US', 'DE').
        market_settings: Scalar market settings dict.
        category_thresholds: {category: accuracy_threshold}.
        model_cost_tiers: {model_type: cost_tier} for ALL models.
        model_param_spaces: {model_type: param_space_dict} for ALL models.
        disabled_models: Models to EXCLUDE from the seeded config.
        category_skill_thresholds: {category: min_skill_vs_baseline}; omitted => the
            absolute accuracy_threshold gate applies.

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

    # 6. category_skill_thresholds — single JSON object (relative gate), optional
    if category_skill_thresholds:
        rows.append(_v2_row(market_id, "category_skill_thresholds", category_skill_thresholds))

    return rows


# ══════════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════════

def seed_default_market(spark: "SparkSession", catalog: str) -> None:
    """Idempotently insert the demo-market ("CA" — Kaggle M5's state_id, see
    data/kaggle_m5/loader.py) bootstrap rows into v2 config (~12 rows).

    Run once per environment (or re-run — merge_upsert makes it idempotent).
    """
    from utils.io import merge_upsert

    all_rows = build_market_rows(
        "CA",
        market_settings=DEFAULT_MARKET_SETTINGS,
        category_thresholds=DEFAULT_CATEGORY_THRESHOLDS,
        model_cost_tiers=_DEFAULT_MODEL_COST_TIERS,
        model_param_spaces=_MODEL_PARAM_SPACES,
        disabled_models=_DEFAULT_DISABLED_MODELS,
        category_skill_thresholds=DEFAULT_CATEGORY_SKILL_THRESHOLDS,
    )
    merge_upsert(
        spark,
        spark.createDataFrame(all_rows, schema=_ROW_SCHEMA),
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
    category_skill_thresholds: dict[str, float] | None = None,
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
        category_skill_thresholds=category_skill_thresholds,
    )
    merge_upsert(
        spark,
        spark.createDataFrame(all_rows, schema=_ROW_SCHEMA),
        catalog,
        _CONFIG_SCHEMA,
        _CONFIG_TABLE,
        merge_keys=_MERGE_KEYS,
    )
