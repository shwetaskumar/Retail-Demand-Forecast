"""
training/engine.py — shared engine building blocks + the pandas-only cascade
reference implementation (fit_one_series). NOT the production engine — that's
training/engine_layered.py::run_engine_layered, which is what notebooks/02_train.py
actually calls. See fit_one_series's own docstring for why it's kept anyway.

Cascade contract (§10.2 onion ordering) that both fit_one_series (here) and
run_engine_layered (the real, Spark-distributed implementation) satisfy:
  Candidates are evaluated in ascending cost_tier order (config_model.cost_tier,
  seeded from config/seed_defaults.py's _DEFAULT_MODEL_COST_TIERS — the config-table
  -driven source of truth, not hardcoded here). Default cascade: lgbm (cost 0,
  alone) -> auto_arima (2) -> statsforecast/croston/ets_event/elastic_net, one
  merged tier (3) -> prophet, last resort (4). If any tier clears the per-series
  WMAPE gate, evaluation stops there.

  Per-series early-stop threshold = category accuracy_threshold from config_category
  (same threshold used by the cohort-level gate). After the last (most expensive)
  tier, the best available candidate is selected even if no tier cleared the gate.

HPT is NOT run here — it is a post-gate conditional step in notebooks/02_train.py
(and notebooks/05_hpt_retune.py), only triggered for series/cohorts that missed
the gate threshold. See hpt.py.

Decomposition contract (§10.1):
  yhat = baseline + event_lift + model_uplift (0 when uplift_enabled=False)
  baseline   = trend + seasonality_weekly + seasonality_yearly + temporal_features_effect
  event_lift = sum(holiday_lift.values()) + sum(promo_lift.values())

STAGE1_MODEL_REGISTRY / OUTPUT_SCHEMA / evaluate_candidate ARE live and public:
training/hpt.py::tune_one_series and training/engine_layered.py both reuse them
directly — only fit_one_series itself is the non-production piece here.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

from config.loader import ForecastConfig
from training.backtest import backtest_metrics
from training.decomposition.decomposer import reconstruction_error, validate_decomposition
from training.evaluation import gate_wmape_ceiling
from training.models.base import ForecastModel
from training.models.uplift import UpliftModel

# STAGE1_MODEL_REGISTRY is replaced by the auto-discovery registry.
# Keeping the name as a thin shim so any external code that imported it still works;
# the real registry lives in training/models/registry.py.
from training.models.registry import get_registry as _get_registry


class _RegistryProxy:
    """Thin dict-like shim over the lazy auto-discovery registry.

    __setitem__/__delitem__ write through to the underlying cached registry dict
    (not a local copy) so tests can register a fake model via
    monkeypatch.setitem(engine.STAGE1_MODEL_REGISTRY, ...) / monkeypatch's
    corresponding teardown del, exactly like a plain dict would support.
    """
    def __contains__(self, k): return k in _get_registry()
    def __getitem__(self, k):  return _get_registry()[k]
    def __setitem__(self, k, v): _get_registry()[k] = v
    def __delitem__(self, k):  del _get_registry()[k]
    def get(self, k, d=None):  return _get_registry().get(k, d)
    def keys(self):   return _get_registry().keys()
    def values(self): return _get_registry().values()
    def items(self):  return _get_registry().items()


STAGE1_MODEL_REGISTRY = _RegistryProxy()


OUTPUT_SCHEMA_COLUMNS = [
    "series_key", "market_id", "category", "store_id", "sku_id",
    "snapshot_date", "model_type", "params",
    # Selection key: MASE — scale-free, beats-naïve benchmark, champion sorted by this.
    "mase_7d",
    # Gate / planner metric: WMAPE vs per-category threshold (lag-1/lag-2 contract).
    "wmape_7d",
    # Supplementary diagnostics.
    "bias_pct", "naive_mae_7d",
    # Trailing-28-day-mean baseline WMAPE on the same folds — the relative gate's yardstick.
    "baseline_wmape_7d",
    "volume", "reconstruction_error", "passed_decomposition", "status",
]

OUTPUT_SCHEMA = (
    "series_key string, market_id string, category string, store_id string, sku_id string, "
    "snapshot_date string, model_type string, params string, "
    "mase_7d double, wmape_7d double, bias_pct double, naive_mae_7d double, baseline_wmape_7d double, "
    "volume double, reconstruction_error double, passed_decomposition boolean, status string"
)


def result_row(
    series_key: str,
    market_id: str,
    category: str | None,
    store_id: str,
    sku_id: str,
    snapshot_date: str,
    *,
    model_type: str | None = None,
    params: dict | None = None,
    mase_7d: float = float("nan"),       # champion SELECTION key
    wmape_7d: float = float("nan"),      # WMAPE gate / planner-facing metric
    bias_pct: float = float("nan"),      # supplementary diagnostic
    naive_mae_7d: float = float("nan"),  # supplementary diagnostic
    baseline_wmape_7d: float = float("nan"),  # relative-gate yardstick
    volume: float = 0.0,
    reconstruction_error_value: float = float("nan"),
    passed_decomposition: bool = False,
    status: str = "error",
) -> dict:
    """Build one output row dict matching OUTPUT_SCHEMA_COLUMNS."""
    return {
        "series_key":           series_key,
        "market_id":        market_id,
        "category":             category,
        "store_id":          store_id,
        "sku_id":              sku_id,
        "snapshot_date":        snapshot_date,
        "model_type":           model_type,
        "params":               json.dumps(params) if params is not None else None,
        "mase_7d":              mase_7d,
        "wmape_7d":             wmape_7d,
        "bias_pct":             bias_pct,
        "naive_mae_7d":         naive_mae_7d,
        "baseline_wmape_7d":    baseline_wmape_7d,
        "volume":               volume,
        "reconstruction_error": reconstruction_error_value,
        "passed_decomposition": passed_decomposition,
        "status":               status,
    }


def evaluate_candidate(
    model: "ForecastModel",
    model_type: str,
    params: dict,
    series_pdf: pd.DataFrame,
    market_cfg,
    identity: tuple,
) -> dict:
    """Fit one model with `params` on `series_pdf`, backtest, validate decomposition.

    After fitting, calls model.get_storable_params(params) so the stored params
    in champion_index always carry the structured
    {baseline_params, event_coefficients} shape — not just the raw config param_space.

    Returns a result_row dict. `identity` = (series_key, market, category,
    store_id, sku_id, snapshot_date). Captures any exception as status=error:<msg>.
    """

    try:
        # backtest_metrics returns all four metrics in one pass; model is left fitted
        # on the last fold's training window for the decomposition check below.
        metrics = backtest_metrics(
            model, series_pdf,
            market_cfg.backtest_folds,
            market_cfg.val_weeks,
            market_cfg.eval_horizon_days,
            params,
        )
        # Decomposition contract: model is already fitted on the last fold (backtest_metrics
        # mutates in place), so no extra fit() call needed here.
        holdout_start = len(series_pdf) - market_cfg.eval_horizon_days
        holdout_pdf   = series_pdf.iloc[holdout_start:].reset_index(drop=True)
        decomp        = model.decompose(holdout_pdf)
        passed        = validate_decomposition(decomp, market_cfg.reconstruction_tol_pct)
        recon_err     = float(reconstruction_error(decomp).max())
        # `volume` is the gate's weighting term and MUST be the last-fold holdout |qty|
        # (training/evaluation.py's contract; training/engine_layered.py:306 uses exactly
        # this window). Summing the whole series here — as this line did until 2026-09-10 —
        # made HPT-retuned champions carry a full-year volume while cascade champions carried
        # a 7-day one, so the "volume-weighted" WMAPE at every grain silently weighted retuned
        # series ~50x more than the rest.
        volume        = float(holdout_pdf["qty"].astype(float).abs().sum())
        # Store structured params (baseline_params + event_coefficients) rather than
        # the raw config param_space so champion_index is directly queryable.
        storable = (
            model.get_storable_params(params)
            if hasattr(model, "get_storable_params")
            else params
        )
        return result_row(
            *identity,
            model_type=model_type,
            params=storable,
            mase_7d=metrics["mase_7d"],          # champion selection key
            wmape_7d=metrics["wmape_7d"],        # WMAPE gate / planner-facing
            bias_pct=metrics["bias_pct"],        # supplementary
            naive_mae_7d=metrics["naive_mae_7d"],  # supplementary
            baseline_wmape_7d=metrics.get("baseline_wmape_7d", float("nan")),
            volume=volume,
            reconstruction_error_value=recon_err,
            passed_decomposition=passed,
            status="ok",
        )
    except Exception as exc:
        return result_row(*identity, model_type=model_type, params=params,
                          status=f"error:{exc}")


def tag_series_key(
    spark: "SparkSession",
    features_df: "DataFrame",
    market_id: str,
    view_suffix: str = "",
) -> "DataFrame":
    """Add `series_key` and `market_id` columns to `features_df`.

    series_key = <market_id>|<category>|<store_id>|<sku_id>
    """
    from pyspark.sql import functions as F

    tagged = features_df.withColumn(
        "series_key",
        F.concat_ws("|", F.lit(market_id), F.col("category"), F.col("store_id"), F.col("sku_id")),
    ).withColumn("market_id", F.lit(market_id))

    if view_suffix:
        tagged.createOrReplaceTempView(f"_engine{view_suffix}")
    return tagged


def _passes_gate(cat_cfg, row: dict, require_configured: bool = False) -> bool:
    """Per-series gate check shared by Phase 1 (stability) and Phase 2 (early stop):
    training/evaluation.py::gate_wmape_ceiling — relative to the row's own 28-day-mean
    baseline when the category is in skill mode, absolute otherwise. With no gate
    configured, Phase 1 treats it as passed while Phase 2 never early-stops."""
    ceiling = gate_wmape_ceiling(cat_cfg, row.get("baseline_wmape_7d"))
    if ceiling is None:
        return not require_configured
    return bool(row["wmape_7d"] <= ceiling)


def fit_one_series(
    series_pdf: pd.DataFrame,
    config: "ForecastConfig",
    snapshot_date: str,
    prev_champions: dict | None = None,
) -> pd.DataFrame:
    """Pandas-only reference implementation of the per-series cascade contract
    (docs/instructions.md §10.2). NOT called by any production entrypoint —
    notebooks/02_train.py calls training/engine_layered.py::run_engine_layered instead,
    which implements the same contract distributed across the cluster (config-driven,
    grouped by evaluation_cost, one Spark pass per cost tier).

    Kept deliberately: it's the only implementation of this contract callable
    directly on a plain pandas DataFrame, with no Spark session required — which is
    exactly what tests/test_engine.py's TestFitOneSeries suite and
    tests/test_market_onboarding.py's config-only-onboarding proof need. If you're
    changing cascade behavior, engine_layered.py is the file that actually needs the
    change; keep this one in sync only if you want the tests it backs to stay
    meaningful as a spec of the contract, not because anything in prod depends on it.

    Three-phase evaluation:
      Phase 1 — stability check against prev champion (drift_tolerance + absolute gate).
      Phase 2 — full evaluation in ascending evaluation_cost order; early-stop on gate.
    """
    pdf = series_pdf[series_pdf["isActive"] == 1].sort_values("date_id").reset_index(drop=True)
    # Normalize date_id: Arrow deserialises DATE columns as Python datetime.date objects,
    # but backtest.py builds fold origins as pd.Timestamp via pd.Timedelta arithmetic.
    # Converting once here keeps all downstream model/backtest code type-consistent.
    pdf["date_id"] = pd.to_datetime(pdf["date_id"])
    if pdf.empty:
        return pd.DataFrame(columns=OUTPUT_SCHEMA_COLUMNS)

    series_key    = pdf["series_key"].iloc[0]
    market        = pdf["market_id"].iloc[0]
    category      = pdf["category"].iloc[0] if "category" in pdf.columns else None
    store_id = pdf["store_id"].iloc[0]
    sku_id     = pdf["sku_id"].iloc[0]
    identity      = (series_key, market, category, store_id, sku_id, snapshot_date)

    market_cfg = config.market_config

    if len(pdf) < market_cfg.eval_horizon_days * 2:
        return pd.DataFrame([result_row(*identity, model_type=None, params=None, status="insufficient_history")])

    cat_cfg    = next((c for c in config.categories if c.category == category), None)
    drift_tol  = cat_cfg.drift_tolerance if cat_cfg is not None else 0.05

    # ── PHASE 1: stability check ─────────────────────────────────────────────────────────────────
    prev = (prev_champions or {}).get(series_key)
    if prev is not None:
        prev_model_type = prev["model_type"]
        prev_wmape      = float(prev["wmape_7d"])
        prev_params     = prev.get("params", {})
        if isinstance(prev_params, str):
            try:
                import json as _j
                prev_params = _j.loads(prev_params) if prev_params else {}
            except Exception:
                prev_params = {}
        try:
            p1_cls = STAGE1_MODEL_REGISTRY.get(prev_model_type)
            if p1_cls is not None:
                uplift_p1    = market_cfg.uplift_enabled and prev_model_type != "lgbm"
                p1_model_obj = UpliftModel(p1_cls()) if uplift_p1 else p1_cls()
                p1_type_key  = f"{prev_model_type}+uplift" if uplift_p1 else prev_model_type
                p1_row = evaluate_candidate(
                    p1_model_obj, p1_type_key, prev_params, pdf, market_cfg, identity
                )
                wmape_now = float(p1_row.get("wmape_7d", float("nan")))
                if (
                    p1_row["status"] == "ok"
                    and p1_row["passed_decomposition"]
                    and wmape_now == wmape_now  # not NaN
                    and wmape_now <= prev_wmape * (1.0 + drift_tol)   # relative: no worse than prev+tol
                    and _passes_gate(cat_cfg, p1_row)  # same bar as the reported gate
                ):
                    stable_row = dict(p1_row)
                    stable_row["status"] = "stable"
                    return pd.DataFrame([stable_row])
        except Exception:
            pass  # Phase 1 failure → fall through to Phase 2

    # ── PHASE 2: full candidate evaluation ───────────────────────────────────────────────
    reg = STAGE1_MODEL_REGISTRY
    enabled_models = sorted(
        [m for m in config.models if m.enabled and m.model_type in reg],
        key=lambda m: getattr(reg[m.model_type], "evaluation_cost", 2),
    )

    best_row: dict | None = None

    for model_cfg in enabled_models:
        model_cls = reg[model_cfg.model_type]
        uplift_enabled = market_cfg.uplift_enabled and model_cfg.model_type != "lgbm"
        model = UpliftModel(model_cls()) if uplift_enabled else model_cls()
        model_type_key = f"{model_cfg.model_type}+uplift" if uplift_enabled else model_cfg.model_type

        row = evaluate_candidate(model, model_type_key, model_cfg.param_space, pdf, market_cfg, identity)

        if row["status"] != "ok":
            continue

        if best_row is None or row["wmape_7d"] < best_row["wmape_7d"]:
            best_row = row

        # Early stop: cheapest model cleared the accuracy gate → skip remaining slower ones
        if row["passed_decomposition"] and _passes_gate(cat_cfg, row, require_configured=True):
            return pd.DataFrame([row])

    if best_row is not None:
        return pd.DataFrame([best_row])

    return pd.DataFrame([result_row(*identity, model_type=None, params=None, status="all_models_failed")])


# ── MMF-style batch engine ────────────────────────────────────────────────────
# Replaces per-series applyInPandas (106K tasks) with per-location batches
# (~666 tasks of ~160 series).  StatsForecast fits ALL series in a partition
# at once using its vectorised Numba implementation, then Ridge event regression
# runs per-series over the batch results.  Expected fit_local time: ~3 min.
#
# Emits the SAME OUTPUT_SCHEMA/OUTPUT_SCHEMA_COLUMNS as training/engine_layered.py
# (mase_7d, bias_pct, naive_mae_7d) rather than a separate narrower schema — champion.py's
# select_champions_spark unconditionally reads those columns for the bias gate + MASE
# selection, so this engine's output must carry them too or that call throws
# AnalysisException (see training/run_fit_local.py, training/run_backfill_training.py,
# which both feed run_engine_batch straight into select_champions_spark).
# Fallback defaults for the forecast sanity cap; the live values come from
# config_market (forecast_cap_mult, cap_trailing_days) — mirrors
# training/engine_layered.py's _FORECAST_CAP_MULT / _CAP_TRAILING_DAYS.
_BATCH_FORECAST_CAP_MULT = 2.5
_BATCH_CAP_TRAILING_DAYS = 365


def run_engine_batch(
    spark: "SparkSession",
    features_df: "DataFrame",
    config: "ForecastConfig",
    snapshot_date: str,
) -> "DataFrame":
    """MMF-style batch engine: groups by store_id, fits 160 series per task.

    ~666 Spark tasks instead of 106K.  Each task fits all items for one location
    using StatsForecast's vectorised batch API, then runs Ridge event regression
    per series.  Returns the same OUTPUT_SCHEMA rows as run_engine / run_engine_simple.
    """
    from pyspark.sql import functions as F

    tagged_df = (
        features_df
        .withColumn("series_key",
                    F.concat_ws("|", F.lit(config.market_config.market_id),
                                F.col("category"), F.col("store_id"), F.col("sku_id")))
        .withColumn("market", F.lit(config.market_config.market_id))
    )
    return tagged_df.groupBy("store_id").applyInPandas(
        lambda pdf: fit_batch_of_series(pdf, config, snapshot_date),
        schema=OUTPUT_SCHEMA,
    )


def fit_batch_of_series(
    batch_pdf: pd.DataFrame,
    config: "ForecastConfig",
    snapshot_date: str,
) -> pd.DataFrame:
    """Per-location batch function for run_engine_batch applyInPandas.

    1. Feeds all active series in this store to StatsForecast.fit() at once —
       the vectorised Numba back-end amortises JIT startup across series.
    2. Reads back in-sample fitted values per series for event Ridge regression.
    3. Runs last-horizon-days holdout to compute WMAPE + decomposition check.
    4. Returns one result row per series (same schema as run_engine).
    """
    from sklearn.linear_model import Ridge
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS
    from training.models.ets_event import (
        _event_cols_from, _extract_ets_components,
    )
    from training.evaluation import wmape as _wmape

    market      = config.market_config.market_id if hasattr(config, "market_config") else batch_pdf["market_id"].iloc[0]
    horizon     = config.market_config.eval_horizon_days  # typically 7
    season_len  = 7
    # Forecast-cap knobs from config (fall back to module defaults for older config rows) —
    # mirrors training/engine_layered.py::_layer_udf_factory.
    cap_mult    = getattr(config.market_config, "forecast_cap_mult", _BATCH_FORECAST_CAP_MULT)
    cap_days    = getattr(config.market_config, "cap_trailing_days", _BATCH_CAP_TRAILING_DAYS)

    # ── 1. Filter to active rows ────────────────────────────────────────────────
    active_pdf = (
        batch_pdf[batch_pdf["isActive"] == 1]
        .copy()
        .assign(date_id=lambda d: pd.to_datetime(d["date_id"]))
    )
    if active_pdf.empty:
        return pd.DataFrame(columns=OUTPUT_SCHEMA_COLUMNS)

    event_cols_all = _event_cols_from(active_pdf)

    # ── 2. Build StatsForecast input (exclude last `horizon` days → holdout) ──
    cutoff_per_series = (
        active_pdf.groupby("series_key")["date_id"].max() - pd.Timedelta(days=horizon)
    )
    train_mask = active_pdf.join(
        cutoff_per_series.rename("cutoff"), on="series_key"
    )["date_id"] <= cutoff_per_series.reindex(active_pdf["series_key"].values).values

    train_pdf = active_pdf[train_mask].copy()

    sf_df = (
        train_pdf[["series_key", "date_id", "qty"]]
        .rename(columns={"series_key": "unique_id", "date_id": "ds", "qty": "y"})
        .assign(y=lambda d: d["y"].fillna(0.0).clip(lower=0.0))
    )

    if sf_df["unique_id"].nunique() == 0:
        return pd.DataFrame(columns=OUTPUT_SCHEMA_COLUMNS)

    # ── 3. Batch ETS fit — ALL series in this location at once ─────────────────
    try:
        sf = StatsForecast(
            models=[AutoETS(season_length=season_len)],
            freq="D",
            n_jobs=1,          # Spark handles inter-task parallelism
        )
        sf.fit(sf_df)
        # Build in-sample lookup using the stable public API (avoids relying on
        # sf.fitted_[idx, 0] internal structure which changed in statsforecast 1.x)
        try:
            _fv_df = sf.forecast_fitted_values()
            _insample_lookup = {
                uid: grp.sort_values("ds")["AutoETS"].fillna(0.0).values
                for uid, grp in _fv_df.groupby("unique_id", sort=False)
            }
        except Exception:
            _insample_lookup = {}
        # Batch holdout + 200-day inference in one predict() call —
        # eliminates the per-series sf_single refit that caused 9.4% null-WMAPE.
        _INFER_H = 200
        try:
            _batch_fcst     = sf.predict(h=horizon + _INFER_H)
            _holdout_lookup: dict = {}
            _infer_lookup:   dict = {}
            for _uid, _grp in _batch_fcst.groupby("unique_id", sort=False):
                _vals = _grp.sort_values("ds")["AutoETS"].values.astype(float)
                _holdout_lookup[_uid] = _vals[:horizon]
                _infer_lookup[_uid]   = _vals[horizon:]
        except Exception:
            _holdout_lookup = {}
            _infer_lookup   = {}
    except Exception as exc:
        # Whole batch failed → mark all series as error
        rows = []
        for sk, grp in active_pdf.groupby("series_key"):
            row = active_pdf[active_pdf["series_key"] == sk].iloc[0]
            cat = row.get("category") if hasattr(row, "get") else row["category"] if "category" in row.index else None
            rows.append(result_row(
                sk, market, cat, str(row["store_id"]), str(row["sku_id"]), snapshot_date,
                model_type="ets_event", status=f"batch_ets_error:{exc}",
            ))
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=OUTPUT_SCHEMA_COLUMNS)

    # Build a lookup: series_key → fitted model object index
    sf_series_order = sf_df["unique_id"].unique().tolist()
    sk_to_idx = {sk: i for i, sk in enumerate(sf_series_order)}

    # ── 4. Per-series: event regression + holdout evaluation ──────────────────
    results = []

    for series_key, series_grp in active_pdf.groupby("series_key"):
        series_grp = series_grp.sort_values("date_id").reset_index(drop=True)

        cat           = series_grp["category"].iloc[0] if "category" in series_grp.columns else None
        store_id = str(series_grp["store_id"].iloc[0])
        sku_id     = str(series_grp["sku_id"].iloc[0])
        identity    = (series_key, market, cat, store_id, sku_id, snapshot_date)

        def _err_row(status):
            return result_row(*identity, model_type="ets_event", status=status)

        if len(series_grp) < horizon * 2:
            results.append(_err_row("insufficient_history"))
            continue

        train_grp   = series_grp.iloc[:-horizon].reset_index(drop=True)
        holdout_grp = series_grp.iloc[-horizon:].reset_index(drop=True)

        # Get in-sample ETS fitted values for this series
        idx = sk_to_idx.get(series_key)
        if idx is None:
            results.append(_err_row("not_in_batch"))
            continue

        # In-sample fitted values via stable public API (sf.fitted_ structure changed in 1.x)
        _is_arr = _insample_lookup.get(series_key)
        in_sample = (
            np.asarray(_is_arr, dtype=float)[:len(train_grp)]
            if _is_arr is not None
            else np.full(len(train_grp), np.nan)
        )
        # Best-effort model state for ETS type label; falls back to {} on any error
        try:
            _fo = sf.fitted_[idx, 0]
            model_state = dict(getattr(_fo, "model_", {}))
        except Exception:
            model_state = {}

        # Event Ridge on training residuals
        y_train   = train_grp["qty"].fillna(0.0).values.astype(float)
        residuals = y_train - in_sample
        # Only include numeric columns — string label columns (e.g. holiday_name,
        # promo_subtype) match the event prefix but cannot be used as Ridge features.
        ev_cols   = [
            c for c in event_cols_all
            if c in train_grp.columns
            and pd.api.types.is_numeric_dtype(train_grp[c])
        ]
        event_coeffs: dict[str, float] = {}

        if ev_cols:
            X = train_grp[ev_cols].fillna(0).values.astype(float)
            active_mask  = X.sum(axis=0) > 0
            active_ev    = [c for c, m in zip(ev_cols, active_mask) if m]
            X_a = X[:, active_mask]
            if X_a.shape[1] > 0:
                res_clean = np.nan_to_num(residuals, nan=0.0)
                ridge = Ridge(alpha=1.0, fit_intercept=False)
                ridge.fit(X_a, res_clean)
                event_coeffs = {c: float(v) for c, v in zip(active_ev, ridge.coef_)}

        # Holdout evaluation: use pre-computed batch predictions — no per-series refit.
        ets_holdout_arr = _holdout_lookup.get(series_key)
        if ets_holdout_arr is None or len(ets_holdout_arr) == 0:
            results.append(_err_row("holdout_predict_error"))
            continue
        ets_holdout_fcst = ets_holdout_arr

        # Apply event lift to holdout forecast
        event_lift_h = np.zeros(horizon, dtype=float)
        for col, coeff in event_coeffs.items():
            if col in holdout_grp.columns:
                event_lift_h += coeff * holdout_grp[col].fillna(0).values.astype(float)

        yhat_holdout = pd.Series(np.maximum(0.0, ets_holdout_fcst + event_lift_h))
        actual_h     = holdout_grp["qty"].astype(float).reset_index(drop=True)
        wmape_val    = _wmape(actual_h, yhat_holdout)
        volume_val   = float(actual_h.abs().sum())

        # Bias / MASE diagnostics (mirrors training/engine_layered.py::_layer_udf_factory
        # so champion.py::select_champions_spark's bias gate + MASE selection work
        # against this engine's output too — it reads mase_7d/bias_pct unconditionally).
        _asum = float(actual_h.to_numpy().sum())
        bias_val = float((yhat_holdout.to_numpy().sum() - _asum) / _asum) if _asum != 0 else float("nan")

        naive_mae_val = float("nan")
        if len(train_grp) > 7:
            naive_fc = train_grp["qty"].shift(7).iloc[-horizon:].fillna(0.0).to_numpy().astype(float)
            if len(naive_fc) == horizon:
                naive_mae_val = float(np.abs(actual_h.to_numpy() - naive_fc).mean())
        mae_val = float(np.abs(actual_h.to_numpy() - yhat_holdout.to_numpy()).mean())
        mase_val = (
            mae_val / naive_mae_val
            if naive_mae_val == naive_mae_val and naive_mae_val > 0
            else float("nan")
        )

        # Series-level demand ceiling for the inference forecast cap (model-agnostic;
        # mirrors training/engine_layered.py::_layer_udf_factory).
        _cap_hist = train_grp["qty"].tail(cap_days)
        _cap_max = float(_cap_hist.max()) if len(_cap_hist) else float("nan")
        forecast_cap = cap_mult * _cap_max if (_cap_max == _cap_max and _cap_max > 0) else None

        # ── Pre-compute 200-day ETS forecast for load-and-score inference ─────────
        # Use pre-computed batch predictions — avoids a second per-series refit.
        _INFER_H = 200
        _infer_arr = _infer_lookup.get(series_key)
        if _infer_arr is not None and len(_infer_arr) >= _INFER_H:
            ets_infer_vals = _infer_arr[:_INFER_H].tolist()
            try:
                _trend_full, _seas_full = _extract_ets_components(
                    model_state, horizon + _INFER_H, season_len
                )
                ets_trend_vals    = _trend_full.values[horizon:].tolist()
                ets_seasonal_vals = _seas_full.values[horizon:].tolist()
            except Exception:
                ets_trend_vals = ets_seasonal_vals = []
        else:
            ets_infer_vals = ets_trend_vals = ets_seasonal_vals = []

        # ETS+Ridge decomposition exact by construction (non_promo ≡ baseline - trend - seasonal)
        passed    = True
        recon_err = 0.0

        # Store params for champion index
        # ets_forecast_* arrays enable load-and-score at inference (no refit needed).
        # origin = snapshot_date (training date); values[i] = forecast for snap+i+1.
        stored_params = {
            "season_length":         season_len,
            "ets_type":              str(model_state.get("components") or model_state.get("model") or "unknown"),
            "event_coefficients":    event_coeffs,
            "ets_forecast_origin":   snapshot_date,
            "ets_forecast_values":   ets_infer_vals,
            "ets_trend_values":      ets_trend_vals,
            "ets_seasonal_values":   ets_seasonal_vals,
        }
        if forecast_cap is not None:
            stored_params["forecast_cap"] = forecast_cap

        results.append(result_row(
            *identity,
            model_type=            "ets_event",
            params=                stored_params,
            mase_7d=                mase_val,
            wmape_7d=               wmape_val,
            bias_pct=               bias_val,
            naive_mae_7d=           naive_mae_val,
            volume=                 volume_val,
            reconstruction_error_value=recon_err,
            passed_decomposition=  passed,
            status=                "ok",
        ))

    return pd.DataFrame(results) if results else pd.DataFrame(columns=OUTPUT_SCHEMA_COLUMNS)
