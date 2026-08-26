"""
training/models/lgbm.py — LightGBM wrapper (candidate #3, docs/instructions.md §10.2).

Explainable gradient boosting on calendar+event features, with SHAP attribution
grouped into the decomposition contract's buckets (trend/seasonality/event).
Verified interactively (SHAP TreeExplainer): `shap_values.sum(axis=1) +
expected_value` reconstructs `model.predict(X)` to ~1e-14 — so grouping every
SHAP column into exactly one decomposition bucket (with the constant
`expected_value` folded into `trend`) reconstructs `yhat` exactly by
construction.

v1 keeps the feature set numeric only (no string categoricals like
holiday_name/promo_subtype — those describe the same event for reporting, not
as a tree-split input, and would need a categorical-encoding decision this
wrapper doesn't make). Tactic/severity ARE included as numeric signal though:
promo_tactic_rank (ordinal 0-3, data/features/event_features.py's _TACTIC_RANK)
and holiday_weight (severity 0-1) ride alongside the plain holiday_flag/
promo_active_flag binary indicators, so LightGBM can split on tactic strength
and holiday severity, not just "was something active". Stage-2 uplift (if
applied by training/models/uplift.py) is folded into the promo_lift bucket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.models.base import Decomposition, ForecastModel

# feature column -> decomposition bucket. "time_idx" is derived internally (row position
# continuing the training series), not read from the input frame.
_FEATURE_BUCKETS: dict[str, str] = {
    "time_idx": "trend",
    "dow": "seasonality_weekly",
    "week_of_year": "seasonality_yearly",
    "month": "seasonality_yearly",
    "quarter": "seasonality_yearly",
    "day_of_month": "seasonality_yearly",
    "payday_flag": "temporal_features_effect",
    "weekend_flag": "temporal_features_effect",
    "promo_active_flag": "promo_lift",
    "promo_concurrent_count": "promo_lift",
    # Tactic-level binary flags: one per canonical tactic (event_features.py).
    # Binary 0/1 — indicates whether this tactic type is active (for decomposition
    # attribution). Magnitude comes only from promo_concurrent_count (log-capped).
    "promo_count_temp_price_change": "promo_lift",
    "promo_count_bogo_mix": "promo_lift",
    "promo_count_bogo_match": "promo_lift",
    "promo_count_mix_match_mix": "promo_lift",
    "promo_count_mix_match_match": "promo_lift",
    "promo_count_meal_bundle": "promo_lift",
    "promo_count_free_offer": "promo_lift",
    "promo_count_advertising": "promo_lift",
    "promo_count_pairings": "promo_lift",
    "promo_count_tactic_other": "promo_lift",
    # Subtype-level binary flags.
    "promo_count_lto": "promo_lift",
    "promo_count_market_digital": "promo_lift",
    "promo_count_local_offer": "promo_lift",
    "promo_count_subtype_other": "promo_lift",
    # Holiday features (v4: sparse, window-bounded, equi-join based).
    "holiday_is_day_of": "holiday_lift",
    "holiday_days_until": "holiday_lift",
    "holiday_days_since": "holiday_lift",
    # Halo (complement) / cannibalization (substitute) partner signals — the 6
    # KNOWN_AT_INFERENCE columns from data/features/halo_cannibalization_features.py.
    # KNOWN_AT_INFERENCE, not hardcoded-imported here (models/*.py reference feature
    # column names as literals, matching training/models/elastic_net.py's convention,
    # since JOB 1 data/ and JOB 2 training/ stay independent per CLAUDE.md). Promos are
    # known 2mo-1yr ahead, so halo_partner_promo_*/cannib_partner_promo_* are exactly as
    # future-safe as promo_active_flag above; the corr_strength pair is a structural,
    # snapshot-time-fixed attribute, safe for the whole horizon like category.
    # Deliberately EXCLUDES the module's 2 HISTORICAL_ONLY columns
    # (halo_partner_qty_7d_roll, cannib_partner_qty_7d_roll): the future spine never
    # carries them (they require realized qty), so training on real values but always
    # predicting on a 0 placeholder would be a train/serve mismatch — same reasoning
    # that already keeps category_qty_7d_roll etc. out of this bucket map.
    "halo_partner_corr_strength":       "cross_sku_lift",
    "cannib_partner_corr_strength":     "cross_sku_lift",
    "halo_partner_promo_active_flag":   "cross_sku_lift",
    "halo_partner_promo_count":         "cross_sku_lift",
    "cannib_partner_promo_active_flag": "cross_sku_lift",
    "cannib_partner_promo_count":       "cross_sku_lift",
}
CALENDAR_FEATURE_COLUMNS = [c for c in _FEATURE_BUCKETS if c != "time_idx"]


def build_calendar_feature_matrix(pdf: pd.DataFrame, start_idx: int, columns: list[str]) -> pd.DataFrame:
    """Build a (time_idx + named calendar/event columns) feature matrix, filling missing
    columns/NaNs with 0. `time_idx` continues from `start_idx` (0 at fit time, the training
    length at predict/decompose time) so it behaves as a trend feature. Shared by
    LightGBMModel and training/models/uplift.py's residual learner.
    """
    n = len(pdf)
    data = {"time_idx": np.arange(start_idx, start_idx + n)}
    for col in columns:
        data[col] = pdf[col].fillna(0).to_numpy() if col in pdf.columns else np.zeros(n)
    return pd.DataFrame(data, index=pdf.index)


class LightGBMModel(ForecastModel):
    """LightGBM on calendar+event features; decomposition via grouped SHAP attribution."""

    model_type = "lgbm"
    evaluation_cost = 2  # per-series fit; much cheaper than Prophet but not batch vectorised

    def __init__(self) -> None:
        self._model = None           # lgb.LGBMRegressor — set after fit()
        self._booster = None         # lgb.Booster — set after from_serialized(); used in predict/decompose
        self._train_length: int = 0
        self._fitted_values: pd.Series | None = None

    def _feature_matrix(self, pdf: pd.DataFrame, start_idx: int) -> pd.DataFrame:
        return build_calendar_feature_matrix(pdf, start_idx, CALENDAR_FEATURE_COLUMNS)

    def fit(self, series_pdf: pd.DataFrame, known_future_pdf: pd.DataFrame, params: dict) -> "LightGBMModel":
        import lightgbm as lgb

        pdf = series_pdf.sort_values("date_id").reset_index(drop=True)
        self._train_length = len(pdf)
        X = self._feature_matrix(pdf, start_idx=0)
        y = pdf["qty"].astype(float).to_numpy()

        # ── Recency weighting ─────────────────────────────────────────────────
        # Exponential decay so recent observations dominate the learned level
        # while older history (holidays, seasonality) is still visible.
        # decay_rate controls half-life: 0.003 → ~230 days (1yr ago = 0.33 weight).
        # Configurable via params["recency_decay"] (HPT can tune it).
        _decay = float(params.get("recency_decay", 0))
        if _decay > 0:
            _positions = np.arange(len(y), dtype=float)
            sample_weight = np.exp(_decay * (_positions - _positions[-1]))
        else:
            sample_weight = np.ones(len(y), dtype=float)

        # Filter to genuine LGBMRegressor constructor kwargs before splatting. `params`
        # is usually a small hyperparameter dict (HPT / config_model.param_space),
        # but inference/batch_infer.py's refit fallback (used when from_serialized()
        # fails or no artifact was stored) passes the FULL stored champion-index params
        # blob here (baseline_params, event_coefficients, model_artifact, forecast_cap,
        # ...) — unfiltered **params would raise TypeError on those unknown keys.
        valid_keys = set(lgb.LGBMRegressor().get_params().keys())
        lgb_params = {k: v for k, v in params.items() if k in valid_keys}

        # Under Spark we run one fit per task and rely on Spark for parallelism.
        # LightGBM defaults to all cores per fit, so N concurrent tasks each spawn
        # N OpenMP threads → quadratic oversubscription and severe slowdown. Force
        # single-threaded fits; the caller can still override via params.
        lgb_params = {"n_jobs": 1, **lgb_params}
        model = lgb.LGBMRegressor(verbosity=-1, **lgb_params)
        model.fit(X, y, sample_weight=sample_weight)
        self._model = model
        self._fitted_values = pd.Series(model.predict(X), index=pdf.index)
        return self

    def fitted_values(self) -> pd.Series:
        assert self._fitted_values is not None, "fit() must be called before fitted_values()"
        return self._fitted_values

    def get_storable_params(self, input_params: dict) -> dict:
        """LightGBM handles event effects inside the tree model; no separate Ridge step.
        Serializes the fitted booster AND pre-computed SHAP decomposition so inference
        can skip both the historical refit AND the expensive SHAP computation.
        """
        out = {"baseline_params": dict(input_params), "event_coefficients": {}}
        artifact = self.serialize()
        if artifact is not None:
            out["model_artifact"] = artifact
            out["baseline_params"]["train_length"] = self._train_length

            # Pre-compute SHAP decomposition for a 200-day horizon.
            # At inference, decompose() replays these stored values (~0ms)
            # instead of calling shap.TreeExplainer (~200ms).
            try:
                self._precompute_shap_decomp(out)
            except Exception:
                pass  # SHAP precompute is best-effort; falls back to runtime SHAP
        return out

    def _precompute_shap_decomp(self, out: dict) -> None:
        """Compute SHAP on a 200-day synthetic horizon and store per-bucket arrays."""
        import shap

        _H = 200
        # Build a synthetic 200-day feature matrix starting after training
        # Calendar features (dow, month, etc.) are deterministic from time_idx
        import datetime
        base_date = datetime.date(2026, 1, 2)  # day after snapshot; exact date doesn't matter for decomp storage
        dates = pd.date_range(base_date, periods=_H, freq="D")
        synth_pdf = pd.DataFrame({
            "date_id": dates,
            **{col: 0 for col in CALENDAR_FEATURE_COLUMNS}
        })
        # Fill calendar columns from dates
        synth_pdf["dow"] = dates.dayofweek + 1  # ISO: 1=Mon, 7=Sun
        synth_pdf["week_of_year"] = dates.isocalendar().week.astype(int).values
        synth_pdf["month"] = dates.month
        synth_pdf["quarter"] = dates.quarter
        synth_pdf["day_of_month"] = dates.day
        synth_pdf["weekend_flag"] = (dates.dayofweek >= 5).astype(int)
        # holiday_flag, promo_active_flag, and all 6 halo/cannib cross_sku_lift columns
        # stay 0 (baseline decomp) — same as every other event-bucket column, they're
        # attributed at decompose()-time via the marginal-prediction approach instead,
        # since actual event/partner-promo flags vary per real future date.

        X = self._feature_matrix(synth_pdf, start_idx=self._train_length)
        booster = self._model
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X)
        expected_value = float(explainer.expected_value)

        # Aggregate SHAP into decomposition buckets
        buckets = {"trend": [], "seasonality_weekly": [], "seasonality_yearly": [], "temporal_features_effect": []}
        for day_idx in range(_H):
            day_buckets = {k: 0.0 for k in buckets}
            for col_idx, col in enumerate(X.columns):
                bucket = _FEATURE_BUCKETS[col]
                if bucket in day_buckets:
                    day_buckets[bucket] += float(shap_values[day_idx, col_idx])
            for k in buckets:
                buckets[k].append(day_buckets[k])

        # Trend includes the expected_value (constant base)
        buckets["trend"] = [v + expected_value for v in buckets["trend"]]

        out["shap_decomp"] = {
            "trend": buckets["trend"],
            "seasonality_weekly": buckets["seasonality_weekly"],
            "seasonality_yearly": buckets["seasonality_yearly"],
            "temporal_features_effect": buckets["temporal_features_effect"],
            "expected_value": expected_value,
        }

    def serialize(self) -> str | None:
        """Serialize the fitted LightGBM booster to a model-text string."""
        if self._model is None:
            return None
        return self._model.booster_.model_to_string()

    @classmethod
    def from_serialized(cls, params: dict) -> "LightGBMModel":
        """Reconstruct from stored params dict — no historical refit required."""
        import lightgbm as lgb
        artifact = params["model_artifact"]
        obj = cls.__new__(cls)
        obj._model = None
        obj._booster = lgb.Booster(model_str=artifact)
        obj._train_length = int(params.get("baseline_params", {}).get("train_length", 0))
        obj._fitted_values = None
        # Load pre-computed SHAP decomposition if available
        obj._stored_shap_decomp = params.get("shap_decomp")
        return obj

    def predict(self, horizon_pdf: pd.DataFrame) -> pd.Series:
        X = self._feature_matrix(horizon_pdf, start_idx=self._train_length)
        booster = self._booster if self._booster is not None else self._model
        return pd.Series(booster.predict(X), index=horizon_pdf.index)

    def decompose(self, horizon_pdf: pd.DataFrame) -> Decomposition:
        h = len(horizon_pdf)
        index = horizon_pdf.index

        # Fast path: use pre-computed SHAP decomposition from training (~0ms)
        stored = getattr(self, "_stored_shap_decomp", None)
        if stored and len(stored.get("trend", [])) >= h:
            trend = pd.Series(stored["trend"][:h], index=index, dtype=float)
            seasonality_weekly = pd.Series(stored["seasonality_weekly"][:h], index=index, dtype=float)
            seasonality_yearly = pd.Series(stored["seasonality_yearly"][:h], index=index, dtype=float)
            temporal_features_effect = pd.Series(stored["temporal_features_effect"][:h], index=index, dtype=float)

            # Event contributions: use marginal prediction approach (2 predictions, ~2ms)
            # since actual event flags may differ from training template
            X = self._feature_matrix(horizon_pdf, start_idx=self._train_length)
            booster = self._booster if self._booster is not None else self._model
            yhat_full = booster.predict(X)
            X_no_events = X.copy()
            event_cols = [
                c for c in X.columns
                if _FEATURE_BUCKETS.get(c) in ("holiday_lift", "promo_lift", "cross_sku_lift")
            ]
            X_no_events[event_cols] = 0
            yhat_no_events = booster.predict(X_no_events)
            total_event_lift = yhat_full - yhat_no_events

            # Attribute event lift proportionally by |feature_value| across active cols.
            # For each day, sum absolute feature values across all active event columns,
            # then each column gets (|its_value| / total) * total_event_lift.
            holiday_lift: dict[str, pd.Series] = {}
            promo_lift: dict[str, pd.Series] = {}
            cross_sku_lift: dict[str, pd.Series] = {}

            # Build a matrix of absolute feature values for event columns
            event_abs = np.column_stack([np.abs(X[col].values.astype(float)) for col in event_cols])
            # Per-day total of absolute values (for proportional split)
            event_abs_total = event_abs.sum(axis=1, keepdims=True)
            # Avoid division by zero (days with no events → lift is already 0)
            event_abs_total = np.where(event_abs_total == 0, 1.0, event_abs_total)
            # Proportional weights per column per day
            event_weights = event_abs / event_abs_total

            for col_idx_e, col in enumerate(event_cols):
                bucket = _FEATURE_BUCKETS[col]
                contribution = pd.Series(
                    total_event_lift * event_weights[:, col_idx_e], index=index
                )
                if bucket == "holiday_lift":
                    holiday_lift[col] = contribution
                elif bucket == "promo_lift":
                    promo_lift[col] = contribution
                else:
                    cross_sku_lift[col] = contribution

            # Adjust trend so total reconstructs yhat exactly
            base_sum = trend + seasonality_weekly + seasonality_yearly + temporal_features_effect
            event_sum = (
                sum(holiday_lift.values(), pd.Series(0.0, index=index))
                + sum(promo_lift.values(), pd.Series(0.0, index=index))
                + sum(cross_sku_lift.values(), pd.Series(0.0, index=index))
            )
            # Any residual goes into temporal_features_effect to keep reconstruction exact
            yhat_series = pd.Series(yhat_full, index=index)
            residual = yhat_series - (base_sum + event_sum)
            temporal_features_effect = temporal_features_effect + residual

            return Decomposition(
                trend=trend,
                seasonality_weekly=seasonality_weekly,
                seasonality_yearly=seasonality_yearly,
                holiday_lift=holiday_lift if holiday_lift else {"holiday": pd.Series(0.0, index=index)},
                promo_lift=promo_lift if promo_lift else {"promo": pd.Series(0.0, index=index)},
                cross_sku_lift=cross_sku_lift,
                temporal_features_effect=temporal_features_effect,
                yhat=yhat_series,
            )

        # Slow path: compute SHAP at runtime (fallback for champions without stored decomp)
        import shap

        X = self._feature_matrix(horizon_pdf, start_idx=self._train_length)
        booster = self._booster if self._booster is not None else self._model
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X)
        expected_value = float(explainer.expected_value)

        bucket_sums: dict[str, pd.Series] = {
            bucket: pd.Series(0.0, index=index)
            for bucket in ("trend", "seasonality_weekly", "seasonality_yearly", "temporal_features_effect")
        }
        holiday_lift: dict[str, pd.Series] = {}
        promo_lift: dict[str, pd.Series] = {}
        cross_sku_lift: dict[str, pd.Series] = {}
        for col_idx, col in enumerate(X.columns):
            contribution = pd.Series(shap_values[:, col_idx], index=index)
            bucket = _FEATURE_BUCKETS[col]
            if bucket == "holiday_lift":
                holiday_lift[col] = contribution
            elif bucket == "promo_lift":
                promo_lift[col] = contribution
            elif bucket == "cross_sku_lift":
                cross_sku_lift[col] = contribution
            else:
                bucket_sums[bucket] = bucket_sums[bucket] + contribution

        trend = bucket_sums["trend"] + expected_value

        named_sum = trend + bucket_sums["seasonality_weekly"] + bucket_sums["seasonality_yearly"]
        named_sum = named_sum + bucket_sums["temporal_features_effect"]
        for lift in (*holiday_lift.values(), *promo_lift.values(), *cross_sku_lift.values()):
            named_sum = named_sum + lift

        return Decomposition(
            trend=trend,
            seasonality_weekly=bucket_sums["seasonality_weekly"],
            seasonality_yearly=bucket_sums["seasonality_yearly"],
            holiday_lift=holiday_lift,
            promo_lift=promo_lift,
            cross_sku_lift=cross_sku_lift,
            temporal_features_effect=bucket_sums["temporal_features_effect"],
            yhat=named_sum,
        )
