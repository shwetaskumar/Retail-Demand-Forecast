"""
training/models/prophet.py — Prophet wrapper (candidate #1, docs/instructions.md §10.2).

Native additive decomposition (trend + weekly/yearly seasonality + holiday/promo
regressors). Current top-down baseline system's library; Stage-1 candidate here.

`Decomposition.yhat` is always Prophet's own authoritative `forecast["yhat"]`
(not a re-sum of the named components), and `non_promo_effect` absorbs
whatever gap remains between `yhat` and the individually-named terms (e.g. any
extra_regressors_additive term or floating-point rounding Prophet folds into
`yhat` that we didn't extract by name). That guarantees the decomposition
contract's reconstruction holds exactly, by construction, regardless of
Prophet-internal accounting we didn't explicitly model. `model_uplift` stays 0
here — this is a Stage-1-only model; Stage-2 wraps it (training/models/uplift.py).
"""
from __future__ import annotations

import pandas as pd

from training.models.base import Decomposition, ForecastModel

# Prefixes used to auto-discover event regressor columns from the training DataFrame.
# Mirrors the ETS+Ridge approach: every matching numeric column with at least one
# non-zero observation becomes an additive Prophet regressor — no hardcoded list.
_EVENT_PREFIXES = ("holiday_", "promo_", "bridge_", "payday_")

# Pre-fit regressor pruning: columns that fire fewer than _MIN_REGRESSOR_SUPPORT
# times in the training window are excluded BEFORE fitting Prophet. With fewer
# observations than this, Stan cannot reliably estimate a coefficient — the result
# is dominated by noise and adds variance to the forecast without predictive signal.
#
# Benefit: fewer regressors = faster Stan LBFGS convergence (Prophet fit time scales
# with parameter count), AND prevents overfitting to 1-2 random event days.
# This is the same statistical reasoning as ETS's support-adjusted threshold but
# applied PRE-fit (because Prophet refit is ~2-5s, too expensive to waste on a
# fit-then-prune-then-refit cycle).
_MIN_REGRESSOR_SUPPORT = 5  # minimum non-zero observations to trust a regressor

# Coefficient of variation guard (same logic as ETS's _CV_GUARD): regressors that
# are near-constant (CV < threshold) add no information and can confuse Stan's
# optimizer (e.g. promo_active_flag=1 for 98% of rows during a long-running promo).
_PROPHET_CV_GUARD = 0.05


class ProphetModel(ForecastModel):
    """Prophet with holiday & promo regressors in additive mode."""

    model_type = "prophet"
    evaluation_cost = 4  # Stan MCMC: ~2-5s per series; evaluated last in the cascade

    def __init__(self) -> None:
        self._model = None
        self._event_regressors: list[str] = []   # discovered at fit time
        self._fitted_values: pd.Series | None = None

    def fit(self, series_pdf: pd.DataFrame, known_future_pdf: pd.DataFrame, params: dict) -> "ProphetModel":
        import inspect

        from prophet import Prophet

        pdf = series_pdf.sort_values("date_id").reset_index(drop=True)

        # Recency truncation (config-driven via params["recency_decay"])
        _decay = float(params.get("recency_decay", 0))
        if _decay > 0:
            from training.models.base import apply_recency_truncation
            pdf = apply_recency_truncation(pdf, _decay)

        # ── Recency truncation ─────────────────────────────────────────────────
        train_df = pd.DataFrame({"ds": pd.to_datetime(pdf["date_id"]), "y": pdf["qty"].astype(float).to_numpy()})

        # Discover event columns with sufficient support for this specific series.
        # Three-stage pre-fit pruning (mirrors ETS’s approach but applied before
        # fitting to save Stan optimization time):
        #   1. Prefix match + numeric dtype (same as before)
        #   2. Minimum support: column must fire >= _MIN_REGRESSOR_SUPPORT times.
        #      With <5 observations Stan can't reliably estimate the coefficient,
        #      and the regressor just adds noise + slows convergence.
        #   3. CV guard: near-constant columns (CV < _PROPHET_CV_GUARD) excluded.
        #      e.g. promo_active_flag=1 for 98% of rows adds no discriminative signal.
        self._event_regressors = []
        for c in pdf.columns:
            if not any(c.startswith(p) for p in _EVENT_PREFIXES):
                continue
            if not pd.api.types.is_numeric_dtype(pdf[c]):
                continue
            col_vals = pdf[c].fillna(0)
            n_fires = int((col_vals != 0).sum())
            if n_fires < _MIN_REGRESSOR_SUPPORT:
                continue  # too sparse — can't estimate a reliable coefficient
            col_mean = col_vals.mean()
            if col_mean == 0:
                continue  # all-zero
            if col_vals.std() / abs(col_mean) < _PROPHET_CV_GUARD:
                continue  # near-constant — no discriminative signal
            self._event_regressors.append(c)
        for col in self._event_regressors:
            train_df[col] = pdf[col].fillna(0).astype(float).to_numpy()

        # Filter to genuine Prophet constructor kwargs before splatting. `params` is
        # usually a small hyperparameter dict, but inference/batch_infer.py's refit
        # fallback (used when from_serialized() fails or no artifact was stored) passes
        # the FULL stored champion-index params blob here (baseline_params,
        # model_artifact, event_regressors, ...) — unfiltered **params would raise
        # TypeError on those unknown keys. Introspects Prophet's own __init__ rather
        # than hardcoding a kwarg list so this stays correct across prophet versions.
        valid_keys = set(inspect.signature(Prophet.__init__).parameters.keys()) - {"self"}
        prophet_params = {k: v for k, v in params.items() if k in valid_keys}
        model = Prophet(seasonality_mode="additive", **prophet_params)
        for col in self._event_regressors:
            model.add_regressor(col, mode="additive")
        model.fit(train_df, iter=1000)   # cap LBFGS at 1000 iters — prevents infinite hang on executor

        self._model = model
        self._fitted_values = pd.Series(model.predict(train_df)["yhat"].to_numpy(), index=pdf.index)
        return self

    def fitted_values(self) -> pd.Series:
        assert self._fitted_values is not None, "fit() must be called before fitted_values()"
        return self._fitted_values

    def get_storable_params(self, input_params: dict) -> dict:
        """Prophet handles holiday/promo regressors natively — no separate Ridge step.
        Serializes the fitted model so inference can skip the historical refit.
        """
        out = {"baseline_params": dict(input_params), "event_coefficients": {}}
        artifact = self.serialize()
        if artifact is not None:
            out["model_artifact"] = artifact
            out["event_regressors"] = self._event_regressors  # needed by _raw_predict at inference
        return out

    def serialize(self) -> str | None:
        """Serialize the fitted Prophet model to a JSON string via prophet.serialize."""
        if self._model is None:
            return None
        from prophet.serialize import model_to_json
        return model_to_json(self._model)

    @classmethod
    def from_serialized(cls, params: dict) -> "ProphetModel":
        """Reconstruct from stored params dict — no historical refit required."""
        from prophet.serialize import model_from_json
        artifact = params["model_artifact"]
        obj = cls.__new__(cls)
        obj._model = model_from_json(artifact)
        obj._event_regressors = params.get("event_regressors", [])
        obj._promo_regressors = [c for c in obj._event_regressors if c.startswith("promo_")]
        obj._holiday_regressors = [c for c in obj._event_regressors
                                   if c.startswith(("holiday_", "bridge_", "payday_"))]
        obj._fitted_values = None
        return obj

    def _raw_predict(self, horizon_pdf: pd.DataFrame) -> pd.DataFrame:
        future_df = pd.DataFrame({"ds": pd.to_datetime(horizon_pdf["date_id"])})
        for col in self._event_regressors:
            future_df[col] = horizon_pdf[col].fillna(0).to_numpy() if col in horizon_pdf.columns else 0.0
        return self._model.predict(future_df)

    def predict(self, horizon_pdf: pd.DataFrame) -> pd.Series:
        forecast = self._raw_predict(horizon_pdf)
        return pd.Series(forecast["yhat"].to_numpy(), index=horizon_pdf.index)

    def decompose(self, horizon_pdf: pd.DataFrame) -> Decomposition:
        forecast = self._raw_predict(horizon_pdf)
        index = horizon_pdf.index

        yhat = pd.Series(forecast["yhat"].to_numpy(), index=index)
        trend = pd.Series(forecast["trend"].to_numpy(), index=index)
        seasonality_weekly = pd.Series(
            forecast["weekly"].to_numpy() if "weekly" in forecast.columns else 0.0, index=index
        )
        seasonality_yearly = pd.Series(
            forecast["yearly"].to_numpy() if "yearly" in forecast.columns else 0.0, index=index
        )
        # Route each event regressor to its decomposition bucket by prefix.
        holiday_lift = {
            col: pd.Series(forecast[col].to_numpy() if col in forecast.columns else 0.0, index=index)
            for col in self._event_regressors if col.startswith(("holiday_", "bridge_", "payday_"))
        }
        promo_lift = {
            col: pd.Series(forecast[col].to_numpy() if col in forecast.columns else 0.0, index=index)
            for col in self._event_regressors if col.startswith("promo_")
        }
        named_sum = trend + seasonality_weekly + seasonality_yearly
        for lift in (*holiday_lift.values(), *promo_lift.values()):
            named_sum = named_sum + lift
        temporal_features_effect = yhat - named_sum  # reconciliation: guarantees exact reconstruction

        return Decomposition(
            trend=trend,
            seasonality_weekly=seasonality_weekly,
            seasonality_yearly=seasonality_yearly,
            holiday_lift=holiday_lift,
            promo_lift=promo_lift,
            temporal_features_effect=temporal_features_effect,
            yhat=yhat,
        )
