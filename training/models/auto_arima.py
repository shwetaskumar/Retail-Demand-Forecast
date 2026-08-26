"""
training/models/auto_arima.py — AutoARIMA + Ridge event regression (two-stage).

Architecture (same two-stage pattern as ets_event.py):
  Stage 1 — AutoARIMA (statsforecast): fits on raw qty series capturing
             trend + autocorrelation + seasonality via automatic order selection.
  Stage 2 — Ridge regression on Stage 1 in-sample residuals against event
             indicator columns: produces per-event lift coefficients.

WHY AutoARIMA alongside AutoETS:
  AutoETS handles level/trend/multiplicative seasonality well but can struggle
  with series that have significant residual autocorrelation (ARMA terms) or
  structural non-stationarity resolved by differencing.  AutoARIMA covers this
  complementary class of series at the same speed tier (statsforecast
  vectorised Numba backend, ~50-150 ms/series).  The two models are evaluated
  independently; the one with lower WMAPE + passing decomposition becomes champion.

DECOMPOSITION CONTRACT (§10.1):
  yhat = trend + seasonality_weekly + seasonality_yearly
       + Σ holiday_lift + Σ promo_lift
       + temporal_features_effect

   How each term is derived:
     trend              = full ARIMA point forecast (level + AR/MA dynamics)
     seasonality_weekly = 0  (seasonal ARIMA handles it inside the model;
                              attributing it separately requires inspecting fitted
                              SARIMA seasonal component — deferred, absorbed into trend)
     seasonality_yearly = 0  (same reason)
     holiday/promo_lift = Ridge_coeff × event_indicator (future known)
     temporal_features_effect = 0  (reconstruction is exact: trend + event_lifts = yhat)

  Reconstruction check: trend + Σ event_lifts = yhat  (exact by construction).

evaluation_cost = 1: statsforecast AutoARIMA is batch-capable via the same
StatsForecast(...).fit() API as AutoETS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.models.base import Decomposition, ForecastModel
from training.models.ets_event import (
    _HOLIDAY_PREFIXES,
    _event_cols_from,
)

_DEFAULT_SEASON_LENGTH = 7


class AutoARIMAModel(ForecastModel):
    """AutoARIMA baseline + Ridge event regression (two-stage, same pattern as ETSEventModel)."""

    model_type      = "auto_arima"
    evaluation_cost = 1  # statsforecast AutoARIMA: batch-capable, same cost tier as ets_event

    def __init__(self, season_length: int = _DEFAULT_SEASON_LENGTH) -> None:
        self.season_length          = season_length
        self._sf                    = None
        self._event_coeffs: dict[str, float] = {}
        self._n_train               = 0

    # ── fit ─────────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        series_pdf: pd.DataFrame,
        known_future_pdf: pd.DataFrame,
        params: dict,
    ) -> "AutoARIMAModel":
        from sklearn.linear_model import Ridge
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA

        season_length = int(params.get("season_length", self.season_length))

        # At inference time, stored event_coefficients are passed via params —
        # reuse them directly, skip Ridge re-fitting (same pattern as ets_event.py).
        stored_coeffs = params.get("event_coefficients")
        _need_ridge   = stored_coeffs is None
        if not _need_ridge:
            self._event_coeffs = stored_coeffs

        pdf = series_pdf.sort_values("date_id").reset_index(drop=True)

        # Recency truncation (config-driven via params["recency_decay"])
        _decay = float(params.get("recency_decay", 0))
        if _decay > 0:
            from training.models.base import apply_recency_truncation
            pdf = apply_recency_truncation(pdf, _decay)

        y   = pdf["qty"].fillna(0.0).clip(lower=0.0).values.astype(float)
        ds  = pd.to_datetime(pdf["date_id"])

        sf_df = pd.DataFrame({"unique_id": "s", "ds": ds, "y": y})
        self._sf = StatsForecast(
            models=[AutoARIMA(season_length=season_length, approximation=True)],
            freq="D",
            n_jobs=1,   # Spark handles inter-task parallelism
        )
        self._sf.fit(sf_df)
        self._n_train = len(sf_df)

        # Stage 2: Ridge on in-sample residuals to capture event effects
        if _need_ridge:
            ev_cols = _event_cols_from(series_pdf)
            if ev_cols:
                try:
                    fv_df = self._sf.forecast_fitted_values()
                    fitted_vals = (
                        fv_df[fv_df["unique_id"] == "s"]
                        .sort_values("ds")["AutoARIMA"]
                        .values.astype(float)
                    )
                except Exception:
                    fitted_vals = np.full(len(y), np.nan)
                residuals  = y - fitted_vals[: len(y)]
                X          = series_pdf[ev_cols].fillna(0).values.astype(float)
                active_mask = X.sum(axis=0) > 0
                active_cols = [c for c, m in zip(ev_cols, active_mask) if m]
                X_a = X[:, active_mask]
                if X_a.shape[1] > 0:
                    res_clean = np.nan_to_num(residuals, nan=0.0)
                    ridge = Ridge(alpha=1.0, fit_intercept=False)
                    ridge.fit(X_a, res_clean)
                    self._event_coeffs = {
                        c: float(v) for c, v in zip(active_cols, ridge.coef_)
                    }

        return self

    # ── fitted_values (for UpliftModel residual computation) ──────────────────────

    def fitted_values(self) -> pd.Series:
        if self._sf is None:
            raise RuntimeError("Must call fit() first")
        try:
            fv_df = self._sf.forecast_fitted_values()
            vals  = (
                fv_df[fv_df["unique_id"] == "s"]
                .sort_values("ds")["AutoARIMA"]
                .values.astype(float)
            )
            return pd.Series(vals)
        except Exception:
            return pd.Series([], dtype=float)

    # ── decompose ───────────────────────────────────────────────────────────────────────────

    def decompose(self, horizon_pdf: pd.DataFrame) -> Decomposition:
        h = len(horizon_pdf)
        # Use stored forecast values if available (from_serialized path)
        if hasattr(self, "_forecast_values") and self._forecast_values is not None:
            yhat_arima = pd.Series(self._forecast_values[:h].copy())
        else:
            arima_fcst = self._sf.predict(h=h)
            yhat_arima = pd.Series(arima_fcst["AutoARIMA"].values.astype(float))

        # ARIMA structural decomposition:
        # The full ARIMA point forecast is attributed to trend; we do not attempt
        # to split out a weekly seasonal component because AutoARIMA encodes
        # seasonality via seasonal differencing / SARIMA terms rather than
        # an explicit additive seasonal component — any split would be approximate.
        # Exact reconstruction is guaranteed: trend + event_lifts = yhat.
        trend              = yhat_arima.copy()
        seasonality_weekly = pd.Series(np.zeros(h, dtype=float))
        seasonality_yearly = pd.Series(np.zeros(h, dtype=float))

        # Event lift decomposition (identical to ets_event.py)
        holiday_lift: dict[str, pd.Series] = {}
        promo_lift:   dict[str, pd.Series] = {}
        for col, coeff in self._event_coeffs.items():
            vals = (
                horizon_pdf[col].fillna(0).values.astype(float)
                if col in horizon_pdf.columns
                else np.zeros(h)
            )
            lift = pd.Series(coeff * vals)
            if any(col.startswith(p) for p in _HOLIDAY_PREFIXES):
                holiday_lift[col] = lift
            else:
                promo_lift[col]   = lift

        zeros = pd.Series(np.zeros(h, dtype=float))
        if not holiday_lift:
            holiday_lift = {"holiday": zeros.copy()}
        if not promo_lift:
            promo_lift   = {"promo":   zeros.copy()}

        total_event = (
            sum(holiday_lift.values(), zeros.copy())
            + sum(promo_lift.values(), zeros.copy())
        )
        yhat_final = yhat_arima + total_event

        return Decomposition(
            trend              = trend,
            seasonality_weekly = seasonality_weekly,
            seasonality_yearly = seasonality_yearly,
            holiday_lift       = holiday_lift,
            promo_lift         = promo_lift,
            temporal_features_effect = zeros.copy(),   # reconstruction is exact without a residual term
            yhat               = yhat_final,
        )

    # ── storable params ──────────────────────────────────────────────────────────────────

    def get_storable_params(self, input_params: dict) -> dict:
        """Structured params for champion_index: AutoARIMA baseline + Ridge event coefficients.

        Stores pre-computed forecast values (200 steps) so inference can skip refit.
        """
        # Pre-compute and store forecast values for inference fast path
        forecast_values = []
        if self._sf is not None:
            try:
                fc = self._sf.predict(h=200)
                forecast_values = fc["AutoARIMA"].values.astype(float).tolist()
            except Exception:
                pass
        return {
            "baseline_params":       {"season_length": self.season_length, "n_train": self._n_train},
            "event_coefficients":    self._event_coeffs,
            "arima_forecast_values": forecast_values,
            "model_artifact":        "auto_arima_forecast" if forecast_values else None,
        }

    def serialize(self) -> str | None:
        """AutoARIMA stores forecast values as the artifact (statsforecast object not picklable)."""
        if self._sf is None:
            return None
        return "auto_arima_forecast"

    @classmethod
    def from_serialized(cls, params: dict) -> "AutoARIMAModel":
        """Reconstruct from stored forecast values + event coefficients — no refit needed."""
        forecast_values = params.get("arima_forecast_values", [])
        if not forecast_values:
            raise NotImplementedError("No stored arima_forecast_values")
        bp = params.get("baseline_params", {})
        obj = cls.__new__(cls)
        obj.season_length = int(bp.get("season_length", _DEFAULT_SEASON_LENGTH))
        obj._n_train = int(bp.get("n_train", 0))
        obj._event_coeffs = params.get("event_coefficients", {})
        obj._sf = None
        obj._forecast_values = np.array(forecast_values, dtype=float)
        return obj

    def predict(self, horizon_pdf: pd.DataFrame) -> pd.Series:
        h = len(horizon_pdf)
        # Use stored forecast values if available (from_serialized path)
        if hasattr(self, "_forecast_values") and self._forecast_values is not None:
            yhat_base = self._forecast_values[:h]
        else:
            arima_fcst = self._sf.predict(h=h)
            yhat_base = arima_fcst["AutoARIMA"].values.astype(float)

        event_lift = np.zeros(h, dtype=float)
        for col, coeff in self._event_coeffs.items():
            if col in horizon_pdf.columns:
                event_lift += coeff * horizon_pdf[col].fillna(0).values.astype(float)

        return pd.Series(np.maximum(0.0, yhat_base + event_lift))
