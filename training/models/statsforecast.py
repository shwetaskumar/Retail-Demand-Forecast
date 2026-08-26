"""
training/models/statsforecast.py — statsforecast wrapper (candidate #2, docs/instructions.md §10.2).

Additive trend + Fourier seasonality + event-dummy model, per §10.2's "Additive
GAM / regularized linear" description, built on the `statsforecast` library's
`MSTL` (STL decomposition) for the historical trend/weekly-seasonal split, plus
a small linear regression of the MSTL remainder on known-at-inference event
dummies (holiday/promo) and a yearly Fourier pair.

Why not MSTL's own `.predict()`/`.forecast()`: those return only a single
blended point forecast, not a decomposed one, and its default trend_forecaster
(AutoETS) tends to flatten a smooth STL trend to "no trend" in-sample (checked
interactively) — an unreliable base for a 140-day horizon. Instead:
- trend: linear extrapolation of the STL trend's recent slope (robust, and
  trivially defined for any future date — no library forecast call needed).
- seasonality_weekly: the last observed `season_length`-day STL seasonal
  cycle, tiled forward by day-of-week (seasonal-naive extrapolation).
- seasonality_yearly + holiday_lift + promo_lift: a linear regression of the
  STL remainder against holiday_flag, holiday_weight, promo_active_flag,
  promo_diversity, and a yearly Fourier pair (sin/cos of day-of-year) — all
  known-at-inference for any future date. holiday_weight (severity 0-1) and
  promo_diversity (ordinal 0-3, data/features/event_features.py's
  _TACTIC_RANK) ride alongside the plain binary flags so different holiday
  severities and promo tactics get distinct learned lift, not one flat
  coefficient shared by every event.

Because every term above is deterministic given a future date_id (no term
needs a "forecast" of itself), decompose()'s components sum to `yhat` exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.models.base import Decomposition, ForecastModel

_HOLIDAY_COLUMN = "holiday_flag"
_PROMO_COLUMN = "promo_active_flag"
_HOLIDAY_WEIGHT_COLUMN = "holiday_weight"
_PROMO_DIVERSITY_COLUMN = "promo_active_flag"
_DEFAULT_SEASON_LENGTH = 7
_DEFAULT_TREND_WINDOW_DAYS = 90
_YEARLY_PERIOD_DAYS = 365.25


class StatsForecastModel(ForecastModel):
    """Additive trend + Fourier seasonality + event-dummy model via `statsforecast`'s MSTL."""

    model_type = "statsforecast"
    evaluation_cost = 1  # MSTL vectorisable; fast per-series in batch

    def __init__(self) -> None:
        self._n: int = 0
        self._season_length: int = _DEFAULT_SEASON_LENGTH
        self._trend_slope: float = 0.0
        self._trend_intercept: float = 0.0
        self._last_season_cycle: np.ndarray | None = None
        self._last_dow: int = 0
        self._coefs: dict[str, float] = {}
        self._fitted_values: pd.Series | None = None
        # Batch-engine path: stored forecast arrays (no coef-based reconstruction needed)
        self._stored_forecast: np.ndarray | None = None
        self._forecast_origin: str | None = None

    def fit(self, series_pdf: pd.DataFrame, known_future_pdf: pd.DataFrame, params: dict) -> "StatsForecastModel":
        from statsforecast.models import MSTL

        pdf = series_pdf.sort_values("date_id").reset_index(drop=True)

        # Recency truncation (config-driven via params["recency_decay"])
        _decay = float(params.get("recency_decay", 0))
        if _decay > 0:
            from training.models.base import apply_recency_truncation
            pdf = apply_recency_truncation(pdf, _decay)

        y = pdf["qty"].astype(float).to_numpy()
        n = len(y)
        self._n = n
        self._season_length = int(params.get("season_length", _DEFAULT_SEASON_LENGTH))

        mstl = MSTL(season_length=[self._season_length]).fit(y)
        trend_h = mstl.model_["trend"].to_numpy()
        seasonal_h = mstl.model_["seasonal"].to_numpy()
        remainder_h = mstl.model_["remainder"].to_numpy()

        trend_window = min(n, int(params.get("trend_window_days", _DEFAULT_TREND_WINDOW_DAYS)))
        t_idx = np.arange(n)
        self._trend_slope, self._trend_intercept = np.polyfit(
            t_idx[-trend_window:], trend_h[-trend_window:], deg=1
        )

        self._last_season_cycle = seasonal_h[-self._season_length :]
        self._last_dow = int(pd.to_datetime(pdf["date_id"].iloc[-1]).dayofweek)

        doy = pd.to_datetime(pdf["date_id"]).dt.dayofyear.to_numpy()
        holiday = pdf[_HOLIDAY_COLUMN].fillna(0).to_numpy() if _HOLIDAY_COLUMN in pdf.columns else np.zeros(n)
        promo = pdf[_PROMO_COLUMN].fillna(0).to_numpy() if _PROMO_COLUMN in pdf.columns else np.zeros(n)
        holiday_weight = (
            pdf[_HOLIDAY_WEIGHT_COLUMN].fillna(0).to_numpy() if _HOLIDAY_WEIGHT_COLUMN in pdf.columns else np.zeros(n)
        )
        promo_diversity = (
            pdf[_PROMO_DIVERSITY_COLUMN].fillna(0).to_numpy() if _PROMO_DIVERSITY_COLUMN in pdf.columns else np.zeros(n)
        )
        design = np.column_stack(
            [
                np.ones(n),
                holiday,
                promo,
                holiday_weight,
                promo_diversity,
                np.sin(2 * np.pi * doy / _YEARLY_PERIOD_DAYS),
                np.cos(2 * np.pi * doy / _YEARLY_PERIOD_DAYS),
            ]
        )
        coefs, *_ = np.linalg.lstsq(design, remainder_h, rcond=None)
        self._coefs = {
            "intercept": float(coefs[0]),
            "holiday": float(coefs[1]),
            "promo": float(coefs[2]),
            "holiday_weight": float(coefs[3]),
            "promo_diversity": float(coefs[4]),
            "yearly_sin": float(coefs[5]),
            "yearly_cos": float(coefs[6]),
        }
        fitted_remainder = design @ coefs
        self._fitted_values = pd.Series(trend_h + seasonal_h + fitted_remainder, index=pdf.index)
        return self

    def fitted_values(self) -> pd.Series:
        assert self._fitted_values is not None, "fit() must be called before fitted_values()"
        return self._fitted_values

    def get_storable_params(self, input_params: dict) -> dict:
        """Return MSTL state + Ridge event coefficients. Fully describes the model
        for inference — no library object or refit needed."""
        return {
            "baseline_params": {
                "season_length":     self._season_length,
                "trend_window_days": int(input_params.get("trend_window_days", _DEFAULT_TREND_WINDOW_DAYS)),
                "train_length":      self._n,
                "trend_slope":       self._trend_slope,
                "trend_intercept":   self._trend_intercept,
                "last_dow":          self._last_dow,
            },
            "last_season_cycle": self._last_season_cycle.tolist() if self._last_season_cycle is not None else [],
            "coefs": self._coefs,
            "event_coefficients": {
                _HOLIDAY_COLUMN:            self._coefs.get("holiday", 0.0),
                _PROMO_COLUMN:              self._coefs.get("promo",   0.0),
                _HOLIDAY_WEIGHT_COLUMN:     self._coefs.get("holiday_weight",     0.0),
                _PROMO_DIVERSITY_COLUMN:  self._coefs.get("promo_diversity",  0.0),
            },
            "model_artifact": "statsforecast_stateless",
        }

    def serialize(self) -> str | None:
        """MSTL is fully described by slope, intercept, season cycle, and coefs."""
        return "statsforecast_stateless"

    @classmethod
    def from_serialized(cls, params: dict) -> "StatsForecastModel":
        """Reconstruct fitted MSTL from stored params — no refit needed.

        Supports two formats:
          1. Coefficient-based (per-series engine): baseline_params + last_season_cycle + coefs
          2. Forecast-array (batch engine / fit_store_cv): ets_forecast_values + ets_forecast_origin
        """
        obj = cls.__new__(cls)
        obj._fitted_values = None
        obj._stored_forecast = None
        obj._forecast_origin = None

        # Batch-engine path: raw MSTL forecast arrays stored by fit_store_cv
        forecast_vals = params.get("ets_forecast_values")
        if forecast_vals and len(forecast_vals) > 0 and not params.get("baseline_params"):
            obj._stored_forecast = np.asarray(forecast_vals, dtype=float)
            obj._forecast_origin = params.get("ets_forecast_origin")
            obj._n = 0
            obj._season_length = int(params.get("season_length", _DEFAULT_SEASON_LENGTH))
            obj._trend_slope = 0.0
            obj._trend_intercept = 0.0
            obj._last_dow = 0
            obj._last_season_cycle = np.zeros(obj._season_length)
            obj._coefs = {"intercept": 0.0, "holiday": 0.0, "promo": 0.0,
                          "holiday_weight": 0.0, "promo_diversity": 0.0,
                          "yearly_sin": 0.0, "yearly_cos": 0.0}
            return obj

        # Per-series engine path: coefficient-based reconstruction
        bp = params.get("baseline_params", {})
        obj._n = int(bp.get("train_length", 0))
        obj._season_length = int(bp.get("season_length", _DEFAULT_SEASON_LENGTH))
        obj._trend_slope = float(bp.get("trend_slope", 0.0))
        obj._trend_intercept = float(bp.get("trend_intercept", 0.0))
        obj._last_dow = int(bp.get("last_dow", 0))
        obj._last_season_cycle = np.asarray(
            params.get("last_season_cycle", [0.0] * obj._season_length), dtype=float
        )
        obj._coefs = params.get("coefs", {
            "intercept": 0.0, "holiday": 0.0, "promo": 0.0,
            "holiday_weight": 0.0, "promo_diversity": 0.0,
            "yearly_sin": 0.0, "yearly_cos": 0.0,
        })
        return obj

    def _components(self, horizon_pdf: pd.DataFrame) -> dict[str, pd.Series]:
        """Every future term is a deterministic function of the future date (or a known-at-inference
        regressor), so nothing here needs its own forecast — assumes horizon_pdf is contiguous
        immediately after the training series (true for both real scoring and backtest folds).

        When _stored_forecast is available (batch-engine path), uses the pre-computed MSTL
        forecast directly and derives trend/seasonal via rolling-mean split.
        """
        index = horizon_pdf.index
        h = len(horizon_pdf)

        # ─ Fast path: stored forecast arrays from batch engine ────────────────
        if self._stored_forecast is not None:
            n = len(self._stored_forecast)
            # Re-anchor to inference dates using forecast origin
            offset = 0
            if self._forecast_origin and "date_id" in horizon_pdf.columns and h:
                try:
                    first_ts = pd.Timestamp(horizon_pdf["date_id"].iloc[0])
                    offset = (first_ts - pd.Timestamp(self._forecast_origin)).days - 1
                except Exception:
                    offset = 0
            offset = max(0, min(offset, n - h))
            base = self._stored_forecast[offset:offset + h]

            # Derive trend/seasonal from baseline via rolling mean
            w = max(2, self._season_length)
            base_s = pd.Series(base, index=index)
            trend = base_s.rolling(window=w, center=True, min_periods=1).mean()
            seasonality_weekly = base_s - trend
            # Event lift from horizon features
            holiday_flag = (
                horizon_pdf[_HOLIDAY_COLUMN].fillna(0).to_numpy() if _HOLIDAY_COLUMN in horizon_pdf.columns else np.zeros(h)
            )
            promo_flag = (
                horizon_pdf[_PROMO_COLUMN].fillna(0).to_numpy() if _PROMO_COLUMN in horizon_pdf.columns else np.zeros(h)
            )
            holiday_weight_vals = (
                horizon_pdf[_HOLIDAY_WEIGHT_COLUMN].fillna(0).to_numpy()
                if _HOLIDAY_WEIGHT_COLUMN in horizon_pdf.columns else np.zeros(h)
            )
            promo_diversity_vals = (
                horizon_pdf[_PROMO_DIVERSITY_COLUMN].fillna(0).to_numpy()
                if _PROMO_DIVERSITY_COLUMN in horizon_pdf.columns else np.zeros(h)
            )
            # Use event_coefficients from params (stored by fit_store_cv). .get(..., 0.0)
            # so champions stored before holiday_weight/promo_diversity were added still
            # reconstruct fine (both terms just contribute 0).
            ev_coefs = self._coefs
            holiday_lift = {
                _HOLIDAY_COLUMN:        pd.Series(ev_coefs.get("holiday", 0.0) * holiday_flag, index=index),
                _HOLIDAY_WEIGHT_COLUMN: pd.Series(ev_coefs.get("holiday_weight", 0.0) * holiday_weight_vals, index=index),
            }
            promo_lift = {
                _PROMO_COLUMN:             pd.Series(ev_coefs.get("promo", 0.0) * promo_flag, index=index),
                _PROMO_DIVERSITY_COLUMN: pd.Series(ev_coefs.get("promo_diversity", 0.0) * promo_diversity_vals, index=index),
            }
            seasonality_yearly = pd.Series(0.0, index=index)
            temporal_features_effect = pd.Series(0.0, index=index)
            yhat = (
                base_s
                + holiday_lift[_HOLIDAY_COLUMN] + holiday_lift[_HOLIDAY_WEIGHT_COLUMN]
                + promo_lift[_PROMO_COLUMN] + promo_lift[_PROMO_DIVERSITY_COLUMN]
            )
            return {
                "trend": trend,
                "seasonality_weekly": seasonality_weekly,
                "seasonality_yearly": seasonality_yearly,
                "holiday_lift": holiday_lift,
                "promo_lift": promo_lift,
                "temporal_features_effect": temporal_features_effect,
                "yhat": yhat,
            }

        # ─ Coefficient path: per-series engine reconstruction ─────────────────
        future_t = np.arange(self._n, self._n + h)
        trend = pd.Series(self._trend_slope * future_t + self._trend_intercept, index=index)

        assert self._last_season_cycle is not None
        start_offset = (self._last_dow + 1) % self._season_length
        seasonality_weekly = pd.Series(
            [self._last_season_cycle[(start_offset + i) % self._season_length] for i in range(h)], index=index
        )

        doy = pd.to_datetime(horizon_pdf["date_id"]).dt.dayofyear.to_numpy()
        seasonality_yearly = pd.Series(
            self._coefs["yearly_sin"] * np.sin(2 * np.pi * doy / _YEARLY_PERIOD_DAYS)
            + self._coefs["yearly_cos"] * np.cos(2 * np.pi * doy / _YEARLY_PERIOD_DAYS),
            index=index,
        )

        holiday_flag = (
            horizon_pdf[_HOLIDAY_COLUMN].fillna(0).to_numpy() if _HOLIDAY_COLUMN in horizon_pdf.columns else np.zeros(h)
        )
        promo_flag = (
            horizon_pdf[_PROMO_COLUMN].fillna(0).to_numpy() if _PROMO_COLUMN in horizon_pdf.columns else np.zeros(h)
        )
        holiday_weight_vals = (
            horizon_pdf[_HOLIDAY_WEIGHT_COLUMN].fillna(0).to_numpy()
            if _HOLIDAY_WEIGHT_COLUMN in horizon_pdf.columns else np.zeros(h)
        )
        promo_diversity_vals = (
            horizon_pdf[_PROMO_DIVERSITY_COLUMN].fillna(0).to_numpy()
            if _PROMO_DIVERSITY_COLUMN in horizon_pdf.columns else np.zeros(h)
        )
        # .get(..., 0.0): champions fit before holiday_weight/promo_diversity were added
        # to the design matrix have no such keys in self._coefs — both terms contribute 0.
        holiday_lift = {
            _HOLIDAY_COLUMN:        pd.Series(self._coefs["holiday"] * holiday_flag, index=index),
            _HOLIDAY_WEIGHT_COLUMN: pd.Series(self._coefs.get("holiday_weight", 0.0) * holiday_weight_vals, index=index),
        }
        promo_lift = {
            _PROMO_COLUMN:             pd.Series(self._coefs["promo"] * promo_flag, index=index),
            _PROMO_DIVERSITY_COLUMN: pd.Series(self._coefs.get("promo_diversity", 0.0) * promo_diversity_vals, index=index),
        }
        temporal_features_effect = pd.Series(self._coefs["intercept"], index=index)

        yhat = (
            trend
            + seasonality_weekly
            + seasonality_yearly
            + holiday_lift[_HOLIDAY_COLUMN] + holiday_lift[_HOLIDAY_WEIGHT_COLUMN]
            + promo_lift[_PROMO_COLUMN] + promo_lift[_PROMO_DIVERSITY_COLUMN]
            + temporal_features_effect
        )
        return {
            "trend": trend,
            "seasonality_weekly": seasonality_weekly,
            "seasonality_yearly": seasonality_yearly,
            "holiday_lift": holiday_lift,
            "promo_lift": promo_lift,
            "temporal_features_effect": temporal_features_effect,
            "yhat": yhat,
        }

    def predict(self, horizon_pdf: pd.DataFrame) -> pd.Series:
        return self._components(horizon_pdf)["yhat"]

    def decompose(self, horizon_pdf: pd.DataFrame) -> Decomposition:
        c = self._components(horizon_pdf)
        return Decomposition(
            trend=c["trend"],
            seasonality_weekly=c["seasonality_weekly"],
            seasonality_yearly=c["seasonality_yearly"],
            holiday_lift=c["holiday_lift"],
            promo_lift=c["promo_lift"],
            temporal_features_effect=c["temporal_features_effect"],
            yhat=c["yhat"],
        )
