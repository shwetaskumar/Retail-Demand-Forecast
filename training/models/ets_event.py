"""
training/models/ets_event.py — Fast decomposable model: AutoETS + Ridge event regression.

WHY THIS EXISTS — the Prophet bottleneck:
  Prophet uses Stan MCMC sampling: ~2-5 s per series, cannot be vectorised.
  With 106K series that means 35+ minutes even on 256 parallel Spark cores.

  StatsForecast AutoETS uses Numba/Cython vectorised math: ~8 ms per series
  inside a batch call. With the batch engine (666 tasks of ~160 series each)
  the same 106K series fit in ~3 minutes.

DECOMPOSITION CONTRACT (docs §10.1) — exactly satisfied:
  yhat = trend + seasonality_weekly + seasonality_yearly
       + Σ holiday_lift + Σ promo_lift
       + temporal_features_effect

   How:
     trend              = ETS level component projected h steps ahead
     seasonality_weekly = ETS 7-day seasonal component projected h steps ahead
     seasonality_yearly = 0  (ETS handles one seasonality; yearly absorbed into trend)
     holiday_lift       = {event_name: Ridge_coeff × holiday_indicator_series}
     promo_lift         = {event_name: Ridge_coeff × promo_indicator_series}
     temporal_features_effect = ETS_forecast - (trend + seasonality_weekly)
                          (≈0 for AAA models; absorbs multiplicative rounding)
    yhat               = ETS_forecast + Σ holiday_lift + Σ promo_lift

  Reconstruction is exact by construction:
    trend + seasonality_weekly + temporal_features_effect = ETS_forecast  (by definition)
    → LHS = ETS_forecast + event_lifts = yhat  ✓

EVENT REGRESSION:
  After fitting ETS on the raw qty series (including event days), the in-sample
  ETS residuals (qty − ETS_fitted) carry the unexplained signal.  On event days
  this residual is dominated by the event effect.  We fit a Ridge regression
  (no intercept) of event indicator columns against these residuals.  The
  resulting coefficients are the per-event lift estimates used at inference.

  Event columns are auto-detected from the feature table by prefix:
    is_holiday*, holiday_*  → holiday_lift decomp bucket
    is_promo*,   promo_*    → promo_lift decomp bucket

INFERENCE (batch_infer.py _score_one_series):
  At inference time ETSEventModel.from_serialized() reconstructs the model from
  stored params (forecast arrays + event coefficients).  predict() and decompose()
  re-anchor the stored forecast to the actual inference dates via ets_forecast_origin,
  so daily scoring off a monthly model returns the correct dates (not index 0).
  When trend/seasonal component arrays weren't stored (batch-fit path), decompose()
  derives them from the combined baseline via _split_baseline() — reconstruction
  stays exact (trend + seasonal = baseline) with zero extra storage.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

from training.models.base import Decomposition, ForecastModel

_HOLIDAY_PREFIXES = ("is_holiday", "holiday_")
_PROMO_PREFIXES   = ("is_promo",   "promo_")
_EVENT_PREFIXES   = ("is_holiday", "is_promo", "holiday_", "promo_", "event_")


def _event_cols_from(pdf: pd.DataFrame) -> list[str]:
    """Event columns eligible for the Ridge design matrix: prefix-matched AND numeric.

    The event-feature prefix also matches string label columns (holiday_name,
    holiday_type, holiday_subtype, promo_subtype, promo_tactic) that describe the
    same event in human-readable form for reporting/BI, not as a model input —
    casting them to float crashes the Ridge fit the moment any real holiday/promo
    is present in a series' training window. The numeric survivors still include
    the tactic/severity signal (promo_tactic_rank, holiday_weight) alongside the
    plain binary flags, so tactic-level differentiation reaches the Ridge fit.
    """
    return [
        c for c in pdf.columns
        if any(c.startswith(p) for p in _EVENT_PREFIXES) and pd.api.types.is_numeric_dtype(pdf[c])
    ]


def _split_baseline(base, season_length: int):
    """Split a combined baseline forecast into (trend, seasonal) with trend+seasonal == base.

    Trend = centered rolling mean over the seasonal period; seasonal = base - trend.
    Used at inference when only the combined baseline array was stored (batch-fit path),
    so ETS decomposition still yields a real trend/seasonality split with no extra storage.
    """
    b = pd.Series(np.asarray(base, dtype=float))
    w = max(2, int(season_length))
    trend = b.rolling(window=w, center=True, min_periods=1).mean().to_numpy()
    seasonal = b.to_numpy() - trend
    return trend, seasonal


def _extract_ets_components(model_state: dict, h: int, season_length: int):
    """Project ETS level/slope/seasonal h steps ahead.

    Returns (trend_series, seasonal_series) both of length h.
    trend + seasonal = ETS_forecast for additive models.
    For multiplicative seasonal, seasonal = yhat - level_forecast so that
    trend + seasonal = yhat exactly.

    This function is intentionally defensive: if a model variant doesn't
    expose a given state array, it falls back to zero gracefully.
    """
    level    = np.asarray(model_state.get("level",  [0.0]), dtype=float)
    slope    = np.asarray(model_state.get("slope",  [0.0]), dtype=float)
    seasonal = np.asarray(model_state.get("season", []),    dtype=float)
    ets_type = str(model_state.get("model", "ANN"))   # e.g. "AAA", "MNM", "AAN"

    L = float(level[-1])   if len(level)  > 0 else 0.0
    b = float(slope[-1])   if len(slope)  > 0 else 0.0
    has_trend    = len(ets_type) > 1 and ets_type[1] in ("A", "d")
    has_seasonal = len(ets_type) > 2 and ets_type[2] in ("A", "M")
    mult_seasonal = len(ets_type) > 2 and ets_type[2] == "M"

    # Last full cycle of seasonal indices
    if has_seasonal and len(seasonal) >= season_length:
        last_seasons = seasonal[-season_length:]
    elif has_seasonal and len(seasonal) > 0:
        last_seasons = seasonal
    else:
        last_seasons = np.zeros(season_length)

    trend_vals    = []
    seasonal_vals = []

    for t in range(1, h + 1):
        L_t = L + b * t if has_trend else L
        trend_vals.append(float(L_t))

        if has_seasonal:
            s_t = float(last_seasons[(t - 1) % len(last_seasons)])
            if mult_seasonal:
                # Multiplicative: yhat = L_t * s_t → seasonality contribution = L_t*(s_t-1)
                seasonal_vals.append(float(L_t * (s_t - 1.0)))
            else:
                seasonal_vals.append(float(s_t))
        else:
            seasonal_vals.append(0.0)

    return pd.Series(trend_vals, dtype=float), pd.Series(seasonal_vals, dtype=float)


class ETSEventModel(ForecastModel):
    """AutoETS baseline + Ridge event regression.

    See module docstring for full architecture description.
    """

    model_type = "ets_event"
    evaluation_cost = 1  # batch-capable via statsforecast AutoETS

    def __init__(self, season_length: int = 7):
        self.season_length    = season_length
        self._sf              = None
        self._event_coeffs    = {}    # {col_name: float}
        self._model_state     = {}    # raw ETS state dict for decompose (refit path)
        self._n_train         = 0
        # Load-and-score: pre-computed ETS arrays from training (no refit needed)
        self._stored_forecast: np.ndarray | None = None
        self._stored_trend:    np.ndarray | None = None
        self._stored_seasonal: np.ndarray | None = None
        self._forecast_origin: str | None = None  # training origin; stored[0] = origin + 1 day

    def load_stored_state(self, params: dict, snapshot_date: str, first_horizon_date: str) -> bool:
        """Populate stored ETS arrays from champion params if available.

        Slices the stored 200-day arrays to align with the requested horizon.
        Returns True if stored state was loaded, False if fallback refit is needed.
        """
        vals     = params.get("ets_forecast_values")
        trend    = params.get("ets_trend_values")
        seasonal = params.get("ets_seasonal_values")
        origin   = params.get("ets_forecast_origin")
        if not vals or not origin:
            return False
        try:
            origin_ts = pd.Timestamp(origin)
            first_ts  = pd.Timestamp(first_horizon_date)
            offset    = (first_ts - origin_ts).days - 1   # vals[0] = origin+1
            if offset < 0 or offset >= len(vals):
                return False
            v = np.asarray(vals,     dtype=float)[offset:]
            t = np.asarray(trend,    dtype=float)[offset:] if trend    else np.zeros(len(v))
            s = np.asarray(seasonal, dtype=float)[offset:] if seasonal else np.zeros(len(v))
            self._stored_forecast = v
            self._stored_trend    = t
            self._stored_seasonal = s
            self._forecast_origin = origin
            self._event_coeffs    = params.get("event_coefficients", {})
            return True
        except Exception:
            return False

    # ── fit ────────────────────────────────────────────────────────────────────

    def fit(self, series_pdf: pd.DataFrame, known_future_pdf: pd.DataFrame, params: dict) -> "ETSEventModel":
        from sklearn.linear_model import Ridge
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        season_length = int(params.get("season_length", self.season_length))

        # At inference time, stored event_coefficients are passed via params —
        # skip Ridge re-fitting and use them directly.
        stored_coeffs = params.get("event_coefficients")
        if stored_coeffs is not None:
            self._event_coeffs = stored_coeffs
            _need_ridge = False
        else:
            _need_ridge = True

        # Recency truncation (config-driven via params["recency_decay"])
        _decay = float(params.get("recency_decay", 0))
        if _decay > 0:
            from training.models.base import apply_recency_truncation
            _sorted_pdf = series_pdf.sort_values("date_id").reset_index(drop=True)
            _sorted_pdf = apply_recency_truncation(_sorted_pdf, _decay)
        else:
            _sorted_pdf = series_pdf

        y  = _sorted_pdf["qty"].fillna(0.0).clip(lower=0.0).values.astype(float)
        ds = pd.to_datetime(_sorted_pdf["date_id"])

        sf_df = pd.DataFrame({"unique_id": "s", "ds": ds, "y": y})

        self._sf = StatsForecast(
            models=[AutoETS(season_length=season_length)],
            freq="D",
            n_jobs=1,
        )
        self._sf.fit(sf_df)
        self._n_train = len(sf_df)

        # Cache ETS model state for decompose()
        fitted_obj = self._sf.fitted_[0, 0]
        self._model_state = dict(getattr(fitted_obj, "model_", {}))

        # Event Ridge regression on in-sample residuals
        if _need_ridge:
            ev_cols = _event_cols_from(series_pdf)
            if ev_cols:
                in_sample_pred = np.asarray(
                    self._model_state.get("fitted", np.full(len(y), np.nan)), dtype=float
                )[:len(y)]
                residuals = y - in_sample_pred
                X = series_pdf[ev_cols].fillna(0).values.astype(float)
                active_mask = X.sum(axis=0) > 0
                active_cols = [c for c, m in zip(ev_cols, active_mask) if m]
                X_a = X[:, active_mask]
                if X_a.shape[1] > 0:
                    res_clean = np.nan_to_num(residuals, nan=0.0)
                    ridge = Ridge(alpha=1.0, fit_intercept=False)
                    if _decay > 0:
                        from training.models.base import compute_recency_weights
                        _rw = compute_recency_weights(len(res_clean), _decay)
                        ridge.fit(X_a, res_clean, sample_weight=_rw)
                    else:
                        ridge.fit(X_a, res_clean)
                    self._event_coeffs = {c: float(v) for c, v in zip(active_cols, ridge.coef_)}

        return self

    # ── stored-state alignment ──────────────────────────────────────────────

    def _aligned_stored(self, horizon_pdf: pd.DataFrame):
        """Return (baseline, trend, seasonal) stored arrays aligned to horizon_pdf's dates.

        Re-anchors the stored forecast to the inference origin using ets_forecast_origin
        (stored[0] corresponds to origin + 1 day), so daily scoring off a monthly-trained
        model returns the correct dates instead of a shifted forecast.  Derives trend/seasonal
        from the baseline when component arrays were not stored (batch-fit path).  Returns
        None when there is no usable stored window.
        """
        arr = self._stored_forecast
        if arr is None:
            return None
        h = len(horizon_pdf)
        n = len(arr)
        if n < h:
            return None
        offset = 0
        origin = self._forecast_origin
        if origin and "date_id" in horizon_pdf.columns and h:
            try:
                first_ts = pd.Timestamp(horizon_pdf["date_id"].iloc[0])
                offset = (first_ts - pd.Timestamp(origin)).days - 1  # stored[0] = origin + 1
            except Exception:
                offset = 0
        offset = max(0, min(offset, n - h))  # clamp into the stored window
        base = np.asarray(arr, dtype=float)[offset:offset + h]

        trend, seas = self._stored_trend, self._stored_seasonal
        have_components = (
            trend is not None and seas is not None
            and len(trend) >= offset + h and len(seas) >= offset + h
            and (np.any(trend != 0) or np.any(seas != 0))
        )
        if have_components:
            trend_arr = np.asarray(trend, dtype=float)[offset:offset + h]
            seas_arr  = np.asarray(seas,  dtype=float)[offset:offset + h]
        else:
            # Derive trend/seasonal from baseline — reconstruction stays exact
            trend_arr, seas_arr = _split_baseline(base, self.season_length)
        return base, trend_arr, seas_arr

    # ── predict ────────────────────────────────────────────────────────────────

    def predict(self, horizon_pdf: pd.DataFrame) -> pd.Series:
        h = len(horizon_pdf)

        # Fast path: re-anchored stored baseline (no refit at inference)
        aligned = self._aligned_stored(horizon_pdf)
        if aligned is not None:
            yhat_base = aligned[0]
        else:
            ets_fcst  = self._sf.predict(h=h)
            yhat_base = ets_fcst["AutoETS"].values.astype(float)

        event_lift = np.zeros(h, dtype=float)
        for col, coeff in self._event_coeffs.items():
            if col in horizon_pdf.columns:
                event_lift += coeff * horizon_pdf[col].fillna(0).values.astype(float)

        return pd.Series(np.maximum(0.0, yhat_base + event_lift))

    # ── decompose ──────────────────────────────────────────────────────────────

    def decompose(self, horizon_pdf: pd.DataFrame) -> Decomposition:
        h = len(horizon_pdf)

        # ETS baseline + structural components — re-anchored to inference origin;
        # trend/seasonal derived from the baseline when component arrays weren't stored.
        aligned = self._aligned_stored(horizon_pdf)
        if aligned is not None:
            base_arr, trend_arr, seasonal_arr = aligned
            yhat_ets   = pd.Series(base_arr)
            trend_s    = pd.Series(trend_arr)
            seasonal_s = pd.Series(seasonal_arr)
        else:
            ets_fcst = self._sf.predict(h=h)
            yhat_ets = pd.Series(ets_fcst["AutoETS"].values.astype(float))
            trend_s, seasonal_s = _extract_ets_components(
                self._model_state, h, self.season_length
            )

        # temporal_features_effect absorbs rounding gap so reconstruction is exact
        non_promo = yhat_ets - (trend_s + seasonal_s)

        # Event lift decomposition
        holiday_lift: dict[str, pd.Series] = {}
        promo_lift:   dict[str, pd.Series] = {}

        for col, coeff in self._event_coeffs.items():
            vals = horizon_pdf[col].fillna(0).values.astype(float) if col in horizon_pdf.columns else np.zeros(h)
            lift = pd.Series(coeff * vals)
            if any(col.startswith(p) for p in _HOLIDAY_PREFIXES):
                holiday_lift[col] = lift
            else:
                promo_lift[col] = lift

        # Ensure dicts are non-empty (decomposer sums over .values())
        zeros = pd.Series(np.zeros(h, dtype=float))
        if not holiday_lift:
            holiday_lift = {"holiday": zeros.copy()}
        if not promo_lift:
            promo_lift = {"promo": zeros.copy()}

        # Total event lift
        total_event_lift = sum(holiday_lift.values(), pd.Series(np.zeros(h, dtype=float)))
        total_event_lift = total_event_lift + sum(promo_lift.values(), pd.Series(np.zeros(h, dtype=float)))

        yhat_final = yhat_ets + total_event_lift

        return Decomposition(
            trend              = trend_s,
            seasonality_weekly = seasonal_s,
            seasonality_yearly = zeros.copy(),
            temporal_features_effect = non_promo,
            holiday_lift       = holiday_lift,
            promo_lift         = promo_lift,
            yhat               = yhat_final,
        )

    # ── fitted_values (for uplift.py residual computation) ────────────────────

    def fitted_values(self) -> pd.Series:
        if self._sf is None:
            raise RuntimeError("Must call fit() first")
        fv = np.asarray(self._model_state.get("fitted", []), dtype=float)
        return pd.Series(fv)

    # ── params for champion index (stored as JSON) ─────────────────────────────

    def get_storable_params(self, input_params: dict) -> dict:
        """Return structured params for champion_index.params.

        Stores pre-computed 200-step forecast + trend + seasonal arrays so
        inference can skip refit entirely via from_serialized().
        """
        _H = 200  # store enough steps for any horizon (140 + safety margin)
        forecast_values = []
        trend_values = []
        seasonal_values = []
        if self._sf is not None:
            try:
                fc = self._sf.predict(h=_H)
                forecast_values = fc["AutoETS"].values.astype(float).tolist()
                t, s = _extract_ets_components(self._model_state, _H, self.season_length)
                trend_values = t.tolist()
                seasonal_values = s.tolist()
            except Exception:
                pass
        return {
            "baseline_params": {
                "season_length": self.season_length,
                "ets_type":      str(self._model_state.get("model", "unknown")),
            },
            "event_coefficients":  self._event_coeffs,
            "ets_forecast_origin": input_params.get("ets_forecast_origin"),
            "ets_forecast_values": forecast_values,
            "ets_trend_values":    trend_values,
            "ets_seasonal_values": seasonal_values,
            "model_artifact":      "ets_stored_forecast" if forecast_values else None,
        }

    def serialize(self) -> str | None:
        """ETS stores forecast values as its artifact."""
        return "ets_stored_forecast" if self._sf is not None else None

    @classmethod
    def from_serialized(cls, params: dict) -> "ETSEventModel":
        """Reconstruct from stored forecast values — no refit needed."""
        forecast_values = params.get("ets_forecast_values", [])
        if not forecast_values:
            raise NotImplementedError("No stored ets_forecast_values")
        obj = cls.__new__(cls)
        bp = params.get("baseline_params", {})
        obj.season_length = int(bp.get("season_length", 7))
        obj._sf = None
        obj._model_state = {}
        obj._n_train = 0
        obj._event_coeffs = params.get("event_coefficients", {})
        obj._forecast_origin = params.get("ets_forecast_origin")
        obj._stored_forecast = np.asarray(forecast_values, dtype=float)
        obj._stored_trend = np.asarray(params.get("ets_trend_values", []), dtype=float)
        obj._stored_seasonal = np.asarray(params.get("ets_seasonal_values", []), dtype=float)
        if len(obj._stored_trend) == 0:
            obj._stored_trend = np.zeros(len(obj._stored_forecast))
        if len(obj._stored_seasonal) == 0:
            obj._stored_seasonal = np.zeros(len(obj._stored_forecast))
        return obj
