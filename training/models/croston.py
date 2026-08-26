"""
training/models/croston.py — Croston/SBA model for intermittent demand (docs/instructions.md §10.2).

WHY THIS EXISTS:
  D-category (and many C-category) series have a large fraction of zero-demand
  days.  Standard models (ETS, LGBM, Prophet) fit the envelope of non-zero days
  and produce smoothly positive forecasts, which means they almost always
  over-forecast intermittent series — exactly the catastrophic +1932% bias
  seen for Prophet C in the Jan-2026 evaluation snapshot.

  Croston's method (1972) explicitly models two separate processes:
    - z_t: the demand SIZE on non-zero demand days (exponential smoothing on
           the sub-series of non-zero observations).
    - p_t: the demand INTERVAL between non-zero days (exponential smoothing on
           the inter-arrival times).
  The per-period expected demand is then z_t / p_t.

  The Syntetos-Boylan Approximation (SBA, 2005) adds a bias-correction factor
  (1 - alpha/2) that empirically reduces the systematic over-forecast inherent
  in basic Croston.

DECOMPOSITION CONTRACT (docs §10.1):
  Croston/SBA does not have explicit trend/seasonality components in the
  classical sense.  We satisfy the decomposition contract by:
    trend              = yhat_base (the SBA level forecast, flat for intermittent
                         series — these are typically stationary in their
                         non-zero level between demand events)
    seasonality_weekly = 0  (intermittent demand has no reliable weekly pattern)
    seasonality_yearly = 0
    holiday_lift       = event_coeff × holiday_flag  (Ridge, same as ETSEvent)
    promo_lift         = event_coeff × promo_flag    (Ridge)
    temporal_features_effect = 0
    yhat               = trend + Σ holiday_lift + Σ promo_lift

  This satisfies the additive identity by construction.  The trade-off:
  seasonality is sacrificed — appropriate for D-category series where the
  "seasonal" signal is noise rather than pattern at daily grain.

WHEN TO USE:
  The model_type = "croston" row in config_model is enabled by default.
  The layered engine's category routing (training/engine_layered.py) should
  place Croston as the FIRST candidate for D and zero-heavy C series by setting
  a low evaluation_cost — the stats layer already does this; Croston slips in
  as an additional candidate at cost=0 (even faster than ETS).

PARAMETERS (stored in champion_index.params):
  alpha_z: float [0, 1] — smoothing for demand size (default 0.1)
  alpha_p: float [0, 1] — smoothing for inter-arrival time (default 0.1)
  event_coefficients: dict[str, float] — Ridge event lifts (same as ETSEvent)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.models.base import Decomposition, ForecastModel

_DEFAULT_ALPHA = 0.1       # smoothing parameter for both z and p processes
_HOLIDAY_PREFIXES = ("is_holiday", "holiday_")
_PROMO_PREFIXES = ("is_promo", "promo_")

# Minimum non-zero demand observations to fit Croston reliably.
# Series with fewer non-zero days fall back to a simple demand-rate average.
_MIN_NONZERO_OBS = 5


def _detect_event_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (holiday_cols, promo_cols) from df.columns using prefix matching, numeric only.

    The prefix also matches string label columns (holiday_name, promo_subtype,
    promo_tactic, ...) that describe the same event for reporting, not as a Ridge
    input — see training/models/ets_event.py::_event_cols_from for the crash this
    guards against. The numeric survivors include promo_tactic_rank/holiday_weight,
    so tactic-level differentiation still reaches this model's Ridge fit.
    """
    holiday_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in _HOLIDAY_PREFIXES) and pd.api.types.is_numeric_dtype(df[c])
    ]
    promo_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in _PROMO_PREFIXES) and pd.api.types.is_numeric_dtype(df[c])
    ]
    return holiday_cols, promo_cols


class CrostonModel(ForecastModel):
    """Croston/SBA model for intermittent (zero-inflated) demand series.

    Appropriate for D-category and zero-heavy C-category series where standard
    smooth-demand models systematically over-forecast.
    """

    model_type = "croston"
    evaluation_cost = 0   # Faster than ETS: pure NumPy, no external library call.

    def __init__(self) -> None:
        self._alpha_z: float = _DEFAULT_ALPHA
        self._alpha_p: float = _DEFAULT_ALPHA
        self._z_level: float = 0.0          # demand size estimate at last non-zero obs
        self._p_level: float = 1.0          # inter-arrival time estimate
        self._event_coefs: dict[str, float] = {}
        self._holiday_cols: list[str] = []
        self._promo_cols: list[str] = []
        self._fitted_values: pd.Series | None = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        series_pdf: pd.DataFrame,
        known_future_pdf: pd.DataFrame,
        params: dict,
    ) -> "CrostonModel":
        """Fit Croston/SBA on `series_pdf.qty`.

        1. Run the Croston update equations to estimate z (demand size) and
           p (inter-arrival time) at the end of the training window.
        2. Fit a Ridge regression of event indicators against (qty - yhat_sba)
           on event days to isolate holiday/promo lift — same approach as ETSEvent.
        """
        pdf = series_pdf.sort_values("date_id").reset_index(drop=True)

        # Recency truncation (config-driven via params["recency_decay"])
        _decay = float(params.get("recency_decay", 0))
        if _decay > 0:
            from training.models.base import apply_recency_truncation
            pdf = apply_recency_truncation(pdf, _decay)

        y = pdf["qty"].fillna(0).astype(float).to_numpy()
        n = len(y)

        self._alpha_z = float(params.get("alpha_z", _DEFAULT_ALPHA))
        self._alpha_p = float(params.get("alpha_p", _DEFAULT_ALPHA))

        # ── Croston update equations ──────────────────────────────────────────
        nonzero_idx = np.where(y > 0)[0]
        if len(nonzero_idx) < _MIN_NONZERO_OBS:
            # Degenerate: use mean of non-zero days as flat level.
            self._z_level = float(y[y > 0].mean()) if (y > 0).any() else 0.0
            self._p_level = float(n / max(len(nonzero_idx), 1))
        else:
            z = float(y[nonzero_idx[0]])  # initialise with first non-zero demand
            p = float(nonzero_idx[0] + 1) if nonzero_idx[0] > 0 else 1.0
            prev_nonzero = nonzero_idx[0]
            for k in nonzero_idx[1:]:
                interval = float(k - prev_nonzero)
                z = self._alpha_z * y[k] + (1 - self._alpha_z) * z
                p = self._alpha_p * interval + (1 - self._alpha_p) * p
                prev_nonzero = k
            self._z_level = z
            self._p_level = max(p, 1e-6)

        # SBA bias-correction factor: (1 - alpha/2)
        sba_factor = 1.0 - self._alpha_z / 2.0
        yhat_base = (sba_factor * self._z_level / self._p_level)

        # ── Event lift via Ridge (same pattern as ETSEvent) ──────────────────
        # Event lift via Ridge (no pruning — all event features kept)
        self._holiday_cols, self._promo_cols = _detect_event_cols(pdf)
        event_cols = self._holiday_cols + self._promo_cols

        if event_cols:
            residuals = y - yhat_base
            X = pdf[event_cols].fillna(0).astype(float).to_numpy()
            XtX = X.T @ X
            alpha_ridge = float(params.get("ridge_alpha", 1.0))
            coefs, *_ = np.linalg.lstsq(
                XtX + alpha_ridge * np.eye(len(event_cols)),
                X.T @ residuals,
                rcond=None,
            )
            self._event_coefs = {col: float(c) for col, c in zip(event_cols, coefs)}
        else:
            self._event_coefs = {}

        # Store fitted values for backtest scoring
        event_lift_train = np.zeros(n)
        for col, coef in self._event_coefs.items():
            if col in pdf.columns:
                event_lift_train += coef * pdf[col].fillna(0).astype(float).to_numpy()
        self._fitted_values = pd.Series(yhat_base + event_lift_train, index=pdf.index)
        return self

    def fitted_values(self) -> pd.Series:
        assert self._fitted_values is not None, "fit() must be called before fitted_values()"
        return self._fitted_values

    def get_storable_params(self, input_params: dict) -> dict:
        return {
            "baseline_params": {
                "alpha_z": self._alpha_z,
                "alpha_p": self._alpha_p,
                "z_level": self._z_level,
                "p_level": self._p_level,
            },
            "event_coefficients": self._event_coefs,
            "model_artifact": "croston_stateless",  # marker: model is fully defined by params above
        }

    def serialize(self) -> str | None:
        """Croston is fully described by z_level, p_level, event_coefs — no heavy artifact."""
        return "croston_stateless"

    @classmethod
    def from_serialized(cls, params: dict) -> "CrostonModel":
        """Reconstruct fitted Croston from stored params — no historical refit needed."""
        obj = cls.__new__(cls)
        bp = params.get("baseline_params", {})
        obj._alpha_z = float(bp.get("alpha_z", _DEFAULT_ALPHA))
        obj._alpha_p = float(bp.get("alpha_p", _DEFAULT_ALPHA))
        obj._z_level = float(bp.get("z_level", 0.0))
        obj._p_level = float(bp.get("p_level", 1.0))
        obj._event_coefs = params.get("event_coefficients", {})
        obj._holiday_cols = [c for c in obj._event_coefs if any(c.startswith(p) for p in _HOLIDAY_PREFIXES)]
        obj._promo_cols = [c for c in obj._event_coefs if any(c.startswith(p) for p in _PROMO_PREFIXES)]
        obj._fitted_values = None
        return obj

    # ── predict + decompose ───────────────────────────────────────────────────

    def _components(self, horizon_pdf: pd.DataFrame) -> dict:
        index = horizon_pdf.index

        # Flat SBA level (Croston has no trend extrapolation — appropriate for intermittent)
        sba_factor = 1.0 - self._alpha_z / 2.0
        yhat_base_val = sba_factor * self._z_level / max(self._p_level, 1e-6)
        trend = pd.Series(yhat_base_val, index=index)

        seasonality_weekly = pd.Series(0.0, index=index)
        seasonality_yearly = pd.Series(0.0, index=index)
        temporal_features_effect = pd.Series(0.0, index=index)

        holiday_lift: dict[str, pd.Series] = {}
        promo_lift: dict[str, pd.Series] = {}
        yhat = trend.copy()

        for col in self._holiday_cols:
            coef = self._event_coefs.get(col, 0.0)
            vals = horizon_pdf[col].fillna(0).astype(float) if col in horizon_pdf.columns else pd.Series(0.0, index=index)
            lift = pd.Series(coef * vals.to_numpy(), index=index)
            holiday_lift[col] = lift
            yhat = yhat + lift

        for col in self._promo_cols:
            coef = self._event_coefs.get(col, 0.0)
            vals = horizon_pdf[col].fillna(0).astype(float) if col in horizon_pdf.columns else pd.Series(0.0, index=index)
            lift = pd.Series(coef * vals.to_numpy(), index=index)
            promo_lift[col] = lift
            yhat = yhat + lift

        # Floor at zero: intermittent series cannot have negative demand.
        yhat = yhat.clip(lower=0.0)

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
