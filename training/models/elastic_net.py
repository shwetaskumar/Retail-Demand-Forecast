"""
training/models/elastic_net.py — ElasticNet model for low-to-medium volume series
(docs/instructions.md §10.2).

WHY THIS EXISTS:
  LGBM is a powerful model but requires enough data to avoid overfitting: it
  needs ~200+ non-zero rows to learn reliable splits reliably.  For B/C-category
  series (many with <50 non-zero days per year), the tree complexity becomes a
  liability — it overfits to noise.

  Elastic Net (L1 + L2 regularization on a linear model) works well here because:
    - It is strongly regularized — a sparse solution is forced even with many
      event feature columns, preventing the feature explosion problem.
    - It is interpretable: each coefficient maps directly to a decomposition component.
    - It is fast: scikit-learn coordinate descent is ~1ms per series.
    - It benefits directly from the new promo/holiday features (holiday_weight,
      promo_tactic_rank, promo_lead_flag, competing_promo_count) introduced in
      data/features/event_features.py and category_features.py, and from the
      halo/cannibalization partner features in data/features/halo_cannibalization_features.py.

DECOMPOSITION CONTRACT (docs §10.1):
  yhat = trend + seasonality_weekly + seasonality_yearly
       + Σ holiday_lift + Σ promo_lift + temporal_features_effect

  Implementation:
    trend              = intercept + slope × t  (linear trend from OLS pre-pass)
    seasonality_weekly = Σ{dow coefs} × dow dummies (6 dummies, one per DOW)
    seasonality_yearly = sin_coef × sin(2π×doy/365.25) + cos_coef × cos(2π×doy/365.25)
    holiday_lift       = holiday_weight coef × holiday_weight feature
                       + holiday_flag coef × holiday_flag (one per named event column)
    promo_lift         = promo_active_flag coef × promo_active_flag
                       + promo_tactic_rank coef × promo_tactic_rank
                       + promo_lead_flag coef × promo_lead_flag
    temporal_features_effect = 0  (absorbed into intercept)

  Reconstruction is exact: yhat = sum of above by construction.

FEATURE COLUMNS USED:
  Calendar:   trend (t index), DOW dummies (6), yearly Fourier pair (sin/cos doy)
  Event:      holiday_flag, holiday_weight (new), promo_active_flag,
              promo_tactic_rank (new), promo_lead_flag (new), competing_promo_count (new),
              all holiday_*/promo_* prefix columns (from ETSEvent pattern)
  Contextual: category_qty_7d_roll, other_category_qty_7d_roll, halo_partner_qty_7d_roll,
              cannib_partner_qty_7d_roll — HISTORICAL_ONLY, so only used during training
              (not available in future_spine at inference); they enter fit() but not
              predict()/decompose().
  Halo/cannib (known-at-inference): halo_partner_corr_strength, cannib_partner_corr_strength,
              halo_partner_promo_active_flag/count, cannib_partner_promo_active_flag/count
              (data/features/halo_cannibalization_features.py) — used in both fit() and
              predict()/decompose(), same treatment as promo_active_flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.models.base import Decomposition, ForecastModel

_DEFAULT_ALPHA = 0.01   # ElasticNet regularisation strength; tuned via HPT
_DEFAULT_L1_RATIO = 0.5  # 0=Ridge, 1=LASSO; 0.5=balanced default
_YEARLY_PERIOD = 365.25
_HOLIDAY_PREFIXES = ("is_holiday", "holiday_", "bridge_")
_PROMO_PREFIXES = ("is_promo", "promo_", "item_specific_promo_")

# Features that are historical-only (NOT available in future spine — never use in predict)
_HIST_ONLY_FEATURES = frozenset({
    "category_qty_7d_roll", "other_category_qty_7d_roll", "category_active_sku_count",
    "halo_partner_qty_7d_roll", "cannib_partner_qty_7d_roll",
})


def _detect_event_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Prefix-matched AND numeric event columns — see training/models/ets_event.py's
    `_event_cols_from` for why the numeric guard is required (string label columns
    like holiday_name/promo_subtype/promo_tactic share the same prefix but crash a
    float design matrix). The numeric survivors include promo_tactic_rank and
    holiday_weight, so tactic-level differentiation reaches the ElasticNet fit.
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


class ElasticNetModel(ForecastModel):
    """Regularised linear model (Elastic Net) for low-to-medium volume B/C series.

    Provides strong regularisation on a rich event feature set, making it
    superior to LGBM when training data is sparse (< ~200 non-zero observations).
    """

    model_type = "elastic_net"
    evaluation_cost = 1   # scikit-learn coordinate descent: ~1ms/series, same tier as ETS

    def __init__(self) -> None:
        self._alpha: float = _DEFAULT_ALPHA
        self._l1_ratio: float = _DEFAULT_L1_RATIO
        self._n: int = 0
        # Decomposition coefficient groups
        self._trend_intercept: float = 0.0
        self._trend_slope: float = 0.0
        self._dow_coefs: np.ndarray = np.zeros(6)   # 6 DOW dummies (Monday=ref)
        self._yearly_sin_coef: float = 0.0
        self._yearly_cos_coef: float = 0.0
        self._holiday_coefs: dict[str, float] = {}
        self._promo_coefs: dict[str, float] = {}
        self._holiday_cols: list[str] = []
        self._promo_cols: list[str] = []
        self._extra_event_cols: list[str] = []   # promo_tactic_rank, competing_promo_count, etc.
        self._extra_event_coefs: dict[str, float] = {}
        self._fitted_values: pd.Series | None = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        series_pdf: pd.DataFrame,
        known_future_pdf: pd.DataFrame,
        params: dict,
    ) -> "ElasticNetModel":
        from sklearn.linear_model import ElasticNet

        pdf = series_pdf.sort_values("date_id").reset_index(drop=True)
        y = pdf["qty"].fillna(0).astype(float).to_numpy()
        n = len(y)
        self._n = n
        self._alpha = float(params.get("alpha", _DEFAULT_ALPHA))
        self._l1_ratio = float(params.get("l1_ratio", _DEFAULT_L1_RATIO))

        t = np.arange(n, dtype=float)
        dates = pd.to_datetime(pdf["date_id"])
        doy = dates.dt.dayofyear.to_numpy().astype(float)
        dow = dates.dt.dayofweek.to_numpy()  # 0=Monday

        # ── design matrix ────────────────────────────────────────────────────
        # Calendar block: trend, DOW dummies (6, Monday omitted as reference),
        # yearly Fourier
        dow_dummies = np.zeros((n, 6))
        for d in range(1, 7):
            dow_dummies[:, d - 1] = (dow == d).astype(float)

        yearly_sin = np.sin(2 * np.pi * doy / _YEARLY_PERIOD)
        yearly_cos = np.cos(2 * np.pi * doy / _YEARLY_PERIOD)

        # Event/promo/holiday/halo features (no pruning — all features kept)
        self._holiday_cols, self._promo_cols = _detect_event_cols(pdf)
        event_cols = self._holiday_cols + self._promo_cols
        extra_candidates = [
            "competing_promo_count",
            "halo_partner_corr_strength", "cannib_partner_corr_strength",
            "halo_partner_promo_active_flag", "halo_partner_promo_count",
            "cannib_partner_promo_active_flag", "cannib_partner_promo_count",
        ]
        self._extra_event_cols = [c for c in extra_candidates if c in pdf.columns]
        all_event_cols = event_cols + self._extra_event_cols
        ctx_cols = [c for c in _HIST_ONLY_FEATURES if c in pdf.columns]
        feature_cols = all_event_cols + ctx_cols
        X_event = pdf[feature_cols].fillna(0).astype(float).to_numpy() if feature_cols else np.zeros((n, 0))

        X = np.column_stack([
            np.ones(n),   # intercept (trend at t=0)
            t,            # linear trend
            dow_dummies,
            yearly_sin,
            yearly_cos,
            X_event,
        ])

        col_names = (
            ["intercept", "slope"]
            + [f"dow_{d}" for d in range(1, 7)]
            + ["yearly_sin", "yearly_cos"]
            + feature_cols
        )

        # ── Recency weighting ───────────────────────────────────────────────────
        # Recency weighting (config-driven via params["recency_decay"])
        _decay = float(params.get("recency_decay", 0))
        if _decay > 0:
            from training.models.base import compute_recency_weights
            _sw = compute_recency_weights(n, _decay)
        else:
            _sw = None

        en = ElasticNet(
            alpha=self._alpha,
            l1_ratio=self._l1_ratio,
            fit_intercept=False,  # intercept is the first column in X
            max_iter=2000,
        )
        en.fit(X, y, sample_weight=_sw) if _sw is not None else en.fit(X, y)
        coefs = dict(zip(col_names, en.coef_))

        # ── unpack into decomposition groups ─────────────────────────────────
        self._trend_intercept = float(coefs.get("intercept", 0.0))
        self._trend_slope = float(coefs.get("slope", 0.0))
        self._dow_coefs = np.array([float(coefs.get(f"dow_{d}", 0.0)) for d in range(1, 7)])
        self._yearly_sin_coef = float(coefs.get("yearly_sin", 0.0))
        self._yearly_cos_coef = float(coefs.get("yearly_cos", 0.0))
        self._holiday_coefs = {c: float(coefs.get(c, 0.0)) for c in self._holiday_cols}
        self._promo_coefs = {c: float(coefs.get(c, 0.0)) for c in self._promo_cols}
        self._extra_event_coefs = {c: float(coefs.get(c, 0.0)) for c in self._extra_event_cols}

        self._fitted_values = pd.Series(X @ en.coef_, index=pdf.index)
        return self

    def fitted_values(self) -> pd.Series:
        assert self._fitted_values is not None
        return self._fitted_values

    def get_storable_params(self, input_params: dict) -> dict:
        return {
            "baseline_params": {
                "alpha": self._alpha,
                "l1_ratio": self._l1_ratio,
                "trend_intercept": self._trend_intercept,
                "trend_slope": self._trend_slope,
                "train_length": self._n,
            },
            "dow_coefs": self._dow_coefs.tolist(),
            "yearly_coefs": {
                "sin": self._yearly_sin_coef,
                "cos": self._yearly_cos_coef,
            },
            "event_coefficients": {**self._holiday_coefs, **self._promo_coefs},
            "extra_event_coefs": self._extra_event_coefs,
            "model_artifact": "elastic_net_stateless",  # marker: fully defined by coefficients above
        }

    def serialize(self) -> str | None:
        """ElasticNet is fully described by its coefficients — no heavy artifact needed."""
        return "elastic_net_stateless"

    @classmethod
    def from_serialized(cls, params: dict) -> "ElasticNetModel":
        """Reconstruct fitted ElasticNet from stored coefficients — no refit needed."""
        obj = cls.__new__(cls)
        bp = params.get("baseline_params", {})
        obj._alpha = float(bp.get("alpha", _DEFAULT_ALPHA))
        obj._l1_ratio = float(bp.get("l1_ratio", _DEFAULT_L1_RATIO))
        obj._n = int(bp.get("train_length", 0))
        obj._trend_intercept = float(bp.get("trend_intercept", 0.0))
        obj._trend_slope = float(bp.get("trend_slope", 0.0))
        obj._dow_coefs = np.array(params.get("dow_coefs", [0.0] * 6), dtype=float)
        yearly = params.get("yearly_coefs", {})
        obj._yearly_sin_coef = float(yearly.get("sin", 0.0))
        obj._yearly_cos_coef = float(yearly.get("cos", 0.0))
        ec = params.get("event_coefficients", {})
        obj._holiday_coefs = {c: v for c, v in ec.items() if any(c.startswith(p) for p in _HOLIDAY_PREFIXES)}
        obj._promo_coefs = {c: v for c, v in ec.items() if any(c.startswith(p) for p in _PROMO_PREFIXES)}
        obj._holiday_cols = list(obj._holiday_coefs.keys())
        obj._promo_cols = list(obj._promo_coefs.keys())
        obj._extra_event_coefs = params.get("extra_event_coefs", {})
        obj._extra_event_cols = list(obj._extra_event_coefs.keys())
        obj._fitted_values = None
        return obj

    # ── predict + decompose ───────────────────────────────────────────────────

    def _components(self, horizon_pdf: pd.DataFrame) -> dict:
        index = horizon_pdf.index
        h = len(horizon_pdf)
        future_t = np.arange(self._n, self._n + h, dtype=float)

        dates = pd.to_datetime(horizon_pdf["date_id"])
        doy = dates.dt.dayofyear.to_numpy().astype(float)
        dow = dates.dt.dayofweek.to_numpy()

        trend = pd.Series(
            self._trend_intercept + self._trend_slope * future_t,
            index=index,
        )

        dow_effect = np.zeros(h)
        for i, d in enumerate(range(1, 7)):
            dow_effect += self._dow_coefs[i] * (dow == d).astype(float)
        seasonality_weekly = pd.Series(dow_effect, index=index)

        seasonality_yearly = pd.Series(
            self._yearly_sin_coef * np.sin(2 * np.pi * doy / _YEARLY_PERIOD)
            + self._yearly_cos_coef * np.cos(2 * np.pi * doy / _YEARLY_PERIOD),
            index=index,
        )

        holiday_lift: dict[str, pd.Series] = {}
        promo_lift: dict[str, pd.Series] = {}
        yhat = trend + seasonality_weekly + seasonality_yearly

        for col, coef in self._holiday_coefs.items():
            vals = horizon_pdf[col].fillna(0).astype(float) if col in horizon_pdf.columns else pd.Series(0.0, index=index)
            lift = pd.Series(coef * vals.to_numpy(), index=index)
            holiday_lift[col] = lift
            yhat = yhat + lift

        for col, coef in self._promo_coefs.items():
            vals = horizon_pdf[col].fillna(0).astype(float) if col in horizon_pdf.columns else pd.Series(0.0, index=index)
            lift = pd.Series(coef * vals.to_numpy(), index=index)
            promo_lift[col] = lift
            yhat = yhat + lift

        # Extra event columns (promo_tactic_rank, promo_lead_flag, etc.) —
        # bucket into promo_lift for decomp transparency.
        for col, coef in self._extra_event_coefs.items():
            if col in _HIST_ONLY_FEATURES:
                continue  # historical-only: skip at inference
            vals = horizon_pdf[col].fillna(0).astype(float) if col in horizon_pdf.columns else pd.Series(0.0, index=index)
            lift = pd.Series(coef * vals.to_numpy(), index=index)
            promo_lift[col] = lift
            yhat = yhat + lift

        temporal_features_effect = pd.Series(0.0, index=index)

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
