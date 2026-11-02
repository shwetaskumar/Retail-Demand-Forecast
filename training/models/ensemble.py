"""
training/models/ensemble.py — equal-weight forecast combination (docs/instructions.md §10.2).

Averages several Stage-1 wrappers (default: auto_arima + ets_event + lgbm). Forecast
combination is the most reliable accuracy lever on noisy, intermittent store×sku
demand: on a 150-series Kaggle M5 sample (CA, 3-fold rolling backtest) this three-model
average reached 38.3% pooled 7-day WMAPE vs 39.9–42.1% for its members on their own.
It is the production default candidate (config/seed_defaults.py) because, over 750
out-of-sample series-weeks, always using it beat every per-series selection rule
(demo/compare_selection_rules.py).

Explainability is preserved exactly, not approximated: every member's decomposition is
additive, and the mean of additive decompositions is additive — each component here is
the plain average of the members' same-named component (missing named lifts count as 0),
so trend + seasonality + Σ lifts + temporal_features_effect still reconstructs yhat
(§10.1) whenever each member's own decomposition does.

Members and their params are config-driven (`params["member_types"]`,
`params["member_params"]`); a member that fails to fit is dropped rather than failing
the whole ensemble, as long as at least one member fits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.models.base import Decomposition, ForecastModel

_DEFAULT_MEMBER_TYPES: tuple[str, ...] = ("auto_arima", "ets_event", "lgbm")

_SCALAR_COMPONENTS = ("trend", "seasonality_weekly", "seasonality_yearly", "temporal_features_effect", "yhat")
_DICT_COMPONENTS = ("holiday_lift", "promo_lift", "cross_sku_lift")


def _artifact_present(storable: dict) -> bool:
    """Mirror of inference/batch_infer.py's has_serialized check, per member."""
    return bool(storable.get("model_artifact") or storable.get("ets_forecast_values"))


def _member_types(params: dict) -> list[str]:
    explicit = params.get("member_types") or params.get("baseline_params", {}).get("member_types")
    return list(explicit) if explicit else list(_DEFAULT_MEMBER_TYPES)


class EnsembleModel(ForecastModel):
    """Equal-weight average of several ForecastModel wrappers, decomposition averaged per component."""

    model_type = "ensemble"
    evaluation_cost = 4  # refits every member, so it runs after the single-model tiers

    def __init__(self) -> None:
        self._members: dict[str, ForecastModel] = {}
        self._member_params: dict[str, dict] = {}

    def fit(self, series_pdf: pd.DataFrame, known_future_pdf: pd.DataFrame, params: dict) -> "EnsembleModel":
        from training.models.registry import build

        # params["member_params"] = {member_type: hyperparams}. Present both on a fresh
        # fit (config/HPT) and in the stored champion blob (get_storable_params below), so
        # inference/batch_infer.py's refit fallback reproduces the trained ensemble.
        member_params = params.get("member_params") or {}
        self._members, self._member_params = {}, {}
        errors: dict[str, str] = {}
        for member_type in _member_types(params):
            p = dict(member_params.get(member_type, {}))
            try:
                self._members[member_type] = build(member_type).fit(series_pdf, known_future_pdf, p)
                self._member_params[member_type] = p
            except Exception as exc:  # noqa: BLE001 — one failing member must not sink the ensemble
                errors[member_type] = f"{type(exc).__name__}: {exc}"
        if not self._members:
            raise RuntimeError(f"ensemble: every member failed to fit: {errors}")
        return self

    def predict(self, horizon_pdf: pd.DataFrame) -> pd.Series:
        preds = [np.asarray(m.predict(horizon_pdf), dtype=float) for m in self._members.values()]
        return pd.Series(np.mean(preds, axis=0), index=horizon_pdf.index)

    def decompose(self, horizon_pdf: pd.DataFrame) -> Decomposition:
        decomps = [m.decompose(horizon_pdf) for m in self._members.values()]
        n, index = len(horizon_pdf), horizon_pdf.index

        def _mean(arrays: list[np.ndarray]) -> pd.Series:
            return pd.Series(np.mean(arrays, axis=0), index=index, dtype=float)

        scalars = {
            name: _mean([np.asarray(getattr(d, name), dtype=float) for d in decomps])
            for name in _SCALAR_COMPONENTS
        }
        dicts: dict[str, dict[str, pd.Series]] = {}
        for name in _DICT_COMPONENTS:
            keys = sorted({k for d in decomps for k in getattr(d, name)})
            dicts[name] = {
                k: _mean([
                    np.asarray(getattr(d, name)[k], dtype=float) if k in getattr(d, name) else np.zeros(n)
                    for d in decomps
                ])
                for k in keys
            }
        return Decomposition(**scalars, **dicts)

    def fitted_values(self) -> pd.Series:
        fitted = [m.fitted_values() for m in self._members.values()]
        return pd.Series(np.mean([np.asarray(f, dtype=float) for f in fitted], axis=0), index=fitted[0].index)

    def get_storable_params(self, input_params: dict) -> dict:
        members = {
            mt: (m.get_storable_params(self._member_params[mt]) if hasattr(m, "get_storable_params")
                 else {"baseline_params": dict(self._member_params[mt]), "event_coefficients": {}})
            for mt, m in self._members.items()
        }
        out = {
            "baseline_params": {"member_types": list(members)},
            "event_coefficients": {},
            "member_params": {mt: dict(p) for mt, p in self._member_params.items()},
            "members": members,
        }
        # Only advertise a reloadable artifact when EVERY member has one — otherwise
        # inference must refit (from_serialized would raise for the missing member anyway).
        if all(_artifact_present(s) for s in members.values()):
            out["model_artifact"] = "ensemble_members"
        return out

    @classmethod
    def from_serialized(cls, params: dict) -> "EnsembleModel":
        from training.models.registry import get_registry

        members = params.get("members") or {}
        if not members:
            raise NotImplementedError("ensemble: no stored member artifacts")
        reg = get_registry()
        obj = cls()
        for member_type in _member_types(params):
            if member_type not in members:
                continue
            obj._members[member_type] = reg[member_type].from_serialized(members[member_type])
            obj._member_params[member_type] = members[member_type].get("baseline_params", {})
        if not obj._members:
            raise NotImplementedError("ensemble: stored members do not match member_types")
        return obj
