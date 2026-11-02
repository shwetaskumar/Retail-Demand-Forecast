"""
training/models/base.py — the one fit/predict/decompose interface all wrappers implement.

Spec: docs/instructions.md §8 (in-house engine, not mmf_sa) / §10.2 (candidate
models) / §10.1 (decomposition contract). Each wrapper (prophet.py,
statsforecast.py, lgbm.py, uplift.py) wraps one mainstream, swappable library
behind this interface so training/engine.py never depends on a library directly.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Decomposition:
    """Additive components for one series (docs/instructions.md §10.1).

    Must satisfy: trend + seasonality_weekly + seasonality_yearly
    + sum(holiday_lift.values()) + sum(promo_lift.values()) + sum(cross_sku_lift.values())
    + temporal_features_effect == yhat, within reconstruction_tol_pct from config.

    Note: Stage-2 uplift corrections are folded into promo_lift (key "uplift")
    rather than a separate bucket, keeping the consumer-facing decomposition simple.
    """

    trend: pd.Series
    seasonality_weekly: pd.Series
    seasonality_yearly: pd.Series
    holiday_lift: dict[str, pd.Series]
    promo_lift: dict[str, pd.Series]
    temporal_features_effect: pd.Series
    yhat: pd.Series
    # Halo (complement) / cannibalization (substitute) partner attribution — one entry per
    # data/features/halo_cannibalization_features.py KNOWN_AT_INFERENCE column. Defaulted to
    # {} (not required) so every existing wrapper (prophet/statsforecast/auto_arima/croston/
    # ets_event/elastic_net), none of which produce this bucket, keeps constructing
    # Decomposition(...) unchanged. Only training/models/lgbm.py populates it today.
    cross_sku_lift: dict[str, pd.Series] = field(default_factory=dict)


class ForecastModel(abc.ABC):
    """Interface every candidate model wrapper implements (Prophet/statsforecast/LightGBM/uplift).

    Class attributes every subclass must declare:
      model_type      (str)  — unique key used in config_model.model_type and registry lookup.
      evaluation_cost (int)  — autonomous cascade ordering: 0 = lgbm (evaluated first, alone —
                               nothing else shares cost 0/1), 2 = auto_arima, 3 = statsforecast/
                               croston/ets_event/elastic_net (one merged tier), 4 = prophet
                               (Stan MCMC, evaluated last).  Engine sorts ascending; this class
                               default is only a fallback — config/seed_defaults.py's
                               _DEFAULT_MODEL_COST_TIERS (seeded into config_model.cost_tier)
                               is the actual config-table-driven source of truth.
    """

    model_type: str
    evaluation_cost: int = 2  # default: per-series/medium; override in each subclass

    @abc.abstractmethod
    def fit(self, series_pdf: pd.DataFrame, known_future_pdf: pd.DataFrame, params: dict) -> "ForecastModel":
        """Fit on one series' history (`series_pdf`) plus its known-at-inference future features."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, horizon_pdf: pd.DataFrame) -> pd.Series:
        """Predict `yhat` over the given future horizon frame (dense daily spine)."""
        raise NotImplementedError

    @abc.abstractmethod
    def decompose(self, horizon_pdf: pd.DataFrame) -> Decomposition:
        """Return the additive decomposition for the given horizon (must reconstruct yhat, §10.1)."""
        raise NotImplementedError

    def fitted_values(self) -> pd.Series:
        """In-sample fitted `yhat` for the training data passed to the most recent `fit()` call.

        Used by training/models/uplift.py (§10.3 step 2: "compute residuals") to get
        Stage-1 residuals. Deliberately not `predict()` on the training frame again:
        statsforecast.py/lgbm.py's `predict()` assumes its input is a *future*
        continuation of the training series (their internal `time_idx` continues
        from `len(training)`), so re-passing the training frame there would silently
        compute the wrong thing. Concrete Stage-1 wrappers cache this during `fit()`.
        """
        raise NotImplementedError

    def get_storable_params(self, input_params: dict) -> dict:
        """Return the params dict to store in champion_index after fitting.

        Always returns a dict with at least two top-level keys so downstream consumers
        (champion_index queries, batch_infer, Phase 1 stability check) have a
        stable contract:

            {
              "baseline_params":    {...},   # model hyperparameters (season_length, etc.)
              "event_coefficients": {...},   # {event_col: coeff} for Ridge-based wrappers;
                                             # empty dict for models that handle events
                                             # internally (Prophet, LightGBM).
              "model_artifact":     "...",  # (optional) serialized fitted model text/JSON —
                                             # present when the model supports serialize().
                                             # Allows inference to skip refitting entirely.
            }

        Default implementation wraps input_params as baseline_params with no event
        coefficients — suitable for simple wrappers that store no fitted state.
        Override in any wrapper that fits a separate Ridge event regression.
        """
        return {"baseline_params": dict(input_params), "event_coefficients": {}}

    def serialize(self) -> str | None:
        """Return a serialized (text/JSON) snapshot of the fitted model state.

        Stored under the 'model_artifact' key in get_storable_params() output.
        When non-None, inference can reconstruct the model via from_serialized()
        and skip the historical refit entirely — significantly faster inference.

        Returns None by default (model does not support artifact serialization).
        Override in subclasses that can checkpoint their fitted state.
        """
        return None

    @classmethod
    def from_serialized(cls, params: dict) -> "ForecastModel":
        """Reconstruct a fitted model from the params dict stored in champion_index.

        The returned object has predict() and decompose() available without calling fit().
        `params` is the full stored dict (including baseline_params, model_artifact, etc.).

        Raises NotImplementedError by default — override in subclasses that implement serialize().
        """
        raise NotImplementedError(f"{cls.__name__} does not implement from_serialized()")


# ═══════════════════════════════════════════════════════════════════════════════════════════
# RECENCY WEIGHTING UTILITIES (config-driven — no hardcoded defaults here)
# ═══════════════════════════════════════════════════════════════════════════════════════════
# These are pure math helpers. The actual decay value MUST come from params
# (populated by HPT from the config table's param_space). If a model receives
# recency_decay=0 (or it's absent from params), it should skip recency entirely.

_RECENCY_MIN_ROWS = 60  # never truncate below this many rows


def compute_recency_weights(n: int, decay: float) -> np.ndarray:
    """Exponential decay weights: index 0 = oldest (lowest), index n-1 = newest (1.0).

    weight_i = exp(decay * (i - (n-1)))
    -> newest row always gets weight 1.0, oldest gets exp(-decay*(n-1)).
    """
    positions = np.arange(n, dtype=float)
    return np.exp(decay * (positions - (n - 1)))


def apply_recency_truncation(pdf, decay: float, min_weight: float = 0.10):
    """Drop oldest rows where exponential weight < min_weight.

    Keeps at least _RECENCY_MIN_ROWS rows regardless. PDF must be sorted
    by date_id ascending (oldest first). Returns truncated copy.
    """
    n = len(pdf)
    if n <= _RECENCY_MIN_ROWS:
        return pdf
    import math
    cutoff_idx = int(math.ceil((n - 1) + math.log(min_weight) / decay))
    cutoff_idx = max(0, cutoff_idx)
    cutoff_idx = min(cutoff_idx, n - _RECENCY_MIN_ROWS)
    if cutoff_idx <= 0:
        return pdf
    return pdf.iloc[cutoff_idx:].reset_index(drop=True)
