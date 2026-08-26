"""
training/models/uplift.py — bounded Stage-2 residual learner (candidate #4, docs/instructions.md §10.2-10.3).

Wraps any Stage-1 ForecastModel: fits a secondary LightGBM learner on the
Stage-1 in-sample residuals (`stage1.fitted_values()`, §10.3 step 2) using
calendar/event features (`training/models/lgbm.py::CALENDAR_FEATURE_COLUMNS` —
including the 6 halo/cannibalization KNOWN_AT_INFERENCE columns from
data/features/halo_cannibalization_features.py, so the residual learner can pick
up cross-item spillover the interpretable Stage-1 base can't model), then at
predict/decompose time adds a CLIPPED contribution (`uplift_clip_frac` of
|Stage-1 yhat|, from config) folded into the `promo_lift` dict (key "uplift")
— bounded so it can never dominate or break interpretability (§10.2). This stays
a single opaque bounded term (no per-column attribution) even though halo/cannib
columns influence it — unlike lgbm.py's own decompose(), which exposes them as
a named `cross_sku_lift` bucket. Only `uplift_learner="lgbm_global"` is
implemented; `"chronos"` needs a pretrained foundation-model download, out of
scope for this authoring-only build.
"""
from __future__ import annotations

import pandas as pd

from training.models.base import Decomposition, ForecastModel
from training.models.lgbm import CALENDAR_FEATURE_COLUMNS, build_calendar_feature_matrix

_DEFAULT_CLIP_FRAC = 0.25


class UpliftModel(ForecastModel):
    """Bounded residual learner wrapping a Stage-1 ForecastModel (per
    config_market.uplift_learner / uplift_clip_frac)."""

    model_type = "uplift"

    def __init__(self, stage1_model: ForecastModel) -> None:
        self._stage1 = stage1_model
        self._learner = None
        self._clip_frac = _DEFAULT_CLIP_FRAC
        self._train_length = 0
        self._stage1_params: dict = {}
        self._learner_params: dict = {}
        self._uplift_learner = "lgbm_global"

    def fit(self, series_pdf: pd.DataFrame, known_future_pdf: pd.DataFrame, params: dict) -> "UpliftModel":
        import lightgbm as lgb

        pdf = series_pdf.sort_values("date_id").reset_index(drop=True)
        self._train_length = len(pdf)
        self._clip_frac = float(params.get("uplift_clip_frac", _DEFAULT_CLIP_FRAC))
        # Retain the settings so get_storable_params() can round-trip this champion
        # through champion_index and inference can refit it identically.
        self._stage1_params = dict(params.get("stage1_params", {}))
        self._learner_params = dict(params.get("learner_params", {}))
        self._uplift_learner = params.get("uplift_learner", "lgbm_global")

        learner_kind = params.get("uplift_learner", "lgbm_global")
        if learner_kind != "lgbm_global":
            raise NotImplementedError(
                f"uplift_learner={learner_kind!r} not implemented; only 'lgbm_global' is available in v1"
            )

        self._stage1.fit(pdf, known_future_pdf, params.get("stage1_params", {}))
        residual = pdf["qty"].astype(float).to_numpy() - self._stage1.fitted_values().to_numpy()

        X = build_calendar_feature_matrix(pdf, start_idx=0, columns=CALENDAR_FEATURE_COLUMNS)
        # n_jobs=1 by default: in the per-series applyInPandas pattern each Spark
        # task already occupies a core, so letting LightGBM spawn all-core threads
        # oversubscribes the executor and slows the whole tier. Matches lgbm.py.
        learner_params = {"n_jobs": 1, **params.get("learner_params", {})}
        learner = lgb.LGBMRegressor(verbosity=-1, **learner_params)
        learner.fit(X, residual)
        self._learner = learner
        return self

    def serialize(self) -> str | None:
        """Serialize the fitted Stage-2 residual booster to a LightGBM model string.

        Returns None if the learner has not been fit. The Stage-1 base serializes
        separately via its own get_storable_params()/from_serialized() and is carried
        under 'stage1_storable', so from_serialized() rebuilds BOTH stages with no refit.
        """
        if self._learner is None:
            return None
        return self._learner.booster_.model_to_string()

    def get_storable_params(self, input_params: dict) -> dict:
        """Params written to champion_index for a '<base>+uplift' champion.

        Stores everything needed to reconstruct the wrapper at inference with ZERO
        fitting: the Stage-1 base's own storable params (including its serialized
        artifact) nested under 'stage1_storable', plus the Stage-2 residual booster
        string under 'uplift_learner_artifact'. Emitting 'model_artifact' makes
        batch_infer take the from_serialized() fast path — reconstruct + predict —
        so inference never calls fit(). The index also stays queryable for the
        interpretable Stage-1 part via baseline_params / event_coefficients.
        """
        stage1_storable = (
            self._stage1.get_storable_params(self._stage1_params)
            if hasattr(self._stage1, "get_storable_params")
            else {}
        )
        learner_artifact = self.serialize()
        return {
            "baseline_params":    stage1_storable.get("baseline_params", {}),
            "event_coefficients": stage1_storable.get("event_coefficients", {}),
            "stage1_model_type":  getattr(self._stage1, "model_type", None),
            "stage1_storable":    stage1_storable,
            "uplift_clip_frac":   self._clip_frac,
            "uplift_learner":     self._uplift_learner,
            "stage1_params":      self._stage1_params,
            "learner_params":     self._learner_params,
            "train_length":       self._train_length,
            "uplift_learner_artifact": learner_artifact,
            # Marker → batch_infer.has_serialized == True → from_serialized() fast
            # path (reconstruct + predict). Never refit at inference.
            "model_artifact":     "uplift_serialized" if learner_artifact else None,
        }

    @classmethod
    def from_serialized(cls, params: dict) -> "UpliftModel":
        """Reconstruct a fitted UpliftModel from stored params — no fit at inference.

        Rebuilds the Stage-1 base via its own from_serialized() (using the nested
        'stage1_storable' dict) and the Stage-2 residual learner from its stored
        LightGBM booster string. predict()/decompose() are available immediately;
        fit() is never called. Raises NotImplementedError if the residual artifact
        or the Stage-1 base type is missing, so batch_infer logs it and falls back
        rather than silently misbehaving.
        """
        import lightgbm as lgb

        from training.models.registry import get_registry

        artifact = params.get("uplift_learner_artifact")
        if not artifact:
            raise NotImplementedError("No stored uplift_learner_artifact — cannot reconstruct")

        stage1_type = params.get("stage1_model_type")
        base_cls = get_registry().get(stage1_type)
        if base_cls is None:
            raise NotImplementedError(f"Unknown stage1_model_type {stage1_type!r} in registry")
        base = base_cls.from_serialized(params.get("stage1_storable", {}))

        obj = cls.__new__(cls)
        obj._stage1 = base
        obj._learner = lgb.Booster(model_str=artifact)
        obj._clip_frac = float(params.get("uplift_clip_frac", _DEFAULT_CLIP_FRAC))
        obj._train_length = int(params.get("train_length", 0))
        obj._stage1_params = params.get("stage1_params", {})
        obj._learner_params = params.get("learner_params", {})
        obj._uplift_learner = params.get("uplift_learner", "lgbm_global")
        return obj

    def _clipped_uplift(self, horizon_pdf: pd.DataFrame, baseline: pd.Series) -> pd.Series:
        """The learner's raw prediction, clipped to +/- `uplift_clip_frac` of |baseline|
        so Stage-2 can only nudge the Stage-1 forecast, never dominate it (§10.2)."""
        X = build_calendar_feature_matrix(horizon_pdf, start_idx=self._train_length, columns=CALENDAR_FEATURE_COLUMNS)
        raw_uplift = pd.Series(self._learner.predict(X), index=horizon_pdf.index)
        bound = self._clip_frac * baseline.abs()
        return raw_uplift.clip(lower=-bound, upper=bound)

    def predict(self, horizon_pdf: pd.DataFrame) -> pd.Series:
        stage1_yhat = self._stage1.predict(horizon_pdf)
        return stage1_yhat + self._clipped_uplift(horizon_pdf, stage1_yhat)

    def decompose(self, horizon_pdf: pd.DataFrame) -> Decomposition:
        """Stage-1's named components pass through unchanged; the clipped Stage-2
        contribution is folded into promo_lift (key 'uplift') to keep the
        consumer-facing decomposition simple."""
        stage1_decomp = self._stage1.decompose(horizon_pdf)
        uplift = self._clipped_uplift(horizon_pdf, stage1_decomp.yhat)
        yhat = stage1_decomp.yhat + uplift
        # Merge uplift into promo_lift dict
        merged_promo_lift = {**stage1_decomp.promo_lift, "uplift": uplift}
        return Decomposition(
            trend=stage1_decomp.trend,
            seasonality_weekly=stage1_decomp.seasonality_weekly,
            seasonality_yearly=stage1_decomp.seasonality_yearly,
            holiday_lift=stage1_decomp.holiday_lift,
            promo_lift=merged_promo_lift,
            cross_sku_lift=stage1_decomp.cross_sku_lift,
            temporal_features_effect=stage1_decomp.temporal_features_effect,
            yhat=yhat,
        )