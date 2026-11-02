"""
tests/test_models.py — model wrapper unit tests (Step 4), on small synthetic pandas series.

Exercises the real prophet/statsforecast/lightgbm/shap libraries (installed
locally for this) — no Spark, no Databricks. Fits each candidate on a small
synthetic series with a trend, weekly seasonality, and sparse holiday/promo
flags, then checks the additive decomposition contract (docs/instructions.md
§10.1): the named components must sum to `yhat`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.decomposition.decomposer import validate_decomposition
from training.models.elastic_net import ElasticNetModel
from training.models.ensemble import EnsembleModel
from training.models.lgbm import LightGBMModel
from training.models.prophet import ProphetModel
from training.models.statsforecast import StatsForecastModel
from training.models.uplift import UpliftModel

_HORIZON_DAYS = 14


def _synthetic_frame(n_total: int = 200, start_date: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_total, freq="D")
    t = np.arange(n_total)
    dow = dates.dayofweek.to_numpy() + 1  # ISO: 1=Mon, 7=Sun

    holiday_flag = np.zeros(n_total)
    # Sparse, deterministic. Period 40 (not 47) so it clears training/models/prophet.py's
    # _MIN_REGRESSOR_SUPPORT=5 within the 186-row training window (200 - _HORIZON_DAYS):
    # indices 0,40,80,120,160 = 5 fires. A sparser period would get pruned as unreliable.
    holiday_flag[::40] = 1
    promo_active_flag = np.zeros(n_total)
    promo_active_flag[::23] = 1  # sparse, deterministic, different phase than holidays

    qty = (
        50
        + 0.05 * t
        + 4 * np.sin(2 * np.pi * dow / 7)
        + 6 * holiday_flag
        + 4 * promo_active_flag
        + rng.normal(0, 0.5, n_total)
    )
    qty = np.clip(qty, 0, None)

    return pd.DataFrame(
        {
            "store_id": "S1",
            "sku_id": "ITEM_A",
            "date_id": dates,
            "qty": qty,
            "dow": dow,
            "week_of_year": dates.isocalendar().week.to_numpy(),
            "month": dates.month.to_numpy(),
            "quarter": dates.quarter.to_numpy(),
            "day_of_month": dates.day.to_numpy(),
            "payday_flag": (dates.day == 1).astype(int),
            "weekend_flag": (dow >= 6).astype(int),  # ISO: 6=Sat, 7=Sun
            "holiday_flag": holiday_flag,
            "promo_active_flag": promo_active_flag,
        }
    )


def _train_horizon_split(n_total: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _synthetic_frame(n_total)
    train = frame.iloc[: n_total - _HORIZON_DAYS]
    horizon = frame.iloc[n_total - _HORIZON_DAYS :]
    return train, horizon


def _assert_reconstructs(decomp, atol: float = 1e-4, rtol: float = 0.02) -> None:
    """docs/instructions.md §10.1: components must sum to yhat within reconstruction_tolerance."""
    named_sum = decomp.trend + decomp.seasonality_weekly + decomp.seasonality_yearly
    named_sum = named_sum + decomp.temporal_features_effect
    for lift in (*decomp.holiday_lift.values(), *decomp.promo_lift.values(), *decomp.cross_sku_lift.values()):
        named_sum = named_sum + lift
    diff = (named_sum - decomp.yhat).abs()
    threshold = atol + rtol * decomp.yhat.abs()
    assert (diff <= threshold).all(), f"reconstruction gap exceeds tolerance:\n{diff[diff > threshold]}"


@pytest.fixture(scope="module")
def train_horizon():
    return _train_horizon_split()


class TestProphetModel:
    def test_fits_and_reconstructs(self, train_horizon):
        train, horizon = train_horizon
        model = ProphetModel().fit(train, horizon, params={})
        decomp = model.decompose(horizon)
        _assert_reconstructs(decomp)

    def test_predict_matches_decompose_yhat(self, train_horizon):
        train, horizon = train_horizon
        model = ProphetModel().fit(train, horizon, params={})
        predicted = model.predict(horizon)
        decomp = model.decompose(horizon)
        pd.testing.assert_series_equal(predicted, decomp.yhat, check_names=False)

    def test_stage1_has_no_uplift_bucket(self, train_horizon):
        train, horizon = train_horizon
        model = ProphetModel().fit(train, horizon, params={})
        decomp = model.decompose(horizon)
        assert "uplift" not in decomp.promo_lift

    def test_holiday_and_promo_lift_are_named(self, train_horizon):
        train, horizon = train_horizon
        model = ProphetModel().fit(train, horizon, params={})
        decomp = model.decompose(horizon)
        # payday_flag also matches prophet.py's holiday-bucket prefix set
        # (_HOLIDAY_PREFIXES / decompose()'s ("holiday_", "bridge_", "payday_") check).
        assert set(decomp.holiday_lift.keys()) == {"holiday_flag", "payday_flag"}
        assert set(decomp.promo_lift.keys()) == {"promo_active_flag"}


class TestStatsForecastModel:
    def test_fits_and_reconstructs(self, train_horizon):
        train, horizon = train_horizon
        model = StatsForecastModel().fit(train, horizon, params={})
        decomp = model.decompose(horizon)
        _assert_reconstructs(decomp, atol=1e-6, rtol=1e-6)  # fully additive by construction

    def test_predict_matches_decompose_yhat(self, train_horizon):
        train, horizon = train_horizon
        model = StatsForecastModel().fit(train, horizon, params={})
        predicted = model.predict(horizon)
        decomp = model.decompose(horizon)
        pd.testing.assert_series_equal(predicted, decomp.yhat, check_names=False)

    def test_captures_positive_trend_slope(self, train_horizon):
        train, horizon = train_horizon
        model = StatsForecastModel().fit(train, horizon, params={})
        decomp = model.decompose(horizon)
        # the synthetic series has slope +0.05/day; the extrapolated trend must follow suit
        assert decomp.trend.iloc[-1] > decomp.trend.iloc[0]

    def test_fitted_values_length_matches_training_data(self, train_horizon):
        train, _ = train_horizon
        model = StatsForecastModel().fit(train, train, params={})
        assert len(model.fitted_values()) == len(train)


def _synthetic_frame_with_halo_cannib(n_total: int = 200, start_date: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    """Same generative process as _synthetic_frame, plus halo/cannibalization columns
    (data/features/halo_cannibalization_features.py) present on the TRAINING portion only
    — mirrors production exactly: batch_infer.py's future spine never carries the
    HISTORICAL_ONLY halo_partner_qty_7d_roll / cannib_partner_qty_7d_roll columns (they
    require realized qty), only the KNOWN_AT_INFERENCE ones.
    """
    frame = _synthetic_frame(n_total, start_date, seed)
    rng = np.random.default_rng(seed + 1)
    n = len(frame)
    frame["halo_partner_qty_7d_roll"] = rng.uniform(0, 20, n)          # HISTORICAL_ONLY
    frame["cannib_partner_qty_7d_roll"] = rng.uniform(0, 20, n)        # HISTORICAL_ONLY
    frame["halo_partner_corr_strength"] = 0.5                          # KNOWN_AT_INFERENCE
    frame["cannib_partner_corr_strength"] = 0.4                        # KNOWN_AT_INFERENCE
    frame["halo_partner_promo_active_flag"] = (np.arange(n) % 17 == 0).astype(int)   # KNOWN_AT_INFERENCE
    frame["halo_partner_promo_count"] = frame["halo_partner_promo_active_flag"]      # KNOWN_AT_INFERENCE
    frame["cannib_partner_promo_active_flag"] = (np.arange(n) % 31 == 0).astype(int)  # KNOWN_AT_INFERENCE
    frame["cannib_partner_promo_count"] = frame["cannib_partner_promo_active_flag"]   # KNOWN_AT_INFERENCE
    return frame


class TestElasticNetModel:
    """Real sklearn ElasticNet on synthetic pandas data — no Spark (docs/instructions.md
    §17 leakage-control tests, extended to the halo/cannibalization columns wired in
    training/models/elastic_net.py)."""

    def test_fits_and_reconstructs(self):
        frame = _synthetic_frame_with_halo_cannib()
        train = frame.iloc[: len(frame) - _HORIZON_DAYS]
        horizon = frame.iloc[len(frame) - _HORIZON_DAYS :]
        # Horizon carries only the columns future_spine.py actually produces: drop the
        # HISTORICAL_ONLY halo/cannib columns to mirror production exactly.
        horizon_known_only = horizon.drop(columns=["halo_partner_qty_7d_roll", "cannib_partner_qty_7d_roll"])
        model = ElasticNetModel().fit(train, horizon_known_only, params={})
        decomp = model.decompose(horizon_known_only)
        _assert_reconstructs(decomp, atol=1e-4, rtol=0.05)

    def test_predict_matches_decompose_yhat(self):
        frame = _synthetic_frame_with_halo_cannib()
        train = frame.iloc[: len(frame) - _HORIZON_DAYS]
        horizon = frame.iloc[len(frame) - _HORIZON_DAYS :].drop(
            columns=["halo_partner_qty_7d_roll", "cannib_partner_qty_7d_roll"]
        )
        model = ElasticNetModel().fit(train, horizon, params={})
        predicted = model.predict(horizon)
        decomp = model.decompose(horizon)
        np.testing.assert_allclose(predicted.to_numpy(), decomp.yhat.to_numpy(), atol=1e-4)

    def test_historical_only_halo_cannib_columns_are_not_used_at_predict_time(self):
        """The core leakage-safety invariant for the two new HISTORICAL_ONLY columns:
        removing them from the horizon frame must not change yhat (they must never be
        read at predict time), and — unlike the KNOWN_AT_INFERENCE halo/cannib columns —
        must not appear in the fitted event-coefficient set used in decompose()."""
        frame = _synthetic_frame_with_halo_cannib()
        train = frame.iloc[: len(frame) - _HORIZON_DAYS]
        horizon = frame.iloc[len(frame) - _HORIZON_DAYS :]
        model = ElasticNetModel().fit(train, horizon, params={})

        assert "halo_partner_qty_7d_roll" not in model._extra_event_coefs
        assert "cannib_partner_qty_7d_roll" not in model._extra_event_coefs
        assert "halo_partner_promo_active_flag" in model._extra_event_coefs
        assert "cannib_partner_promo_active_flag" in model._extra_event_coefs

        horizon_with_hist = horizon.copy()
        horizon_without_hist = horizon.drop(columns=["halo_partner_qty_7d_roll", "cannib_partner_qty_7d_roll"])
        # Perturbing the historical-only columns must not move the prediction at all.
        horizon_with_hist["halo_partner_qty_7d_roll"] = 999_999.0
        horizon_with_hist["cannib_partner_qty_7d_roll"] = 999_999.0

        pred_with = model.predict(horizon_with_hist)
        pred_without = model.predict(horizon_without_hist)
        np.testing.assert_allclose(pred_with.to_numpy(), pred_without.to_numpy(), atol=1e-9)

    def test_stage1_has_no_uplift_bucket(self):
        frame = _synthetic_frame_with_halo_cannib()
        train = frame.iloc[: len(frame) - _HORIZON_DAYS]
        horizon = frame.iloc[len(frame) - _HORIZON_DAYS :].drop(
            columns=["halo_partner_qty_7d_roll", "cannib_partner_qty_7d_roll"]
        )
        model = ElasticNetModel().fit(train, horizon, params={})
        decomp = model.decompose(horizon)
        assert "uplift" not in decomp.promo_lift


class TestLightGBMModel:
    def test_fits_and_reconstructs(self, train_horizon):
        train, horizon = train_horizon
        model = LightGBMModel().fit(train, horizon, params={"n_estimators": 50, "max_depth": 4})
        decomp = model.decompose(horizon)
        _assert_reconstructs(decomp, atol=1e-4, rtol=1e-6)

    def test_predict_matches_decompose_yhat(self, train_horizon):
        train, horizon = train_horizon
        model = LightGBMModel().fit(train, horizon, params={"n_estimators": 50, "max_depth": 4})
        predicted = model.predict(horizon)
        decomp = model.decompose(horizon)
        np.testing.assert_allclose(predicted.to_numpy(), decomp.yhat.to_numpy(), atol=1e-4)

    def test_stage1_has_no_uplift_bucket(self, train_horizon):
        train, horizon = train_horizon
        model = LightGBMModel().fit(train, horizon, params={"n_estimators": 50, "max_depth": 4})
        decomp = model.decompose(horizon)
        assert "uplift" not in decomp.promo_lift

    _CROSS_ITEM_COLS = {
        "halo_partner_corr_strength", "cannib_partner_corr_strength",
        "halo_partner_promo_active_flag", "halo_partner_promo_count",
        "cannib_partner_promo_active_flag", "cannib_partner_promo_count",
    }

    def _cross_item_train_horizon(self):
        """Horizon carries only the columns future_spine.py actually produces: drop the
        2 HISTORICAL_ONLY halo/cannib columns, matching production (mirrors the
        ElasticNet test above using the same _synthetic_frame_with_halo_cannib helper)."""
        frame = _synthetic_frame_with_halo_cannib()
        train = frame.iloc[: len(frame) - _HORIZON_DAYS]
        horizon = frame.iloc[len(frame) - _HORIZON_DAYS :].drop(
            columns=["halo_partner_qty_7d_roll", "cannib_partner_qty_7d_roll"]
        )
        return train, horizon

    def test_cross_sku_lift_bucket_is_populated_on_slow_runtime_shap_path(self):
        # decompose() called directly on a freshly-fit model (no serialize/from_serialized
        # round-trip) has no _stored_shap_decomp -> takes the runtime-SHAP path.
        train, horizon = self._cross_item_train_horizon()
        model = LightGBMModel().fit(train, horizon, params={"n_estimators": 50, "max_depth": 4})
        assert not hasattr(model, "_stored_shap_decomp")
        decomp = model.decompose(horizon)
        assert set(decomp.cross_sku_lift.keys()) == self._CROSS_ITEM_COLS
        _assert_reconstructs(decomp, atol=1e-4, rtol=1e-6)

    def test_cross_sku_lift_bucket_is_populated_on_fast_reconstruct_path(self):
        # The from_serialized() path inference/batch_infer.py's fast path uses: reload
        # from get_storable_params()'s stored booster + precomputed SHAP, no refit.
        train, horizon = self._cross_item_train_horizon()
        model = LightGBMModel().fit(train, horizon, params={"n_estimators": 50, "max_depth": 4})
        storable = model.get_storable_params({})
        assert storable.get("shap_decomp") is not None
        reconstructed = LightGBMModel.from_serialized(storable)
        decomp = reconstructed.decompose(horizon)
        assert set(decomp.cross_sku_lift.keys()) == self._CROSS_ITEM_COLS
        _assert_reconstructs(decomp, atol=1e-4, rtol=1e-6)

    def test_cross_sku_lift_columns_are_excluded_from_other_buckets(self):
        """The 6 halo/cannib columns must land ONLY in cross_sku_lift, not silently
        folded into temporal_features_effect or elsewhere — guards against a bucket-mapping typo."""
        train, horizon = self._cross_item_train_horizon()
        model = LightGBMModel().fit(train, horizon, params={"n_estimators": 50, "max_depth": 4})
        decomp = model.decompose(horizon)
        assert self._CROSS_ITEM_COLS.isdisjoint(decomp.holiday_lift.keys())
        assert self._CROSS_ITEM_COLS.isdisjoint(decomp.promo_lift.keys())


class TestUpliftModel:
    def test_fits_and_reconstructs(self, train_horizon):
        train, horizon = train_horizon
        stage1 = LightGBMModel()
        model = UpliftModel(stage1).fit(
            train, horizon, params={"uplift_clip_frac": 0.25, "stage1_params": {"n_estimators": 50, "max_depth": 4}}
        )
        decomp = model.decompose(horizon)
        _assert_reconstructs(decomp, atol=1e-4, rtol=1e-6)

    def test_uplift_is_bounded_by_clip_frac(self, train_horizon):
        train, horizon = train_horizon
        stage1 = LightGBMModel()
        clip_frac = 0.05
        model = UpliftModel(stage1).fit(
            train,
            horizon,
            params={"uplift_clip_frac": clip_frac, "stage1_params": {"n_estimators": 50, "max_depth": 4}},
        )
        decomp = model.decompose(horizon)
        stage1_yhat = stage1.decompose(horizon).yhat
        bound = clip_frac * stage1_yhat.abs() + 1e-9
        assert (decomp.promo_lift["uplift"].abs() <= bound).all()

    def test_rejects_unimplemented_learner(self, train_horizon):
        train, horizon = train_horizon
        model = UpliftModel(LightGBMModel())
        with pytest.raises(NotImplementedError):
            model.fit(train, horizon, params={"uplift_learner": "chronos"})

    def test_predict_matches_decompose_yhat(self, train_horizon):
        train, horizon = train_horizon
        model = UpliftModel(LightGBMModel()).fit(
            train, horizon, params={"uplift_clip_frac": 0.25, "stage1_params": {"n_estimators": 50, "max_depth": 4}}
        )
        predicted = model.predict(horizon)
        decomp = model.decompose(horizon)
        np.testing.assert_allclose(predicted.to_numpy(), decomp.yhat.to_numpy(), atol=1e-4)

    def test_cross_sku_lift_passes_through_from_lgbm_stage1(self):
        """UpliftModel.decompose() must not silently drop the Stage-1 base's
        cross_sku_lift bucket (data/features/halo_cannibalization_features.py columns) —
        the residual learner ALSO sees these columns as raw features via
        CALENDAR_FEATURE_COLUMNS, but they must additionally survive the passthrough."""
        frame = _synthetic_frame_with_halo_cannib()
        train = frame.iloc[: len(frame) - _HORIZON_DAYS]
        horizon = frame.iloc[len(frame) - _HORIZON_DAYS :].drop(
            columns=["halo_partner_qty_7d_roll", "cannib_partner_qty_7d_roll"]
        )
        stage1 = LightGBMModel()
        model = UpliftModel(stage1).fit(
            train, horizon, params={"uplift_clip_frac": 0.25, "stage1_params": {"n_estimators": 50, "max_depth": 4}}
        )
        decomp = model.decompose(horizon)
        assert set(decomp.cross_sku_lift.keys()) == TestLightGBMModel._CROSS_ITEM_COLS
        _assert_reconstructs(decomp, atol=1e-4, rtol=1e-6)


class TestValidateDecompositionOnRealWrappers:
    """Step 5 integration check: training/decomposition/decomposer.py's validator (built and
    tested against a synthetic fixture) must also accept every Step 4 wrapper's real output."""

    @pytest.mark.parametrize(
        "model_factory,params",
        [
            (lambda: ProphetModel(), {}),
            (lambda: StatsForecastModel(), {}),
            (lambda: LightGBMModel(), {"n_estimators": 50, "max_depth": 4}),
            (
                lambda: UpliftModel(LightGBMModel()),
                {"uplift_clip_frac": 0.25, "stage1_params": {"n_estimators": 50, "max_depth": 4}},
            ),
        ],
        ids=["prophet", "statsforecast", "lgbm", "uplift"],
    )
    def test_validate_decomposition_accepts_real_wrapper_output(self, train_horizon, model_factory, params):
        train, horizon = train_horizon
        model = model_factory().fit(train, horizon, params=params)
        decomp = model.decompose(horizon)
        assert validate_decomposition(decomp, reconstruction_tol_pct=0.02) is True


class TestEnsembleModel:
    _MEMBERS = {"member_types": ["elastic_net", "lgbm"], "member_params": {"lgbm": {"n_estimators": 50}}}

    def test_fits_and_reconstructs(self, train_horizon):
        train, horizon = train_horizon
        decomp = EnsembleModel().fit(train, horizon, self._MEMBERS).decompose(horizon)
        _assert_reconstructs(decomp)
        assert validate_decomposition(decomp, 0.02)

    def test_forecast_is_mean_of_members(self, train_horizon):
        train, horizon = train_horizon
        ens = EnsembleModel().fit(train, horizon, self._MEMBERS)
        en = ElasticNetModel().fit(train, horizon, {})
        lg = LightGBMModel().fit(train, horizon, {"n_estimators": 50})
        expected = (en.predict(horizon).to_numpy() + lg.predict(horizon).to_numpy()) / 2
        np.testing.assert_allclose(ens.predict(horizon).to_numpy(), expected, rtol=1e-9)

    def test_failing_member_is_dropped(self, train_horizon):
        train, horizon = train_horizon
        params = {"member_types": ["elastic_net", "not_a_model"]}
        ens = EnsembleModel().fit(train, horizon, params)
        np.testing.assert_allclose(
            ens.predict(horizon).to_numpy(),
            ElasticNetModel().fit(train, horizon, {}).predict(horizon).to_numpy(),
        )

    def test_all_members_failing_raises(self, train_horizon):
        train, horizon = train_horizon
        with pytest.raises(RuntimeError):
            EnsembleModel().fit(train, horizon, {"member_types": ["not_a_model"]})

    def test_storable_params_round_trip_through_refit(self, train_horizon):
        """inference/batch_infer.py's refit fallback passes the stored blob straight back into fit()."""
        train, horizon = train_horizon
        ens = EnsembleModel().fit(train, horizon, self._MEMBERS)
        stored = ens.get_storable_params(self._MEMBERS)
        assert stored["baseline_params"]["member_types"] == ["elastic_net", "lgbm"]
        refit = EnsembleModel().fit(train, horizon, stored)
        np.testing.assert_allclose(refit.predict(horizon).to_numpy(), ens.predict(horizon).to_numpy(), rtol=1e-6)

