"""
tests/test_reconstruction.py — decomposition reconstruction unit tests (docs/instructions.md §10.1).

Exercises training/decomposition/decomposer.py against the `decomposition_pdf`
fixture in tests/conftest.py (a flat row whose components sum exactly to
`yhat`), converted into a real `Decomposition` object — the same type every
training/models/*.py wrapper's decompose() returns.
"""
from __future__ import annotations

import pandas as pd
import pytest

from training.decomposition.decomposer import (
    named_component_sum,
    reconstruction_error,
    validate_decomposition,
)
from training.models.base import Decomposition


def _decomposition_from_row(row: pd.Series) -> Decomposition:
    """Wrap a single flat row (as tests/conftest.py's decomposition_pdf provides) into a
    Decomposition, matching what a real training/models/*.py wrapper's decompose() returns."""

    def s(value: float) -> pd.Series:
        return pd.Series([value])

    return Decomposition(
        trend=s(row["trend"]),
        seasonality_weekly=s(row["seasonality_weekly"]),
        seasonality_yearly=s(row["seasonality_yearly"]),
        holiday_lift={"holiday": s(row["holiday_lift"])},
        # Stage-2 uplift is folded into promo_lift (key "uplift"), not a separate field
        # (training/models/uplift.py) — mirror that here.
        promo_lift={"promo": s(row["promo_lift"]), "uplift": s(row["model_uplift"])},
        temporal_features_effect=s(row["temporal_features_effect"]),
        yhat=s(row["yhat"]),
    )


def test_named_component_sum_matches_the_fixtures_yhat(decomposition_pdf):
    decomp = _decomposition_from_row(decomposition_pdf.iloc[0])
    total = named_component_sum(decomp)
    assert total.iloc[0] == pytest.approx(decomposition_pdf.iloc[0]["yhat"])


def test_reconstruction_error_is_zero_for_the_fixture(decomposition_pdf):
    decomp = _decomposition_from_row(decomposition_pdf.iloc[0])
    error = reconstruction_error(decomp)
    assert error.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_validate_decomposition_passes_for_the_fixture(decomposition_pdf):
    decomp = _decomposition_from_row(decomposition_pdf.iloc[0])
    assert validate_decomposition(decomp, reconstruction_tol_pct=0.005) is True


def test_validate_decomposition_fails_when_a_component_is_perturbed(decomposition_pdf):
    row = decomposition_pdf.iloc[0].copy()
    row["trend"] = row["trend"] + 5.0  # breaks the sum well beyond 0.5% of |yhat|=18
    decomp = _decomposition_from_row(row)
    assert validate_decomposition(decomp, reconstruction_tol_pct=0.005) is False


def test_validate_decomposition_respects_the_absolute_floor_near_zero_yhat():
    """docs/instructions.md §10.1: tolerance = max(absolute_floor, pct * |yhat|) — near
    yhat=0, pct*|yhat| is ~0, so a tiny mismatch must still pass via the absolute floor."""
    tiny_error = 5e-7  # smaller than the default 1e-6 floor
    decomp = Decomposition(
        trend=pd.Series([tiny_error]),
        seasonality_weekly=pd.Series([0.0]),
        seasonality_yearly=pd.Series([0.0]),
        holiday_lift={},
        promo_lift={},
        temporal_features_effect=pd.Series([0.0]),
        yhat=pd.Series([0.0]),
    )
    assert validate_decomposition(decomp, reconstruction_tol_pct=0.005) is True


def test_reconstruction_sums_across_multiple_holiday_and_promo_events():
    """A real wrapper (e.g. Prophet with several named regressors) can return holiday_lift/
    promo_lift dicts with more than one key — every one of them must be summed in."""
    decomp = Decomposition(
        trend=pd.Series([10.0]),
        seasonality_weekly=pd.Series([1.0]),
        seasonality_yearly=pd.Series([1.0]),
        holiday_lift={"christmas": pd.Series([2.0]), "easter": pd.Series([0.5])},
        promo_lift={"bogo": pd.Series([1.0]), "discount": pd.Series([0.5])},
        temporal_features_effect=pd.Series([0.0]),
        yhat=pd.Series([16.0]),  # 10+1+1+2+0.5+1+0.5+0
    )
    assert reconstruction_error(decomp).iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert validate_decomposition(decomp, reconstruction_tol_pct=0.005) is True


def test_reconstruction_fails_when_an_event_lift_is_missing_from_yhat():
    decomp = Decomposition(
        trend=pd.Series([10.0]),
        seasonality_weekly=pd.Series([1.0]),
        seasonality_yearly=pd.Series([1.0]),
        holiday_lift={"christmas": pd.Series([2.0]), "easter": pd.Series([0.5])},
        promo_lift={},
        temporal_features_effect=pd.Series([0.0]),
        yhat=pd.Series([12.0]),  # forgets the +2.5 holiday lift
    )
    assert validate_decomposition(decomp, reconstruction_tol_pct=0.005) is False
