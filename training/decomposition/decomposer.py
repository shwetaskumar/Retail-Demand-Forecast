"""
training/decomposition/decomposer.py — additive decomposition contract + reconstruction check.

Spec: docs/instructions.md §10.1. A model cannot be champion unless its
Decomposition reconstructs `yhat` within `reconstruction_tol_pct` from config
(<=0.5% of |yhat|, or a 1e-6 absolute floor when |yhat| is near zero). This is
the validator any of training/models/*.py's `decompose()` output must pass
before training/champion.py can consider that candidate.
"""
from __future__ import annotations

import pandas as pd

from training.models.base import Decomposition

_DEFAULT_ABSOLUTE_FLOOR = 1e-6


def named_component_sum(decomp: Decomposition) -> pd.Series:
    """Sum every named component (trend + seasonality_* + all holiday/promo/cross-item
    lifts + temporal_features_effect) — the left-hand side of the §10.1
    reconstruction equation."""
    total = decomp.trend + decomp.seasonality_weekly + decomp.seasonality_yearly
    total = total + decomp.temporal_features_effect
    for lift in (*decomp.holiday_lift.values(), *decomp.promo_lift.values(), *decomp.cross_sku_lift.values()):
        total = total + lift
    return total


def reconstruction_error(decomp: Decomposition) -> pd.Series:
    """Per-row |yhat - sum(named components)| (docs/instructions.md §10.1)."""
    return (decomp.yhat - named_component_sum(decomp)).abs()


def validate_decomposition(
    decomp: Decomposition,
    reconstruction_tol_pct: float,
    absolute_floor: float = _DEFAULT_ABSOLUTE_FLOOR,
) -> bool:
    """True iff every row's reconstruction_error is within tolerance:
    max(absolute_floor, reconstruction_tol_pct * |yhat|).

    A model can't be champion unless this holds for its whole decomposition
    (docs/instructions.md §10.1) — call right after any wrapper's decompose().
    """
    error = reconstruction_error(decomp)
    tolerance = (reconstruction_tol_pct * decomp.yhat.abs()).clip(lower=absolute_floor)
    return bool((error <= tolerance).all())
