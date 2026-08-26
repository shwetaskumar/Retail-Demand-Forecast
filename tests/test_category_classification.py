"""
tests/test_category_classification.py — static ABC/D/SEASONAL category rule unit tests (Step 6 follow-up).

Per user direction 2026-07-01: no source table names a per-sku category
classification, so data/features/category_classification.py computes one from
trailing sales volume. Pure Python — no Spark needed for the banding/override
rules themselves (compute_sku_category's SQL mirrors them, per module docstring).
"""
from __future__ import annotations

import pytest

from data.features.category_classification import (
    A_CUM_SHARE,
    B_CUM_SHARE,
    SLOW_MOVER_MAX_QTY,
    classify_volume_band,
    resolve_category,
)


class TestClassifyVolumeBand:
    def test_zero_volume_is_slow_mover_d(self):
        assert classify_volume_band(total_qty=0.0, cum_share=0.01) == "D"

    def test_at_slow_mover_threshold_is_d(self):
        assert classify_volume_band(total_qty=SLOW_MOVER_MAX_QTY, cum_share=0.05) == "D"

    def test_just_above_slow_mover_threshold_is_not_d(self):
        result = classify_volume_band(total_qty=SLOW_MOVER_MAX_QTY + 0.01, cum_share=0.05)
        assert result != "D"

    def test_high_volume_low_cum_share_is_a(self):
        assert classify_volume_band(total_qty=10_000.0, cum_share=0.10) == "A"

    def test_at_a_boundary_is_a_inclusive(self):
        assert classify_volume_band(total_qty=1000.0, cum_share=A_CUM_SHARE) == "A"

    def test_just_above_a_boundary_is_b(self):
        assert classify_volume_band(total_qty=1000.0, cum_share=A_CUM_SHARE + 1e-9) == "B"

    def test_at_b_boundary_is_b_inclusive(self):
        assert classify_volume_band(total_qty=1000.0, cum_share=B_CUM_SHARE) == "B"

    def test_just_above_b_boundary_is_c(self):
        assert classify_volume_band(total_qty=1000.0, cum_share=B_CUM_SHARE + 1e-9) == "C"

    def test_full_cum_share_is_c(self):
        assert classify_volume_band(total_qty=1000.0, cum_share=1.0) == "C"

    def test_slow_mover_override_beats_high_cum_share_band(self):
        # a sku could theoretically have a tiny cum_share yet still exceed the volume floor's
        # inverse relationship in degenerate data; the point is D always wins on total_qty alone
        assert classify_volume_band(total_qty=1.0, cum_share=0.01) == "D"


class TestResolveCategory:
    def test_new_item_is_always_seasonal(self):
        assert resolve_category(is_new_item=True, volume_band="A") == "SEASONAL"
        assert resolve_category(is_new_item=True, volume_band="D") == "SEASONAL"

    def test_non_new_item_uses_volume_band(self):
        assert resolve_category(is_new_item=False, volume_band="B") == "B"


@pytest.mark.parametrize("band", ["A", "B", "C", "D"])
def test_resolve_category_passes_through_every_band_when_not_new(band):
    assert resolve_category(is_new_item=False, volume_band=band) == band
