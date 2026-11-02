"""
tests/test_halo_cannibalization_features.py — pure-Python unit tests for
data/features/halo_cannibalization_features.py.

The feature-construction logic itself is Spark SQL (no local Spark session —
no Spark in the test suite), same as its siblings event_features.py /
category_features.py, neither of which has Spark-execution tests either. What IS
testable locally, pure Python, is `classify_relation` — the exact mirror of the SQL
WHERE-clause thresholds used to label a pair 'halo' / 'cannib' / no relation
(same pattern as data/features/category_classification.py's classify_volume_band) —
plus the module's declared feature-class contract (HISTORICAL_ONLY vs
KNOWN_AT_INFERENCE column lists), which config/seed_defaults.py depends on.
"""
from __future__ import annotations

from data.features.halo_cannibalization_features import (
    HISTORICAL_ONLY_COLUMNS,
    KNOWN_AT_INFERENCE_COLUMNS,
    classify_relation,
)


class TestClassifyRelation:
    def test_strong_positive_correlation_is_halo(self):
        assert classify_relation(0.9) == "halo"

    def test_strong_negative_correlation_is_cannib(self):
        assert classify_relation(-0.9) == "cannib"

    def test_weak_correlation_is_no_relation(self):
        assert classify_relation(0.05) is None
        assert classify_relation(-0.05) is None

    def test_at_halo_threshold_is_inclusive(self):
        assert classify_relation(0.30, halo_threshold=0.30) == "halo"

    def test_just_below_halo_threshold_is_no_relation(self):
        assert classify_relation(0.30 - 1e-9, halo_threshold=0.30) is None

    def test_at_cannib_threshold_is_inclusive(self):
        assert classify_relation(-0.30, cannib_threshold=-0.30) == "cannib"

    def test_just_above_cannib_threshold_is_no_relation(self):
        assert classify_relation(-0.30 + 1e-9, cannib_threshold=-0.30) is None

    def test_zero_correlation_is_no_relation(self):
        assert classify_relation(0.0) is None

    def test_custom_thresholds_are_respected(self):
        assert classify_relation(0.5, halo_threshold=0.6) is None
        assert classify_relation(0.65, halo_threshold=0.6) == "halo"


class TestFeatureClassContract:
    """config/seed_defaults.py imports these lists directly; a silent drift here would
    silently mis-tag a feature as leakage-safe when it isn't (or vice versa)."""

    def test_historical_only_and_known_at_inference_are_disjoint(self):
        assert set(HISTORICAL_ONLY_COLUMNS).isdisjoint(set(KNOWN_AT_INFERENCE_COLUMNS))

    def test_no_duplicate_column_names_within_each_list(self):
        assert len(HISTORICAL_ONLY_COLUMNS) == len(set(HISTORICAL_ONLY_COLUMNS))
        assert len(KNOWN_AT_INFERENCE_COLUMNS) == len(set(KNOWN_AT_INFERENCE_COLUMNS))

    def test_historical_only_columns_are_demand_rolling_features(self):
        # Historical-only features require realized qty and must never be joined onto
        # the future spine — naming convention makes that reviewable at a glance.
        for col in HISTORICAL_ONLY_COLUMNS:
            assert col.endswith("_qty_7d_roll"), col

    def test_known_at_inference_columns_cover_strength_and_promo_signals(self):
        assert "halo_partner_corr_strength" in KNOWN_AT_INFERENCE_COLUMNS
        assert "cannib_partner_corr_strength" in KNOWN_AT_INFERENCE_COLUMNS
        assert "halo_partner_promo_active_flag" in KNOWN_AT_INFERENCE_COLUMNS
        assert "cannib_partner_promo_active_flag" in KNOWN_AT_INFERENCE_COLUMNS
