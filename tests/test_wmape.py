"""
tests/test_wmape.py — WMAPE math + 3-grain gate unit tests (docs/instructions.md §3, §7).

Pure pandas. `gate`'s market_sku/category_cohort aggregation is volume-weighted
averaging of per-series WMAPE (training/evaluation.py's module docstring
explains why that's exact, not an approximation) — this file hand-verifies
the arithmetic for a small, fully worked fixture.
"""
from __future__ import annotations

import pandas as pd
import pytest

from config.loader import CategoryConfig
from training.evaluation import gate, gate_wmape_ceiling, wmape


class TestWmape:
    def test_perfect_prediction_is_zero(self):
        actual = pd.Series([10.0, 20.0, 30.0])
        assert wmape(actual, actual.copy()) == pytest.approx(0.0)

    def test_matches_hand_computed_value_for_a_partial_week(self):
        # n < 7 -> one partial "week" spanning all rows (wmape() chunks into 7-day
        # weeks first, see its docstring) -> collapses to sum(actual) vs sum(predicted).
        actual = pd.Series([10.0, 20.0, 30.0])
        predicted = pd.Series([12.0, 18.0, 33.0])
        # sum(actual)=60, sum(predicted)=63 -> |60-63|/60
        assert wmape(actual, predicted) == pytest.approx(3.0 / 60.0)

    def test_matches_hand_computed_value_for_a_full_week(self):
        # A single 7-day holdout is the standard backtest fold; wmape()'s docstring says
        # this simplifies to |sum(actual) - sum(predicted)| / sum(actual).
        actual = pd.Series([10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 0.0])
        predicted = pd.Series([12.0, 18.0, 33.0, 0.0, 0.0, 0.0, 0.0])
        # sum(actual)=60, sum(predicted)=63 -> |60-63|/60
        assert wmape(actual, predicted) == pytest.approx(3.0 / 60.0)

    def test_zero_total_volume_is_nan(self):
        actual = pd.Series([0.0, 0.0, 0.0])
        predicted = pd.Series([1.0, -1.0, 2.0])
        assert pd.isna(wmape(actual, predicted))

    def test_errors_cancel_within_a_week_by_design(self):
        # wmape() aggregates into 7-day weeks BEFORE taking the absolute difference
        # (its docstring: "intra-week over/under-forecasts cancel if the weekly total
        # is correct") — an overshoot and undershoot of equal magnitude in the SAME
        # week must net to zero.
        actual = pd.Series([10.0] * 7)
        predicted = pd.Series([10.0] * 5 + [12.0, 8.0])  # week sum still 70
        assert wmape(actual, predicted) == pytest.approx(0.0)

    def test_errors_across_different_weeks_are_absolute_not_signed(self):
        # Week 1 over-forecasts by 2, week 2 under-forecasts by 2 -- these must NOT
        # cancel ACROSS weeks even though same-week daily errors do (see above).
        actual = pd.Series([10.0] * 7 + [10.0] * 7)                    # week sums: 70, 70
        predicted = pd.Series([10.0] * 6 + [12.0] + [10.0] * 6 + [8.0])  # week sums: 72, 68
        # |70-72| + |70-68| = 4; sum(actual) = 140
        assert wmape(actual, predicted) == pytest.approx(4.0 / 140.0)


def _champion_row(series_key, market, category, sku_id, wmape_7d, volume) -> dict:
    return {
        "series_key": series_key,
        "market_id": market,
        "category": category,
        "sku_id": sku_id,
        "wmape_7d": wmape_7d,
        "volume": volume,
    }


@pytest.fixture
def categories() -> list[CategoryConfig]:
    return [
        CategoryConfig(
            market_id="US", category="A", accuracy_threshold=0.85, gate_grains=[],
            series_grain="market_store_sku", drift_tolerance=0.05,
        ),
    ]


@pytest.fixture
def champions_pdf() -> pd.DataFrame:
    # series A: ITEM1@S1, series B: ITEM1@S2 (same sku, different store), series C: ITEM2@S1
    return pd.DataFrame(
        [
            _champion_row("US|A|S1|ITEM1", "US", "A", "ITEM1", wmape_7d=0.10, volume=100.0),
            _champion_row("US|A|S2|ITEM1", "US", "A", "ITEM1", wmape_7d=0.20, volume=300.0),
            _champion_row("US|A|S1|ITEM2", "US", "A", "ITEM2", wmape_7d=0.05, volume=200.0),
        ]
    )


class TestGate:
    def test_market_store_sku_grain_equals_per_series_wmape(self, champions_pdf, categories):
        gated = gate(champions_pdf, categories)
        finest = gated[gated["grain"] == "market_store_sku"].set_index("entity_key")
        assert finest.loc["US|A|S1|ITEM1", "wmape"] == pytest.approx(0.10)
        assert finest.loc["US|A|S2|ITEM1", "wmape"] == pytest.approx(0.20)
        assert finest.loc["US|A|S1|ITEM2", "wmape"] == pytest.approx(0.05)

    def test_market_sku_grain_is_volume_weighted(self, champions_pdf, categories):
        gated = gate(champions_pdf, categories)
        item1 = gated[(gated["grain"] == "market_sku") & (gated["entity_key"] == "ITEM1")].iloc[0]
        # (0.10*100 + 0.20*300) / (100+300) = 70/400 = 0.175
        assert item1["wmape"] == pytest.approx(0.175)
        assert item1["volume"] == pytest.approx(400.0)

    def test_category_cohort_grain_is_volume_weighted_across_all_skus(self, champions_pdf, categories):
        gated = gate(champions_pdf, categories)
        cohort = gated[gated["grain"] == "category_cohort"].iloc[0]
        # (0.10*100 + 0.20*300 + 0.05*200) / 600 = 80/600
        assert cohort["wmape"] == pytest.approx(80.0 / 600.0)
        assert cohort["entity_key"] == "A"
        assert cohort["volume"] == pytest.approx(600.0)

    def test_passed_true_when_accuracy_meets_threshold(self, champions_pdf, categories):
        gated = gate(champions_pdf, categories)
        finest = gated[gated["grain"] == "market_store_sku"].set_index("entity_key")
        # 1 - 0.10 = 0.90 >= 0.85
        assert bool(finest.loc["US|A|S1|ITEM1", "passed"]) is True

    def test_passed_false_when_below_threshold(self, champions_pdf, categories):
        gated = gate(champions_pdf, categories)
        finest = gated[gated["grain"] == "market_store_sku"].set_index("entity_key")
        # 1 - 0.20 = 0.80 < 0.85
        assert bool(finest.loc["US|A|S2|ITEM1", "passed"]) is False

    def test_cohort_can_pass_even_when_a_constituent_series_fails(self, champions_pdf, categories):
        gated = gate(champions_pdf, categories)
        cohort = gated[gated["grain"] == "category_cohort"].iloc[0]
        # 1 - 80/600 = 0.8667 >= 0.85, even though series B (S2|ITEM1) individually fails
        assert bool(cohort["passed"]) is True

    def test_horizon_is_always_7(self, champions_pdf, categories):
        gated = gate(champions_pdf, categories)
        assert (gated["horizon"] == 7).all()

    def test_missing_category_threshold_yields_null_passed(self, champions_pdf):
        gated = gate(champions_pdf, categories=[])  # no threshold configured for "A"
        assert gated["threshold"].isna().all()
        assert gated["passed"].isna().all()

    def test_empty_champions_returns_empty_frame_with_expected_columns(self, categories):
        empty = pd.DataFrame(columns=["series_key", "market_id", "category", "sku_id", "wmape_7d", "volume"])
        gated = gate(empty, categories)
        assert gated.empty
        assert list(gated.columns) == [
            "market_id", "category", "grain", "entity_key", "horizon", "wmape", "volume", "threshold",
            "baseline_wmape", "skill", "min_skill", "passed", "bias_pct", "mase_7d",
        ]


def _skill_categories(min_skill: float) -> list[CategoryConfig]:
    return [
        CategoryConfig(
            market_id="US", category="A", accuracy_threshold=0.85, gate_grains=[],
            series_grain="market_store_sku", drift_tolerance=0.05, min_skill_vs_baseline=min_skill,
        ),
    ]


class TestSkillGate:
    """Relative gate: pass iff 1 - wmape / baseline_wmape >= min_skill_vs_baseline."""

    @pytest.fixture
    def champions_with_baseline(self, champions_pdf) -> pd.DataFrame:
        # Model WMAPEs 0.10 / 0.20 / 0.05 vs baselines 0.20 / 0.21 / 0.10:
        # per-series skill 50% / ~4.8% / 50%.
        return champions_pdf.assign(baseline_wmape_7d=[0.20, 0.21, 0.10])

    def test_series_skill_and_pass_fail(self, champions_with_baseline):
        gated = gate(champions_with_baseline, _skill_categories(0.10))
        finest = gated[gated["grain"] == "market_store_sku"].set_index("entity_key")
        assert finest.loc["US|A|S1|ITEM1", "skill"] == pytest.approx(0.5)
        assert finest.loc["US|A|S2|ITEM1", "skill"] == pytest.approx(1 - 0.20 / 0.21)
        assert bool(finest.loc["US|A|S1|ITEM1", "passed"]) is True
        assert bool(finest.loc["US|A|S2|ITEM1", "passed"]) is False

    def test_relative_gate_overrides_absolute_threshold(self):
        """30% WMAPE fails the absolute 85% target (70% accuracy) but beats a 40%-WMAPE
        baseline by 25% — in skill mode the relative result decides."""
        row = _champion_row("US|A|S1|ITEM1", "US", "A", "ITEM1", wmape_7d=0.30, volume=50.0)
        pdf = pd.DataFrame([{**row, "baseline_wmape_7d": 0.40}])
        assert bool(gate(pdf, _skill_categories(0.10)).iloc[0]["passed"]) is True
        assert bool(gate(pdf, _skill_categories(0.30)).iloc[0]["passed"]) is False
        absolute = [CategoryConfig(
            market_id="US", category="A", accuracy_threshold=0.85, gate_grains=[],
            series_grain="market_store_sku", drift_tolerance=0.05,
        )]
        assert bool(gate(pdf, absolute).iloc[0]["passed"]) is False

    def test_cohort_skill_is_volume_weighted(self, champions_with_baseline):
        gated = gate(champions_with_baseline, _skill_categories(0.0))
        cohort = gated[gated["grain"] == "category_cohort"].iloc[0]
        model_w = (0.10 * 100 + 0.20 * 300 + 0.05 * 200) / 600
        base_w = (0.20 * 100 + 0.21 * 300 + 0.10 * 200) / 600
        assert cohort["baseline_wmape"] == pytest.approx(base_w)
        assert cohort["skill"] == pytest.approx(1 - model_w / base_w)
        assert cohort["min_skill"] == 0.0

    def test_falls_back_to_absolute_gate_without_baseline(self, champions_pdf):
        gated = gate(champions_pdf, _skill_categories(0.10))
        finest = gated[gated["grain"] == "market_store_sku"].set_index("entity_key")
        # no baseline column -> absolute 85% target: 0.10 passes (90%), 0.20 fails (80%)
        assert bool(finest.loc["US|A|S1|ITEM1", "passed"]) is True
        assert bool(finest.loc["US|A|S2|ITEM1", "passed"]) is False

    def test_zero_last_fold_volume_series_still_gets_skill(self):
        row = _champion_row("US|A|S1|ITEM1", "US", "A", "ITEM1", wmape_7d=0.30, volume=0.0)
        gated = gate(pd.DataFrame([{**row, "baseline_wmape_7d": 0.40}]), _skill_categories(0.10))
        finest = gated[gated["grain"] == "market_store_sku"].iloc[0]
        assert finest["skill"] == pytest.approx(0.25)
        assert bool(finest["passed"]) is True


class TestGateWmapeCeiling:
    def test_relative_ceiling_uses_series_baseline(self):
        assert gate_wmape_ceiling(_skill_categories(0.10)[0], 0.40) == pytest.approx(0.36)

    def test_absolute_ceiling_when_no_baseline(self):
        assert gate_wmape_ceiling(_skill_categories(0.10)[0], float("nan")) == pytest.approx(0.15)

    def test_none_without_category(self):
        assert gate_wmape_ceiling(None, 0.4) is None

