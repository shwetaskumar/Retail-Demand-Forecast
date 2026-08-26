"""
tests/test_champion.py — champion selection unit tests (Step 6, docs/instructions.md §11).

Pure pandas — training/champion.py::select_champions is the canonical logic
(select_champions_spark mirrors it in SQL for the real, no-collect path).
"""
from __future__ import annotations

import pandas as pd

from training.champion import select_champions


def _row(series_key, model_type, wmape_7d, mase_7d=None, bias_pct=float("nan"),
         reconstruction_error=0.0, passed=True, status="ok") -> dict:
    """`mase_7d` defaults to mirroring `wmape_7d` when not given — select_champions()
    sorts on mase_7d (not wmape_7d) since the MASE-based selection migration, so every
    test below that only varies wmape_7d still exercises its intended ordering."""
    return {
        "series_key": series_key,
        "market": "US",
        "category": "A",
        "store_id": "S1",
        "sku_id": "ITEM_A",
        "snapshot_date": "2024-06-01",
        "model_type": model_type,
        "params": "{}",
        "mase_7d": mase_7d if mase_7d is not None else wmape_7d,
        "wmape_7d": wmape_7d,
        "bias_pct": bias_pct,
        "reconstruction_error": reconstruction_error,
        "passed_decomposition": passed,
        "status": status,
    }


def test_picks_lowest_wmape_among_valid_candidates():
    pdf = pd.DataFrame(
        [
            _row("S1", "prophet", wmape_7d=0.30),
            _row("S1", "lgbm", wmape_7d=0.15),
            _row("S1", "statsforecast", wmape_7d=0.25),
        ]
    )
    champions = select_champions(pdf)
    assert len(champions) == 1
    assert champions.iloc[0]["model_type"] == "lgbm"


def test_ignores_candidates_that_failed_decomposition():
    pdf = pd.DataFrame(
        [
            _row("S1", "lgbm", wmape_7d=0.05, passed=False),  # best wmape but invalid decomposition
            _row("S1", "prophet", wmape_7d=0.30, passed=True),
        ]
    )
    champions = select_champions(pdf)
    assert champions.iloc[0]["model_type"] == "prophet"


def test_ignores_candidates_with_error_status():
    pdf = pd.DataFrame(
        [
            _row("S1", "lgbm", wmape_7d=0.05, status="error:boom"),
            _row("S1", "prophet", wmape_7d=0.30, status="ok"),
        ]
    )
    champions = select_champions(pdf)
    assert champions.iloc[0]["model_type"] == "prophet"


def test_tie_break_by_reconstruction_error():
    pdf = pd.DataFrame(
        [
            _row("S1", "prophet", wmape_7d=0.20, reconstruction_error=0.01),
            _row("S1", "lgbm", wmape_7d=0.20, reconstruction_error=0.001),
        ]
    )
    champions = select_champions(pdf)
    assert champions.iloc[0]["model_type"] == "lgbm"


def test_tie_break_prefers_non_uplift_over_uplift():
    pdf = pd.DataFrame(
        [
            _row("S1", "lgbm+uplift", wmape_7d=0.20, reconstruction_error=0.0),
            _row("S1", "lgbm", wmape_7d=0.20, reconstruction_error=0.0),
        ]
    )
    champions = select_champions(pdf)
    assert champions.iloc[0]["model_type"] == "lgbm"


def test_final_tie_break_is_alphabetical_model_type():
    pdf = pd.DataFrame(
        [
            _row("S1", "statsforecast", wmape_7d=0.20, reconstruction_error=0.0),
            _row("S1", "lgbm", wmape_7d=0.20, reconstruction_error=0.0),
        ]
    )
    champions = select_champions(pdf)
    assert champions.iloc[0]["model_type"] == "lgbm"


def test_returns_empty_frame_when_no_candidate_is_valid():
    pdf = pd.DataFrame([_row("S1", "lgbm", wmape_7d=0.20, passed=False)])
    champions = select_champions(pdf)
    assert champions.empty


def test_selects_independently_per_series_key():
    pdf = pd.DataFrame(
        [
            _row("S1", "lgbm", wmape_7d=0.10),
            _row("S1", "prophet", wmape_7d=0.30),
            _row("S2", "prophet", wmape_7d=0.05),
            _row("S2", "lgbm", wmape_7d=0.40),
        ]
    )
    champions = select_champions(pdf).set_index("series_key")
    assert champions.loc["S1", "model_type"] == "lgbm"
    assert champions.loc["S2", "model_type"] == "prophet"
