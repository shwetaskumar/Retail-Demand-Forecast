"""
conftest.py — local test harness (NO Databricks / NO real Spark required).

Purpose: let `pytest` run on a laptop with no cluster. Core transforms must
accept an injected `spark` and use pandas-friendly logic, or be unit-tested
against these fakes. Real Spark/dbutils only exist on Databricks at runtime.
"""
from unittest.mock import MagicMock
import pandas as pd
import pytest


@pytest.fixture
def fake_dbutils():
    """Stand-in for dbutils; extend as needed (widgets, secrets, fs)."""
    dbu = MagicMock(name="dbutils")
    dbu.widgets.get.side_effect = lambda k: {"env": "retail_forecast", "market": "US"}.get(k, "")
    dbu.secrets.get.return_value = "fake-secret"
    return dbu


@pytest.fixture
def fake_spark():
    """
    Minimal Spark stand-in. Prefer testing pure logic on pandas fixtures;
    use this only to assert wiring (method calls), not real computation.
    """
    return MagicMock(name="SparkSession")


@pytest.fixture
def sales_pdf():
    """Tiny deterministic sales frame for transform/leakage/reconstruction tests."""
    return pd.DataFrame(
        {
            "store_id": ["S1", "S1", "S1", "S1"],
            "sku_id": ["ITEM_A"] * 4,
            "date_id": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            "qty": [10.0, 12.0, 0.0, 15.0],
            "isActive": [1, 1, 1, 1],
        }
    )


@pytest.fixture
def decomposition_pdf():
    """Components that must reconstruct yhat within tolerance (for §10.1 test)."""
    return pd.DataFrame(
        {
            "trend": [10.0],
            "seasonality_weekly": [1.5],
            "seasonality_yearly": [0.5],
            "holiday_lift": [2.0],
            "promo_lift": [3.0],
            "temporal_features_effect": [0.0],
            "model_uplift": [1.0],
            "yhat": [18.0],  # == sum of components
        }
    )
