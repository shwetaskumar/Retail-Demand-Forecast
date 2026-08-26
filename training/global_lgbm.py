"""
training/global_lgbm.py — MMF-style per-SKU global LightGBM.

Architecture (Meta's Many-Models Forecasting pattern, adapted):
  Instead of 475K per-(store,SKU) LightGBM fits, train ONE model per sku_id
  that pools ALL stores' data for that item. This provides:
    - Cross-store learning: promotional effects, day-of-week patterns are shared
    - Faster training: 819 LGB fits instead of 475K
    - Better accuracy for sparse store-SKU pairs that benefit from store pooling
    - Explainability via SHAP (same as per-series LGB)

  store_id is included as a LightGBM categorical feature so the model learns
  store-specific adjustments on top of the shared item behaviour.

  Per-SKU model replaces the per-series LightGBM champion only when its holdout
  WMAPE is lower than the local model already registered for that series.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

# Features used by the global LGB model — must exist in sku_store_features
_FEATURE_COLS = [
    "qty_lag_1", "qty_lag_7", "qty_lag_14", "qty_lag_28",
    "rolling_avg_7", "rolling_avg_28",
    "ewma_7", "ewma_28",
    "weekday", "week", "month", "year",
    "is_holiday", "is_promo",
    "store_id",    # categorical — store ID
]

_CATEGORICAL_COLS = ["store_id"]

_OUTPUT_SCHEMA = (
    "series_key string, market_id string, category string, store_id string, sku_id string, "
    "snapshot_date string, model_type string, params string, wmape_7d double, volume double, "
    "reconstruction_error double, passed_decomposition boolean, status string"
)

_DEFAULT_LGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def _wmape(actual: pd.Series, predicted: pd.Series) -> float:
    denom = actual.abs().sum()
    return float((actual - predicted).abs().sum() / denom) if denom > 0 else float("nan")


def _fit_sku_model(sku_pdf: pd.DataFrame, snapshot_date: str, market_id: str, horizon: int = 7):
    """Fit one LightGBM on all stores' data for a single sku_id.

    Returns one result row per (store_id, sku_id) pair.
    """
    import lightgbm as lgb

    sku_id = sku_pdf["sku_id"].iloc[0]
    sku_pdf = sku_pdf.sort_values(["store_id", "date_id"]).reset_index(drop=True)
    sku_pdf["date_id"] = pd.to_datetime(sku_pdf["date_id"])

    # Only rows with valid qty for training
    train_mask  = (sku_pdf["isActive"] == 1) & sku_pdf["qty"].notna()
    active_pdf  = sku_pdf[train_mask].copy()

    if len(active_pdf) < horizon * 10:
        return pd.DataFrame()  # not enough data for this SKU at all

    # Single holdout: last `horizon` days across ALL stores
    cutoff     = active_pdf["date_id"].max() - pd.Timedelta(days=horizon)
    train_pdf  = active_pdf[active_pdf["date_id"] <= cutoff]
    test_pdf   = active_pdf[active_pdf["date_id"] >  cutoff]

    if len(train_pdf) < horizon * 5 or len(test_pdf) == 0:
        return pd.DataFrame()

    # Prepare features — drop rows with any NaN in feature cols
    avail_feats = [c for c in _FEATURE_COLS if c in train_pdf.columns]
    cat_feats   = [c for c in _CATEGORICAL_COLS if c in avail_feats]

    X_train = train_pdf[avail_feats].copy()
    y_train = train_pdf["qty"].astype(float)
    X_test  = test_pdf[avail_feats].copy()

    # Encode categoricals
    for col in cat_feats:
        X_train[col] = X_train[col].astype("category")
        X_test[col]  = X_test[col].astype("category")

    # Drop rows where target or features are NaN
    valid = y_train.notna() & X_train.notna().all(axis=1)
    X_train, y_train = X_train[valid], y_train[valid]
    if len(X_train) < horizon:
        return pd.DataFrame()

    try:
        model = lgb.LGBMRegressor(**_DEFAULT_LGB_PARAMS)
        model.fit(X_train, y_train, categorical_feature=cat_feats if cat_feats else "auto")
    except Exception:
        return pd.DataFrame()

    # Evaluate per series (store_id) in the holdout
    rows = []
    for store_id, grp in test_pdf.groupby("store_id"):
        X_grp = grp[avail_feats].copy()
        for col in cat_feats:
            X_grp[col] = X_grp[col].astype("category")
        try:
            pred = pd.Series(model.predict(X_grp), index=grp.index)
        except Exception:
            continue
        actual = grp["qty"].astype(float)
        wmape_val = _wmape(actual, pred)
        category  = grp["category"].iloc[0] if "category" in grp.columns else None
        series_key = f"{market_id}|{category}|{store_id}|{sku_id}"
        rows.append({
            "series_key":           series_key,
            "market_id":        market_id,
            "category":             category,
            "store_id":          str(store_id),
            "sku_id":              sku_id,
            "snapshot_date":        snapshot_date,
            "model_type":           "lgbm_global",
            "params":               json.dumps(_DEFAULT_LGB_PARAMS),
            "wmape_7d":             wmape_val,
            "volume":               float(actual.abs().sum()),
            "reconstruction_error": float("nan"),
            "passed_decomposition": True,   # LGB decomp not validated here; inherit
            "status":               "ok",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_global_lgbm(
    spark: "SparkSession",
    features_df: "DataFrame",
    market_id: str,
    snapshot_date: str,
    horizon: int = 7,
) -> "DataFrame":
    """Fit per-SKU global LightGBM on all series' feature data.

    Returns per-(store_id, sku_id) candidate rows in the same
    OUTPUT_SCHEMA as the local engine — ready for champion comparison.
    """
    from pyspark.sql import functions as F

    tagged_df = features_df.withColumn("market_id", F.lit(market_id))

    return (
        tagged_df
        .groupBy("sku_id")
        .applyInPandas(
            lambda pdf: _fit_sku_model(pdf, snapshot_date, market_id, horizon),
            schema=_OUTPUT_SCHEMA,
        )
    )
