"""
training/backtest.py — rolling-origin backtest folds (docs/instructions.md §11).

4-6 folds over the last 8-12 weeks, 7-day horizon per fold, computed
point-in-time as of snapshot_date (no fold's training data includes its own
holdout dates or anything later).

Metrics produced per fold and averaged across folds:
  wmape_7d   — volume-weighted MAPE (existing; retained as the WMAPE gate input
               so planners' lag-1/lag-2 thresholds stay intact).
  mase_7d    — MAE / naive_mae where naive = lag-7 seasonal naïve (same weekday
               prior week). Scale-free; defined for zero-volume series. Used as
               the champion SELECTION criterion (training/champion.py).
  bias_pct   — (sum_yhat - sum_actual) / sum_actual. Positive = over-forecast.
               Supplementary diagnostic stored in backtest_metrics.
  naive_mae_7d — raw denominator; stored so downstream code can recompute MASE
               at different aggregation levels without re-running backtest.
  baseline_wmape_7d — WMAPE of the trailing-28-day-mean baseline on the same folds;
               the denominator of the relative "skill" gate (training/evaluation.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.evaluation import wmape as _wmape_fn
from training.models.base import ForecastModel


def rolling_origin_folds(
    series_pdf: pd.DataFrame,
    n_folds: int,
    val_weeks: int,
    eval_horizon_days: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return up to `n_folds` (train_pdf, holdout_pdf) pairs; each holdout is `eval_horizon_days`
    long, with origins spaced across the last `val_weeks` weeks of `series_pdf` (sorted by date_id).

    Point-in-time: a fold's `train_pdf` only ever contains rows at or before that fold's
    origin date — never the holdout's own dates or anything later. Folds with too little
    training or holdout data (short series, or a `val_weeks` window wider than the series'
    actual history) are silently dropped rather than padded.
    """
    pdf = series_pdf.sort_values("date_id").reset_index(drop=True)
    if pdf.empty:
        return []
    # Ensure date_id is Timestamp (Arrow may deliver datetime.date from Spark DateType)
    pdf["date_id"] = pd.to_datetime(pdf["date_id"])

    last_date = pd.Timestamp(pdf["date_id"].max())
    val_start = last_date - pd.Timedelta(days=val_weeks * 7)
    latest_origin = last_date - pd.Timedelta(days=eval_horizon_days)
    earliest_origin = min(val_start, latest_origin)

    if n_folds <= 1:
        origins = [latest_origin]
    else:
        total_days = (latest_origin - earliest_origin).days
        step_days = total_days // (n_folds - 1)
        origins = [earliest_origin + pd.Timedelta(days=step_days * k) for k in range(n_folds)]

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for origin in origins:
        train_pdf = pdf[pdf["date_id"] <= origin]
        holdout_pdf = pdf[
            (pdf["date_id"] > origin) & (pdf["date_id"] <= origin + pd.Timedelta(days=eval_horizon_days))
        ]
        if len(train_pdf) < eval_horizon_days or len(holdout_pdf) < eval_horizon_days:
            continue
        folds.append((train_pdf.reset_index(drop=True), holdout_pdf.reset_index(drop=True)))
    return folds


# Trailing window of the level baseline the relative gate is measured against: the mean
# daily qty over the last 28 training days, held flat across the holdout. Deliberately a
# strong naive benchmark (on Kaggle M5 it beats the lag-7 seasonal naive and several
# per-series models outright), so "beats the baseline" is a meaningful bar.
BASELINE_WINDOW_DAYS = 28


def _nanmean_or_nan(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return float(np.mean(finite)) if finite else float("nan")


def baseline_wmape(train_pdf: pd.DataFrame, holdout_pdf: pd.DataFrame) -> float:
    """WMAPE of the trailing-BASELINE_WINDOW_DAYS-mean forecast on one fold's holdout."""
    history = train_pdf["qty"].astype(float).tail(BASELINE_WINDOW_DAYS)
    if history.empty:
        return float("nan")
    actual = holdout_pdf["qty"].astype(float).reset_index(drop=True)
    return _wmape_fn(actual, pd.Series(np.full(len(actual), history.mean())))


def backtest_metrics(
    model: ForecastModel,
    series_pdf: pd.DataFrame,
    n_folds: int,
    val_weeks: int,
    eval_horizon_days: int,
    params: dict,
) -> dict:
    """Fit+predict across all folds; return averaged metrics dict.

    Mutates `model` in place fold-by-fold — after this returns, `model` is fitted
    on the LAST fold's training window, so `engine.evaluate_candidate` can call
    `model.decompose(holdout)` without a redundant re-fit.

    Returns:
        wmape_7d    — volume-weighted MAPE averaged across folds (WMAPE gate input).
        mase_7d     — MAE / naive_mae averaged across folds (champion SELECTION key).
        bias_pct    — directional error averaged across folds (diagnostic).
        naive_mae_7d — lag-7 naïve MAE averaged across folds (MASE denominator).
        baseline_wmape_7d — WMAPE of the trailing-28-day-mean baseline on the same folds
                    (denominator of the relative "skill" gate, training/evaluation.py).
    All values are NaN when there are no usable folds.
    """
    folds = rolling_origin_folds(series_pdf, n_folds, val_weeks, eval_horizon_days)
    if not folds:
        nan = float("nan")
        return {"wmape_7d": nan, "mase_7d": nan, "bias_pct": nan, "naive_mae_7d": nan,
                "baseline_wmape_7d": nan,
                "osc_ratio": nan, "actual_ac7": nan}

    fold_wmape, fold_mase, fold_bias, fold_naive_mae = [], [], [], []
    fold_baseline_wmape = [baseline_wmape(train_pdf, holdout_pdf) for train_pdf, holdout_pdf in folds]
    for train_pdf, holdout_pdf in folds:
        model.fit(train_pdf, holdout_pdf, params)
        predicted = model.predict(holdout_pdf).reset_index(drop=True)
        actual    = holdout_pdf["qty"].astype(float).reset_index(drop=True)
        h         = len(actual)

        # ── WMAPE (existing gate metric) ────────────────────────────────────
        fold_wmape.append(_wmape_fn(actual, predicted))

        # ── Bias: (sum_yhat - sum_actual) / sum_actual ──────────────────────
        denom = float(actual.sum())
        fold_bias.append(
            float((predicted.sum() - actual.sum()) / denom) if denom != 0 else float("nan")
        )

        # ── Naive MAE: lag-7 seasonal naïve (last h rows of training window) ─
        # naive[i] = train_pdf["qty"].iloc[-(h - i)] — same weekday prior week.
        naive_vals = train_pdf["qty"].astype(float).iloc[-h:].reset_index(drop=True)
        if len(naive_vals) == h:
            naive_mae = float((actual - naive_vals).abs().mean())
            fold_naive_mae.append(naive_mae)
            # MASE: MAE / naive_mae
            mae_val = float((actual - predicted).abs().mean())
            fold_mase.append(mae_val / naive_mae if naive_mae > 0 else float("nan"))
        else:
            fold_mase.append(float("nan"))

    return {
        "wmape_7d":     float(np.nanmean(fold_wmape)),
        "mase_7d":      float(np.nanmean(fold_mase)),
        "bias_pct":     float(np.nanmean(fold_bias)),
        "naive_mae_7d": float(np.nanmean(fold_naive_mae)) if fold_naive_mae else float("nan"),
        "baseline_wmape_7d": _nanmean_or_nan(fold_baseline_wmape),
    }


def backtest_model(
    model: ForecastModel,
    series_pdf: pd.DataFrame,
    n_folds: int,
    val_weeks: int,
    eval_horizon_days: int,
    params: dict,
) -> float:
    """Backward-compatible wrapper — returns only wmape_7d (float).

    Prefer `backtest_metrics()` for new code; this shim exists so training/hpt.py
    and any external callers that expect a float are not broken by the metrics
    expansion.
    """
    return backtest_metrics(model, series_pdf, n_folds, val_weeks, eval_horizon_days, params)["wmape_7d"]
