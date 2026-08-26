"""
demo/run_local_demo.py — no-Spark local demo.

Generates a synthetic multi-store x multi-sku dataset (data/synthetic/generator.py),
fits real candidate models (Prophet/statsforecast/LightGBM/ElasticNet/Croston) via the
same pandas-only functions the test suite exercises — training/engine.py::fit_one_series,
training/champion.py::select_champions, training/evaluation.py::gate/wmape — and writes
the README's accuracy table + example forecast charts to docs/assets/.

Not part of the deployed pipeline (no Spark/Databricks here — this is exactly what
CLAUDE.md's "authoring-only" environment supports). Re-run whenever the README's
numbers need refreshing:

    pip install -r requirements.txt
    python demo/run_local_demo.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# __file__ is available here (a plain script, never a notebook cell) — resolve the
# repo root from this file's own location so `python demo/run_local_demo.py` works
# from any cwd, no hardcoded path.
_REPO_ROOT = str(_Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
del _sys, _Path, _REPO_ROOT

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.loader import CategoryConfig, ForecastConfig, MarketConfig, ModelConfig
from config.seed_defaults import DEFAULT_CATEGORY_THRESHOLDS
from data.synthetic.generator import generate_synthetic_source
from training.engine import fit_one_series
from training.evaluation import gate
from training.models.registry import build as build_model

_SEED = 42
_N_STORES = 12
_N_SKUS = 40
_N_DAYS = 730
_HORIZON_DAYS = 14
_MAX_SERIES = 40
_CANDIDATE_MODELS = ["prophet", "statsforecast", "lgbm", "elastic_net", "croston"]
_OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"


def _build_config() -> ForecastConfig:
    """A minimal ForecastConfig for the demo — same dataclasses config/loader.py
    returns in production, just built inline instead of read from Delta."""
    market_config = MarketConfig(
        market_id="US",
        active=True,
        horizon_days=140,
        eval_horizon_days=7,
        history_start="2024-01-01",
        incremental_read_days=750,
        incremental_write_days=365,
        backtest_folds=3,
        val_weeks=6,
        hpt_trigger="threshold_miss",
        hpt_trials=10,
        uplift_enabled=False,
        uplift_learner="lgbm_global",
        uplift_clip_frac=0.25,
        reconstruction_tol_pct=0.02,
        run_budget_usd=100.0,
    )
    categories = [
        CategoryConfig(
            market_id="US", category=cat, accuracy_threshold=thresh,
            gate_grains=["category_cohort", "market_sku", "market_store_sku"],
            series_grain="market_store_sku", drift_tolerance=0.05,
        )
        for cat, thresh in DEFAULT_CATEGORY_THRESHOLDS.items()
    ]
    models = [
        ModelConfig(market_id="US", model_type=mt, enabled=True, param_space={})
        for mt in _CANDIDATE_MODELS
    ]
    return ForecastConfig(market_config=market_config, categories=categories, models=models, features=[])


def _calendar_with_event_flags(cal_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Pandas-only re-derivation of the calendar/holiday/promo columns
    data/features/temporal_features.py + event_features.py build via Spark SQL —
    Spark isn't available locally, so the demo derives the same handful of
    known-at-inference columns directly in pandas. Not a parallel production
    path: this is demo-only feature construction, deliberately simpler than
    the real feature pipeline (no lags/rolling stats — those aren't needed to
    exercise the model wrappers' decomposition contract)."""
    cal_df = cal_df.copy()
    cal_df["date_id"] = pd.to_datetime(cal_df["date_id"])
    cal_df["dow"] = np.where(cal_df["weekday"] == 1, 7, cal_df["weekday"] - 1)
    cal_df["week_of_year"] = cal_df["week"]
    cal_df["day_of_month"] = cal_df["day"]
    cal_df["weekend_flag"] = cal_df["weekday_name"].isin(["Saturday", "Sunday"]).astype(int)
    cal_df["payday_flag"] = (
        ((cal_df["day"] >= 25) | (cal_df["day"] <= 5)) & (cal_df["weekend_flag"] == 0)
    ).astype(int)

    blocks = []
    for market_id, mkt_events in events_df.groupby("market_id"):
        holiday_flag = pd.Series(0, index=cal_df.index)
        promo_flag = pd.Series(0, index=cal_df.index)
        for _, ev in mkt_events.iterrows():
            mask = (cal_df["date_id"] >= pd.Timestamp(ev["start_date"])) & (
                cal_df["date_id"] <= pd.Timestamp(ev["end_date"])
            )
            if ev["event_type"] == "HOLIDAY":
                holiday_flag = holiday_flag | mask.astype(int)
            else:
                promo_flag = promo_flag | mask.astype(int)
        block = cal_df.copy()
        block["market_id"] = market_id
        block["holiday_flag"] = holiday_flag
        block["promo_active_flag"] = promo_flag
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _build_series_frames(frames: dict[str, pd.DataFrame]) -> dict[tuple[str, str], pd.DataFrame]:
    """One pandas DataFrame per (store_id, sku_id) series, shaped for
    training/engine.py::fit_one_series (series_key/market_id/category/store_id/
    sku_id/date_id/qty/isActive + the calendar/event feature columns)."""
    cal_flags = _calendar_with_event_flags(frames["calendar_date"], frames["events"])
    sku_cat = dict(zip(frames["sku"]["sku_id"], frames["sku"]["category"]))

    sales_df = frames["sales"].copy()
    sales_df["date_id"] = pd.to_datetime(sales_df["date_id"])
    sales_df["qty"] = sales_df["units_sold"] + sales_df["units_promotional"] + sales_df["units_shrinkage"]

    merged = sales_df.merge(cal_flags, on=["market_id", "date_id"], how="left")
    merged["category"] = merged["sku_id"].map(sku_cat).fillna("D")
    merged["series_key"] = (
        merged["market_id"] + "|" + merged["category"] + "|" + merged["store_id"] + "|" + merged["sku_id"]
    )
    merged["isActive"] = 1

    keep_cols = [
        "series_key", "market_id", "category", "store_id", "sku_id", "date_id", "qty", "isActive",
        "dow", "week_of_year", "month", "quarter", "day_of_month",
        "payday_flag", "weekend_flag", "holiday_flag", "promo_active_flag",
    ]
    return {
        (store_id, sku_id): g.sort_values("date_id").reset_index(drop=True)[keep_cols]
        for (store_id, sku_id), g in merged.groupby(["store_id", "sku_id"])
    }


def _plot_series(series_key: str, series_pdf: pd.DataFrame, model_type: str, out_path: Path) -> None:
    """Actual-vs-forecast + additive decomposition stack for one series' holdout window —
    refits the champion model type on a plain train/holdout split (same pattern as
    tests/test_models.py) purely to get a decompose() to plot; the champion SELECTION
    above already ran the real backtest via fit_one_series."""
    train = series_pdf.iloc[: -_HORIZON_DAYS]
    horizon = series_pdf.iloc[-_HORIZON_DAYS:]

    model = build_model(model_type)
    model.fit(train, horizon, params={})
    decomp = model.decompose(horizon)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[1.3, 1])

    tail = series_pdf.iloc[-(_HORIZON_DAYS + 45):]
    ax1.plot(tail["date_id"], tail["qty"], label="actual", color="#333333", linewidth=1.5)
    ax1.plot(horizon["date_id"], decomp.yhat, label="forecast", color="#1f77b4", linewidth=2, linestyle="--")
    ax1.axvline(horizon["date_id"].iloc[0], color="#999999", linestyle=":", linewidth=1)
    ax1.set_title(f"{series_key} — {model_type} (actual vs. forecast)")
    ax1.legend(loc="upper left")
    ax1.set_ylabel("qty")

    # .to_numpy() (not the raw Series) — decomp's Series carry horizon_pdf's original
    # row index (e.g. 716-729), which would silently reindex-to-NaN against the
    # DataFrame's date index below if passed through directly.
    components = {
        "trend": decomp.trend.to_numpy(),
        "seasonality_weekly": decomp.seasonality_weekly.to_numpy(),
        "seasonality_yearly": decomp.seasonality_yearly.to_numpy(),
        "temporal_features_effect": decomp.temporal_features_effect.to_numpy(),
    }
    for name, series in decomp.holiday_lift.items():
        components[f"holiday:{name}"] = series.to_numpy()
    for name, series in decomp.promo_lift.items():
        components[f"promo:{name}"] = series.to_numpy()
    # Drop components that never move (e.g. LightGBM's SHAP path emits one bucket per
    # possible promo tactic/subtype, most unused by this demo's simplified feature set) —
    # a real production chart with the full feature set would populate more of these.
    components = {k: v for k, v in components.items() if np.abs(v).max() > 1e-6}
    comp_df = pd.DataFrame(components, index=horizon["date_id"].values)
    # matplotlib's stackplot (not pandas' .plot.area) — tolerates components that go
    # negative (e.g. seasonality_yearly, a holiday dip), which a real decomposition can.
    ax2.stackplot(comp_df.index, *[comp_df[c] for c in comp_df.columns], labels=comp_df.columns, alpha=0.85)
    ax2.set_title("additive decomposition (holdout window)")
    ax2.legend(loc="upper left", fontsize=7, ncol=2)
    ax2.set_ylabel("contribution")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_demo() -> dict:
    t0 = time.time()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = generate_synthetic_source(
        seed=_SEED, n_stores=_N_STORES, n_skus=_N_SKUS, n_days=_N_DAYS, markets=["US"]
    )
    series_frames = _build_series_frames(frames)
    config = _build_config()
    snapshot_date = str(frames["calendar_date"]["date_id"].max())

    all_keys = list(series_frames.keys())
    rng = np.random.default_rng(_SEED)
    sample_idx = rng.choice(len(all_keys), size=min(_MAX_SERIES, len(all_keys)), replace=False)
    sample_keys = [all_keys[i] for i in sample_idx]

    fit_t0 = time.time()
    result_rows = []
    for store_id, sku_id in sample_keys:
        series_pdf = series_frames[(store_id, sku_id)]
        if len(series_pdf) < config.market_config.eval_horizon_days * 4:
            continue
        result = fit_one_series(series_pdf, config, snapshot_date)
        row = result.iloc[0].to_dict()
        if row["status"] in ("ok", "stable"):
            result_rows.append(row)
    fit_elapsed = time.time() - fit_t0

    champions_pdf = pd.DataFrame(result_rows)
    gated = gate(champions_pdf, config.categories)
    series_gate = gated[gated["grain"] == "market_store_sku"].copy()

    category_summary = (
        gated[gated["grain"] == "category_cohort"][["entity_key", "wmape", "threshold", "passed", "volume"]]
        .rename(columns={"entity_key": "category"})
        .sort_values("category")
        .to_dict("records")
    )

    # Representative series for the README's example charts: lowest-WMAPE series for up
    # to 3 distinct champion model types (favors a diverse showcase over 3 near-identical
    # picks, which is what plain best/median/worst tends to produce when one cheap model
    # dominates the low-volume tail).
    result_by_key = {r["series_key"]: r for r in result_rows}
    ranked = series_gate.sort_values("wmape").reset_index(drop=True)
    picks, seen_models, seen_keys = [], set(), set()
    for _, row in ranked.iterrows():
        model_type = result_by_key[row["entity_key"]]["model_type"]
        if model_type in seen_models:
            continue
        seen_models.add(model_type)
        seen_keys.add(row["entity_key"])
        picks.append(row)
        if len(picks) == 3:
            break
    for _, row in ranked.iterrows():
        if len(picks) == 3:
            break
        if row["entity_key"] not in seen_keys:
            seen_keys.add(row["entity_key"])
            picks.append(row)

    chart_files = []
    for i, pick in enumerate(picks):
        champ_row = result_by_key[pick["entity_key"]]
        store_id, sku_id = champ_row["store_id"], champ_row["sku_id"]
        model_type = champ_row["model_type"]
        out_path = _OUT_DIR / f"forecast_{i}_{sku_id}_{store_id}.png"
        _plot_series(champ_row["series_key"], series_frames[(store_id, sku_id)], model_type, out_path)
        chart_files.append({
            "file": out_path.name,
            "series_key": champ_row["series_key"],
            "model_type": model_type,
            "wmape_7d": float(champ_row["wmape_7d"]),
        })

    total_elapsed = time.time() - t0
    metrics = {
        "seed": _SEED,
        "n_stores": _N_STORES,
        "n_skus": _N_SKUS,
        "n_series_sampled": len(sample_keys),
        "n_series_evaluated": len(result_rows),
        "candidate_models": _CANDIDATE_MODELS,
        "overall_wmape_7d": float(np.average(series_gate["wmape"], weights=series_gate["volume"])) if len(series_gate) else None,
        "category_summary": category_summary,
        "charts": chart_files,
        "timing": {
            "fit_and_select_seconds": round(fit_elapsed, 2),
            "total_seconds": round(total_elapsed, 2),
            "series_per_second": round(len(result_rows) / fit_elapsed, 3) if fit_elapsed > 0 else None,
        },
    }

    with open(_OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    m = run_demo()
    print(json.dumps(m, indent=2))
