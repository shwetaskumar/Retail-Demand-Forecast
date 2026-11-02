"""
demo/run_local_demo.py — no-Spark local demo, run against the real Kaggle M5
Forecasting Accuracy dataset (Walmart daily unit sales).

Runs the production default candidate set (config/seed_defaults.py: the equal-weight
auto_arima + ets_event + lgbm ensemble — see that file for why single models are off by
default) through the pandas-only functions the test suite exercises —
training/engine.py::fit_one_series, training/evaluation.py::gate — then scores the
champions on a held-out week against the trailing-28-day-mean baseline, and writes the
README's accuracy table + example forecast charts to docs/assets/.
demo/compare_selection_rules.py is the companion experiment comparing champion-selection
rules over several held-out weeks.

Requires the M5 dataset downloaded locally first (see data/kaggle_m5/loader.py's
module docstring for the Kaggle CLI steps — M5 is a competition, not a plain
dataset, so this needs your own Kaggle account/credentials and cannot be fetched
automatically):

    pip install -r requirements.txt
    kaggle competitions download -c m5-forecasting-accuracy -p data/m5_raw
    unzip data/m5_raw/m5-forecasting-accuracy.zip -d data/m5_raw
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
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.loader import CategoryConfig, ForecastConfig, MarketConfig, ModelConfig
from config.seed_defaults import DEFAULT_CATEGORY_SKILL_THRESHOLDS, DEFAULT_CATEGORY_THRESHOLDS
from data.features.category_classification import classify_volume_band, resolve_category
from data.kaggle_m5.loader import load_m5_source
from training.backtest import rolling_origin_folds
from training.engine import fit_one_series
from training.evaluation import gate
from training.models.registry import build as build_model

_M5_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "m5_raw"
_STATES = ["CA"]          # M5 state_id -> our market_id; CA alone keeps the local run fast
_MAX_SKUS = 300           # sampled before the (expensive) wide->long melt — see loader docstring
_TRAILING_DAYS = 730      # ~2 years of history, most recent
_NEW_ITEM_MAX_DAYS = 365  # data/features/lifecycle.py's is_new_item threshold, mirrored here
# Held-out evaluation week (docs/instructions.md §9: "evaluate champions on the next 7
# days"). Champions are selected by fit_one_series on history that EXCLUDES this week, then
# scored on it — so the README's headline numbers are out-of-sample, not the in-sample
# backtest score of a best-of-N pick (which flatters per-series selection; see run_demo).
_HOLDOUT_DAYS = 7
_BASELINE_DAYS = 28  # training/backtest.py::BASELINE_WINDOW_DAYS — the gate's yardstick
_CHART_MIN_WEEKLY_UNITS = 7  # charts skip near-zero series (nothing visible to plot)
# 150 series: enough that the category numbers stop swinging with which handful of series
# the sampler drew (a 40-series sample once reported 24.6% overall WMAPE; 150 puts it ~40%).
_MAX_SERIES = 150
# Mirrors the production default config: config/seed_defaults.py's cost tiers minus
# _DEFAULT_DISABLED_MODELS.
_CANDIDATE_MODELS = ["ensemble"]
_OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"


def _build_config(markets: list[str]) -> ForecastConfig:
    """A minimal ForecastConfig for the demo — same dataclasses config/loader.py
    returns in production, just built inline instead of read from Delta. One
    CategoryConfig per (market, category) combo, same thresholds for every market
    (a real multi-market deployment would seed per-market thresholds via the app)."""
    market_config = MarketConfig(
        market_id=markets[0],
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
            market_id=market_id, category=cat, accuracy_threshold=thresh,
            gate_grains=["category_cohort", "market_sku", "market_store_sku"],
            series_grain="market_store_sku", drift_tolerance=0.05,
            min_skill_vs_baseline=DEFAULT_CATEGORY_SKILL_THRESHOLDS.get(cat),
        )
        for market_id in markets
        for cat, thresh in DEFAULT_CATEGORY_THRESHOLDS.items()
    ]
    models = [
        ModelConfig(market_id=markets[0], model_type=mt, enabled=True, param_space={})
        for mt in _CANDIDATE_MODELS
    ]
    return ForecastConfig(market_config=market_config, categories=categories, models=models, features=[])


def _classify_categories(sales_df: pd.DataFrame) -> dict[str, str]:
    """Market-wide {sku_id: category} using the SAME rule data/features/
    category_classification.py applies in production (ported to pandas here since
    that module's compute_sku_category is Spark SQL) — M5 ships no category column;
    category is a volume-derived classification either way, real or synthetic data."""
    by_sku = sales_df.groupby("sku_id")["qty"].sum().rename("total_qty").reset_index()
    by_sku = by_sku.sort_values(["total_qty", "sku_id"], ascending=[False, True])
    by_sku["cum_share"] = by_sku["total_qty"].cumsum() / by_sku["total_qty"].sum()

    n_days_present = sales_df[sales_df["qty"] > 0].groupby("sku_id")["date_id"].nunique()

    categories = {}
    for _, row in by_sku.iterrows():
        sku_id = row["sku_id"]
        volume_band = classify_volume_band(row["total_qty"], row["cum_share"])
        is_new_item = n_days_present.get(sku_id, 0) < _NEW_ITEM_MAX_DAYS
        categories[sku_id] = resolve_category(is_new_item, volume_band)
    return categories


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

    sales_df = frames["sales"].copy()
    sales_df["date_id"] = pd.to_datetime(sales_df["date_id"])
    sales_df["qty"] = sales_df["units_sold"] + sales_df["units_promotional"] + sales_df["units_shrinkage"]

    sku_cat = _classify_categories(sales_df)

    merged = sales_df.merge(cal_flags, on=["market_id", "date_id"], how="left")
    merged["category"] = merged["sku_id"].map(sku_cat).fillna("D")
    merged["series_key"] = (
        merged["market_id"] + "|" + merged["category"] + "|" + merged["store_id"] + "|" + merged["sku_id"]
    )
    # M5's sku_store active window (data/kaggle_m5/loader.py, from sell_prices.csv listing
    # weeks) tells us the true active range; treat every row inside it as active — this
    # demo skips is_new_store/is_delisted nuance the real pipeline's lifecycle.py handles.
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


def _holdout_forecast(model_type: str, train: pd.DataFrame, horizon: pd.DataFrame):
    """Refit `model_type` on `train` (the same history champion selection saw) and return
    (yhat, decomposition) for the held-out `horizon`."""
    model = build_model(model_type)
    model.fit(train, horizon, params={})
    decomp = model.decompose(horizon)
    return np.asarray(decomp.yhat, dtype=float), decomp


def _plot_series(series_key: str, series_pdf: pd.DataFrame, model_type: str, decomp, wmape_value: float,
                 out_path: Path) -> None:
    """Actual-vs-forecast + additive decomposition stack for one series' held-out week."""
    horizon = series_pdf.iloc[-_HOLDOUT_DAYS:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[1.3, 1])

    tail = series_pdf.iloc[-(_HOLDOUT_DAYS + 45):]
    ax1.plot(tail["date_id"], tail["qty"], label="actual", color="#333333", linewidth=1.5)
    ax1.plot(horizon["date_id"], decomp.yhat, label="forecast", color="#1f77b4", linewidth=2, linestyle="--")
    ax1.axvline(horizon["date_id"].iloc[0], color="#999999", linestyle=":", linewidth=1)
    ax1.set_title(f"{series_key} — {model_type}, held-out week WMAPE {wmape_value:.0%}")
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
    ax2.set_title("additive decomposition (held-out week)")
    ax2.legend(loc="upper left", fontsize=7, ncol=2)
    ax2.set_ylabel("contribution")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _poisson_mean_abs_dev(lam: float) -> float:
    """E|X - lam| for X ~ Poisson(lam): the error even a perfect forecast of the mean makes."""
    if lam <= 0:
        return 0.0
    k = np.arange(0, int(lam * 4 + 50))
    log_pmf = k * math.log(lam) - lam - np.array([math.lgamma(x + 1) for x in k])
    return float((np.abs(k - lam) * np.exp(log_pmf)).sum())


def _noise_floor(series_frames: dict, keys: list, config: ForecastConfig) -> dict[str, float]:
    """Per-category irreducible weekly WMAPE if weekly demand were Poisson around each
    series' mean weekly volume on the backtest folds — a lower bound no model can beat,
    used to put the gate's numbers in context (not to gate anything)."""
    mc = config.market_config
    err: dict[str, float] = {}
    vol: dict[str, float] = {}
    for key in keys:
        pdf = series_frames[key].assign(date_id=lambda d: pd.to_datetime(d["date_id"]))
        weekly = [h["qty"].sum() for _, h in rolling_origin_folds(pdf, mc.backtest_folds, mc.val_weeks, mc.eval_horizon_days)]
        if not weekly:
            continue
        cat, lam = pdf["category"].iloc[0], float(np.mean(weekly))
        err[cat] = err.get(cat, 0.0) + _poisson_mean_abs_dev(lam) * len(weekly)
        vol[cat] = vol.get(cat, 0.0) + float(np.sum(weekly))
    floors = {c: err[c] / vol[c] for c in err if vol[c] > 0}
    if vol:
        floors["ALL"] = sum(err.values()) / sum(vol.values())
    return floors


def _pooled_wmape(actual: pd.Series, forecast: pd.Series) -> float | None:
    """sum|A - F| / sum A over weekly totals — training/evaluation.py::wmape pooled across series."""
    total = float(actual.sum())
    return float((actual - forecast).abs().sum() / total) if total > 0 else None


def _skill(model_wmape: float | None, baseline_wmape: float | None) -> float | None:
    if model_wmape is None or not baseline_wmape:
        return None
    return 1.0 - model_wmape / baseline_wmape


def run_demo() -> dict:
    t0 = time.time()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_m5_source(
        _M5_DATA_DIR, states=_STATES, max_skus=_MAX_SKUS, trailing_days=_TRAILING_DAYS,
    )
    markets = sorted(frames["store"]["market_id"].unique().tolist())
    series_frames = _build_series_frames(frames)
    config = _build_config(markets)

    all_keys = list(series_frames.keys())
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(all_keys), size=min(_MAX_SERIES, len(all_keys)), replace=False)
    sample_keys = [all_keys[i] for i in sample_idx]

    fit_t0 = time.time()
    result_rows, holdout_rows, decomps = [], [], {}
    for store_id, sku_id in sample_keys:
        series_pdf = series_frames[(store_id, sku_id)]
        if len(series_pdf) < config.market_config.eval_horizon_days * 4 + _HOLDOUT_DAYS:
            continue
        train, held = series_pdf.iloc[:-_HOLDOUT_DAYS], series_pdf.iloc[-_HOLDOUT_DAYS:]
        snapshot_date = str(pd.Timestamp(train["date_id"].iloc[-1]).date())
        row = fit_one_series(train, config, snapshot_date).iloc[0].to_dict()
        if row["status"] not in ("ok", "stable"):
            continue
        result_rows.append(row)

        # Out-of-sample: champion vs. the 28-day-mean baseline, both forecasting the same
        # unseen week from the same history.
        champ_yhat, champ_decomp = _holdout_forecast(row["model_type"], train, held)
        decomps[row["series_key"]] = champ_decomp
        holdout_rows.append({
            "series_key": row["series_key"], "category": row["category"],
            "store_id": store_id, "sku_id": sku_id, "model_type": row["model_type"],
            "actual": float(held["qty"].sum()),
            "champion": float(champ_yhat.sum()),
            "baseline": float(train["qty"].tail(_BASELINE_DAYS).mean() * _HOLDOUT_DAYS),
        })
    fit_elapsed = time.time() - fit_t0

    # ── Backtest gate (what the production pipeline gates and tunes on) ──────────────
    champions_pdf = pd.DataFrame(result_rows)
    gated = gate(champions_pdf, config.categories)
    gate_summary = (
        gated[gated["grain"] == "category_cohort"][
            ["market_id", "entity_key", "wmape", "baseline_wmape", "skill", "min_skill", "passed"]
        ]
        .rename(columns={"entity_key": "category"})
        .sort_values(["market_id", "category"])
        .to_dict("records")
    )
    grain_pass_rate = {
        grain: float(g["passed"].dropna().astype(bool).mean()) if g["passed"].notna().any() else None
        for grain, g in gated.groupby("grain")
    }

    # ── Held-out week (the README's headline accuracy) ───────────────────────────────
    hold = pd.DataFrame(holdout_rows)
    noise_floor = _noise_floor(series_frames, [(r["store_id"], r["sku_id"]) for r in holdout_rows], config)

    def _summary(g: pd.DataFrame, floor: float | None) -> dict:
        champ, base = (_pooled_wmape(g["actual"], g[c]) for c in ("champion", "baseline"))
        return {
            "n_series": int(len(g)),
            "median_weekly_units": float(g["actual"].median()),
            "champion_wmape": champ, "baseline_wmape": base,
            "champion_skill": _skill(champ, base),
            "noise_floor_wmape": floor,
        }

    holdout_by_category = {
        cat: _summary(g, noise_floor.get(cat)) for cat, g in hold.groupby("category")
    }
    holdout_overall = _summary(hold, noise_floor.get("ALL"))

    # Example charts: TYPICAL series, not the best ones — the 3 series closest to the median
    # held-out champion WMAPE (one per distinct champion model and category), among series
    # selling at least _CHART_MIN_WEEKLY_UNITS units that week (a near-zero series has
    # nothing to see).
    hold["champion_wmape"] = (hold["actual"] - hold["champion"]).abs() / hold["actual"].where(hold["actual"] > 0)
    chartable = hold[hold["actual"] >= _CHART_MIN_WEEKLY_UNITS].dropna(subset=["champion_wmape"])
    median_wmape = float(chartable["champion_wmape"].median()) if len(chartable) else float("nan")
    ranked = chartable.assign(_dist=(chartable["champion_wmape"] - median_wmape).abs()).sort_values("_dist")
    picks, seen = [], set()
    for _, row in ranked.iterrows():
        tag = (row["model_type"], row["category"])
        if tag in seen:
            continue
        seen.add(tag)
        picks.append(row)
        if len(picks) == 3:
            break

    chart_files = []
    for i, pick in enumerate(picks):
        out_path = _OUT_DIR / f"forecast_{i}_{pick['sku_id']}_{pick['store_id']}.png"
        _plot_series(pick["series_key"], series_frames[(pick["store_id"], pick["sku_id"])], pick["model_type"],
                     decomps[pick["series_key"]], float(pick["champion_wmape"]), out_path)
        chart_files.append({
            "file": out_path.name,
            "series_key": pick["series_key"],
            "model_type": pick["model_type"],
            "holdout_wmape": float(pick["champion_wmape"]),
            "holdout_weekly_units": float(pick["actual"]),
        })

    total_elapsed = time.time() - t0
    metrics = {
        "dataset": "Kaggle M5 Forecasting Accuracy (Walmart)",
        "states": _STATES,
        "max_skus": _MAX_SKUS,
        "trailing_days": _TRAILING_DAYS,
        "n_series_sampled": len(sample_keys),
        "n_series_evaluated": len(holdout_rows),
        "candidate_models": _CANDIDATE_MODELS,
        "metric": "7-day weekly-total WMAPE, pooled: sum|A-F| / sum A over series",
        "holdout_overall": holdout_overall,
        "holdout_by_category": holdout_by_category,
        "backtest_gate_by_category": gate_summary,
        "backtest_gate_pass_rate_by_grain": grain_pass_rate,
        "champion_model_counts": champions_pdf["model_type"].value_counts().to_dict(),
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
