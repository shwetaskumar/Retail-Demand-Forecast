"""
demo/compare_selection_rules.py — out-of-sample comparison of champion-selection rules
on Kaggle M5 (no Spark). Companion to demo/run_local_demo.py; this is the experiment
behind config/seed_defaults.py's default candidate set.

For a random sample of store×sku series it forecasts _N_WEEKS consecutive weeks with
every single-model wrapper, the equal-weight ensemble, and the trailing-28-day-mean
baseline. Then, for every target week after the first _LOOKBACK, each rule picks a
forecast per series using ONLY the _LOOKBACK preceding weeks' errors (exactly how a
backtest-driven champion would be chosen) and is scored on the target week it never saw.
Reports pooled 7-day WMAPE, skill vs the baseline, and a series-bootstrap 90% interval.

Takes ~25 min on a laptop for the default 150 series x 8 weeks:

    python demo/compare_selection_rules.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = str(_Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
del _sys, _Path, _REPO_ROOT

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from demo.run_local_demo import _M5_DATA_DIR, _MAX_SKUS, _STATES, _TRAILING_DAYS, _build_series_frames
from data.kaggle_m5.loader import load_m5_source
from training.backtest import BASELINE_WINDOW_DAYS, rolling_origin_folds
from training.models.ensemble import _DEFAULT_MEMBER_TYPES
from training.models.registry import build as build_model

_N_SERIES = 150
_N_WEEKS = 8
_LOOKBACK = 3          # backtest weeks a rule may look at (run_local_demo.py uses 3 folds)
_SEED = 7              # different draw from run_local_demo.py's sample (seed 42)
_SINGLES = ["lgbm", "auto_arima", "statsforecast", "croston", "ets_event"]
_OVERRIDE_MARGINS = (0.10, 0.20, 0.30)
_BOOTSTRAP = 500
_OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "selection_rules.json"

logger = logging.getLogger(__name__)


def _weekly_forecasts(series: dict, keys: list) -> pd.DataFrame:
    """One row per (series, week): weekly actual + weekly forecast of every candidate."""
    rows = []
    for n, key in enumerate(keys, 1):
        pdf = series[key].assign(date_id=lambda d: pd.to_datetime(d["date_id"]))
        for week, (train, hold) in enumerate(rolling_origin_folds(pdf, _N_WEEKS, _N_WEEKS, 7)):
            row = {
                "key": key, "category": pdf["category"].iloc[0], "week": week,
                "actual": float(hold["qty"].sum()),
                "baseline": float(train["qty"].tail(BASELINE_WINDOW_DAYS).mean() * len(hold)),
            }
            for model_type in _SINGLES:
                try:
                    model = build_model(model_type).fit(train, hold, {})
                    row[model_type] = float(np.sum(model.predict(hold)))
                except Exception:  # noqa: BLE001 — a failed fit is a missing forecast, not a crash
                    row[model_type] = np.nan
            # The ensemble is the equal-weight mean of its members' forecasts (training/models/
            # ensemble.py), so it is computed from the member fits above instead of refitting.
            row["ensemble"] = float(np.mean([row[m] for m in _DEFAULT_MEMBER_TYPES]))
            rows.append(row)
        if n % 25 == 0:
            logger.info("forecast %d/%d series", n, len(keys))
    return pd.DataFrame(rows)


def _rules() -> dict:
    def best(cands, bt):
        return min(cands, key=lambda c: bt[c] if bt[c] == bt[c] else np.inf)

    rules = {
        "baseline (28-day mean)": lambda bt: "baseline",
        "always ensemble": lambda bt: "ensemble",
        "always lgbm": lambda bt: "lgbm",
        "best candidate per series": lambda bt: best(_SINGLES + ["ensemble"], bt),
    }
    for m in _OVERRIDE_MARGINS:
        rules[f"ensemble unless a single beats it by {m:.0%}"] = (
            lambda bt, m=m: (lambda s: s if bt[s] <= bt["ensemble"] * (1 - m) else "ensemble")(best(_SINGLES, bt))
        )
    return rules


def _apply_rules(weekly: pd.DataFrame, rules: dict) -> pd.DataFrame:
    cands = _SINGLES + ["ensemble", "baseline"]
    out = []
    for _, g in weekly.groupby("key"):
        g = g.sort_values("week").reset_index(drop=True)
        ape = {c: (g["actual"] - g[c]).abs() / g["actual"].where(g["actual"] > 0) for c in cands}
        for t in range(_LOOKBACK, len(g)):
            bt = {c: float(np.nanmean(ape[c].iloc[t - _LOOKBACK:t])) if ape[c].iloc[t - _LOOKBACK:t].notna().any()
                  else np.nan for c in cands}
            target = g.iloc[t]
            out.append({"key": target["key"], "category": target["category"], "actual": target["actual"],
                        **{name: target[rule(bt)] for name, rule in rules.items()}})
    return pd.DataFrame(out)


def _pooled(g: pd.DataFrame, col: str) -> float:
    return float((g["actual"] - g[col]).abs().sum() / g["actual"].sum())


def run() -> dict:
    warnings.filterwarnings("ignore")
    frames = load_m5_source(_M5_DATA_DIR, states=_STATES, max_skus=_MAX_SKUS, trailing_days=_TRAILING_DAYS)
    series = _build_series_frames(frames)
    all_keys = list(series)
    rng = np.random.default_rng(_SEED)
    keys = [all_keys[i] for i in rng.choice(len(all_keys), size=min(_N_SERIES, len(all_keys)), replace=False)]

    rules = _rules()
    scored = _apply_rules(_weekly_forecasts(series, keys), rules)

    by_key = {k: g for k, g in scored.groupby("key")}
    uniq = list(by_key)
    boot = [np.random.default_rng(i).choice(len(uniq), len(uniq)) for i in range(_BOOTSTRAP)]
    results = {}
    for name in rules:
        wmape, base = _pooled(scored, name), _pooled(scored, "baseline (28-day mean)")
        skills = []
        for idx in boot:
            g = pd.concat([by_key[uniq[i]] for i in idx])
            skills.append(1 - _pooled(g, name) / _pooled(g, "baseline (28-day mean)"))
        lo, hi = np.percentile(skills, [5, 95])
        results[name] = {
            "wmape": wmape, "skill_vs_baseline": 1 - wmape / base, "skill_ci90": [float(lo), float(hi)],
            "wmape_by_category": {c: _pooled(g, name) for c, g in scored.groupby("category")},
        }
    out = {"n_series": len(uniq), "n_series_weeks": len(scored), "lookback_weeks": _LOOKBACK, "rules": results}
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run()
    logger.info("%d series-weeks over %d series", res["n_series_weeks"], res["n_series"])
    for rule_name, r in res["rules"].items():
        logger.info("%-44s WMAPE %.1f%%  skill %+.1f%%  90%% CI [%+.1f%%, %+.1f%%]", rule_name, 100 * r["wmape"],
                    100 * r["skill_vs_baseline"], 100 * r["skill_ci90"][0], 100 * r["skill_ci90"][1])
