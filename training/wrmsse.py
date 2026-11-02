"""
training/wrmsse.py — M5 competition's WRMSSE, implemented to the official spec.

WHY THIS EXISTS
  The engine's own champion metric is 7-day volume-weighted WMAPE at three grains
  (docs/instructions.md §3, training/evaluation.py). That is the right metric for the
  planning use case, but it is NOT the metric the M5 competition scored, so a WMAPE
  number cannot be placed next to an M5 leaderboard position. Anyone who recognises
  this dataset will make that comparison regardless, so this module computes the
  competition's actual metric on the competition's actual protocol.

THE METRIC (Makridakis et al., "The M5 Accuracy competition: results, findings and
conclusions"; https://github.com/Mcompetitions/M5-methods)

  Per series i, over a forecast horizon of h days following a training sample of n days:

                      sqrt( (1/h) * SUM_{t=n+1..n+h} (Y_t - Yhat_t)^2 )
      RMSSE_i  =  ---------------------------------------------------------
                      sqrt( (1/(n-1)) * SUM_{t=2..n} (Y_t - Y_{t-1})^2 )

  The denominator is the in-sample one-step NAIVE (lag-1, NOT lag-7) squared error.
  This is what makes the score scale-free and roughly interpretable against 1.0:
  a model exactly as good as the naive random walk scores ~1.0.

  Two spec details that are easy to get wrong and both materially change the score:
    1. The denominator is computed only from the series' ACTIVE window — starting at
       its first non-zero observation. M5 items are not stocked for the whole history,
       and counting leading structural zeros deflates the naive error, which inflates
       every RMSSE built on it.
    2. The horizon mean uses h (the full horizon), while the training mean uses n-1.

  The competition aggregates across 12 hierarchy levels (total, state, store, category,
  department, the state/store x category/department crosses, item, item-state,
  item-store). Each level contributes EQUALLY — one twelfth — regardless of how many
  series it contains:

      WRMSSE  =  SUM_{levels l} (1/12) * SUM_{series i in l} w_{l,i} * RMSSE_{l,i}

  Within a level the weights w_{l,i} are that series' share of total REVENUE
  (units sold x sell_price) over the LAST 28 DAYS of the training window, and sum to 1.
  Revenue-weighted, not unit-weighted — which is why sell_price has to be persisted
  from the raw M5 files for this to be reproducible (data/kaggle_m5/loader.py).

SCOPE HONESTY
  Run over a subset of the catalogue (a few stores, a sample of SKUs), the levels that
  collapse to a single group — 'total' always, 'state' when only one state is ingested —
  carry no information but still take their 1/12 share. `compute_wrmsse` therefore
  reports the per-level breakdown alongside the headline, and `LEVELS` marks which
  levels are degenerate for a given frame, so a subset result can be quoted as what it
  is rather than as a leaderboard-equivalent number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The 12 official M5 aggregation levels, as the grouping keys each one rolls up to.
# An empty tuple is the grand total (level 1). Order matches the competition's Table 1.
LEVELS: list[tuple[str, tuple[str, ...]]] = [
    ("total", ()),
    ("state", ("state_id",)),
    ("store", ("store_id",)),
    ("category", ("cat_id",)),
    ("department", ("dept_id",)),
    ("state_category", ("state_id", "cat_id")),
    ("state_department", ("state_id", "dept_id")),
    ("store_category", ("store_id", "cat_id")),
    ("store_department", ("store_id", "dept_id")),
    ("item", ("item_id",)),
    ("item_state", ("item_id", "state_id")),
    ("item_store", ("item_id", "store_id")),
]

_WEIGHT_WINDOW_DAYS = 28  # M5: weights come from the last 28 days of the training window


def naive_scale(train_qty: np.ndarray) -> float:
    """Denominator of RMSSE: RMS of the in-sample lag-1 naive error, over the ACTIVE window.

    Active window = from the first non-zero observation onward (see module docstring,
    spec detail 1). Returns NaN when the series never sells or has < 2 active points,
    which leaves RMSSE undefined for it — such a series carries ~0 revenue weight
    anyway, so callers drop it rather than substituting a value.
    """
    y = np.asarray(train_qty, dtype=float)
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return float("nan")
    active = y[nz[0]:]
    if active.size < 2:
        return float("nan")
    diffs = np.diff(active)
    return float(np.sqrt(np.mean(diffs ** 2)))


def rmsse(actual: np.ndarray, predicted: np.ndarray, scale: float) -> float:
    """RMSSE for one series: RMS forecast error divided by `scale` (naive_scale's output)."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(predicted, dtype=float)
    if a.size == 0 or not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - f) ** 2)) / scale)


def _aggregate(df: pd.DataFrame, keys: tuple[str, ...], value_cols: list[str]) -> pd.DataFrame:
    """Sum `value_cols` up to `keys` (empty keys = grand total, one row)."""
    if not keys:
        out = df[value_cols].sum().to_frame().T
        out["_grp"] = "total"
        return out.set_index("_grp")
    return df.groupby(list(keys), observed=True)[value_cols].sum()


def compute_wrmsse(
    train_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    price_df: pd.DataFrame,
    levels: list[tuple[str, tuple[str, ...]]] | None = None,
) -> dict:
    """Compute M5 WRMSSE plus its per-level breakdown.

    Args:
        train_df:    long training history, one row per (item_id, store_id, date_id) with
                     `qty`, plus the hierarchy columns (state_id, cat_id, dept_id).
        actual_df:   realised demand over the horizon, same shape as train_df.
        forecast_df: predictions over the same horizon, with `yhat` instead of `qty`.
        price_df:    (item_id, store_id, date_id, sell_price) covering at least the last
                     `_WEIGHT_WINDOW_DAYS` of train_df — M5 weights are revenue-based.
        levels:      defaults to all 12 official levels.

    Returns:
        {"wrmsse": float, "levels": [{level, rmsse_weighted, n_series, degenerate}, ...]}
        where `degenerate` flags a level that collapsed to a single group for this
        subset (e.g. 'state' when only one state was ingested) and therefore measures
        nothing, while still taking its 1/12 share of the headline.
    """
    levels = levels or LEVELS

    # ── Revenue weights: last 28 training days, units x price ────────────────────
    cutoff = train_df["date_id"].max() - pd.Timedelta(days=_WEIGHT_WINDOW_DAYS - 1)
    window = train_df[train_df["date_id"] >= cutoff].merge(
        price_df, on=["item_id", "store_id", "date_id"], how="left"
    )
    window["revenue"] = window["qty"].astype(float) * window["sell_price"].astype(float).fillna(0.0)

    per_level = []
    total = 0.0
    for name, keys in levels:
        # Roll actuals/forecasts/history up to this level, then score each group.
        a = _aggregate(actual_df, keys, ["qty"]).rename(columns={"qty": "actual"})
        f = _aggregate(forecast_df, keys, ["yhat"])
        joined = a.join(f, how="left").fillna({"yhat": 0.0})

        # Per-group training series, in date order, for the naive scale.
        tkeys = list(keys) + ["date_id"]
        t = train_df.groupby(tkeys, observed=True)["qty"].sum().reset_index() if keys else \
            train_df.groupby("date_id")["qty"].sum().reset_index()
        t = t.sort_values("date_id")

        # Horizon series per group, in date order, so error is computed over aligned days.
        akeys = list(keys) + ["date_id"]
        ah = actual_df.groupby(akeys, observed=True)["qty"].sum().reset_index() if keys else \
            actual_df.groupby("date_id")["qty"].sum().reset_index()
        fh = forecast_df.groupby(akeys, observed=True)["yhat"].sum().reset_index() if keys else \
            forecast_df.groupby("date_id")["yhat"].sum().reset_index()

        w = _aggregate(window, keys, ["revenue"])
        w_total = float(w["revenue"].sum())

        rows = []
        groups = [()] if not keys else list(joined.index)
        for g in groups:
            if keys:
                mask_t = np.ones(len(t), dtype=bool)
                mask_a = np.ones(len(ah), dtype=bool)
                gvals = g if isinstance(g, tuple) else (g,)
                for k, v in zip(keys, gvals):
                    mask_t &= (t[k] == v).to_numpy()
                    mask_a &= (ah[k] == v).to_numpy()
                tr = t.loc[mask_t].sort_values("date_id")["qty"].to_numpy()
                av = ah.loc[mask_a].sort_values("date_id")["qty"].to_numpy()
                fv = fh.loc[mask_a].sort_values("date_id")["yhat"].to_numpy()
                weight = float(w.loc[g, "revenue"]) if g in w.index else 0.0
            else:
                tr = t.sort_values("date_id")["qty"].to_numpy()
                av = ah.sort_values("date_id")["qty"].to_numpy()
                fv = fh.sort_values("date_id")["yhat"].to_numpy()
                weight = w_total

            s = naive_scale(tr)
            r = rmsse(av, fv, s)
            if np.isfinite(r):
                rows.append((r, weight))

        if rows and w_total > 0:
            level_score = sum(r * (wt / w_total) for r, wt in rows)
        else:
            level_score = float("nan")

        per_level.append({
            "level": name,
            "rmsse_weighted": level_score,
            "n_series": len(rows),
            # A level with one group measures nothing on this subset but still takes 1/12.
            "degenerate": len(rows) <= 1,
        })
        if np.isfinite(level_score):
            total += level_score

    return {"wrmsse": total / len(levels), "levels": per_level}
