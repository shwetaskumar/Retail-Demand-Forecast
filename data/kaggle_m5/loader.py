"""
data/kaggle_m5/loader.py — loads Kaggle's M5 Forecasting Accuracy dataset
(Walmart daily unit sales, ~30,490 store×item series across CA/TX/WI) and
transforms it into this project's source-table contracts.

M5 is a Kaggle *competition* (not a plain dataset) — you must have a Kaggle
account, accept the competition rules on kaggle.com, and use your own API
token. Download once:

    pip install kaggle
    # place your kaggle.json (from kaggle.com/settings) at ~/.kaggle/kaggle.json
    kaggle competitions download -c m5-forecasting-accuracy -p data/m5_raw
    unzip data/m5_raw/m5-forecasting-accuracy.zip -d data/m5_raw

`data/m5_raw/` is gitignored — M5's competition data license does not permit
redistribution, so the raw files never get committed to this repo.

Source files (all in the same directory):
  calendar.csv              — date, wm_yr_wk, weekday, wday, month, year, d,
                               event_name_1/2, event_type_1/2, snap_CA/TX/WI
  sales_train_evaluation.csv — id, item_id, dept_id, cat_id, store_id, state_id,
                               d_1..d_1941 (wide, one column per day)
  sell_prices.csv           — store_id, item_id, wm_yr_wk, sell_price
                               (a price row exists only for weeks the item was
                               actually listed at that store — the standard M5
                               proxy for "was this item on sale here", used
                               below to derive active_from/active_to)

Output frame shapes match the same source-table contracts
data/synthetic/generator.py used to (see docs/feature_engineering_spec.md):
  sales:         market_id, store_id, sku_id, date_id, units_sold, units_promotional, units_shrinkage
  sku:           sku_id, category
  store:         store_id, market_id
  sku_store:     market_id, store_id, sku_id, active_from, active_to
  events:        market_id, event_id, event_type, event_subtype, event_nm, start_date, end_date, sku_id, store_id, exclude_flag, tactic
  calendar_date: date_id, weekday, weekday_name, week, day, month, quarter

M5 has no promo/shrinkage split (just one daily unit count) and no per-sku
category (category_classification.py derives that from sales volume, same as
for any other source — see its module docstring) — units_promotional and
units_shrinkage are always 0 here; all of a day's units land in units_sold.
state_id (CA/TX/WI) becomes market_id — a natural 3-market split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# M5's own event_type_1/2 values (Sporting/Cultural/National/Religious) all map to
# our "HOLIDAY" event_type (event_features.py's _events() filters on event_type
# literally equal to 'HOLIDAY' or 'PROMOTION') — the M5 type becomes event_subtype
# for informational granularity.
_SNAP_MARKET_COLUMNS = {"snap_CA": "CA", "snap_TX": "TX", "snap_WI": "WI"}


def _read_calendar(data_dir: Path) -> pd.DataFrame:
    cal = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    return cal


def _build_calendar_date(cal: pd.DataFrame) -> pd.DataFrame:
    """gold.calendar_date — recomputed fresh from calendar.csv's `date` column
    (not M5's own wday/weekday encoding) so it matches our exact contract."""
    dates = pd.DatetimeIndex(cal["date"])
    return pd.DataFrame({
        "date_id": dates.date,
        "weekday": (dates.dayofweek + 1) % 7 + 1,  # 1=Sun..7=Sat, matches temporal_features.py
        "weekday_name": dates.day_name(),
        "week": dates.isocalendar().week.to_numpy(),
        "day": dates.day,
        "month": dates.month,
        "quarter": dates.quarter,
    })


def _build_events(cal: pd.DataFrame, markets: list[str]) -> pd.DataFrame:
    """gold.events — named holidays (broadcast, market_id=None-equivalent i.e. every
    requested market) from event_name_1/2 + event_type_1/2, plus SNAP benefit days
    per state as PROMOTION-type events (SNAP reliably lifts demand in that state —
    the closest fit to our promo/holiday dichotomy, flagged via tactic='SNAP')."""
    rows = []
    event_id = 0

    for suffix in ("1", "2"):
        name_col, type_col = f"event_name_{suffix}", f"event_type_{suffix}"
        named = cal[cal[name_col].notna()]
        for _, r in named.iterrows():
            d = r["date"].date()
            for market_id in markets:
                event_id += 1
                rows.append({
                    "market_id": market_id,
                    "event_id": f"M5HOL{event_id:06d}",
                    "event_type": "HOLIDAY",
                    "event_subtype": r[type_col],  # Sporting | Cultural | National | Religious
                    "event_nm": r[name_col],
                    "start_date": d,
                    "end_date": d,
                    "sku_id": None,
                    "store_id": None,
                    "exclude_flag": 0,
                    "tactic": "HOLIDAY",
                })

    for snap_col, state in _SNAP_MARKET_COLUMNS.items():
        if state not in markets or snap_col not in cal.columns:
            continue
        snap_days = cal[cal[snap_col] == 1]
        for _, r in snap_days.iterrows():
            d = r["date"].date()
            event_id += 1
            rows.append({
                "market_id": state,
                "event_id": f"M5SNAP{event_id:06d}",
                "event_type": "PROMOTION",
                "event_subtype": "SNAP",
                "event_nm": "SNAP benefit day",
                "start_date": d,
                "end_date": d,
                "sku_id": None,
                "store_id": None,
                "exclude_flag": 0,
                "tactic": "SNAP",
            })

    return pd.DataFrame(rows)


def _build_active_window(prices: pd.DataFrame, cal: pd.DataFrame, store_market: dict[str, str]) -> pd.DataFrame:
    """source.sku_store — active_from/active_to derived from sell_prices.csv: a price
    row exists only for weeks the item was actually listed at that store, which is
    the standard M5 proxy for an item's listing window (no explicit listing-window
    table exists in the raw data)."""
    week_bounds = cal.groupby("wm_yr_wk")["date"].agg(["min", "max"])
    span = (
        prices.groupby(["store_id", "item_id"])["wm_yr_wk"]
        .agg(["min", "max"])
        .rename(columns={"min": "wk_start", "max": "wk_end"})
        .reset_index()
    )
    span["active_from"] = span["wk_start"].map(week_bounds["min"]).dt.date
    span["active_to"] = span["wk_end"].map(week_bounds["max"]).dt.date
    span["market_id"] = span["store_id"].map(store_market)
    return span.rename(columns={"item_id": "sku_id"})[
        ["market_id", "store_id", "sku_id", "active_from", "active_to"]
    ]


def load_m5_source(
    data_dir: str | Path,
    states: list[str] | None = None,
    max_skus: int | None = 300,
    trailing_days: int | None = 730,
    seed: int = 42,
    future_horizon_days: int = 140,
) -> dict[str, pd.DataFrame]:
    """Load and transform the M5 dataset into this project's source-table contracts.

    Args:
        data_dir: directory containing calendar.csv, sales_train_evaluation.csv,
            sell_prices.csv (see module docstring for how to download these).
        states: state_id filter (subset of "CA"/"TX"/"WI"); default all three
            (10 stores, ~30,490 series). Filtering to one state keeps the local
            demo fast — melting the wide sales file is the expensive step, and
            it happens AFTER this filter.
        max_skus: randomly sample at most this many item_ids (None = all ~3,049).
            Applied before melting for the same reason.
        trailing_days: only load the most recent N days of history (None = all
            1,941 days from 2011-01-29 to 2016-05-22). Column-sliced before
            melting, not filtered after — this is what actually bounds runtime.
        seed: RNG seed for the sku sample (reproducible).
        future_horizon_days: gold.calendar_date is extended with synthetic rows
            past the last sales day, out to this many days ahead (default 140,
            matching architecture lock #9's forecast horizon). calendar.csv's
            own coverage only extends ~28 days past the last sales day (M5's
            "evaluation" placeholder period) — nowhere near enough for JOB 3's
            future spine, which needs a calendar_date row for every day of the
            horizon (confirmed empirically: without this, inference/future_spine.py
            joins against an empty calendar range and produces zero scored rows).
            Calendar fields (weekday/week/month/quarter) are a pure function of
            the date itself, so synthesizing them needs no real M5 data.

    Returns:
        dict with keys "sales", "sku", "store", "sku_store", "events",
        "calendar_date" — same shape data/synthetic/generator.py used to return,
        so it's a drop-in replacement anywhere that was called.
    """
    data_dir = Path(data_dir)
    for fname in ("calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"):
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"{data_dir / fname} not found — see data/kaggle_m5/loader.py's module "
                "docstring for how to download the M5 dataset via the Kaggle CLI."
            )

    cal = _read_calendar(data_dir)
    sales_wide = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    prices = pd.read_csv(data_dir / "sell_prices.csv")

    all_states = sorted(sales_wide["state_id"].unique())
    states = states or all_states
    sales_wide = sales_wide[sales_wide["state_id"].isin(states)]

    if max_skus is not None:
        rng = np.random.default_rng(seed)
        skus = sales_wide["item_id"].unique()
        if len(skus) > max_skus:
            sampled = rng.choice(skus, size=max_skus, replace=False)
            sales_wide = sales_wide[sales_wide["item_id"].isin(sampled)]

    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    if trailing_days is not None and trailing_days < len(day_cols):
        day_cols = day_cols[-trailing_days:]
    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_wide = sales_wide[id_cols + day_cols]

    prices = prices[prices["store_id"].isin(sales_wide["store_id"].unique())]
    prices = prices[prices["item_id"].isin(sales_wide["item_id"].unique())]
    cal_in_range = cal[cal["d"].isin(day_cols)]

    # ── store / sku masters ──────────────────────────────────────────────
    store_df = (
        sales_wide[["store_id", "state_id"]].drop_duplicates()
        .rename(columns={"state_id": "market_id"})
        .reset_index(drop=True)
    )
    store_market = dict(zip(store_df["store_id"], store_df["market_id"]))
    sku_df = sales_wide[["item_id"]].drop_duplicates().rename(columns={"item_id": "sku_id"}).reset_index(drop=True)

    # ── sales (melt d_* wide -> long, join calendar for real dates) ───────
    d_to_date = dict(zip(cal_in_range["d"], cal_in_range["date"].dt.date))
    melted = sales_wide.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="units_sold")
    melted["date_id"] = melted["d"].map(d_to_date)
    melted = melted.dropna(subset=["date_id"])
    sales_df = melted.rename(columns={"state_id": "market_id", "item_id": "sku_id"})[
        ["market_id", "store_id", "sku_id", "date_id", "units_sold"]
    ]
    sales_df["units_sold"] = sales_df["units_sold"].astype(float)
    sales_df["units_promotional"] = 0.0
    sales_df["units_shrinkage"] = 0.0

    # ── sku_store active window (from sell_prices.csv listing weeks) ──────
    sku_store_df = _build_active_window(prices, cal, store_market)

    # ── events + calendar_date ─────────────────────────────────────────────
    events_df = _build_events(cal_in_range, states)

    # Extend calendar_date past cal_in_range's own max date (see
    # future_horizon_days docstring above) so JOB 3's future spine has a
    # calendar_date row for EVERY day of the forecast horizon, with no gap.
    # Two sources, concatenated in order: (a) any real calendar.csv rows past
    # cal_in_range's own max (M5's evaluation-period placeholder days, which
    # don't reach the full horizon on their own) and (b) synthetic rows for
    # whatever's still short of _needed_end — calendar fields are a pure
    # function of the date, so synthesizing them needs no real M5 data.
    _needed_end = cal_in_range["date"].max() + pd.Timedelta(days=future_horizon_days)
    _real_extra = cal[(cal["date"] > cal_in_range["date"].max()) & (cal["date"] <= _needed_end)][["date"]]
    _covered_through = max(cal_in_range["date"].max(), _real_extra["date"].max()) if len(_real_extra) else cal_in_range["date"].max()
    _cal_for_calendar_date = pd.concat([cal_in_range[["date"]], _real_extra], ignore_index=True)
    if _covered_through < _needed_end:
        _synthetic = pd.date_range(start=_covered_through + pd.Timedelta(days=1), end=_needed_end, freq="D")
        _cal_for_calendar_date = pd.concat(
            [_cal_for_calendar_date, pd.DataFrame({"date": _synthetic})], ignore_index=True
        )
    calendar_date_df = _build_calendar_date(_cal_for_calendar_date)

    return {
        "sales": sales_df.reset_index(drop=True),
        "sku": sku_df,
        "store": store_df,
        "sku_store": sku_store_df,
        "events": events_df,
        "calendar_date": calendar_date_df,
    }


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/m5_raw"
    frames = load_m5_source(data_dir)
    for name, df in frames.items():
        print(f"{name}: {len(df):,} rows")
