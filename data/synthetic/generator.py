"""
data/synthetic/generator.py — synthetic retail demand data generator.

Stands in for the real point-of-sale / ERP source system this project would
normally read from. Produces a small multi-store x multi-sku daily sales
history with realistic trend, weekly/yearly seasonality, promo and holiday
effects, and store/sku lifecycle events (new-store ramp-up, sku
discontinuation) — enough structure to meaningfully exercise
`data/features/active_window.py`, `lifecycle.py`, and `gap_fill.py`, and to
give the modeling engine (Prophet/statsforecast/LightGBM/ElasticNet/Croston)
real signal to fit against.

Deterministic: same `seed` always produces the same data, so numbers/charts
generated from it (e.g. `demo/run_local_demo.py`, the README's accuracy
table) are reproducible.

Pure pandas/numpy — no pyspark import here, so this runs identically on a
laptop (feeding the local demo) and inside a Databricks notebook (feeding
`notebooks/00b_generate_synthetic_source.py`, which writes the returned
frames to Delta tables under the `source`/`gold` schemas).

Output frame shapes match the source-table contracts in
docs/feature_engineering_spec.md (verified against data/features/*.py's actual
Spark SQL reads, not just the spec — e.g. gold.calendar_date needs weekday_name,
and event_type must be exactly "PROMOTION"/"HOLIDAY" to match event_features.py's
_events() filter):
  sales:         market_id, store_id, sku_id, date_id, units_sold, units_promotional, units_shrinkage
  sku:           sku_id, category
  store:         store_id, market_id
  sku_store:     market_id, store_id, sku_id, active_from, active_to
  events:        market_id, event_id, event_type, event_subtype, event_nm, start_date, end_date, sku_id, store_id, exclude_flag, tactic
  calendar_date: date_id, weekday, weekday_name, week, day, month, quarter
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

CATEGORIES = ("A", "B", "C", "D", "SEASONAL")

# Base daily-demand level (log-normal mean) and noise dispersion by category.
# A = fast movers, D = long tail, SEASONAL = strong yearly swing (e.g. holiday goods).
_CATEGORY_PROFILE = {
    "A":        {"base_level": 40.0, "noise_frac": 0.18, "trend_frac": 0.15, "yearly_amp": 0.10},
    "B":        {"base_level": 18.0, "noise_frac": 0.22, "trend_frac": 0.10, "yearly_amp": 0.12},
    "C":        {"base_level": 7.0,  "noise_frac": 0.30, "trend_frac": 0.05, "yearly_amp": 0.15},
    "D":        {"base_level": 2.5,  "noise_frac": 0.45, "trend_frac": 0.00, "yearly_amp": 0.10},
    "SEASONAL": {"base_level": 10.0, "noise_frac": 0.30, "trend_frac": 0.00, "yearly_amp": 0.55},
}

# Category mix weights (must sum to 1.0) — mimics a realistic long-tail assortment.
_CATEGORY_WEIGHTS = {"A": 0.10, "B": 0.20, "C": 0.30, "D": 0.30, "SEASONAL": 0.10}

# Fixed day-of-week multipliers (Mon..Sun), shared across all series for simplicity —
# weekends sell more.
_DOW_MULTIPLIER = np.array([0.90, 0.92, 0.95, 1.00, 1.10, 1.30, 1.20])

# Promo tactic labels — matches data/features/event_features.py's
# _FALLBACK_TACTIC_COLUMN_MAP keys, so generated events exercise the real
# tactic-count feature columns instead of always falling into the "_other" bucket.
_PROMO_TACTICS = (
    "Temporary Price Change",
    "Buy One, Get One - Mix",
    "Meal Bundle",
    "Free Offer",
)

# A small set of fixed holidays (month, day, lift_multiplier, window_days) applied every year.
_HOLIDAYS = [
    ("New Year's Day",  1,  1, 0.5, 1),
    ("Valentine's Day", 2, 14, 1.6, 1),
    ("Independence Day", 7,  4, 1.4, 1),
    ("Thanksgiving",    11, 27, 1.8, 2),
    ("Black Friday",    11, 28, 2.4, 1),
    ("Christmas Eve",   12, 24, 1.7, 1),
    ("Christmas Day",   12, 25, 0.3, 1),
]


def _daterange(start: dt.date, end: dt.date) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="D")


def _build_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """gold.calendar_date — date_id + DOW/week/day/month/quarter/weekday_name
    (weekday: 1=Sun..7=Sat, matching data/features/temporal_features.py's conversion)."""
    return pd.DataFrame({
        "date_id": dates.date,
        "weekday": (dates.dayofweek + 1) % 7 + 1,  # pandas: Mon=0..Sun=6 -> 1=Sun..7=Sat
        "weekday_name": dates.day_name(),
        "week": dates.isocalendar().week.to_numpy(),
        "day": dates.day,
        "month": dates.month,
        "quarter": dates.quarter,
    })


def _build_events(dates: pd.DatetimeIndex, market_ids: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """gold.events — broadcast-level holidays (NULL sku_id/store_id = applies to all) plus
    a handful of random broadcast promo windows per market."""
    rows = []
    event_id = 0
    years = sorted({d.year for d in dates})

    for market_id in market_ids:
        for name, month, day, lift, window_days in _HOLIDAYS:
            for year in years:
                try:
                    d = dt.date(year, month, day)
                except ValueError:
                    continue
                if d < dates[0].date() or d > dates[-1].date():
                    continue
                event_id += 1
                rows.append({
                    "market_id": market_id,
                    "event_id": f"HOL{event_id:05d}",
                    "event_type": "HOLIDAY",
                    "event_subtype": "HOLIDAY",
                    "event_nm": name,
                    "start_date": d - dt.timedelta(days=window_days),
                    "end_date": d + dt.timedelta(days=window_days),
                    "sku_id": None,
                    "store_id": None,
                    "exclude_flag": 0,
                    "tactic": "HOLIDAY",
                })

        # Random broadcast promo windows: ~1 every 5 weeks, 3-7 days long.
        n_promos = max(1, int(len(dates) / 35))
        for i in range(n_promos):
            start_idx = rng.integers(0, max(1, len(dates) - 10))
            duration = int(rng.integers(3, 8))
            start_d = dates[start_idx].date()
            end_d = min(dates[-1].date(), start_d + dt.timedelta(days=duration))
            event_id += 1
            rows.append({
                "market_id": market_id,
                "event_id": f"PROMO{event_id:05d}",
                "event_type": "PROMOTION",
                "event_subtype": rng.choice(["BOGO", "PCT_OFF", "BUNDLE"]),
                "event_nm": f"Promo {i + 1}",
                "start_date": start_d,
                "end_date": end_d,
                "sku_id": None,
                "store_id": None,
                "exclude_flag": 0,
                "tactic": rng.choice(_PROMO_TACTICS),
            })

    return pd.DataFrame(rows)


def _promo_multiplier(dates: pd.DatetimeIndex, events_df: pd.DataFrame, market_id: str) -> np.ndarray:
    """Combined per-day demand multiplier from all broadcast events active for this market."""
    mult = np.ones(len(dates), dtype=float)
    mkt_events = events_df[events_df["market_id"] == market_id]
    date_arr = dates.date
    for _, ev in mkt_events.iterrows():
        active = (date_arr >= ev["start_date"]) & (date_arr <= ev["end_date"])
        if ev["event_type"] == "HOLIDAY":
            name = ev["event_nm"]
            lift = next((h[3] for h in _HOLIDAYS if h[0] == name), 1.0)
        else:
            lift = 1.5  # flat promo lift
        mult[active] *= lift
    return mult


def _yearly_seasonality(dates: pd.DatetimeIndex, amplitude: float) -> np.ndarray:
    doy = dates.dayofyear.to_numpy()
    return 1.0 + amplitude * np.sin(2 * np.pi * (doy - 60) / 365.25)


def generate_synthetic_source(
    seed: int = 42,
    n_stores: int = 30,
    n_skus: int = 300,
    n_days: int = 900,
    markets: list[str] | None = None,
    end_date: dt.date | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate a full synthetic source-table set.

    Args:
        seed: RNG seed — same seed always produces the same data.
        n_stores: number of synthetic stores, split evenly across `markets`.
        n_skus: number of synthetic skus, shared across all stores/markets.
        n_days: length of the daily history (ending at `end_date`).
        markets: market codes to generate (default 2 demo markets).
        end_date: last date of history (default: 2 days ago from "today" —
            passed explicitly since Date.now()-style calls are avoided in
            reproducible-generation contexts; callers should pass a fixed date).

    Returns:
        dict with keys "sales", "sku", "store", "sku_store", "events", "calendar_date" —
        each a pandas DataFrame matching the source-table contracts in
        docs/feature_engineering_spec.md.
    """
    markets = markets or ["US", "DE"]
    end_date = end_date or dt.date(2026, 8, 23)
    start_date = end_date - dt.timedelta(days=n_days - 1)
    dates = _daterange(start_date, end_date)

    rng = np.random.default_rng(seed)

    # ── store / sku masters ──────────────────────────────────────────────
    store_ids = [f"ST{idx:04d}" for idx in range(1, n_stores + 1)]
    store_market = {sid: markets[i % len(markets)] for i, sid in enumerate(store_ids)}
    store_df = pd.DataFrame({
        "store_id": store_ids,
        "market_id": [store_market[sid] for sid in store_ids],
    })

    sku_ids = [f"SKU{idx:05d}" for idx in range(1, n_skus + 1)]
    cats = rng.choice(list(_CATEGORY_WEIGHTS.keys()), size=n_skus, p=list(_CATEGORY_WEIGHTS.values()))
    sku_df = pd.DataFrame({"sku_id": sku_ids, "category": cats})

    # ── lifecycle: a handful of new stores (start selling partway through)
    #    and discontinued skus (stop selling before end_date) ─────────────
    n_new_stores = max(1, n_stores // 15)
    new_store_ids = set(rng.choice(store_ids, size=n_new_stores, replace=False))
    store_start = {
        sid: (start_date if sid not in new_store_ids
              else start_date + dt.timedelta(days=int(rng.integers(n_days // 2, n_days - 30))))
        for sid in store_ids
    }

    n_discontinued = max(1, n_skus // 20)
    discontinued_skus = set(rng.choice(sku_ids, size=n_discontinued, replace=False))
    sku_end = {
        sku: (end_date if sku not in discontinued_skus
              else start_date + dt.timedelta(days=int(rng.integers(n_days // 3, n_days - 30))))
        for sku in sku_ids
    }

    # ── events (holidays + broadcast promos), then per-market promo multiplier series ──
    events_df = _build_events(dates, markets, rng)
    promo_mult_by_market = {m: _promo_multiplier(dates, events_df, m) for m in markets}
    dow_mult_full = _DOW_MULTIPLIER[dates.dayofweek.to_numpy()]

    # ── sales + sku_store windows, one series (store, sku) at a time ──────
    sales_rows = []
    sku_store_rows = []
    cat_by_sku = dict(zip(sku_ids, cats))

    # Assortment: each store carries a random ~70% of the sku catalog (long-tail realism,
    # keeps total series count manageable — n_stores x n_skus x 0.7).
    for store_id in store_ids:
        market_id = store_market[store_id]
        assortment = rng.choice(sku_ids, size=int(n_skus * 0.7), replace=False)
        s_start = store_start[store_id]

        for sku_id in assortment:
            category = cat_by_sku[sku_id]
            profile = _CATEGORY_PROFILE[category]
            active_from = max(s_start, start_date)
            active_to = min(sku_end[sku_id], end_date)
            if active_from >= active_to:
                continue

            sku_store_rows.append({
                "market_id": market_id, "store_id": store_id, "sku_id": sku_id,
                "active_from": active_from, "active_to": active_to,
            })

            mask = (dates.date >= active_from) & (dates.date <= active_to)
            n_active = int(mask.sum())
            if n_active == 0:
                continue

            t = np.arange(n_active, dtype=float)
            trend = 1.0 + profile["trend_frac"] * (t / max(1, n_active))
            yearly = _yearly_seasonality(dates[mask], profile["yearly_amp"])
            dow = dow_mult_full[mask]
            promo = promo_mult_by_market[market_id][mask]

            mean = profile["base_level"] * trend * yearly * dow * promo
            noise = rng.normal(1.0, profile["noise_frac"], size=n_active)
            qty = np.clip(mean * noise, 0, None)
            qty = rng.poisson(np.maximum(qty, 0.01)).astype(float)

            # Split into paid / promotional / shrinkage units (matches the sum-of-3 target
            # contract in docs/feature_engineering_spec.md).
            promo_active = promo > 1.0
            units_promotional = np.where(promo_active, qty * 0.15, 0.0)
            units_shrinkage = rng.binomial(1, 0.01, size=n_active) * rng.uniform(1, 3, size=n_active)
            units_sold = np.maximum(qty - units_promotional - units_shrinkage, 0.0)

            sales_rows.append(pd.DataFrame({
                "market_id": market_id,
                "store_id": store_id,
                "sku_id": sku_id,
                "date_id": dates[mask].date,
                "units_sold": np.round(units_sold, 1),
                "units_promotional": np.round(units_promotional, 1),
                "units_shrinkage": np.round(units_shrinkage, 1),
            }))

    sales_df = pd.concat(sales_rows, ignore_index=True) if sales_rows else pd.DataFrame(
        columns=["market_id", "store_id", "sku_id", "date_id", "units_sold", "units_promotional", "units_shrinkage"]
    )
    sku_store_df = pd.DataFrame(sku_store_rows)
    calendar_df = _build_calendar(dates)

    return {
        "sales": sales_df,
        "sku": sku_df,
        "store": store_df,
        "sku_store": sku_store_df,
        "events": events_df,
        "calendar_date": calendar_df,
    }


if __name__ == "__main__":
    frames = generate_synthetic_source()
    for name, df in frames.items():
        print(f"{name}: {len(df):,} rows")
