"""
inference/future_spine.py — 140-day known-future spine from snapshot_date (docs/instructions.md §9, §13).

Dense daily spine per champion series [snapshot_date+1 .. snapshot_date+horizon_days],
joined to known-at-inference features only (calendar/holiday/promo).
Historical-only lag features for the horizon come from the champion model's
multi-step strategy, never from future rows (leakage control, §17).

Series list comes from champion_index (the authoritative set of what gets
scored) rather than from sku_store_features — champions already carry
category, series_key, model_type and params which downstream batch scoring needs.

Reuses data/features/temporal_features.py and data/features/event_features.py
directly: the JOIN logic is identical for past and future dates; only the
date_id range differs.

Also reuses data/features/halo_cannibalization_features.py's
build_known_at_inference_features: halo/cannibalization partner-promo signals and
partner-relationship strength are KNOWN_AT_INFERENCE (promos are known 2mo-1yr ahead,
and the partner relationship itself is a structural property fixed as of snapshot_date
— see that module's docstring), so they belong on the future spine exactly like
promo/holiday features. Reads the partner list from `halo_cannib_partners`
(written by JOB 1) rather than recomputing correlations — JOB 3 is scoring-only and
never re-reads raw source tables (docs/instructions.md §7 lock #3).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def build_future_spine(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    snapshot_date: str,
    horizon_days: int,
) -> "DataFrame":
    """Build the dense [snapshot_date+1, snapshot_date+horizon_days] spine with
    known-at-inference features for every champion series.

    Series list: champion_index filtered to `market_id` (latest champions,
    any snapshot_date). Carries market_id, category, series_key, store_id,
    sku_id — all needed downstream by batch_infer.score().

    Features joined:
      * Temporal: gold.calendar_date -> dow, week_of_year, month, quarter, day_of_month,
        weekend_flag, season, fiscal_period, payday_flag
      * Promo:    gold.events (event_type='promo') -> promo_active_flag, ...
      * Holiday:  gold.events (event_type='holiday') -> holiday_flag, ...
      * Halo/cannibalization (KNOWN_AT_INFERENCE only): halo_partner_corr_strength,
        cannib_partner_corr_strength, halo_partner_promo_active_flag/count,
        cannib_partner_promo_active_flag/count (data/features/halo_cannibalization_features.py)

      * lag_365: same-date-last-year realised qty. The ONE lag feature that is safe
        across the whole horizon (365 > horizon_days, so d-365 always predates
        snapshot_date); training/models/lgbm.py consumes it. See step 6.

    NOT included (historical-only — leakage if used on future rows):
      lag_1/7/14/28, rolling_*, ewma_*, expanding_*, same_weekday_last_year, yoy_ratio,
      trailing_zero_sales_ratio, isActive, qty.

    Args:
        spark:         Active SparkSession.
        catalog:       UC catalog, resolved from `env` (e.g. dev/prod).
        market_id:     Market code (e.g. 'PL'); used to filter champions and events.
        snapshot_date: ISO date string (YYYY-MM-DD); horizon starts snapshot_date+1.
        horizon_days:  Number of future days to score (e.g. 140).

    Returns:
        DataFrame with columns:
        series_key, market_id, category, store_id, sku_id, date_id,
        + all known-at-inference temporal and event feature columns.
    """
    from utils.dates import future_horizon, resolve_snapshot_date
    from data.features.event_features import build_holiday_features, build_promo_features
    from data.features.halo_cannibalization_features import (
        build_known_at_inference_features,
        read_partner_map,
    )
    from data.features.temporal_features import add_temporal_features

    snap = resolve_snapshot_date(snapshot_date)
    horizon_start, horizon_end = future_horizon(snap, horizon_days)
    safe_market_id = market_id.replace("'", "''")

    # ── 1. Champion series ────────────────────────────────────────────────────
    # If a series has been trained multiple times (different snapshot_dates or
    # versions), pick the latest version to avoid duplicate rows in the spine.
    champions_df = spark.sql(f"""
        WITH ranked AS (
            SELECT market_id, category, series_key, store_id, sku_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY series_key
                       ORDER BY snapshot_date DESC, version DESC
                   ) AS rn
            FROM {catalog}.gold.champion_index
            WHERE market_id = '{safe_market_id}'
        )
        SELECT market_id, category, series_key, store_id, sku_id
        FROM ranked WHERE rn = 1
    """)

    # ── 2. Future date range from gold.calendar_date (canonical calendar) ───────
    # Pulling from gold.calendar_date (rather than generating a sequence) ensures
    # every future date_id has matching temporal columns for the JOIN in Step 4.
    future_dates_df = spark.sql(f"""
        SELECT date_id
        FROM {catalog}.gold.calendar_date
        WHERE date_id BETWEEN DATE('{horizon_start.isoformat()}')
                          AND DATE('{horizon_end.isoformat()}')
    """)

    # ── 3. Dense spine: one row per (series × future date) ───────────────────
    champions_df.createOrReplaceTempView("_spine_champions")
    future_dates_df.createOrReplaceTempView("_spine_future_dates")
    spine_df = spark.sql("""
        SELECT c.market_id, c.category, c.series_key,
               c.store_id, c.sku_id,
               d.date_id
        FROM _spine_champions c
        CROSS JOIN _spine_future_dates d
    """)

    # ── 4. Known-at-inference feature joins ───────────────────────────────────
    # add_temporal_features, build_promo_features, build_holiday_features all do
    # SELECT t.*, <new cols> so they preserve series_key / category / market
    # throughout the chain without extra bookkeeping.
    temporal_df = add_temporal_features(spark, catalog, spine_df)
    promo_df = build_promo_features(spark, catalog, market_id, temporal_df)
    holiday_df = build_holiday_features(spark, catalog, market_id, promo_df)

    # ── 5. Halo/cannibalization KNOWN_AT_INFERENCE features ───────────────────
    # partner_map_df is tiny (bounded by _MAX_CORR_UNIVERSE_ITEMS^2, see
    # halo_cannibalization_features.py) — cheap to read fresh per inference run.
    # Requires config/ddl/migrate_add_halo_cannibalization.sql to have been run once
    # (creates halo_cannib_partners); until JOB 1 has run at least once for this
    # market the table exists but has zero rows, so every halo/cannib column below
    # simply COALESCEs to 0/0.0 rather than failing.
    partner_map_df = read_partner_map(spark, catalog, market_id)
    halo_df = build_known_at_inference_features(spark, catalog, market_id, holiday_df, partner_map_df)

    # ── 6. lag_365 — the one demand-history feature that is horizon-safe ───────
    # training/models/lgbm.py trains on lag_365, so the spine MUST carry it: that
    # wrapper's build_calendar_feature_matrix() fills any absent column with 0, which
    # would silently train on real year-ago demand and then score against zeros.
    #
    # Safe for the entire horizon because 365 > horizon_days: for every horizon date d,
    # d - 365 is at least (365 - horizon_days) days before snapshot_date, so it is
    # always realised history and never a value this forecast is trying to predict.
    # Deliberately NOT generalised to lag_1/7/14/28 or the rolling_* family — those
    # windows do reach into the horizon and are correctly excluded (see this module's
    # docstring). COALESCE to 0.0 mirrors build_calendar_feature_matrix's own fillna(0)
    # so a series with no observation exactly one year back behaves identically in
    # training and scoring.
    halo_df.createOrReplaceTempView("_spine_pre_lag365")
    return spark.sql(f"""
        WITH hist AS (
            SELECT store_id, sku_id, date_id, qty,
                   ROW_NUMBER() OVER (
                       PARTITION BY store_id, sku_id, date_id
                       ORDER BY snapshot_date DESC
                   ) AS rn
            FROM {catalog}.gold.sku_store_features
            WHERE market_id = '{safe_market_id}'
              AND snapshot_date <= DATE('{snap.isoformat()}')
        )
        SELECT s.*, COALESCE(h.qty, 0.0) AS lag_365
        FROM _spine_pre_lag365 s
        LEFT JOIN (SELECT store_id, sku_id, date_id, qty FROM hist WHERE rn = 1) h
               ON s.store_id = h.store_id
              AND s.sku_id   = h.sku_id
              AND h.date_id  = DATE_SUB(s.date_id, 365)
    """)
