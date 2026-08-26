"""
data/features/halo_cannibalization_features.py — Step 6 (halo/cannibalization group),
feature_engineering_spec.md L48 ("Agent may add: sub-events (SEASONAL/bundle cannibalization)").

Genuine, data-driven halo (complement) / cannibalization (substitute) signal — NOT the
coarse same-volume-band proxy in data/features/category_features.py (that module's
`category` is an ABC/D/SEASONAL *volume band* computed market-wide, not a product-family
grouping, since no source table names one — see category_classification.py's docstring.
Two arbitrary items sharing a volume band are not necessarily substitutes/complements).

DESIGN — market-agnostic, no manually-curated item mapping ("not dependent on people"):
  1. `discover_partner_relationships` computes, per market and purely from realized demand,
     which item pairs move together (halo) or oppositely (cannibalization). It correlates
     each item's demand SURPRISE (qty minus its own trailing 7d mean, market-wide, so shared
     calendar/trend/seasonality effects are netted out and the correlation reflects
     idiosyncratic cross-item co-movement — e.g. two items both spiking at Christmas is NOT
     halo; one item spiking exactly when a substitute is discounted IS cannibalization).
  2. Compute cost is bounded independent of catalog size: the correlation universe is capped
     to the top `_MAX_CORR_UNIVERSE_ITEMS` items by trailing volume, market-wide. This makes
     the feature "good for any market" — a 200-item catalog and a 5,000-item catalog cost the
     same to correlate. Long-tail items (outside the cap) get no signal (COALESCE to 0) rather
     than an unreliable correlation estimate from few non-zero days — an honest default, not
     just an engineering shortcut.
  3. The discovered partner list is a per-item *structural* property (recomputed fresh at every
      snapshot_date from data <= snapshot_date only, so it stays point-in-time correct and
      leakage-safe like every other feature here) and is persisted to its own small table,
      `halo_cannib_partners`, so JOB 3 (inference — scoring only, no source-table reads,
     see docs/instructions.md §7 lock #3) can reconstruct the KNOWN_AT_INFERENCE columns for
     the full 140-day horizon without recomputing correlations or touching raw sales history.

Two feature classes (tag matches feature_engineering_spec.md "Leakage control"):
  - HISTORICAL_ONLY: rolling realized demand of a item's halo/cannibalization partners at the
    same store. Requires partner qty that has actually happened — never available for future
    rows, same limitation as every other historical-only feature in this pipeline (lags,
    rolling means, category_qty_7d_roll, ...); the model's own multi-step strategy handles it
    (docs/instructions.md §10.4).
  - KNOWN_AT_INFERENCE: (a) the partner-relationship strength itself (structural, fixed as of
    snapshot_date, so it's valid across the whole horizon like `category`); (b) whether a
    partner item has an active promo today — promos are known 2 months to 1 year ahead
    (feature_engineering_spec.md §6), so this is exactly as future-safe as the existing
    `competing_promo_count`/`promo_active_flag` features. This is the primary cannibalization
    *risk* signal: a substitute going on promo predicts a dip in this item; a complement going
    on promo predicts a lift.

Both feature classes are produced by functions that take a plain feature frame (no dependency
on champion_index or the training feature table), so the exact same code path is used to
compute them for JOB 1's historical build (notebooks/01_build_features.py) and for JOB 3's future
spine (inference/future_spine.py) — the "adapting to halo cannibalization both in history and
future" requirement is satisfied by construction, not by two diverging implementations.
"""
from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

_PARTNER_SCHEMA = "gold"
_PARTNER_TABLE = "halo_cannib_partners"
_EVENTS_TABLE = "events"

# Correlation-universe cap: bounds discover_partner_relationships' self-join to
# _MAX_CORR_UNIVERSE_ITEMS^2 / 2 pairs regardless of market catalog size (feature must be
# "good for any market" — see module docstring point 2).
_MAX_CORR_UNIVERSE_ITEMS = 300
# Trailing window (days, ending at the input frame's max date_id, which is <= snapshot_date)
# used to compute the demand-surprise correlation.
_CORR_LOOKBACK_DAYS = 180
# Minimum co-observed days required before a pair's correlation is trusted.
_MIN_OVERLAP_DAYS = 30
# Top-K halo / top-K cannibalization partners kept per item.
_TOP_K_PARTNERS = 3
# corr >= this -> 'halo' (complements / shared positive demand shock).
_HALO_CORR_THRESHOLD = 0.30
# corr <= this (negative) -> 'cannib' (substitutes / opposite demand shock).
_CANNIB_CORR_THRESHOLD = -0.30

# Exported for config/seed_defaults.py feature seeding — must not drift from the columns
# actually produced below.
HISTORICAL_ONLY_COLUMNS: list[str] = [
    "halo_partner_qty_7d_roll",
    "cannib_partner_qty_7d_roll",
]
KNOWN_AT_INFERENCE_COLUMNS: list[str] = [
    "halo_partner_corr_strength",
    "cannib_partner_corr_strength",
    "halo_partner_promo_active_flag",
    "halo_partner_promo_count",
    "cannib_partner_promo_active_flag",
    "cannib_partner_promo_count",
]


def classify_relation(
    corr: float,
    halo_threshold: float = _HALO_CORR_THRESHOLD,
    cannib_threshold: float = _CANNIB_CORR_THRESHOLD,
) -> str | None:
    """Pure mirror of the WHERE clauses in discover_partner_relationships' SQL: corr >=
    halo_threshold -> 'halo'; corr <= cannib_threshold -> 'cannib'; otherwise None (no
    reliable relation). Kept in sync with the SQL exactly, same pattern as
    data/features/category_classification.py's classify_volume_band/resolve_category.
    """
    if corr >= halo_threshold:
        return "halo"
    if corr <= cannib_threshold:
        return "cannib"
    return None


# ── Phase 1: discover partner relationships from realized demand (JOB 1 only) ─────────────

def discover_partner_relationships(
    spark: "SparkSession",
    df: "DataFrame",
    max_universe_items: int = _MAX_CORR_UNIVERSE_ITEMS,
    lookback_days: int = _CORR_LOOKBACK_DAYS,
    min_overlap_days: int = _MIN_OVERLAP_DAYS,
    top_k: int = _TOP_K_PARTNERS,
    halo_threshold: float = _HALO_CORR_THRESHOLD,
    cannib_threshold: float = _CANNIB_CORR_THRESHOLD,
) -> "DataFrame":
    """Discover each item's top-K halo and top-K cannibalization partners, market-wide.

    Args:
        df: per-series historical frame; must carry store_id, sku_id, date_id,
            qty, isActive. Must NOT contain rows beyond snapshot_date (caller's read
            window already guarantees this — see notebooks/01_build_features.py).

    Returns:
        DataFrame(sku_id, partner_sku_id, corr, n_overlap_days, relation_type)
        with relation_type in ('halo', 'cannib'), at most top_k rows per
        (sku_id, relation_type). An item with no qualifying partner has no rows —
        callers COALESCE the eventual feature join to 0.

    Method: aggregate qty to (sku_id, date_id) market-wide (sums out per-store noise
    and bounds the self-join to items, not item x store pairs); compute each day's demand
    "surprise" as qty_mkt minus its own leakage-safe trailing 7d mean (nets out shared
    calendar/trend/seasonality so two items that both spike at Christmas don't register as
    "halo"); cap to the top `max_universe_items` items by trailing volume (bounds compute
    regardless of catalog size); self-join surprises within that capped universe over the
    trailing `lookback_days` and take Spark's native corr() aggregate per pair.
    """
    df.createOrReplaceTempView("_hc_disco_input")
    return spark.sql(f"""
        WITH market_item_daily AS (
            SELECT sku_id, date_id, SUM(qty) AS qty_mkt
            FROM _hc_disco_input
            WHERE isActive = 1
            GROUP BY sku_id, date_id
        ),
        bounds AS (
            SELECT MAX(date_id) AS max_date FROM market_item_daily
        ),
        market_item_resid AS (
            -- Demand surprise: today's market qty minus the trailing 7d mean (leakage-safe:
            -- excludes the current row, same ROWS BETWEEN ... 1 PRECEDING convention as
            -- data/features/stat_features.py).
            SELECT sku_id, date_id, qty_mkt,
                   qty_mkt - AVG(qty_mkt) OVER (
                       PARTITION BY sku_id ORDER BY date_id
                       ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                   ) AS resid
            FROM market_item_daily
        ),
        item_volume AS (
            SELECT sku_id, SUM(qty_mkt) AS total_qty
            FROM market_item_daily
            GROUP BY sku_id
        ),
        top_items AS (
            SELECT sku_id FROM (
                SELECT sku_id,
                       ROW_NUMBER() OVER (ORDER BY total_qty DESC, sku_id ASC) AS rn
                FROM item_volume
            ) WHERE rn <= {max_universe_items}
        ),
        resid_capped AS (
            SELECT r.sku_id, r.date_id, r.resid
            FROM market_item_resid r
            JOIN top_items t ON r.sku_id = t.sku_id
            CROSS JOIN bounds b
            WHERE r.resid IS NOT NULL
              AND r.date_id >= DATE_SUB(b.max_date, {lookback_days})
        ),
        -- Pre-filter items whose demand residuals are constant (VAR_SAMP = 0) BEFORE
        -- the self-join. corr() in Spark 4.0 ANSI mode (SQLSTATE 22012) throws
        -- DIVIDE_BY_ZERO during the GROUP BY aggregation phase, before any HAVING
        -- filter runs — so HAVING VAR_SAMP > 0 cannot guard it; the problematic rows
        -- must be excluded from the pairs input entirely.
        -- Constant-residual items carry no meaningful correlation signal anyway.
        nonconst_items AS (
            SELECT sku_id FROM resid_capped
            GROUP BY sku_id
            HAVING VAR_SAMP(resid) > 0
        ),
        pairs AS (
            SELECT a.sku_id AS item_a, b.sku_id AS item_b,
                   a.resid AS resid_a, b.resid AS resid_b
            FROM resid_capped a
            JOIN nonconst_items nca ON nca.sku_id = a.sku_id
            JOIN resid_capped b
              ON a.date_id = b.date_id AND a.sku_id < b.sku_id
            JOIN nonconst_items ncb ON ncb.sku_id = b.sku_id
        ),
        pair_corr AS (
            -- Replace corr() with an explicit Pearson formula so ANSI mode
            -- (Spark 4.0 / DBR 17.3 ML LTS, SQLSTATE 22012) never sees a
            -- divide-by-zero: try_divide returns NULL instead of throwing when
            -- stddev_a * stddev_b = 0 (constant residuals on the PAIRED dates).
            -- nonconst_items above already eliminates most such pairs; this is
            -- the belt-and-suspenders guard for the edge case where two items
            -- each have non-zero variance overall but the date intersection has
            -- zero variance on one side.
            -- Mathematically: corr(a,b) = covar_samp(a,b) / (stddev_a * stddev_b)
            SELECT item_a, item_b,
                   try_divide(
                       COVAR_SAMP(resid_a, resid_b),
                       SQRT(VAR_SAMP(resid_a) * VAR_SAMP(resid_b))
                   ) AS corr,
                   COUNT(*) AS n_overlap_days
            FROM pairs
            GROUP BY item_a, item_b
            HAVING COUNT(*) >= {min_overlap_days}
        ),
        -- Pairs are symmetric (item_a < item_b); expand to a directional neighbor list so
        -- every item can look up its own partners regardless of which side of the pair it was.
        directional AS (
            SELECT item_a AS sku_id, item_b AS partner_sku_id, corr, n_overlap_days
            FROM pair_corr
            UNION ALL
            SELECT item_b AS sku_id, item_a AS partner_sku_id, corr, n_overlap_days
            FROM pair_corr
        ),
        ranked_halo AS (
            SELECT sku_id, partner_sku_id, corr, n_overlap_days,
                   ROW_NUMBER() OVER (
                       PARTITION BY sku_id ORDER BY corr DESC, partner_sku_id ASC
                   ) AS rn
            FROM directional
            WHERE corr >= {halo_threshold}
        ),
        ranked_cannib AS (
            SELECT sku_id, partner_sku_id, corr, n_overlap_days,
                   ROW_NUMBER() OVER (
                       PARTITION BY sku_id ORDER BY corr ASC, partner_sku_id ASC
                   ) AS rn
            FROM directional
            WHERE corr <= {cannib_threshold}
        )
        SELECT sku_id, partner_sku_id, corr, n_overlap_days, 'halo' AS relation_type
        FROM ranked_halo WHERE rn <= {top_k}
        UNION ALL
        SELECT sku_id, partner_sku_id, corr, n_overlap_days, 'cannib' AS relation_type
        FROM ranked_cannib WHERE rn <= {top_k}
    """)


def write_partner_map(
    spark: "SparkSession",
    partner_map_df: "DataFrame",
    catalog: str,
    market_id: str,
    snapshot_date: dt.date,
) -> None:
    """Idempotently MERGE the discovered partner map into `halo_cannib_partners`.

    Table must already exist (config/ddl/migrate_add_halo_cannibalization.sql — run once in
    Databricks, same "RUN IN DATABRICKS" contract as every other DDL file in this repo).
    Keyed by (snapshot_date, market_id, sku_id, partner_sku_id, relation_type) so multiple
    snapshots coexist for replay, matching sku_store_features' PK convention.
    """
    from utils.io import merge_upsert

    safe_market_id = market_id.replace("'", "''")
    partner_map_df.createOrReplaceTempView("_hc_partner_map_to_write")
    stamped_df = spark.sql(f"""
        SELECT DATE('{snapshot_date.isoformat()}') AS snapshot_date,
               '{safe_market_id}'                  AS market_id,
               sku_id, partner_sku_id, relation_type, corr, n_overlap_days,
               current_timestamp()                 AS updated_ts
        FROM _hc_partner_map_to_write
    """)
    merge_upsert(
        spark, stamped_df, catalog, _PARTNER_SCHEMA, _PARTNER_TABLE,
        merge_keys=["snapshot_date", "market_id", "sku_id", "partner_sku_id", "relation_type"],
    )


def read_partner_map(spark: "SparkSession", catalog: str, market_id: str) -> "DataFrame":
    """Read the most recent snapshot's partner map for `market_id`.

    Used by both JOB 1 (to attach KNOWN_AT_INFERENCE columns onto the historical frame it
    just discovered relationships from) and JOB 3's future spine (inference/future_spine.py),
    which cannot recompute correlations itself (scoring-only, no source-table reads,
    docs/instructions.md §7 lock #3).
    """
    safe_market_id = market_id.replace("'", "''")
    return spark.sql(f"""
        WITH latest AS (
            SELECT MAX(snapshot_date) AS snap
            FROM {catalog}.{_PARTNER_SCHEMA}.{_PARTNER_TABLE}
            WHERE market_id = '{safe_market_id}'
        )
        SELECT p.sku_id, p.partner_sku_id, p.relation_type, p.corr, p.n_overlap_days
        FROM {catalog}.{_PARTNER_SCHEMA}.{_PARTNER_TABLE} p
        CROSS JOIN latest
        WHERE p.market_id = '{safe_market_id}' AND p.snapshot_date = latest.snap
    """)


# ── Phase 2: KNOWN_AT_INFERENCE features — usable for history AND the future horizon ──────

def attach_partner_corr_strength(spark: "SparkSession", df: "DataFrame", partner_map_df: "DataFrame") -> "DataFrame":
    """Join the static per-item halo/cannib relationship-strength scores onto `df`.

    Structural (fixed as of the snapshot that produced `partner_map_df`), so safe to join
    onto both historical rows and the full future horizon — same "known as of snapshot,
    constant across the horizon" treatment as `category`.
    """
    df.createOrReplaceTempView("_hc_strength_input")
    partner_map_df.createOrReplaceTempView("_hc_strength_partner_map")
    return spark.sql("""
        WITH strength AS (
            SELECT sku_id,
                   AVG(CASE WHEN relation_type = 'halo'   THEN corr        END) AS halo_partner_corr_strength,
                   AVG(CASE WHEN relation_type = 'cannib' THEN ABS(corr)   END) AS cannib_partner_corr_strength
            FROM _hc_strength_partner_map
            GROUP BY sku_id
        )
        SELECT t.*,
               COALESCE(s.halo_partner_corr_strength, 0.0)   AS halo_partner_corr_strength,
               COALESCE(s.cannib_partner_corr_strength, 0.0) AS cannib_partner_corr_strength
        FROM _hc_strength_input t
        LEFT JOIN strength s ON t.sku_id = s.sku_id
    """)


def build_partner_promo_features(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    df: "DataFrame",
    partner_map_df: "DataFrame",
) -> "DataFrame":
    """Attach halo/cannib partner promo signals onto `df` (must carry store_id,
    sku_id, date_id).

    Queries `gold.events` directly (mirroring data/features/event_features.py's `_events()`
    matching logic, keyed by the FOCAL item's partner list instead of its own sku_id)
    rather than self-joining `df` for the partner's promo_active_flag. This is deliberate:
    `df` may not carry a row for the partner item at every (store_id, date) — in JOB 3's future
    spine specifically, rows only exist for series with a trained champion
    (inference/future_spine.py), so a partner item without its own champion at that store
    would silently vanish from a self-join. Going straight to the event source table makes
    this correct for BOTH history and the future horizon regardless of champion coverage.

    Promos are known 2 months to 1 year ahead (feature_engineering_spec.md §6), so this is
    KNOWN_AT_INFERENCE — the primary cannibalization *risk* signal: a substitute item going
    on promo predicts a dip in this item; a complement going on promo predicts a lift.
    """
    safe_market_id = market_id.replace("'", "''")
    df.createOrReplaceTempView("_hc_promo_input")
    partner_map_df.createOrReplaceTempView("_hc_promo_partner_map")
    return spark.sql(f"""
        WITH partner_events AS (
            SELECT event_id, sku_id, store_id, start_date, end_date
            FROM {catalog}.gold.events
            WHERE market_id = '{safe_market_id}'
              AND (upper(event_type) = 'PROMOTION' OR upper(event_subtype) = 'PROMOTION')
        ),
        halo_matches AS (
            SELECT t.store_id, t.sku_id AS focal_item, t.date_id,
                   COUNT(DISTINCT pe.event_id) AS n
            FROM _hc_promo_input t
            JOIN _hc_promo_partner_map pm ON pm.sku_id = t.sku_id AND pm.relation_type = 'halo'
            JOIN partner_events pe
              ON pe.sku_id = pm.partner_sku_id
             AND pe.store_id = t.store_id
             AND t.date_id BETWEEN pe.start_date AND pe.end_date
            GROUP BY t.store_id, t.sku_id, t.date_id
        ),
        cannib_matches AS (
            SELECT t.store_id, t.sku_id AS focal_item, t.date_id,
                   COUNT(DISTINCT pe.event_id) AS n
            FROM _hc_promo_input t
            JOIN _hc_promo_partner_map pm ON pm.sku_id = t.sku_id AND pm.relation_type = 'cannib'
            JOIN partner_events pe
              ON pe.sku_id = pm.partner_sku_id
             AND pe.store_id = t.store_id
             AND t.date_id BETWEEN pe.start_date AND pe.end_date
            GROUP BY t.store_id, t.sku_id, t.date_id
        )
        SELECT t.*,
               COALESCE(h.n, 0) AS halo_partner_promo_count,
               CASE WHEN COALESCE(h.n, 0) > 0 THEN 1 ELSE 0 END AS halo_partner_promo_active_flag,
               COALESCE(c.n, 0) AS cannib_partner_promo_count,
               CASE WHEN COALESCE(c.n, 0) > 0 THEN 1 ELSE 0 END AS cannib_partner_promo_active_flag
        FROM _hc_promo_input t
        LEFT JOIN halo_matches h
          ON t.store_id = h.store_id AND t.sku_id = h.focal_item AND t.date_id = h.date_id
        LEFT JOIN cannib_matches c
          ON t.store_id = c.store_id AND t.sku_id = c.focal_item AND t.date_id = c.date_id
    """)


def build_known_at_inference_features(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    df: "DataFrame",
    partner_map_df: "DataFrame",
) -> "DataFrame":
    """Convenience wrapper: attach every KNOWN_AT_INFERENCE_COLUMNS column onto `df`.

    Single entry point called identically by notebooks/01_build_features.py (JOB 1, historical
    frame) and inference/future_spine.py (JOB 3, future horizon) — see module docstring.
    """
    with_strength = attach_partner_corr_strength(spark, df, partner_map_df)
    return build_partner_promo_features(spark, catalog, market_id, with_strength, partner_map_df)


# ── Phase 3: HISTORICAL_ONLY features — JOB 1 only, never joined into the future spine ────

def build_historical_partner_demand_features(
    spark: "SparkSession",
    df: "DataFrame",
    partner_map_df: "DataFrame",
) -> "DataFrame":
    """Attach rolling realized demand of each item's halo/cannib partners at the same store.

    Args:
        df: per-series historical frame; must carry store_id, sku_id, date_id, qty,
            isActive (realized qty — never call this on the future spine).

    Leakage-safe: ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING excludes the current row, same
    convention as every rolling feature in data/features/stat_features.py.
    """
    df.createOrReplaceTempView("_hc_demand_input")
    partner_map_df.createOrReplaceTempView("_hc_demand_partner_map")
    return spark.sql("""
        WITH halo_daily AS (
            SELECT pm.sku_id AS focal_item, t.store_id, t.date_id,
                   SUM(t.qty) AS partner_qty_today
            FROM _hc_demand_partner_map pm
            JOIN _hc_demand_input t ON t.sku_id = pm.partner_sku_id
            WHERE pm.relation_type = 'halo' AND t.isActive = 1
            GROUP BY pm.sku_id, t.store_id, t.date_id
        ),
        halo_roll AS (
            SELECT focal_item, store_id, date_id,
                   SUM(partner_qty_today) OVER (
                       PARTITION BY focal_item, store_id ORDER BY date_id
                       ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                   ) AS halo_partner_qty_7d_roll
            FROM halo_daily
        ),
        cannib_daily AS (
            SELECT pm.sku_id AS focal_item, t.store_id, t.date_id,
                   SUM(t.qty) AS partner_qty_today
            FROM _hc_demand_partner_map pm
            JOIN _hc_demand_input t ON t.sku_id = pm.partner_sku_id
            WHERE pm.relation_type = 'cannib' AND t.isActive = 1
            GROUP BY pm.sku_id, t.store_id, t.date_id
        ),
        cannib_roll AS (
            SELECT focal_item, store_id, date_id,
                   SUM(partner_qty_today) OVER (
                       PARTITION BY focal_item, store_id ORDER BY date_id
                       ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                   ) AS cannib_partner_qty_7d_roll
            FROM cannib_daily
        )
        SELECT t.*,
               COALESCE(hr.halo_partner_qty_7d_roll, 0.0)   AS halo_partner_qty_7d_roll,
               COALESCE(cr.cannib_partner_qty_7d_roll, 0.0) AS cannib_partner_qty_7d_roll
        FROM _hc_demand_input t
        LEFT JOIN halo_roll hr
          ON t.sku_id = hr.focal_item AND t.store_id = hr.store_id AND t.date_id = hr.date_id
        LEFT JOIN cannib_roll cr
          ON t.sku_id = cr.focal_item AND t.store_id = cr.store_id AND t.date_id = cr.date_id
    """)
