"""
data/features/event_features.py — Step 6 (event group): promo + holiday features, known-at-inference.

Reads gold.events (market, event_id, event_type, event_subtype, event_nm,
start_date, end_date, duration, sku_id, store_id, store_type_cd, exclude_flag,
tactic); matches each day of the input spine against event date ranges,
honoring the broadcast rule (NULL sku_id/store_id on the event matches any
spine row) and `exclude_flag`. Markets may store events at broadcast level
(PL: NULL item/store = national) OR at item-store level (GE: one row per
item×store). The JOIN logic handles both transparently. Promos are known
2 months to 1 year ahead, so the future horizon's promo features are reliable
(feature_engineering_spec.md §6).

Promo: active flag, type/subtype/tactic, duration, days-since-start,
days-until-next, concurrent-promo count, tactic rank (ordinal 0-3),
lead flag (N days before promo starts).
Holiday: flag, name, type/subtype, pre/post +/-N-day window, bridge-day flag
(bridge-day is TODO — see build_holiday_features), holiday_weight (severity float).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

_EVENTS_TABLE = "events"   # parallel owned table
_EVENTS_SCHEMA = "gold"
_DEFAULT_HOLIDAY_WINDOW_DAYS = 3  # TODO(Step 3 refinement): source from config_feature.params
_DEFAULT_PROMO_LEAD_DAYS = 3       # days before promo start where demand pre-loading begins

# ═══════════════════════════════════════════════════════════════════════════════
# PROMO FEATURE CONFIG — config-driven with fallback defaults.
#
# Tactic and subtype mappings are stored in the config table:
#   param_group = 'feature'
#   scope       = 'promo_tactic_counts'
#   param_name  = 'params'
#   param_value = JSON {"tactic_column_map": {...}, "subtype_column_map": {...}}
#
# If a market has no config row, the _FALLBACK_* dicts below are used.
# These are intentionally generic retail tactics; per-market config overrides them.
# ═══════════════════════════════════════════════════════════════════════════════

_FALLBACK_TACTIC_COLUMN_MAP: dict[str, str] = {
    "Temporary Price Change":       "temp_price_change",
    "Buy One, Get One - Mix":       "bogo_mix",
    "Buy One, Get One - Match":     "bogo_match",
    "Mix and Match - Mix":          "mix_match_mix",
    "Mix and Match - Match":        "mix_match_match",
    "Meal Bundle":                  "meal_bundle",
    "Free Offer":                   "free_offer",
    "No Tactic - Advertising Only": "advertising",
    "Pairings (Entree + Side)":     "pairings",
}

_FALLBACK_SUBTYPE_COLUMN_MAP: dict[str, str] = {
    "LIMITED_TIME_OFFER":    "lto",
    "MARKET_OFFER_DIGITAL": "market_digital",
    "LOCAL_OFFER":          "local_offer",
}

# Ordinal severity rank (1-3) used by _tactic_rank_case_sql() to build promo_tactic_rank.
# 0 is reserved for "no promo" (handled separately in the CASE statement); an unrecognised
# tactic falls through to the SQL's ELSE 1 branch. TODO: source from config_feature.params
# per-market like the tactic/subtype column maps above, rather than this static default.
_TACTIC_RANK: dict[str, int] = {
    "Free Offer":                    3,
    "Buy One, Get One - Mix":        3,
    "Buy One, Get One - Match":      3,
    "Meal Bundle":                   2,
    "Mix and Match - Mix":           2,
    "Mix and Match - Match":         2,
    "Pairings (Entree + Side)":      2,
    "Temporary Price Change":        1,
    "No Tactic - Advertising Only":  1,
}


def load_promo_feature_config(
    spark: "SparkSession", catalog: str, market_id: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Load tactic/subtype mappings from the config table for the given market.

    Lookup order:
      1. Market-specific row: market_id=<market>, param_group='feature',
         scope='promo_tactic_counts', param_name='params'
      2. Global fallback row: market_id='_DEFAULT' (same filter otherwise)
      3. Code-level fallback: _FALLBACK_TACTIC_COLUMN_MAP / _FALLBACK_SUBTYPE_COLUMN_MAP

    Returns (tactic_column_map, subtype_column_map).
    """
    import json as _json

    def _try_load(market_key: str) -> dict | None:
        safe_key = market_key.replace("'", "''")
        rows = (
            spark.table(f"{catalog}.{_EVENTS_SCHEMA.replace('gold', 'gold')}.config")
            .filter(f"market_id = '{safe_key}'")
            .filter("param_group = 'feature'")
            .filter("scope = 'promo_tactic_counts'")
            .filter("param_name = 'params'")
            .select("param_value")
            .collect()
        )
        if rows and rows[0]["param_value"]:
            return _json.loads(rows[0]["param_value"])
        return None

    # Try market-specific, then _DEFAULT
    params = _try_load(market_id) or _try_load("_DEFAULT")

    if params:
        tactic_map = params.get("tactic_column_map", _FALLBACK_TACTIC_COLUMN_MAP)
        subtype_map = params.get("subtype_column_map", _FALLBACK_SUBTYPE_COLUMN_MAP)
    else:
        tactic_map = _FALLBACK_TACTIC_COLUMN_MAP
        subtype_map = _FALLBACK_SUBTYPE_COLUMN_MAP

    return tactic_map, subtype_map


def get_promo_feature_columns(
    tactic_map: dict[str, str], subtype_map: dict[str, str]
) -> list[str]:
    """Derive the promo feature column names from the mappings.

    Returns the list of columns that build_promo_features() will produce
    (used by the inference spine to know which columns to expect).
    """
    cols = [
        # Base promo features (always produced)
        "promo_active_flag", "promo_concurrent_count", "promo_days_since_start",
        "promo_days_until_next", "promo_duration",
    ]
    # Tactic count columns
    cols += [f"promo_count_{suffix}" for suffix in tactic_map.values()]
    cols.append("promo_count_tactic_other")
    # Subtype count columns
    cols += [f"promo_count_{suffix}" for suffix in subtype_map.values()]
    cols.append("promo_count_subtype_other")
    # Other promo features
    cols += ["item_specific_promo_count", "promo_days_since_end"]
    return cols

# Holiday severity weights: major (1.0) → national (0.5) → regional (0.25) → minor (0.1).
# These are market-agnostic defaults; override per-market via config_feature.params in future.
# Keeping as a Python dict (not a Delta table) for authoring simplicity — serialised into the SQL
# CASE statement by build_holiday_features so no dynamic table lookup is needed at runtime.
_HOLIDAY_WEIGHT_SQL: str = """
    CASE UPPER(holiday_subtype)
        WHEN 'NATIONAL_HOLIDAY'  THEN 1.0
        WHEN 'PUBLIC_HOLIDAY'    THEN 1.0
        WHEN 'SCHOOL_HOLIDAY'    THEN 0.75
        WHEN 'REGIONAL_HOLIDAY'  THEN 0.5
        WHEN 'OBSERVANCE'        THEN 0.25
        WHEN 'RELIGIOUS'         THEN 0.5
        ELSE 0.5                         -- unknown type: treat as moderate
    END
""".strip()

# Known-at-inference: available for the full 140-day horizon (feature_engineering_spec.md §6).
# The promo portion is derived from config (tactic/subtype maps); holiday portion is static.
# Use get_known_at_inference_columns() for the runtime list (reads config).
# KNOWN_AT_INFERENCE_COLUMNS below is the fallback static list using _FALLBACK_* maps.
_HOLIDAY_KNOWN_AT_INFERENCE: list[str] = [
    "holiday_flag", "holiday_name", "holiday_type", "holiday_subtype",
    "holiday_days_from_event", "holiday_pre_post_window_flag", "bridge_day_flag",
    "holiday_weight",
]

# ═══════════════════════════════════════════════════════════════════════════════
# DISRUPTION (NON_PROMO) FEATURE CONFIG
#
# NON_PROMO events (excluding subtype='HOLIDAY', which is handled by holiday
# features) represent demand disruptions: closures, stock-outs, sales breaks,
# local events, etc. These are known-at-inference (scheduled disruptions are
# visible across the full forecast horizon).
#
# Subtypes are grouped into logical categories that are market-agnostic:
#   closure     = TEMP_CLOSURE | PARTIAL_CLOSURE
#   stock_out   = TEMP_OUT_OF_STOCK
#   sales_break = TEMP_BREAK_IN_SALES
#   local_event = LOCAL_EVENT | (anything else not in above groups)
# ═══════════════════════════════════════════════════════════════════════════════

# Severity weight by subtype (analogous to _HOLIDAY_WEIGHT_SQL).
# closure = full demand loss, stock_out = partial (some items remain), etc.
_DISRUPTION_SEVERITY: dict[str, float] = {
    "TEMP_CLOSURE":          1.0,
    "PARTIAL_CLOSURE":       0.75,
    "TEMP_OUT_OF_STOCK":     0.75,
    "TEMP_BREAK_IN_SALES":   0.5,
    "LOCAL_EVENT":           0.25,
}

_DISRUPTION_KNOWN_AT_INFERENCE: list[str] = [
    "disruption_flag",
    "disruption_severity",
    "disruption_days_elapsed",
]

# Static fallback — derived from _FALLBACK_* maps at import time.
KNOWN_AT_INFERENCE_COLUMNS: list[str] = (
    get_promo_feature_columns(_FALLBACK_TACTIC_COLUMN_MAP, _FALLBACK_SUBTYPE_COLUMN_MAP)
    + _HOLIDAY_KNOWN_AT_INFERENCE
    # Disruption features excluded — closure/stockout/sales-break features were reducing accuracy.
    # + _DISRUPTION_KNOWN_AT_INFERENCE
)


def get_known_at_inference_columns(
    spark: "SparkSession", catalog: str, market_id: str
) -> list[str]:
    """Runtime version: reads config for market-specific tactic/subtype maps.

    Call this instead of KNOWN_AT_INFERENCE_COLUMNS when you have spark available.
    Falls back gracefully if config lookup fails.
    """
    try:
        tactic_map, subtype_map = load_promo_feature_config(spark, catalog, market_id)
    except Exception:
        tactic_map = _FALLBACK_TACTIC_COLUMN_MAP
        subtype_map = _FALLBACK_SUBTYPE_COLUMN_MAP
    return (
        get_promo_feature_columns(tactic_map, subtype_map)
        + _HOLIDAY_KNOWN_AT_INFERENCE
        # Disruption features excluded — closure/stockout/sales-break features were reducing accuracy.
        # + _DISRUPTION_KNOWN_AT_INFERENCE
    )


def _events(spark: "SparkSession", catalog: str, market_id: str, event_type: str) -> "DataFrame":
    """Raw (non-exploded) events rows for `event_type`, filtered to `market_id`, not excluded."""
    safe_market_id = market_id.replace("'", "''")
    safe_event_type = event_type.replace("'", "''")
    return spark.sql(f"""
        SELECT event_id, event_type, event_subtype, event_nm, tactic, UPPER(sku_id) AS sku_id, store_id,
               start_date, end_date, DATEDIFF(end_date, start_date) + 1 AS duration
        FROM {catalog}.{_EVENTS_SCHEMA}.{_EVENTS_TABLE}
        WHERE market_id = '{safe_market_id}'
          AND (upper(event_type) = '{safe_event_type}' OR upper(event_subtype) = '{safe_event_type}')
    """)


def _tactic_rank_case_sql() -> str:
    """Generate a CASE WHEN SQL fragment mapping tactic string → ordinal rank.

    The result is a single SQL expression, inlined into build_promo_features
    so no UDF or Python call happens at scale.
    """
    when_clauses = "\n            ".join(
        f"WHEN UPPER(promo_tactic) = '{k}' THEN {v}"
        for k, v in sorted(_TACTIC_RANK.items(), key=lambda x: -x[1])
    )
    return f"""
        CASE
            WHEN promo_active_flag = 0 OR promo_tactic IS NULL THEN 0
            {when_clauses}
            ELSE 1   -- known-promo but unrecognised tactic: treat as mild
        END
    """.strip()


def build_promo_features(spark: "SparkSession", catalog: str, market_id: str, df: "DataFrame") -> "DataFrame":
    """Join promo features onto `df`, matching each row's date against every promo event's
    [start_date, end_date] range (broadcast: NULL sku_id/store_id on the event matches any).

    New features vs v1:
      promo_lead_flag  — 1 for spine rows within _DEFAULT_PROMO_LEAD_DAYS before a promo
                         start (demand pre-loading window). Known-at-inference (future dates
                         can see upcoming promo schedules).
      promo_tactic_rank — ordinal 0-3 encoding of promo_tactic string (see _TACTIC_RANK).

    `promo_days_until_next` uses a UNION + forward-looking MIN window approach:
      1. Expand each promo's start_date to the set of (store, item) series it applies to
         (handling NULL/broadcast), yielding ~S*P marker rows (S=distinct series, P=promos).
      2. UNION these markers with the spine dates.
      3. A single window pass (MIN of promo markers in FOLLOWING rows) gives the next
         promo start for each spine row — O(N + S*P) instead of O(N*P).
    """
    df.createOrReplaceTempView("_promo_features_input")
    _events(spark, catalog, market_id, "PROMOTION").createOrReplaceTempView("_promo_events")

    lead_days = _DEFAULT_PROMO_LEAD_DAYS

    # ── Load tactic/subtype maps from config (fallback to hardcoded defaults) ──
    try:
        tactic_map, subtype_map = load_promo_feature_config(spark, catalog, market_id)
    except Exception:
        tactic_map = _FALLBACK_TACTIC_COLUMN_MAP
        subtype_map = _FALLBACK_SUBTYPE_COLUMN_MAP

    # ── Dynamically build SQL fragments from config-driven maps ──
    tactic_count_sql_lines = []
    for tactic_str, col_suffix in tactic_map.items():
        safe_tactic = tactic_str.replace("'", "''")
        tactic_count_sql_lines.append(
            f"COUNT(DISTINCT CASE WHEN tactic = '{safe_tactic}' THEN event_id END) AS promo_count_{col_suffix}"
        )
    tactic_not_in_list = ", ".join(f"'{t.replace(chr(39), chr(39)+chr(39))}'" for t in tactic_map)
    tactic_count_sql_lines.append(
        f"COUNT(DISTINCT CASE WHEN tactic NOT IN ({tactic_not_in_list})"
        f" AND tactic IS NOT NULL THEN event_id END) AS promo_count_tactic_other"
    )

    subtype_count_sql_lines = []
    for subtype_str, col_suffix in subtype_map.items():
        safe_subtype = subtype_str.replace("'", "''")
        subtype_count_sql_lines.append(
            f"COUNT(DISTINCT CASE WHEN event_subtype = '{safe_subtype}' THEN event_id END) AS promo_count_{col_suffix}"
        )
    subtype_not_in_list = ", ".join(f"'{s.replace(chr(39), chr(39)+chr(39))}'" for s in subtype_map)
    subtype_count_sql_lines.append(
        f"COUNT(DISTINCT CASE WHEN event_subtype NOT IN ({subtype_not_in_list})"
        f" AND event_subtype IS NOT NULL THEN event_id END) AS promo_count_subtype_other"
    )

    tactic_count_sql = ",\n                   ".join(tactic_count_sql_lines)
    subtype_count_sql = ",\n                   ".join(subtype_count_sql_lines)

    # ── Build COALESCE lines for final SELECT ──
    coalesce_tactic_lines = []
    for col_suffix in tactic_map.values():
        coalesce_tactic_lines.append(f"CASE WHEN COALESCE(a.promo_count_{col_suffix}, 0) > 0 THEN 1 ELSE 0 END AS promo_count_{col_suffix}")
    coalesce_tactic_lines.append("CASE WHEN COALESCE(a.promo_count_tactic_other, 0) > 0 THEN 1 ELSE 0 END AS promo_count_tactic_other")

    coalesce_subtype_lines = []
    for col_suffix in subtype_map.values():
        coalesce_subtype_lines.append(f"CASE WHEN COALESCE(a.promo_count_{col_suffix}, 0) > 0 THEN 1 ELSE 0 END AS promo_count_{col_suffix}")
    coalesce_subtype_lines.append("CASE WHEN COALESCE(a.promo_count_subtype_other, 0) > 0 THEN 1 ELSE 0 END AS promo_count_subtype_other")

    coalesce_tactic_sql = ",\n               ".join(coalesce_tactic_lines)
    coalesce_subtype_sql = ",\n               ".join(coalesce_subtype_lines)

    return spark.sql(f"""
        -- Current-promo features: which promos are active on each spine row's date
        WITH matched AS (
            SELECT t.store_id, t.sku_id, t.date_id,
                   pe.event_id, pe.event_nm, pe.event_subtype, pe.tactic, pe.duration, pe.start_date,
                   pe.sku_id AS event_sku_id
            FROM _promo_features_input t
            JOIN _promo_events pe
              ON (pe.sku_id = t.sku_id)
             AND (pe.store_id = t.store_id)
             AND t.date_id BETWEEN pe.start_date AND pe.end_date
        ),
        agg AS (
            SELECT store_id, sku_id, date_id,
                   COUNT(DISTINCT event_id) AS promo_concurrent_count,
                   -- item_specific_promo_count: targeted promos only (sku_id IS NOT NULL).
                   COUNT(DISTINCT CASE WHEN event_sku_id IS NOT NULL AND event_sku_id != ''
                                       THEN event_id END) AS item_specific_promo_count,
                   MIN(DATEDIFF(date_id, start_date)) AS promo_days_since_start,
                   MAX(duration) AS promo_duration,
                   -- tactic-level counts (config-driven)
                   {tactic_count_sql},
                   -- subtype-level counts (config-driven)
                   {subtype_count_sql}
            FROM matched
            GROUP BY store_id, sku_id, date_id
        ),

        -- Next-promo feature via UNION + forward-looking MIN window.
        -- Step A: expand each promo start_date to the series it applies to.
        promo_starts AS (
            SELECT DISTINCT
                s.store_id, s.sku_id, pe.start_date AS promo_start_date
            FROM (SELECT DISTINCT store_id, sku_id FROM _promo_features_input) s
            JOIN _promo_events pe
              ON pe.sku_id = s.sku_id
             AND pe.store_id = s.store_id
        ),
        -- Step B: UNION spine dates with promo-start markers.
        spine_with_markers AS (
            SELECT store_id, sku_id, date_id, 1 AS is_spine
            FROM _promo_features_input
            UNION ALL
            SELECT store_id, sku_id, promo_start_date AS date_id, 0 AS is_spine
            FROM promo_starts
        ),
        -- Step C: forward-looking MIN window finds the next promo start after each row.
        -- Sort order: promo markers (is_spine=0) BEFORE spine rows (is_spine=1) on the
        -- same date, so ROWS BETWEEN 1 FOLLOWING from a spine row at date D never
        -- includes a promo also starting on date D (matches the original > semantics).
        next_promo AS (
            SELECT store_id, sku_id, date_id, next_start_date
            FROM (
                SELECT store_id, sku_id, date_id, is_spine,
                       MIN(CASE WHEN is_spine = 0 THEN date_id END)
                           OVER (PARTITION BY store_id, sku_id
                                 ORDER BY date_id, is_spine
                                 ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
                           ) AS next_start_date
                FROM spine_with_markers
            )
            WHERE is_spine = 1
        ),

        -- Step D: promo lead flag — rows within N days BEFORE a promo start.
        -- Captures demand pre-loading: planners often see sales lift 2-3 days
        -- before a promotion officially begins.
        -- Note: rows already inside a promo (promo_active_flag=1) are NOT lead days.
        lead_flags AS (
            SELECT DISTINCT
                s.store_id, s.sku_id, s.date_id,
                1 AS promo_lead_flag
            FROM (SELECT DISTINCT store_id, sku_id, date_id FROM _promo_features_input) s
            JOIN _promo_events pe
              ON pe.sku_id = s.sku_id
             AND pe.store_id = s.store_id
             AND s.date_id BETWEEN DATE_SUB(pe.start_date, {lead_days}) AND DATE_SUB(pe.start_date, 1)
        ),

        -- Last-promo-end: backward-looking MAX of promo end_dates per series.
        -- Mirrors promo_days_until_next (forward MIN of start_dates) but looks backward at
        -- the most recent promo expiry. Captures demand decay / return-to-baseline after
        -- a promotion expires (e.g. a buy-one-get-one deal ends; demand drops back to
        -- normal but a model trained during the promo over-forecasts the following week).
        --
        -- Step E: expand each promo end_date to the series it applies to.
        promo_ends AS (
            SELECT DISTINCT
                s.store_id, s.sku_id, pe.end_date AS promo_end_date
            FROM (SELECT DISTINCT store_id, sku_id FROM _promo_features_input) s
            JOIN _promo_events pe
              ON pe.sku_id = s.sku_id
             AND pe.store_id = s.store_id
        ),
        -- Step F: UNION spine dates with promo-end markers.
        -- Sort order: spine rows (is_spine=1) BEFORE markers (is_spine=0) on the same date
        -- (ORDER BY date_id, is_spine DESC), so a promo ending on date D is NOT visible to
        -- the backward window for the spine row at D — D is still covered by the active-promo
        -- join and returns 0. Starting from D+1, the end marker at D enters the window and
        -- DATEDIFF gives the correct days-since-end count.
        spine_with_end_markers AS (
            SELECT store_id, sku_id, date_id, 1 AS is_spine FROM _promo_features_input
            UNION ALL
            SELECT store_id, sku_id, promo_end_date AS date_id, 0 AS is_spine FROM promo_ends
        ),
        -- Step G: backward-looking MAX window finds the most recent promo end before each row.
        last_promo_end AS (
            SELECT store_id, sku_id, date_id, last_end_date
            FROM (
                SELECT store_id, sku_id, date_id, is_spine,
                       MAX(CASE WHEN is_spine = 0 THEN date_id END)
                           OVER (PARTITION BY store_id, sku_id
                                 ORDER BY date_id, is_spine DESC
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                           ) AS last_end_date
                FROM spine_with_end_markers
            )
            WHERE is_spine = 1
        )

        SELECT t.*,
               CASE WHEN a.store_id IS NOT NULL THEN 1 ELSE 0 END AS promo_active_flag,
               LN(1 + COALESCE(a.promo_concurrent_count, 0)) AS promo_concurrent_count,
               COALESCE(a.item_specific_promo_count, 0) AS item_specific_promo_count,
               a.promo_days_since_start, a.promo_duration,
               DATEDIFF(np.next_start_date, t.date_id) AS promo_days_until_next,
               -- Tactic-count features (config-driven)
               {coalesce_tactic_sql},
               -- Subtype-count features (config-driven)
               {coalesce_subtype_sql},

               -- promo_days_since_end: 0 while a promo is active (covered by agg join above);
               -- days since the most recent promo ended for dates with no active promo.
               -- NULL when no prior promo has ever been active for this (store, item) series.
               CASE WHEN a.store_id IS NOT NULL THEN 0
                    ELSE DATEDIFF(t.date_id, le.last_end_date)
               END AS promo_days_since_end
        FROM _promo_features_input t
        LEFT JOIN agg a
          ON t.store_id = a.store_id AND t.sku_id = a.sku_id AND t.date_id = a.date_id
        LEFT JOIN next_promo np
          ON t.store_id = np.store_id AND t.sku_id = np.sku_id AND t.date_id = np.date_id
        LEFT JOIN lead_flags lf
          ON t.store_id = lf.store_id AND t.sku_id = lf.sku_id AND t.date_id = lf.date_id
        LEFT JOIN last_promo_end le
          ON t.store_id = le.store_id AND t.sku_id = le.sku_id AND t.date_id = le.date_id
    """)


def build_holiday_features(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    df: "DataFrame",
    window_days: int = _DEFAULT_HOLIDAY_WINDOW_DAYS,
) -> "DataFrame":
    """Join holiday features onto `df`, matching each row against holiday events within
    +/-`window_days` of [start_date, end_date] (broadcast rule as build_promo_features).

    New feature vs v1:
      holiday_weight — float severity weight (1.0 = major national holiday, down to 0.25
                       for minor observances). Derived from holiday_subtype using
                       _HOLIDAY_WEIGHT_SQL. Lets LGBM/Elastic Net learn that Christmas
                       and Easter have different demand signatures than a bank holiday.
                       Value is 0.0 outside the pre/post window (no holiday effect).

    `bridge_day_flag` is implemented via two small CTEs (holiday_dates, bridge_dates)
    derived entirely from `_holiday_events`. A date is a bridge if it is NOT itself a
    holiday AND has at least one holiday within 2 days BEFORE it AND at least one within
    2 days AFTER it. Holiday dates are extracted by deduplicating on (start_date, end_date)
    across all event rows (regardless of sku_id/store_id), so the logic works for
    both broadcast markets (PL: NULL item/store = national) and item-store markets (GE: one
    row per item×store). No weekend_flag dependency is needed.
    """
    df.createOrReplaceTempView("_holiday_features_input")
    _events(spark, catalog, market_id, "HOLIDAY").createOrReplaceTempView("_holiday_events")
    return spark.sql(f"""
        WITH matched AS (
            SELECT t.store_id, t.sku_id, t.date_id,
                   he.event_id, he.event_nm, he.event_type, he.event_subtype, he.start_date, he.end_date
            FROM _holiday_features_input t
            JOIN _holiday_events he
              ON he.sku_id = t.sku_id
             AND he.store_id = t.store_id
             AND t.date_id BETWEEN DATE_SUB(he.start_date, {window_days}) AND DATE_ADD(he.end_date, {window_days})
        ),
        agg AS (
            SELECT store_id, sku_id, date_id,
                   MAX(CASE WHEN date_id BETWEEN start_date AND end_date THEN 1 ELSE 0 END) AS holiday_flag,
                   MAX(event_nm) AS holiday_name,
                   MAX(event_type) AS holiday_type,
                   MAX(event_subtype) AS holiday_subtype,
                   MIN(
                       CASE WHEN date_id < start_date THEN DATEDIFF(date_id, start_date)
                            WHEN date_id > end_date THEN DATEDIFF(date_id, end_date)
                            ELSE 0 END
                   ) AS holiday_days_from_event
            FROM matched
            GROUP BY store_id, sku_id, date_id
        ),

        -- Bridge-day flag: a non-holiday date flanked by holidays within 2 days each side.
        -- Example: PL 2026 — May 2 (Saturday) sits between Labour Day (May 1, Friday)
        -- and Constitution Day (May 3, Sunday). Both holidays are 1 day away, so May 2
        -- is a bridge day even though it also falls on a weekend.
        -- Deduplicates on (start_date, end_date) so this works for BOTH broadcast markets
        -- (PL: sku_id/store_id NULL) and item-store markets (GE: one row per
        -- item×store — the same holiday appears many times but with the same dates).
        -- No weekend_flag dependency is needed.
        holiday_dates AS (
            SELECT DISTINCT hd AS holiday_date
            FROM (
                SELECT EXPLODE(SEQUENCE(start_date, end_date, INTERVAL 1 DAY)) AS hd
                FROM (SELECT DISTINCT start_date, end_date FROM _holiday_events)
            )
        ),
        bridge_dates AS (
            SELECT DISTINCT s.date_id
            FROM (SELECT DISTINCT date_id FROM _holiday_features_input) s
            -- Spine date must not be a holiday itself
            LEFT JOIN  holiday_dates hd_self   ON hd_self.holiday_date  = s.date_id
            -- Must have at least one holiday within 2 days before
            INNER JOIN holiday_dates hd_before ON hd_before.holiday_date
                           BETWEEN DATE_SUB(s.date_id, 2) AND DATE_SUB(s.date_id, 1)
            -- Must have at least one holiday within 2 days after
            INNER JOIN holiday_dates hd_after  ON hd_after.holiday_date
                           BETWEEN DATE_ADD(s.date_id, 1) AND DATE_ADD(s.date_id, 2)
            WHERE hd_self.holiday_date IS NULL
        )
        SELECT t.*,
               COALESCE(a.holiday_flag, 0) AS holiday_flag,
               a.holiday_name, a.holiday_type, a.holiday_subtype,
               a.holiday_days_from_event,
               CASE WHEN a.holiday_days_from_event IS NOT NULL AND a.holiday_days_from_event <> 0
                    THEN 1 ELSE 0 END AS holiday_pre_post_window_flag,
               CASE WHEN bd.date_id IS NOT NULL THEN 1 ELSE 0 END AS bridge_day_flag,
               -- New v2: severity weight — 0.0 when no holiday in window, else by subtype.
               -- CAST(...AS DOUBLE): Spark SQL infers decimal-point literals (1.0, 0.75, ...)
               -- as DecimalType, not DoubleType, so without this the column lands in the Delta
               -- table itself as DECIMAL — every downstream reader (not just one model wrapper)
               -- gets object-dtype decimal.Decimal in pandas and crashes on any bare
               -- float arithmetic. Cast explicitly so the schema is DOUBLE at the source.
               CAST(
                   CASE WHEN a.holiday_days_from_event IS NULL THEN 0.0
                        ELSE {_HOLIDAY_WEIGHT_SQL}
                   END
               AS DOUBLE) AS holiday_weight
        FROM _holiday_features_input t
        LEFT JOIN agg a
          ON t.store_id = a.store_id AND t.sku_id = a.sku_id AND t.date_id = a.date_id
        LEFT JOIN bridge_dates bd ON t.date_id = bd.date_id
    """)


def build_disruption_features(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    df: "DataFrame",
) -> "DataFrame":
    """Join disruption (NON_PROMO) features onto `df`.

    Produces 3 concise features (known-at-inference):
      disruption_flag          — 1 if any non-holiday NON_PROMO event is active
      disruption_severity      — float weight by subtype (1.0 = full closure,
                                 0.75 = partial/stock-out, 0.5 = break, 0.25 = local).
                                 Takes MAX when multiple overlap. 0.0 if no disruption.
      disruption_days_elapsed  — days since the most severe active disruption started
                                 (NULL if no disruption active)

    Market-agnostic: severity is mapped from event_subtype via _DISRUPTION_SEVERITY.
    Unknown subtypes default to 0.25 (local_event level). Markets with no NON_PROMO
    events (or only HOLIDAY subtype) get all-zero features.
    """
    df.createOrReplaceTempView("_disruption_features_input")
    _events(spark, catalog, market_id, "NON_PROMO").createOrReplaceTempView("_non_promo_events_raw")

    # Build CASE WHEN for severity from _DISRUPTION_SEVERITY dict
    severity_whens = "\n            ".join(
        f"WHEN UPPER(event_subtype) = '{subtype}' THEN {weight}"
        for subtype, weight in _DISRUPTION_SEVERITY.items()
    )
    severity_sql = f"""
        CASE
            {severity_whens}
            ELSE 0.25
        END
    """.strip()

    return spark.sql(f"""
        WITH disruption_events AS (
            SELECT *
            FROM _non_promo_events_raw
            WHERE UPPER(COALESCE(event_subtype, '')) != 'HOLIDAY'
        ),

        matched AS (
            SELECT t.store_id, t.sku_id, t.date_id,
                   de.event_id, de.event_subtype, de.start_date,
                   CAST({severity_sql} AS DOUBLE) AS severity
            FROM _disruption_features_input t
            JOIN disruption_events de
              ON (de.sku_id = t.sku_id)
             AND (de.store_id = t.store_id)
             AND t.date_id BETWEEN de.start_date AND de.end_date
        ),

        agg AS (
            SELECT store_id, sku_id, date_id,
                   MAX(severity) AS disruption_severity,
                   MIN(DATEDIFF(date_id, start_date)) AS disruption_days_elapsed
            FROM matched
            GROUP BY store_id, sku_id, date_id
        )

        SELECT t.*,
               CASE WHEN a.store_id IS NOT NULL THEN 1 ELSE 0 END AS disruption_flag,
               CAST(COALESCE(a.disruption_severity, 0.0) AS DOUBLE) AS disruption_severity,
               a.disruption_days_elapsed
        FROM _disruption_features_input t
        LEFT JOIN agg a
          ON t.store_id = a.store_id AND t.sku_id = a.sku_id AND t.date_id = a.date_id
    """)
