-- =============================================================================
-- data/ddl/migrate_add_halo_cannibalization.sql
--
-- Migration: add halo/cannibalization features (data/features/halo_cannibalization_features.py)
--   1. New columns on sku_store_features (JOB 1's feature table).
--   2. New table halo_cannib_partners — the discovered per-sku halo (complement) /
--      cannibalization (substitute) partner list, persisted so JOB 3's future spine can
--      reconstruct the KNOWN_AT_INFERENCE columns without re-reading source tables
--      (docs/instructions.md §7 lock #3: JOB 3 is scoring-only).
--
-- SAFE TO RE-RUN: ADD COLUMN IF NOT EXISTS / CREATE TABLE IF NOT EXISTS are idempotent
-- on Delta tables. RUN IN DATABRICKS (not executed locally).
--
-- Existing tables are otherwise untouched: only ADD COLUMN (nullable, no default change
-- to existing rows) on sku_store_features, and a brand-new table.
--
-- Parameterize ${catalog} — swap in your Free Edition workspace's catalog name.
-- =============================================================================

-- 1. sku_store_features — 6 KNOWN_AT_INFERENCE + 2 HISTORICAL_ONLY columns.
-- ---------------------------------------------------------------------------
ALTER TABLE ${catalog}.gold.sku_store_features
    ADD COLUMN IF NOT EXISTS halo_partner_qty_7d_roll FLOAT
    COMMENT 'HISTORICAL_ONLY. Rolling 7d realized qty (excl. current row) of this sku\'s top-K halo (complement) partners at the same store, per data/features/halo_cannibalization_features.py. Never present on the future spine.';

ALTER TABLE ${catalog}.gold.sku_store_features
    ADD COLUMN IF NOT EXISTS cannib_partner_qty_7d_roll FLOAT
    COMMENT 'HISTORICAL_ONLY. Rolling 7d realized qty (excl. current row) of this sku\'s top-K cannibalization (substitute) partners at the same store. Never present on the future spine.';

ALTER TABLE ${catalog}.gold.sku_store_features
    ADD COLUMN IF NOT EXISTS halo_partner_corr_strength FLOAT
    COMMENT 'KNOWN_AT_INFERENCE. Avg demand-surprise correlation with this sku\'s halo partners as of the snapshot that discovered them (structural, fixed across the 140-day horizon). 0.0 if no qualifying partner found.';

ALTER TABLE ${catalog}.gold.sku_store_features
    ADD COLUMN IF NOT EXISTS cannib_partner_corr_strength FLOAT
    COMMENT 'KNOWN_AT_INFERENCE. Avg |demand-surprise correlation| with this sku\'s cannibalization partners as of the snapshot that discovered them. 0.0 if no qualifying partner found.';

ALTER TABLE ${catalog}.gold.sku_store_features
    ADD COLUMN IF NOT EXISTS halo_partner_promo_active_flag INT
    COMMENT 'KNOWN_AT_INFERENCE. 1 if >=1 halo partner has an active promo at this store/date_id (promos known 2mo-1yr ahead), else 0.';

ALTER TABLE ${catalog}.gold.sku_store_features
    ADD COLUMN IF NOT EXISTS halo_partner_promo_count INT
    COMMENT 'KNOWN_AT_INFERENCE. Count of distinct halo partners with an active promo at this store/date_id.';

ALTER TABLE ${catalog}.gold.sku_store_features
    ADD COLUMN IF NOT EXISTS cannib_partner_promo_active_flag INT
    COMMENT 'KNOWN_AT_INFERENCE. 1 if >=1 cannibalization (substitute) partner has an active promo at this store/date_id — the primary cannibalization risk signal, else 0.';

ALTER TABLE ${catalog}.gold.sku_store_features
    ADD COLUMN IF NOT EXISTS cannib_partner_promo_count INT
    COMMENT 'KNOWN_AT_INFERENCE. Count of distinct cannibalization (substitute) partners with an active promo at this store/date_id.';

-- ---------------------------------------------------------------------------
-- 2. halo_cannib_partners — new small table, one row per (snapshot_date,
--    market, sku_id, partner_sku_id, relation_type). Bounded size:
--    at most _MAX_CORR_UNIVERSE_ITEMS x _TOP_K_PARTNERS x 2 rows per market per
--    snapshot (see data/features/halo_cannibalization_features.py constants).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ${catalog}.gold.halo_cannib_partners (
  snapshot_date    DATE    NOT NULL COMMENT 'As-of date the partner relationship was discovered from (data <= snapshot_date only).',
  market_id        STRING  NOT NULL,
  sku_id           STRING  NOT NULL COMMENT 'Focal sku.',
  partner_sku_id   STRING  NOT NULL COMMENT 'The halo/cannibalization partner sku.',
  relation_type    STRING  NOT NULL COMMENT '''halo'' (complement, positive demand-surprise correlation) or ''cannib'' (substitute, negative correlation).',
  corr             DOUBLE  COMMENT 'Signed Pearson correlation of market-wide demand-surprise between sku_id and partner_sku_id over the trailing correlation lookback window.',
  n_overlap_days   INT     COMMENT 'Number of co-observed days the correlation was computed over (>= _MIN_OVERLAP_DAYS).',
  updated_ts       TIMESTAMP DEFAULT current_timestamp(),
  CONSTRAINT halo_cannib_partners_relation_type_chk CHECK (relation_type IN ('halo', 'cannib'))
) USING DELTA CLUSTER BY (market_id, snapshot_date, sku_id)
COMMENT 'Per-market, per-snapshot halo (complement) / cannibalization (substitute) partner list, discovered purely from realized demand co-movement (data/features/halo_cannibalization_features.py). Written by JOB 1; read by JOB 1 (to attach KNOWN_AT_INFERENCE columns) and JOB 3\'s future spine (inference/future_spine.py), which cannot recompute correlations itself.'
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');

-- ---------------------------------------------------------------------------
-- Verification: confirm the new columns/table are present.
-- ---------------------------------------------------------------------------
SELECT
    'sku_store_features'      AS tbl,
    COUNT(*)                  AS total_rows,
    COUNT(halo_partner_corr_strength)   AS rows_with_halo_corr_strength,
    COUNT(cannib_partner_promo_count)   AS rows_with_cannib_promo_count
FROM ${catalog}.gold.sku_store_features

UNION ALL

SELECT
    'halo_cannib_partners'  AS tbl,
    COUNT(*)                AS total_rows,
    COUNT(CASE WHEN relation_type = 'halo'   THEN 1 END) AS rows_with_halo_corr_strength,
    COUNT(CASE WHEN relation_type = 'cannib' THEN 1 END) AS rows_with_cannib_promo_count
FROM ${catalog}.gold.halo_cannib_partners;
