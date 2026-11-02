# Feature Engineering Spec — `gold.sku_store_features`

> **Do NOT modify any existing table.** All sources (sales, sku_store, sku, store, events, calendar) are **read-only**. This pipeline writes ONLY to the new parallel `gold.sku_store_features`.
> **This is the DATA job** — separate from training and inference. Parameterized by `snapshot_date`; features are built **point-in-time as of that date** (only source rows with dt ≤ snapshot_date), so any historical snapshot can be replayed. `snapshot_date` is a column on the output.
>
> ROLE: senior Databricks data engineer building a production PySpark feature pipeline on Unity Catalog. Modular, idempotent, parameterized, leakage-safe. Session injected, never created at import. Authored locally (no Spark run); tested on pandas fixtures. `market_id` and `snapshot_date` are parameters applied everywhere — nothing hardcoded.

## Objective
Build `${catalog}.gold.sku_store_features` at grain `store_id (store) × sku_id (sku) × date_id`. Target = `qty`. Daily, incremental writes.

## Parameters
Read from the **config Delta table** (filtered by `market_id`), not YAML: `market_id` (e.g. 'CA' — Kaggle M5's state_id); `snapshot_date` (as-of date; default `current_date()`, overridable to any historical date); `history_start`; `incremental_read_days` (≥400 so t-365 lags & YoY compute); `incremental_write_days` (write last 365). All source reads and lifecycle windows are anchored on `snapshot_date`, never `current_date()` directly.

## Source tables
SALES `source.sales`; SKU_STORE `source.sku_store`; SKU `source.sku`; STORE `source.store`; EVENTS `gold.events`; DATE `gold.calendar_date`.

## Step 1 — Sales base (target)
```sql
SELECT store_id, UPPER(sku_id) AS sku_id,
       CAST(date_id AS DATE) AS date_id,
       CAST(SUM(NVL(units_sold,0)+NVL(units_promotional,0)+NVL(units_shrinkage,0)) AS FLOAT) AS qty
FROM   ${catalog}.source.sales
WHERE  UPPER(market_id)=:market
  AND  CAST(date_id AS DATE) BETWEEN CAST(:history_start AS DATE) AND CAST(:current_date AS DATE)
GROUP BY ALL
```
Standardize keys: `store_id`, `sku_id` (UPPER), `date_id`.

## Step 2 — Active window
Read `source.sku_store` directly (no item/location "type" filtering — generic retail has no BMI/REST-style item/location type distinction); filter market and `active_from<>'1970-01-01'`. `window_active=1` when `active_from<=date_id<=active_to`.

## Step 3 — Lifecycle flags
`is_new_store=1` if store has <365 distinct sales days. `is_new_sku=1` at MARKET level over the 400-day window if sku has <365 distinct sales days; broadcast to all its store rows.

## Step 4 — isActive
`isActive=1` ONLY IF `window_active=1 AND is_new_store=0 AND is_new_sku=0`.

## Step 5 — Gap fill (true zeros)
Dense active-day spine per sku-store `[active_from,active_to]` × calendar; left-join sales; fill `qty=0` ONLY where `isActive=1` and no sales row. Never zero-pad `isActive=0` / new / seasonal / inactive periods.

## Step 6 — Feature groups
**Statistical (leakage-safe, exclude current row):** lags t-1/7/14/28/365; rolling mean/std/min/max/median 7/14/28; EWMA 7,28; expanding mean/std; same-weekday-last-year; YoY ratio; trailing zero-sales ratio.
**Temporal (join `gold.calendar_date`):** DOW, week-of-year, month, quarter, fiscal period, day-of-month, payday flag, weekend flag, season. Prefer existing columns.
**Events (`gold.events`; cols market_id, event_id, event_type, event_subtype, event_nm, start_date, end_date, sku_id, store_id, exclude_flag, tactic):** filter by `market_id`; join sku_id→sku_id, store_id→store_id, date-in-range using broadcast rule (`NULL sku_id/store_id` on the event matches ANY spine row — some markets store events at broadcast/national level, others at sku-store level; the join logic handles both transparently). Honor `exclude_flag`. Split by `event_type`:
- Promo: active flag, type/subtype/tactic, duration, days-since-start, days-until-next, concurrent-promo count, tactic rank (ordinal 0-3), lead flag (N days before start), sku-specific promo count, days-since-end.
- Holiday: flag, name, type/subtype, pre/post ±N-day windows, bridge-day flag, holiday_weight (severity float by subtype).

> Feature lists are a baseline. Extend with: moving-holiday/bridge/closure logic; sub-events (bundle cannibalization); weather (TBD); social/external (TBD); store profile; sku attributes; price signals.

## Leakage control
Tag each feature known-at-inference (promo/holiday/calendar) vs historical-only (lags/rolling). Never compute rolling/lag from current/future rows. Critical for the multi-step horizon.

## Write mode
Read last 400 days ≤ snapshot_date; MERGE-upsert into `sku_store_features` on keys (**snapshot_date**, store_id, sku_id, date_id) so multiple snapshots coexist for replay. Idempotent, Delta, parameterized. Existing tables untouched.

## Output contract
Keys **snapshot_date**/store_id/sku_id/date_id; target qty; flags window_active,is_new_store,is_new_sku,isActive; all statistical/temporal/promo/holiday/other features. Liquid Clustering `CLUSTER BY (snapshot_date, date_id, store_id, sku_id)`; COMMENTs; NOT NULL keys; CHECK(qty>=0).
