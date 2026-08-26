"""
data/features/writer.py — write mode (feature_engineering_spec.md "Write mode").

Stamps the built features with `snapshot_date`, `market_id` (market), and
`updated_ts` and writes into sku_store_features via Feature Store.

The table is registered as a time series feature table with:
  PRIMARY KEY (market_id, store_id, sku_id, date_id, snapshot_date TIMESERIES)
so fe.write_table(mode='merge') merges on all 5 columns — same behavior as
the previous merge_upsert, but with full Feature Store lineage tracking.

`market_id` captures the market code (the market_id filter applied to
all source reads) so rows from multi-market job runs remain filterable in the
single shared feature table. It is included in the PK so that runs for
different markets never collide even if the same store_id appears in two markets.

`updated_ts` is stamped as current_timestamp() on every write so callers can
distinguish the original INSERT from a reprocess (UPDATE). `created_ts` is a
column DEFAULT in the DDL and is deliberately NOT included in the source
DataFrame — Delta only fires the DEFAULT on INSERT, preserving the original
insertion time even after reprocessing.

Existing tables (sku_store_features) are never touched.
"""
from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from utils.dates import write_window

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

_SCHEMA = "gold"
_TABLE = "sku_store_features"


def write_features(
    spark: "SparkSession",
    features_df: "DataFrame",
    catalog: str,
    snapshot_date: dt.date,
    incremental_write_days: int,
    market_id: str,
) -> None:
    """Stamp `features_df` with `snapshot_date`, `market_id` (market), and
    `updated_ts` and idempotently MERGE only the incremental write window into
    `{catalog}.gold.sku_store_features`.

    We still compute features from the longer read window (needed for lag/rolling
    correctness), but persist only the last `incremental_write_days` days for
    this snapshot. Older dates are expected to already exist from prior runs or
    from an initial historical backfill.

    `market_id` = `market` (e.g. 'PL') stored on every row so consumers
    can filter or partition the feature table by market without joining back to a
    separate mapping. Included in the PK so multi-market runs never cross-update
    each other's rows.

    `updated_ts` = current_timestamp() on every write — advances on reprocess so
    callers can detect which rows were re-written and when.

    `created_ts` is preserved for existing rows via COALESCE with the current
    table value. Only genuinely new rows (no match on all 5 PK columns) receive
    current_timestamp(). This matches the original column-DEFAULT semantics.

    After building the stamped DataFrame, all columns are cast to match the
    target table's exact schema (e.g. qty→float, isActive→int) because Spark SQL
    infers wider types (double, bigint) that fe.write_table() would reject.

    Feature Store write_table(mode='merge') merges on the full PK
    (market_id, store_id, sku_id, date_id, snapshot_date) since
    snapshot_date is marked TIMESERIES. Multiple snapshots coexist for replay.

    Write lineage is automatically tracked in Unity Catalog: the Feature Store UI
    shows which job/notebook produced each write.
    """
    from pyspark.sql import functions as F

    safe_market_id = market_id.replace("'", "''")
    write_start, write_end = write_window(snapshot_date, incremental_write_days)
    full_table = f"{catalog}.{_SCHEMA}.{_TABLE}"

    # Stamp snapshot_date, market, and updated_ts — do NOT include created_ts.
    # Delta MERGE semantics handle it correctly without a full-table scan:
    #   WHEN MATCHED UPDATE SET *  → leaves created_ts untouched (not in source)
    #   WHEN NOT MATCHED INSERT *  → uses the column DEFAULT (current_timestamp())
    # The previous approach LEFT-JOINed the entire sku_store_features table
    # (all snapshots × all markets, potentially 370M+ rows) just to recover
    # created_ts for existing rows — that caused O(N) full-table scans on every
    # incremental write and is the root cause of the feature build "blowing up".
    features_df.createOrReplaceTempView("_features_to_write")
    stamped_df = spark.sql(f"""
        SELECT DATE('{snapshot_date.isoformat()}') AS snapshot_date,
               '{safe_market_id}'                  AS market_id,
               current_timestamp()                 AS updated_ts,
               *
        FROM _features_to_write
        WHERE date_id BETWEEN DATE('{write_start.isoformat()}') AND DATE('{write_end.isoformat()}')
    """)

    # Cast DataFrame columns to match the target table's schema where possible.
    # Columns that exist in BOTH source and target are cast to the target type.
    # Columns that exist ONLY in source (new features) pass through uncast —
    # Delta auto-merge will add them to the schema during INSERT.
    target_schema = spark.table(full_table).schema
    _target_field_map = {f.name: f.dataType for f in target_schema}
    cast_exprs = []
    seen = set()
    for col_name in stamped_df.columns:
        if col_name in seen:
            continue
        seen.add(col_name)
        if col_name in _target_field_map:
            cast_exprs.append(F.col(col_name).cast(_target_field_map[col_name]))
        else:
            cast_exprs.append(F.col(col_name))  # new column — keep native type
    stamped_df = stamped_df.select(cast_exprs)

    # Plain Delta MERGE on the 5-column PK.
    # Neither UPDATE SET * nor INSERT * can be used here: Delta resolves ALL
    # target columns for both clauses, including created_ts — which is
    # intentionally absent from src (DDL DEFAULT only, preserves original
    # insertion time on reprocess). Build explicit column lists from
    # stamped_df.columns so created_ts is never referenced in either clause:
    #   UPDATE: columns not in SET are left unchanged (created_ts untouched).
    #   INSERT: columns not listed receive their DDL DEFAULT (current_timestamp()).
    #
    # UPDATE SET can only reference columns that ALREADY exist in the target;
    # new source columns not yet in the target are included in INSERT only —
    # auto-merge adds them to the schema during INSERT of new rows.
    # Matched rows get a follow-up UPDATE for new columns after the MERGE.
    _all_src_cols = stamped_df.columns
    _target_col_names = {f.name for f in target_schema}
    _existing_cols = [c for c in _all_src_cols if c in _target_col_names]
    _set_clause   = ",\n            ".join(f"tgt.{c} = src.{c}" for c in _existing_cols)
    _insert_cols  = ", ".join(_all_src_cols)
    _insert_vals  = ", ".join(f"src.{c}" for c in _all_src_cols)

    stamped_df.createOrReplaceTempView("_features_stamped")
    # Enable schema auto-merge so INSERT can add new columns to the target table.
    spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
    spark.sql(f"""
        MERGE INTO {full_table} tgt
        USING _features_stamped src
        ON  tgt.market_id      = src.market_id
        AND tgt.store_id       = src.store_id
        AND tgt.sku_id         = src.sku_id
        AND tgt.date_id           = src.date_id
        AND tgt.snapshot_date     = src.snapshot_date
        WHEN MATCHED THEN UPDATE SET
            {_set_clause}
        WHEN NOT MATCHED THEN INSERT ({_insert_cols})
            VALUES ({_insert_vals})
    """)

    # For truly new columns (in source but not yet in target before this MERGE),
    # the INSERT clause + auto-merge added them to the schema and populated new
    # rows. But MATCHED rows were only updated with existing-target columns above.
    # Back-fill the new column values for matched rows via a targeted UPDATE.
    _new_cols = [c for c in _all_src_cols if c not in _target_col_names]
    if _new_cols:
        _new_set = ", ".join(f"tgt.{c} = src.{c}" for c in _new_cols)
        spark.sql(f"""
            MERGE INTO {full_table} tgt
            USING _features_stamped src
            ON  tgt.market_id      = src.market_id
            AND tgt.store_id       = src.store_id
            AND tgt.sku_id         = src.sku_id
            AND tgt.date_id        = src.date_id
            AND tgt.snapshot_date  = src.snapshot_date
            WHEN MATCHED THEN UPDATE SET
                {_new_set}
        """)
