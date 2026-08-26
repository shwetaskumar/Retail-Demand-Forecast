"""
utils/io.py — Delta read/write helpers, session injected (docs/instructions.md §18).

Three-level UC names only; idempotent MERGE; no collect()/toPandas() on full
data. All three jobs use these helpers instead of writing raw Spark I/O inline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def read_table(spark: "SparkSession", catalog: str, schema: str, table: str) -> "DataFrame":
    """Read the three-level UC table `catalog.schema.table` as a DataFrame."""
    return spark.table(f"{catalog}.{schema}.{table}")


def merge_upsert(
    spark: "SparkSession",
    df: "DataFrame",
    catalog: str,
    schema: str,
    table: str,
    merge_keys: Sequence[str],
    merge_schema: bool = False,
) -> None:
    """Idempotent MERGE of `df` into `catalog.schema.table` on `merge_keys`.

    Used by JOB 1 (sku_store_features), JOB 2 (champion_index,
    backtest_metrics), and JOB 3 (output, decomposition, dq_audit).
    `delta.tables` is imported lazily so this module stays importable without it.

    `merge_schema=True` allows new columns in `df` to widen the target table
    (`.withSchemaEvolution()`, Delta 3.x+/DBR 14+) — needed for
    sku_store_features, whose feature columns grow as data/features/* adds
    more; leave False for tables with a fixed contract (e.g. output).

    Only the columns present in `df` are updated/inserted.  Target-only columns
    absent from the source (e.g. `created_ts` defined as a column DEFAULT in the
    DDL) are left untouched on UPDATE and receive their DEFAULT / NULL on INSERT
    via the Delta column-defaults feature — so callers do not need to supply every
    column in the target schema.

    For sku_store_features specifically:
      - `updated_ts` IS in df.columns (stamped as current_timestamp() in writer.py)
        → updated on every MERGE action, both INSERT and reprocess UPDATE.
      - `created_ts` is NOT in df.columns (DDL DEFAULT only)
        → set once on first INSERT, never touched on reprocess UPDATE.
    """
    from delta.tables import DeltaTable

    full_name = f"{catalog}.{schema}.{table}"
    merge_condition = " AND ".join(f"target.{key} <=> source.{key}" for key in merge_keys)
    dt = DeltaTable.forName(spark, full_name)
    merge_builder = dt.alias("target").merge(df.alias("source"), merge_condition)
    if merge_schema:
        merge_builder = merge_builder.withSchemaEvolution()
    col_set = {c: f"source.{c}" for c in df.columns}
    if not merge_schema:
        col_set = {c: v for c, v in col_set.items() if c in set(dt.toDF().columns)}
    (
        merge_builder
        .whenMatchedUpdate(set=col_set)
        .whenNotMatchedInsert(values=col_set)
        .execute()
    )


def run_ddl_file(spark: "SparkSession", path: str, catalog: str) -> None:
    """Execute a `.sql` DDL file (config/ddl/*.sql, data/ddl/*.sql), substituting `${catalog}`."""
    with open(path, encoding="utf-8") as f:
        script = f.read()
    for statement in _split_sql_statements(script):
        spark.sql(statement.replace("${catalog}", catalog))


def _split_sql_statements(script: str) -> list[str]:
    """Split a multi-statement SQL script on `;`, respecting single-quoted strings
    and `--` line comments (both full-line and inline).

    Neither a semicolon inside a quoted COMMENT value nor one on a comment line
    is treated as a statement terminator — the naive str.split(';') approach
    breaks on patterns like COMMENT 'foo; bar' or '-- front-end; jobs read...'.
    Trailing comment-only content (e.g. a commented-out ALTER at EOF) is dropped.
    """
    statements: list[str] = []
    buf: list[str] = []
    in_quote = False
    in_line_comment = False
    i = 0
    while i < len(script):
        c = script[i]
        nc = script[i + 1] if i + 1 < len(script) else ""
        if in_line_comment:
            buf.append(c)
            if c == "\n":
                in_line_comment = False
        elif in_quote:
            buf.append(c)
            if c == "'" and nc == "'":  # escaped '' inside a string
                buf.append(nc)
                i += 2
                continue
            elif c == "'":
                in_quote = False
        elif c == "-" and nc == "-":  # start of a -- line comment
            in_line_comment = True
            buf.append(c)
        elif c == "'":
            in_quote = True
            buf.append(c)
        elif c == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(c)
        i += 1
    tail = "".join(buf).strip()
    if tail and not _is_comment_only(tail):
        statements.append(tail)
    return [s for s in statements if s]


def _is_comment_only(stmt: str) -> bool:
    """Return True when `stmt` contains no executable SQL — only `--` comments/whitespace."""
    return all(
        not line.strip() or line.strip().startswith("--")
        for line in stmt.splitlines()
    )
