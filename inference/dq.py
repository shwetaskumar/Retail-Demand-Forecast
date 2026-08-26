"""
inference/dq.py — data-quality checks -> dq_audit (docs/instructions.md §17).

Row counts, null/dup keys, negative qty, isActive integrity, decomposition
reconstruction_error within tolerance, forecast freshness, horizon completeness
(140 rows/series). Fail loudly; block publish on critical failures.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def run_dq_checks(
    spark: "SparkSession",
    output_df: "DataFrame",
    decomposition_df: "DataFrame",
    reconstruction_tol_pct: float,
) -> pd.DataFrame:
    """Run all DQ checks; return one row per check (metric_value, threshold, passed, detail)."""
    # TODO(Step 9)
    raise NotImplementedError


def write_dq_audit(spark: "SparkSession", dq_pdf: pd.DataFrame, catalog: str, run_id: str) -> None:
    """MERGE dq results into `{catalog}.gold.dq_audit`."""
    # TODO(Step 9): utils.io.merge_upsert(...)
    raise NotImplementedError


_SCORING_FAILURES_MERGE_KEYS = ["snapshot_date", "run_id", "series_key"]


def write_scoring_failures(spark: "SparkSession", failed_df: "DataFrame", catalog: str) -> None:
    """Idempotent MERGE of per-series JOB 3 scoring failures into
    `{catalog}.gold.scoring_failures`.

    Durable, SQL-queryable record of WHY a series produced no forecast this run
    (inference.batch_infer._fail_row) — survives cluster termination, unlike the
    executor stderr logs this used to rely on exclusively. One row per
    (snapshot_date, run_id, series_key) that failed to score; a no-op if
    `failed_df` is empty (nothing failed).
    """
    from utils.io import merge_upsert

    if failed_df.take(1) == []:
        return
    merge_upsert(
        spark, failed_df, catalog, "gold", "scoring_failures",
        merge_keys=_SCORING_FAILURES_MERGE_KEYS,
    )
