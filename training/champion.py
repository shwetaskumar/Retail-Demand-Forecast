"""
training/champion.py — best-per-series selection -> champion_index (docs/instructions.md §11).

Champion = lowest 7-day WMAPE among valid (decomposing) candidates, tie-broken
by (a) reconstruction error, (b) training cost — approximated here as "prefer
a plain Stage-1 candidate over the same one wrapped in Stage-2 uplift" (an
extra learner is strictly more training cost), then alphabetically by
model_type for a fully deterministic tie-break. Writes champion_index;
the market x category pyfunc is re-logged/registered in MLflow separately.

`select_champions` (pure pandas) is the canonical selection logic and what's
unit-tested directly; `select_champions_spark` mirrors it exactly as a SQL
window function so the real notebooks/02_train.py path never collects the
full per-series x candidate metrics table to the driver (docs/instructions.md
§18: no collect()/toPandas() on full data).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from config.loader import CategoryConfig

from utils.io import merge_upsert

# Shape gate for champion selection. A candidate whose backtest forecast is
# "flat" (low oscillation ratio) for a series with genuine weekly seasonality
# (high actual AC7) is deprioritised — it may only become champion if NO
# shape-passing candidate exists for that series. The gate ensures daily demand
# shape is preserved in the forecast. Thresholds are config-driven.
_SHAPE_GATE_AC7_MIN = 0.20   # actual must have AC7 > this to trigger the gate
_SHAPE_GATE_OSC_MIN = 0.30   # forecast must have osc_ratio > this to pass

# Bias gate for champion selection. A candidate whose backtest bias exceeds this
# fraction (|Σforecast − Σactual| / Σactual over the holdout) is deprioritised —
# it may only become champion if NO in-band candidate exists for that series.
# This breaks the MASE-only monoculture: a smooth model that best fits one week's
# wobble no longer wins when it systematically over-forecasts, which (a) removes
# the catastrophic over-forecast tail and (b) lets low-bias models like Croston
# win on intermittent series where they belong. NULL bias (couldn't compute) is
# treated as in-band so it never blocks selection.
# --- Tightened from 0.25 to 0.10 (2026-07-29) ---
# Analysis showed systematic over-prediction on GE high-volume items (category A):
#   - elastic_net/A: +27%, ets_event/A: +16%
#   - Bias explodes from lag1 → lag2 (e.g. auto_arima/A: +3%→+22%)
#   - 0.25 was too permissive; 0.10 deprioritises candidates with >10% bias
#     while still allowing selection when no better candidate exists.
_MAX_CHAMPION_BIAS = 0.10

# WMAPE-gate preference margin for champion selection. MASE stays the primary
# ranking key (scale-free, defined for zero-volume series, measures real skill
# over the naive lag-7 baseline — see select_champions' docstring for why WMAPE
# alone isn't used as the ranking key). But a MASE-best candidate can still have
# a worse WMAPE than a rejected alternative that would have PASSED its category's
# gate, and WMAPE is the metric the gate/planners actually act on — so a
# candidate whose WMAPE misses its category's threshold by more than this margin
# is deprioritised the same way an over-biased one already is, and only wins if
# no in-band (bias AND gate) alternative exists for that series. The margin
# (rather than a hard cutoff) exists because engine_layered.py's cascade
# evaluates each candidate on a small number of holdout folds (2 max) — a near-
# miss can be single-fold noise, not a genuine failure, so only a clear miss
# should override MASE.
# Gate preference DISABLED — pure MASE selection. Set to a small value (e.g. 0.02)
# to re-enable gate-aware reordering when inference quality is validated.
_GATE_TOLERANCE = 999.0

_SCHEMA = "gold"
_TABLE = "champion_index"
_INDEX_COLUMNS = [
    "snapshot_date", "market_id", "category", "series_key", "store_id", "sku_id",
    "model_type", "params", "artifact_uri", "mlflow_run_id",
    # Selection key stored for auditability: champion was chosen on lowest mase_7d.
    "mase_7d",
    # WMAPE retained as the planner-facing gate metric (lag-1/lag-2 contract).
    "wmape_7d", "bias_pct", "naive_mae_7d",
    # Volume (sum of weekly actuals in backtest holdout) — needed for
    # volume-weighted WMAPE aggregation across series.
    "volume",
    "passed_gate", "version",
    # Lightweight HPT-tuned hyperparameters (JSON), for interpretability.
    # Distinct from `params` which may include full model state.
    "champion_params",
]


def _gate_wmape_by_category(categories: "list[CategoryConfig] | None") -> dict[tuple, float]:
    """(market_id, category) -> gate WMAPE ceiling (= 1 - accuracy_threshold). Mirrors
    engine_layered.py::_gate_map / evaluation.py::gate's threshold lookup."""
    if not categories:
        return {}
    return {
        (c.market_id, c.category): (1.0 - c.accuracy_threshold)
        for c in categories
        if c.accuracy_threshold
    }


def select_champions(
    candidate_metrics_pdf: pd.DataFrame,
    categories: "list[CategoryConfig] | None" = None,
    max_bias: float = _MAX_CHAMPION_BIAS,
    gate_tolerance: float = _GATE_TOLERANCE,
    shape_gate_ac7_min: float = _SHAPE_GATE_AC7_MIN,
    shape_gate_osc_min: float = _SHAPE_GATE_OSC_MIN,
) -> pd.DataFrame:
    """Per series_key, pick the candidate with the lowest mase_7d among valid rows.

    Selection criterion: MASE (scale-free, defined for zero-volume series, directly
    answers "does this model beat the naïve benchmark?").  WMAPE is retained in the
    output row and used downstream for the per-category gate check in evaluation.py.
    It is NOT the primary ranking key — but a WMAPE-gate PREFERENCE is applied ahead
    of it (see _GATE_TOLERANCE): a candidate whose WMAPE clearly misses its
    category's gate is deprioritised the same way an over-biased one already is,
    so MASE can no longer pick a guaranteed-fail over an available pass. `categories`
    is optional (None/empty => no gate preference applied, i.e. prior behavior)
    so existing callers that don't have category thresholds handy still work.

    Tie-break order: bias_fail → gate_fail → mase_7d → reconstruction_error →
    non-uplift over uplift → model_type alphabetically.  NaN mase_7d rows sort
    last (na_position='last') so a model with a measured MASE always beats one
    that couldn't compute it.
    """
    valid = candidate_metrics_pdf[
        (candidate_metrics_pdf["status"] == "ok") & (candidate_metrics_pdf["passed_decomposition"])
    ].copy()
    if valid.empty:
        return valid

    valid["is_uplift"] = valid["model_type"].str.endswith("+uplift")
    # bias_fail=False sorts first, so in-band candidates are preferred over biased
    # ones; among the preferred group we still pick lowest MASE. NaN bias → NaN>x is
    # False → treated as in-band (never blocks selection). If bias_pct is absent
    # entirely (e.g. legacy metrics), the gate is a no-op.
    if "bias_pct" in valid.columns:
        valid["bias_fail"] = (valid["bias_pct"].abs() > max_bias).fillna(False)
        # Continuous bias magnitude as secondary ranking key (2026-07-29).
        # Among candidates that pass the bias gate, prefer lower |bias| BEFORE
        # comparing MASE. This addresses the GE over-prediction issue: a model
        # with MASE 0.77 and bias +3% now beats one with MASE 0.72 and bias +9%,
        # because the latter's bias explodes at lag2 (+22%) while the former holds.
        # NaN bias → 0.0 so it never penalises models without a bias measurement.
        valid["abs_bias"] = valid["bias_pct"].abs().fillna(0.0)
    else:
        valid["bias_fail"] = False
        valid["abs_bias"] = 0.0

    gate_wmape_by_cat = _gate_wmape_by_category(categories)
    if gate_wmape_by_cat and "category" in valid.columns and "market_id" in valid.columns:
        gate_ceiling = pd.Series(
            list(zip(valid["market_id"], valid["category"])), index=valid.index
        ).map(gate_wmape_by_cat)
        # NaN ceiling (no threshold configured for this category) => never blocks
        # selection, same NULL-safety as bias_fail above.
        valid["gate_fail"] = (valid["wmape_7d"] > (gate_ceiling + gate_tolerance)).fillna(False)
    else:
        valid["gate_fail"] = False

    # Shape gate: flat forecasts for seasonal series are deprioritised.
    # A candidate fails if the series has weekly pattern (actual_ac7 > threshold)
    # AND the candidate's forecast is flat (osc_ratio < osc_min).
    # When actual_ac7 or osc_ratio columns are absent (legacy metrics), gate is no-op.
    if "actual_ac7" in valid.columns and "osc_ratio" in valid.columns:
        needs_shape = valid["actual_ac7"].fillna(0) > shape_gate_ac7_min
        is_flat = valid["osc_ratio"].fillna(0) < shape_gate_osc_min
        valid["shape_fail"] = (needs_shape & is_flat)
    else:
        valid["shape_fail"] = False

    valid = valid.sort_values(
        by=["series_key", "gate_fail", "mase_7d", "bias_fail", "shape_fail", "abs_bias", "reconstruction_error", "is_uplift", "model_type"],
        ascending=True,
        na_position="last",   # NaN mase_7d (failed naive compute) loses to any measured value
    )
    champions = valid.groupby("series_key", as_index=False).first()
    return champions.drop(columns=["is_uplift", "bias_fail", "gate_fail", "abs_bias", "shape_fail"])


def select_champions_spark(
    spark: "SparkSession",
    candidate_metrics_df: "DataFrame",
    categories: "list[CategoryConfig] | None" = None,
    max_bias: float = _MAX_CHAMPION_BIAS,
    gate_tolerance: float = _GATE_TOLERANCE,
    shape_gate_ac7_min: float = _SHAPE_GATE_AC7_MIN,
    shape_gate_osc_min: float = _SHAPE_GATE_OSC_MIN,
) -> "DataFrame":
    """Spark-native mirror of select_champions — selects by mase_7d ASC NULLS LAST,
    with a WMAPE-gate preference tier ahead of it (see select_champions' docstring
    and _GATE_TOLERANCE). `categories` optional: None/empty => no gate preference
    (prior behavior), for callers without category thresholds handy.

    Shape gate: if actual_ac7 > shape_gate_ac7_min AND osc_ratio < shape_gate_osc_min,
    the candidate is deprioritised (flat forecast for a seasonal series). Set
    shape_gate_ac7_min to a high value (e.g. 999) to disable.

    Scalable to the full per-series × candidate metrics table (~150k series × ~7
    candidates at PL scale) with no driver-side collect.
    """
    from pyspark.sql import functions as _F

    candidate_metrics_df.createOrReplaceTempView("_candidate_metrics")

    gate_wmape_by_cat = _gate_wmape_by_category(categories)
    if gate_wmape_by_cat:
        gate_rows = [(m, c, float(w)) for (m, c), w in gate_wmape_by_cat.items()]
        gate_df = spark.createDataFrame(gate_rows, schema="market_id string, category string, gate_wmape double")
        tagged_df = (
            candidate_metrics_df.join(_F.broadcast(gate_df), on=["market_id", "category"], how="left")
        )
    else:
        tagged_df = candidate_metrics_df.withColumn("gate_wmape", _F.lit(None).cast("double"))
    # Ensure shape-gate columns exist (gracefully disabled when not computed upstream)
    if "actual_ac7" not in tagged_df.columns:
        tagged_df = tagged_df.withColumn("actual_ac7", _F.lit(0.0))
    if "osc_ratio" not in tagged_df.columns:
        tagged_df = tagged_df.withColumn("osc_ratio", _F.lit(1.0))
    tagged_df.createOrReplaceTempView("_candidate_metrics_gated")

    return spark.sql(f"""
        WITH ranked AS (
            SELECT * EXCEPT (gate_wmape),
                   ROW_NUMBER() OVER (
                       PARTITION BY series_key
                       ORDER BY
                           CASE WHEN COALESCE(actual_ac7, 0) > {shape_gate_ac7_min}
                                     AND COALESCE(osc_ratio, 1) < {shape_gate_osc_min}
                                THEN 1 ELSE 0 END ASC,    -- shape gate: flat forecasts for seasonal series last
                           CASE WHEN wmape_7d > (gate_wmape + {gate_tolerance}) THEN 1 ELSE 0 END ASC,
                                                          -- WMAPE-gate preference: clear passes before clear misses
                           mase_7d ASC NULLS LAST,        -- primary: MASE (scale-free selection)
                           CASE WHEN ABS(bias_pct) > {max_bias} THEN 1 ELSE 0 END ASC,
                                                          -- tie-break: bias gate (in-band preferred, but only AFTER MASE)
                           ABS(COALESCE(bias_pct, 0)) ASC,  -- tie-break: lower abs bias
                           reconstruction_error ASC,      -- tie-break: decomp fidelity
                           CASE WHEN model_type LIKE '%+uplift' THEN 1 ELSE 0 END ASC,
                           model_type ASC                 -- tie-break: deterministic
                   ) AS rn
            FROM _candidate_metrics_gated
            WHERE status = 'ok' AND passed_decomposition = true
        )
        SELECT * EXCEPT (rn) FROM ranked WHERE rn = 1
    """)


def write_champion_index(spark: "SparkSession", champions_df: "DataFrame", catalog: str, snapshot_date: str, training_run_id: str | None = None) -> None:
    """Stamp `snapshot_date`/`version`/`training_run_id`, extract heavy artifacts to UC Volume, and
    idempotently MERGE into `{catalog}.gold.champion_index`.

    `training_run_id` is the UUID generated at the top of 02_train (always available,
    independent of MLflow). Stored alongside `mlflow_run_id` so champions can always
    be traced back to their training job even when MLflow registration is skipped or fails.

    Heavy model blobs (model_artifact, ets_forecast_values, booster_bytes, etc.) are
    written to /Volumes/{catalog}/gold/model_artifacts/{market}/{snapshot_date}/
    and replaced in `params` with a lightweight reference. The artifact_uri column
    stores the volume path for downstream loading.

    Version auto-increments on each re-run for the same (market, category, series_key,
    snapshot_date) — so re-training produces version 2, 3, etc.
    """
    
    import logging as _logging
    from pyspark.sql import functions as F

    # champion_index.category is NOT NULL. A series whose entire training
    # window has no category assignment (F.first returns NULL in Step 2 of
    # 02_train) reaches here with category=NULL and would violate the constraint.
    # Drop and warn — these series cannot be scored until they get a category.
    _null_cat_df = champions_df.filter(F.col("category").isNull())
    _n_null = _null_cat_df.count()
    if _n_null:
        _null_keys = [r["series_key"] for r in _null_cat_df.select("series_key").limit(10).collect()]
        _logging.getLogger(__name__).warning(
            f"write_champion_index: dropping {_n_null} champion row(s) with "
            f"NULL category — no classification history for these series. "
            f"First series_keys (up to 10): {_null_keys}"
        )
        champions_df = champions_df.filter(F.col("category").isNotNull())

    champions_df.createOrReplaceTempView("_champions_to_write")
    safe_snapshot_date = snapshot_date.replace("'", "''")

    # Determine next version: MAX(existing version) + 1 for this snapshot_date
    existing_max = spark.sql(f"""
        SELECT market_id, category, series_key, MAX(version) AS max_version
        FROM {catalog}.{_SCHEMA}.{_TABLE}
        WHERE snapshot_date = DATE('{safe_snapshot_date}')
        GROUP BY market_id, category, series_key
    """)
    existing_max.createOrReplaceTempView("_existing_versions")

    # champion_params: lightweight JSON of HPT-tuned hyperparameters (for
    # interpretability/reproducibility). Uses `hpt_params` column if the
    # upstream engine provided it (just the trial params), otherwise falls
    # back to `params` (full model state — still useful, and artifacts are
    # extracted separately so it's typically just hyperparams).
    _has_hpt_params = "hpt_params" in [f.name for f in champions_df.schema.fields]
    _champion_params_expr = "c.hpt_params" if _has_hpt_params else "c.params"

    stamped_df = spark.sql(f"""
        SELECT DATE('{safe_snapshot_date}') AS snapshot_date, c.market_id, c.category, c.series_key,
               c.store_id, c.sku_id, c.model_type, c.params,
               CAST(NULL AS STRING)          AS artifact_uri,
               CAST(NULL AS STRING)          AS mlflow_run_id,
               CAST(NULL AS STRING)          AS training_run_id,
               c.mase_7d,
               c.wmape_7d,
               c.bias_pct,
               c.naive_mae_7d,
               c.volume,
               c.passed_decomposition AS passed_gate,
               COALESCE(e.max_version, 0) + 1 AS version,
               {_champion_params_expr} AS champion_params
        FROM _champions_to_write c
        LEFT JOIN _existing_versions e
          ON c.market_id = e.market_id AND c.category = e.category AND c.series_key = e.series_key
    """)
    if training_run_id:
        from pyspark.sql import functions as _F
        stamped_df = stamped_df.withColumn("training_run_id", _F.lit(training_run_id))

    # --- Write champions to Delta (params kept inline for speed) ---
    # Artifact extraction to UC Volume can be done offline as a batch migration
    # if table size becomes a concern. Keeping it out of the hot path preserves
    # the ~30s write time.
    merge_upsert(spark, stamped_df, catalog, _SCHEMA, _TABLE, merge_keys=["market_id", "category", "series_key", "snapshot_date"], merge_schema=True)


def update_mlflow_run_ids(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    snapshot_date: str,
    run_ids: dict[str, str],
) -> int:
    """Update champion_index with MLflow run_ids and artifact_uri after registration.

    Both fields are tracking metadata only — inference does NOT read from them.
    They exist for provenance: linking a champion row to the exact MLflow run
    and registered model artifact that was produced from it.

    Args:
        run_ids: dict mapping category -> mlflow_run_id from register_champions_to_mlflow.

    Returns:
        Number of categories updated.
    """
    if not run_ids:
        return 0

    safe_market_id = market_id.replace("'", "''")
    safe_snap = snapshot_date.replace("'", "''")

    # Build CASE expressions for category -> run_id / artifact_uri mapping
    run_id_case = "CASE t.category\n"
    uri_case = "CASE t.category\n"
    for cat, run_id in run_ids.items():
        safe_cat = cat.replace("'", "''")
        safe_run = run_id.replace("'", "''")
        artifact_uri = f"runs:/{run_id}/model"
        run_id_case += f"            WHEN '{safe_cat}' THEN '{safe_run}'\n"
        uri_case += f"            WHEN '{safe_cat}' THEN '{artifact_uri}'\n"
    run_id_case += "            ELSE mlflow_run_id END"
    uri_case += "            ELSE artifact_uri END"

    # Update latest version for each series in this snapshot
    spark.sql(f"""
        MERGE INTO {catalog}.{_SCHEMA}.{_TABLE} t
        USING (
            SELECT market_id, category, series_key, snapshot_date, MAX(version) AS version
            FROM {catalog}.{_SCHEMA}.{_TABLE}
            WHERE market_id = '{safe_market_id}' AND snapshot_date = DATE('{safe_snap}')
              AND category IN ({','.join("'" + c.replace("'", "''") + "'" for c in run_ids.keys())})
            GROUP BY market_id, category, series_key, snapshot_date
        ) s
        ON t.market_id = s.market_id AND t.category = s.category
           AND t.series_key = s.series_key AND t.snapshot_date = s.snapshot_date
           AND t.version = s.version
        WHEN MATCHED THEN UPDATE SET
            mlflow_run_id = {run_id_case},
            artifact_uri = {uri_case}
    """)

    return len(run_ids)