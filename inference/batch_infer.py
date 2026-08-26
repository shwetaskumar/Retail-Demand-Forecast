"""
inference/batch_infer.py — JOB 3 per-series champion scoring (docs/instructions.md §13).

Mirrors training/engine.py's structure: groups the combined (historical + future
spine) frame by series_key and runs a single applyInPandas pass that fits the
champion model, predicts over the 140-day horizon, and decomposes in one shot.
Output (yhat per date_id) and decomposition (long-format components) are returned
as two separate DataFrames via a record_type discriminator column.

Scoring only — no re-training, no HPT. Reads champions from
champion_index; writes to forecast.output + forecast.decomposition
via merge_upsert.

Per-series scoring failures (no history to refit, from_serialized() error, any
exception during fit/predict/decompose) are captured as durable 'failed' rows in
the same applyInPandas pass and written to scoring_failures — queryable via
SQL, unlike the executor stderr logs this used to rely on exclusively (those are
lost the moment a job cluster terminates or scales down, with no cluster_log_conf
configured).
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from config.loader import ForecastConfig

from data.features.event_features import (
    _FALLBACK_SUBTYPE_COLUMN_MAP,
    _FALLBACK_TACTIC_COLUMN_MAP,
)
from utils.io import merge_upsert
from utils.logging import get_logger
from training.decomposition.decomposer import reconstruction_error
from training.engine import STAGE1_MODEL_REGISTRY  # noqa: F401  (kept for parity/back-compat)
from training.models.uplift import UpliftModel

# Executor-side logger. Messages land in the Spark UI executor stderr logs, which
# is how we surface per-series failures/refits that were previously swallowed.
_worker_log = logging.getLogger("inference.batch_infer.worker")
if not _worker_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [batch_infer.worker] %(message)s"))
    _worker_log.addHandler(_h)
    _worker_log.setLevel(logging.INFO)
    _worker_log.propagate = False

# Round-robin partition count for the scoring applyInPandas pass — see _spread_score's
# docstring for why a plain groupBy("series_key") isn't enough here.
_SCORE_PARTITIONS = 4000

# ── Schemas ──────────────────────────────────────────────────────────────────

# Single applyInPandas schema covering output + decomp + failure rows.
# record_type discriminates: 'output' fills horizon_day/yhat/...,
# 'decomp' fills component_type/component_name/contribution,
# 'failed' fills model_type/fail_stage/fail_reason (see _fail_row).
_COMBINED_SCHEMA = (
    "record_type string, "
    "series_key string, market_id string, category string, "
    "store_id string, sku_id string, date_id date, "
    "horizon_day int, yhat double, yhat_lower double, yhat_upper double, "
    "reconstruction_error double, "
    "component_type string, component_name string, contribution double, "
    "tactic string, sub_type string, "
    "model_type string, fail_stage string, fail_reason string"
)
_COMBINED_COLS = [
    "record_type", "series_key", "market_id", "category",
    "store_id", "sku_id", "date_id",
    "horizon_day", "yhat", "yhat_lower", "yhat_upper", "reconstruction_error",
    "component_type", "component_name", "contribution",
    "tactic", "sub_type",
    "model_type", "fail_stage", "fail_reason",
]

# ── Reverse mapping: feature column name → (tactic label, sub_type label) ─────
# Built from event_features.py's fallback maps.  At runtime, the per-market config
# may override, but for decomposition labelling the fallback is sufficient (the
# feature column names are the same regardless of config — only the event-to-column
# matching changes).
_FEATURE_TO_TACTIC: dict[str, str] = {
    f"promo_count_{col_suffix}": tactic_label
    for tactic_label, col_suffix in _FALLBACK_TACTIC_COLUMN_MAP.items()
}
_FEATURE_TO_SUBTYPE: dict[str, str] = {
    f"promo_count_{col_suffix}": subtype_label
    for subtype_label, col_suffix in _FALLBACK_SUBTYPE_COLUMN_MAP.items()
}

_MERGE_KEYS_OUTPUT = ["snapshot_date", "champion_version", "market_id", "category", "store_id", "sku_id", "date_id"]
_MERGE_KEYS_DECOMP = [
    "snapshot_date", "champion_version", "market_id", "category",
    "store_id", "sku_id", "date_id",
    "component_type", "component_name",
]
_MERGE_KEYS_FAILURES = ["snapshot_date", "run_id", "series_key"]


# ── Model reconstruction ──────────────────────────────────────────────────────────

def _build_model(model_type: str):
    """Instantiate a ForecastModel from the champion model_type string.

    Uses the auto-discovery registry so any model wrapper in training/models/
    is automatically supported — no hardcoded list to maintain.
    Handles plain Stage-1 types and uplift variants ('lgbm+uplift').
    """
    from training.models.registry import build as _registry_build, get_registry

    if model_type.endswith("+uplift"):
        base_type = model_type.replace("+uplift", "")
        base_cls = get_registry()[base_type]
        return UpliftModel(base_cls())
    return _registry_build(model_type)


def _fail_row(identity: dict, model_type: str, fail_stage: str, fail_reason: str) -> pd.DataFrame:
    """One-row 'failed' record — the durable counterpart to the executor-log warning.

    Flows through the same applyInPandas return path as output/decomp rows, so it
    lands in scoring_failures (via score()'s failed_df extraction) and survives
    cluster termination, unlike stderr logs with no cluster_log_conf configured.
    """
    return pd.DataFrame([{
        "record_type": "failed",
        **identity,
        "date_id": None, "horizon_day": None, "yhat": None, "yhat_lower": None,
        "yhat_upper": None, "reconstruction_error": None,
        "component_type": None, "component_name": None, "contribution": None,
        "model_type": model_type, "fail_stage": fail_stage, "fail_reason": fail_reason,
    }], columns=_COMBINED_COLS)


def _spread_score(combined_df: "DataFrame", n_partitions: int, udf, schema) -> "DataFrame":
    """Run _score_one_series over an EVEN round-robin repartition instead of a plain
    groupBy("series_key").applyInPandas(...).

    A plain groupBy hash-partitions series_key into whatever spark.sql.shuffle.partitions
    is set to (200 by default) — coarse relative to how UNEVEN per-series cost is here.
    Most series reconstruct from a stored artifact in sub-millisecond time, but a REFIT
    (no stored artifact) re-fits the full history — the same cost as training, seconds for
    Prophet/LGBM — and even fast-path Prophet/LGBM decompose() calls are materially heavier
    than the stateless models (Stan predict / two booster passes for SHAP marginal
    attribution vs. pure-numpy reconstruction for elastic_net/croston/statsforecast). With
    ~200 coarse partitions, a handful of tasks that happen to land a disproportionate share
    of refits/heavy-decompose series become stragglers, and a Spark stage doesn't finish
    until its slowest task does — so wall-clock ends up dominated by those few tasks even
    though the refit fraction is a modest ~8% on average. This mirrors
    training/engine_layered.py::_spread_apply's fix for the identical class of problem at
    training time (pmod(hash(series_key), n_partitions) so every row of a series still
    lands in the same task, while spreading series far more evenly than the default
    shuffle-partition count) — reimplemented locally rather than imported, since JOB 2
    (training) and JOB 3 (inference) must stay independent (CLAUDE.md lock #3).
    """
    from pyspark.sql import functions as F

    spread = combined_df.withColumn("_spread", F.pmod(F.hash("series_key"), F.lit(int(n_partitions))))
    return (
        spread.repartition(int(n_partitions), "_spread")
        .groupBy("_spread")
        .applyInPandas(udf, schema=schema)
    )


# ── Per-series scoring function ────────────────────────────────────────────────────

def _score_one_series(pdf: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Per-group function (pandas in, pandas out) for applyInPandas.

    Splits combined frame into history (date_id <= snapshot_date) and future
    (date_id > snapshot_date), reconstructs the champion model from its stored
    artifact when possible (fast path) or refits once (fallback), then calls
    predict and decompose to produce output + decomposition rows in a single pass.

    A single series failure returns a one-row 'failed' record (see _fail_row)
    rather than crashing the job — it both logs why to the executor stderr AND
    returns the reason as data, so it's queryable from scoring_failures
    even after the cluster that ran it is gone.
    """
    # ─ identity & champion metadata ───────────────────────────────────────────
    # params may be present on only one row in the group (to avoid replicating a
    # 200-300KB JSON blob across every historical row sent over Arrow).
    def _first_non_null(col: str):
        if col not in pdf.columns:
            return None
        s = pdf[col]
        s = s[s.notna()]
        return s.iloc[0] if not s.empty else None

    identity = {
        "series_key":  _first_non_null("series_key"),
        "market_id": _first_non_null("market_id"),
        "category":    _first_non_null("category"),
        "store_id": _first_non_null("store_id"),
        "sku_id":     _first_non_null("sku_id"),
    }
    series_key = identity["series_key"]
    model_type = str(_first_non_null("model_type") or "")
    params_raw = _first_non_null("params")
    params = json.loads(params_raw) if isinstance(params_raw, str) and params_raw else {}

    # ─ split history vs. future (convert date column ONCE, vectorized) ─────────
    snap_ts = pd.Timestamp(snapshot_date)
    dates = pd.to_datetime(pdf["date_id"])
    history_pdf = pdf[dates <= snap_ts].sort_values("date_id").reset_index(drop=True)
    future_pdf = pdf[dates > snap_ts].sort_values("date_id").reset_index(drop=True)

    has_serialized = bool(params.get("model_artifact") or params.get("ets_forecast_values"))
    if future_pdf.empty or (history_pdf.empty and not has_serialized):
        reason = (
            "future_pdf empty — no horizon rows to score"
            if future_pdf.empty
            else "history_pdf empty and no stored artifact — nothing to fit from"
        )
        _worker_log.warning("series=%s model_type=%s %s", series_key, model_type, reason)
        return _fail_row(identity, model_type, "no_horizon_or_history", reason)

    # ─ reconstruct (fast path) or refit (fallback) ────────────────────────────
    try:
        model = _build_model(model_type)

        _fitted = False
        if has_serialized:
            try:
                model = type(model).from_serialized(params)
                _fitted = True
            except NotImplementedError:
                _worker_log.warning(
                    "series=%s model_type=%s has a stored artifact but from_serialized() "
                    "is not implemented — falling back to REFIT (slow). Fix the model wrapper "
                    "to make inference fast.", series_key, model_type,
                )
            except Exception as e:  # noqa: BLE001 — deliberately broad, but now LOGGED
                _worker_log.warning(
                    "series=%s model_type=%s from_serialized() raised %s: %s — falling back "
                    "to REFIT (slow).", series_key, model_type, type(e).__name__, e,
                )

        if not _fitted:
            if history_pdf.empty:
                reason = "could not load artifact (from_serialized failed/absent) and has no history to refit"
                _worker_log.warning("series=%s model_type=%s %s", series_key, model_type, reason)
                return _fail_row(identity, model_type, "artifact_load_failed_no_history", reason)
            model.fit(history_pdf, future_pdf, params)

        yhat = model.predict(future_pdf)
        decomp = model.decompose(future_pdf)
        recon = reconstruction_error(decomp)

        # ── Forecast sanity cap (guardrail) ───────────────────────────────────
        # Clip each day's yhat to params['forecast_cap'] (set at training =
        # mult × trailing max daily demand) and floor at 0. Catches catastrophic
        # over-forecasts — e.g. a holiday trend extrapolated into a post-holiday
        # trough — before they reach planners. The decomposition rows are left
        # uncapped (diagnostic); only the planner-facing yhat is clipped, so on a
        # rare capped date Σcomponents may exceed the reported yhat.
        _cap = params.get("forecast_cap")
        yhat = (
            yhat.clip(lower=0.0, upper=float(_cap))
            if _cap not in (None, 0)
            else yhat.clip(lower=0.0)
        )
    except Exception as e:  # noqa: BLE001 — single-series failure must not crash the job
        reason = f"{type(e).__name__}: {e}"
        _worker_log.warning("series=%s model_type=%s scoring FAILED %s", series_key, model_type, reason)
        return _fail_row(identity, model_type, "scoring_exception", reason)

    # ─ output rows (one per future date_id) ───────────────────────────────────
    output_rows = [
        {
            "record_type": "output",
            **identity,
            "date_id":              future_pdf["date_id"].iloc[i],
            "horizon_day":          i + 1,
            "yhat":                 float(yhat.iloc[i]),
            "yhat_lower":           None,   # TODO: source from model interval when available
            "yhat_upper":           None,
            "reconstruction_error": float(recon.iloc[i]),
            "component_type":       None,
            "component_name":       None,
            "contribution":         None,
            "model_type":           None,   # joined from _binfer_champions downstream, not self-reported
            "fail_stage":           None,
            "fail_reason":          None,
        }
        for i in range(len(future_pdf))
    ]

    # ─ decomposition rows (long format, one per future date_id × component) ────
    # component_type + component_name convention:
    #   trend / seasonality_weekly / seasonality_yearly / temporal_features_effect
    #       → component_name = "Baseline"
    #   holiday_lift
    #       → component_name = <actual holiday name>
    #   tactic_subtype_lift
    #       → component_name = <feature; post-hoc relabeled to event names via '|'>
    #   cross_sku_lift
    #       → component_name = "Halo" or "Cannibalization"

    def _comp_rows(component_type: str, component_name: str | None, values: pd.Series):
        return [
            {
                "record_type":    "decomp",
                **identity,
                "date_id":        future_pdf["date_id"].iloc[i],
                "horizon_day":    None,
                "yhat":           None,
                "yhat_lower":     None,
                "yhat_upper":     None,
                "reconstruction_error": None,
                "component_type":  component_type,
                "component_name":  component_name,
                "contribution":    float(values.iloc[i]),
                "tactic":          None,
                "sub_type":        None,
                "model_type":      None,
                "fail_stage":      None,
                "fail_reason":     None,
            }
            for i in range(len(values))
        ]

    decomp_rows: list[dict] = []

    # ── 1. BASELINE: separate component_types, all with component_name="Baseline" ─
    decomp_rows += _comp_rows("trend",                     "Baseline", decomp.trend)
    decomp_rows += _comp_rows("seasonality_weekly",        "Baseline", decomp.seasonality_weekly)
    decomp_rows += _comp_rows("seasonality_yearly",        "Baseline", decomp.seasonality_yearly)
    decomp_rows += _comp_rows("temporal_features_effect",  "Baseline", decomp.temporal_features_effect)

    # ── 2. HOLIDAY: component_name = actual holiday name ──────────────────────
    _holiday_names = (
        future_pdf["holiday_name"].fillna("unknown_holiday").values
        if "holiday_name" in future_pdf.columns
        else [None] * len(future_pdf)
    )
    _holiday_total = pd.Series(np.zeros(len(future_pdf)), dtype=float)
    for _col_name, s in decomp.holiday_lift.items():
        _holiday_total += s.fillna(0.0).values
    for i in range(len(future_pdf)):
        contrib = float(_holiday_total.iloc[i])
        if contrib != 0.0 or (_holiday_names[i] is not None and _holiday_names[i] != "unknown_holiday"):
            decomp_rows.append({
                "record_type":    "decomp",
                **identity,
                "date_id":        future_pdf["date_id"].iloc[i],
                "horizon_day":    None,
                "yhat":           None,
                "yhat_lower":     None,
                "yhat_upper":     None,
                "reconstruction_error": None,
                "component_type":  "holiday_lift",
                "component_name":  _holiday_names[i],
                "contribution":    contrib,
                "tactic":          None,
                "sub_type":        None,
                "model_type":      None,
                "fail_stage":      None,
                "fail_reason":     None,
            })

    # ── 3. TACTIC/SUBTYPE LIFT: promo features per day ───────────────────────
    # Post-hoc enrichment relabels component_name to pipe-separated event names.
    for _feat_name, _feat_series in decomp.promo_lift.items():
        _tactic = _FEATURE_TO_TACTIC.get(_feat_name)
        _sub_type = _FEATURE_TO_SUBTYPE.get(_feat_name)
        for i in range(len(future_pdf)):
            contrib = float(_feat_series.iloc[i]) if not pd.isna(_feat_series.iloc[i]) else 0.0
            if contrib != 0.0:
                decomp_rows.append({
                    "record_type":    "decomp",
                    **identity,
                    "date_id":        future_pdf["date_id"].iloc[i],
                    "horizon_day":    None,
                    "yhat":           None,
                    "yhat_lower":     None,
                    "yhat_upper":     None,
                    "reconstruction_error": None,
                    "component_type":  "tactic_subtype_lift",
                    "component_name":  _feat_name,
                    "contribution":    contrib,
                    "tactic":          _tactic,
                    "sub_type":        _sub_type,
                    "model_type":      None,
                    "fail_stage":      None,
                    "fail_reason":     None,
                })

    # ── 4. CROSS-ITEM LIFT: "Halo" or "Cannibalization" ─────────────────────
    for name, s in decomp.cross_sku_lift.items():
        _cross_label = "Cannibalization" if name.startswith("cannib") else "Halo"
        for i in range(len(s)):
            contrib = float(s.iloc[i]) if not pd.isna(s.iloc[i]) else 0.0
            if contrib != 0.0:
                decomp_rows.append({
                    "record_type":    "decomp",
                    **identity,
                    "date_id":        future_pdf["date_id"].iloc[i],
                    "horizon_day":    None,
                    "yhat":           None,
                    "yhat_lower":     None,
                    "yhat_upper":     None,
                    "reconstruction_error": None,
                    "component_type":  "cross_sku_lift",
                    "component_name":  _cross_label,
                    "contribution":    contrib,
                    "tactic":          None,
                    "sub_type":        None,
                    "model_type":      None,
                    "fail_stage":      None,
                    "fail_reason":     None,
                })

    return pd.DataFrame(output_rows + decomp_rows)


def _score_spread_group(pdf: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """applyInPandas entry point for _spread_score's round-robin buckets.

    Each bucket mixes rows from MANY series (unlike the old plain
    groupBy("series_key"), where one call == one series) — split back out and
    score one series at a time, same as _score_one_series always has, then
    concatenate. All branches of _score_one_series (success/fail rows) already
    return the same _COMBINED_COLS shape, so concatenation is safe.
    """
    if pdf is None or pdf.empty:
        return pd.DataFrame(columns=_COMBINED_COLS)
    parts = [
        _score_one_series(g, snapshot_date)
        for _, g in pdf.groupby("series_key", sort=False)
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_COMBINED_COLS)


# ── Public API ───────────────────────────────────────────────────────────────────────

def score(
    spark: "SparkSession",
    catalog: str,
    snapshot_features_df: "DataFrame",
    future_spine_df: "DataFrame",
    config: "ForecastConfig",
    snapshot_date: str,
    run_id: str,
) -> tuple["DataFrame", "DataFrame", "DataFrame"]:
    """Score all champion series, returning (output_df, decomp_df, failed_df).

    Builds a combined history+future frame per series, runs a single
    applyInPandas pass (reconstruct-or-refit → predict → decompose), materializes
    it ONCE (persist), then splits the result into the three output tables. All
    three DataFrames read from the persisted scored result, so the expensive
    pandas UDF runs exactly once regardless of how many counts/merges happen
    downstream. `failed_df` holds one row per series that produced no forecast
    this run, with a durable, queryable reason (see _fail_row/scoring_failures)
    — the caller should write it via inference.dq.write_scoring_failures.
    """
    from pyspark.sql import functions as _F
    from pyspark.sql.window import Window as _Window
    from pyspark.storagelevel import StorageLevel

    logger = get_logger("inference.batch_infer", env=catalog)
    safe_market_id = config.market_config.market_id.replace("'", "''")

    # ─ 1. Load champion metadata (model_type, params) per series ──────────────
    # Filter by snapshot_date <= inference date (no leakage from future-trained
    # champions). When the same series appears at multiple snapshots, pick the
    # LATEST snapshot + highest version (handles re-training on the same date).
    safe_snap = snapshot_date.replace("'", "''")
    champions_df = spark.sql(f"""
        WITH ranked AS (
            SELECT series_key, market_id, category, store_id, sku_id,
                   model_type, params, snapshot_date, version,
                   ROW_NUMBER() OVER (
                       PARTITION BY series_key
                       ORDER BY snapshot_date DESC, version DESC
                   ) AS rn
            FROM {catalog}.gold.champion_index
            WHERE market_id = '{safe_market_id}'
              AND snapshot_date <= DATE('{safe_snap}')
        )
        SELECT series_key, market_id, category, store_id, sku_id,
               model_type, params,
               version AS champion_version,
               CASE
                   WHEN get_json_object(params, '$.model_artifact') IS NOT NULL THEN FALSE
                   WHEN get_json_object(params, '$.ets_forecast_values') IS NOT NULL
                        AND get_json_object(params, '$.ets_forecast_values') <> '[]' THEN FALSE
                   ELSE TRUE
               END AS needs_history
        FROM ranked
        WHERE rn = 1
    """).cache()
    champions_df.createOrReplaceTempView("_binfer_champions")

    # Cheap up-front diagnostic: how many series will REFIT (no artifact) vs
    # fast-path. A high refit fraction is the usual reason inference is as slow
    # as (or slower than) training.
    n_champ = champions_df.count()
    n_refit = champions_df.filter(_F.col("needs_history")).count()
    n_fast = n_champ - n_refit
    logger.info(
        f"Champions: {n_champ:,} total | fast-path (artifact) {n_fast:,} | "
        f"REFIT (no artifact) {n_refit:,}"
    )

    if n_champ and n_refit / n_champ > 0.2:
        logger.warning(
            f"{n_refit:,}/{n_champ:,} ({100*n_refit/n_champ:.0f}%) champion series have NO "
            f"serialized artifact and will be REFIT during inference — this is why scoring is "
            f"slow. Ensure training stores serialize()/model_artifact for these model types."
        )

    # ─ 2. Tag history with champion metadata (only for series that need refit) ─
    champ_meta = champions_df.select(
        "store_id", "sku_id", "series_key", "market_id", "category",
        "model_type", "params", "needs_history"
    )
    h_clean = snapshot_features_df.drop("series_key", "market_id", "category")
    history_tagged = (
        h_clean
        .join(
            champ_meta
            .filter(_F.col("needs_history"))
            .select("store_id", "sku_id", "series_key", "market_id", "category", "model_type"),
            on=["store_id", "sku_id"],
            how="inner",
        )
        .filter(_F.col("isActive") == 1)
        .withColumn("params", _F.lit(None).cast("string"))
    )

    # ─ 3. Tag future spine with model_type + params ───────────────────────────
    #   BROADCAST SAFETY: the champion table is only ~118K rows, so Spark's
    #   row-count estimate makes it look broadcastable — but each row now carries
    #   a serialized artifact (uplift booster + ETS arrays, ~100KB+/row), so
    #   letting `params` into the 16.5M-row future-spine join makes Spark broadcast
    #   a multi-GiB table and blow past the 8 GiB cap (observed: 15.6 GiB). We fix
    #   this WITHOUT disabling broadcast globally (which would force the expensive
    #   history join to sort-merge/shuffle):
    #     (a) broadcast only the lightweight series_key→model_type map onto the spine;
    #     (b) attach `params` to exactly ONE row per series by broadcasting the tiny
    #         118K first-row frame and STREAMING the fat params_map, so `params` is
    #         never the broadcast side. The 16.4M non-first rows never touch params.
    #   `_first_non_null("params")` in the UDF only needs params on one row per group.
    meta_lw    = champ_meta.select("series_key", "model_type")   # no params → tiny, broadcastable
    params_map = champ_meta.select("series_key", "params")       # fat → only ever streamed

    future_lw = (
        future_spine_df
        .join(_F.broadcast(meta_lw), on="series_key", how="inner")
        .withColumn("_rn", _F.row_number().over(_Window.partitionBy("series_key").orderBy("date_id")))
        .withColumn("qty", _F.lit(None).cast("float"))
        .withColumn("isActive", _F.lit(None).cast("int"))
        .persist(StorageLevel.MEMORY_AND_DISK)   # reused by both branches; computes the window once
    )

    # (b) params onto the first row only: broadcast the SMALL first-row frame,
    #     stream the fat params_map (Spark will not broadcast the streamed side).
    first_tagged = (
        _F.broadcast(future_lw.filter(_F.col("_rn") == 1))
        .join(params_map, on="series_key", how="inner")
    )
    rest_tagged = (
        future_lw.filter(_F.col("_rn") != 1)
        .withColumn("params", _F.lit(None).cast("string"))
    )
    future_tagged = first_tagged.unionByName(rest_tagged, allowMissingColumns=True).drop("_rn")

    # ─ 4. Union into one frame: history rows + future rows per series ─────────
    combined_df = history_tagged.unionByName(future_tagged, allowMissingColumns=True)

    # ─ 5. applyInPandas: single reconstruct/refit+predict+decompose pass ──────
    # Round-robin spread (_spread_score), not a plain groupBy("series_key") — see its
    # docstring: with per-series cost this uneven (sub-ms reconstructions mixed with
    # full REFIT and heavy Prophet/LGBM decomposes), the default ~200 shuffle
    # partitions produce straggler tasks that dominate wall-clock even at a modest
    # refit fraction.
    scored_df = _spread_score(
        combined_df, _SCORE_PARTITIONS,
        lambda pdf: _score_spread_group(pdf, snapshot_date),
        _COMBINED_SCHEMA,
    )

    # ── Materialize ONCE. Without this, every downstream count()/MERGE re-runs
    #    the entire pandas UDF (previously ~4×). Persist to disk+memory so the
    #    expensive per-series work happens a single time.
    scored_df = scored_df.persist(StorageLevel.MEMORY_AND_DISK)
    n_rows = scored_df.count()  # the one and only trigger of the UDF
    n_series_scored = scored_df.select("series_key").distinct().count()
    logger.info(
        f"Scored {n_series_scored:,}/{n_champ:,} champion series "
        f"({n_rows:,} output+decomp rows). "
        f"{n_champ - n_series_scored:,} produced nothing — see scoring_failures "
        f"(this run_id={run_id}) for the durable per-series reason, or executor logs "
        f"(tag [batch_infer.worker]) if the cluster is still up."
    )
    scored_df.createOrReplaceTempView("_binfer_scored")

    # ─ 6. Stamp snapshot_date + run_id, split into output and decomp ──────────
    safe_snap = snapshot_date.replace("'", "''")
    safe_run  = run_id.replace("'", "''")

    output_df = spark.sql(f"""
        SELECT DATE('{safe_snap}') AS snapshot_date,
               c.champion_version,
               '{safe_run}'       AS run_id,
               '{safe_market_id}' AS market_id,
               COALESCE(s.category, c.category)           AS category,
               COALESCE(s.store_id, c.store_id)     AS store_id,
               COALESCE(s.sku_id, c.sku_id)             AS sku_id,
               s.date_id,
               s.horizon_day,
               s.yhat AS yhat_raw,
               s.yhat, s.yhat_lower, s.yhat_upper, s.reconstruction_error,
               c.model_type
        FROM _binfer_scored s
        JOIN _binfer_champions c USING (series_key)
        WHERE s.record_type = 'output'
    """)
    decomp_df = spark.sql(f"""
        SELECT DATE('{safe_snap}') AS snapshot_date,
               c.champion_version,
               '{safe_run}'       AS run_id,
               '{safe_market_id}' AS market_id,
               COALESCE(s.category, c.category)           AS category,
               COALESCE(s.store_id, c.store_id)     AS store_id,
               COALESCE(s.sku_id, c.sku_id)             AS sku_id,
               s.date_id,
               s.component_type, s.component_name,
               s.contribution AS contribution_raw,
               s.contribution,
               s.tactic, s.sub_type,
               c.model_type
        FROM _binfer_scored s
        JOIN _binfer_champions c USING (series_key)
        WHERE s.record_type = 'decomp'
    """)
    # No join needed — failure rows self-report model_type (may be '' if the
    # champion lookup itself never resolved a model_type for this series).
    failed_df = spark.sql(f"""
        SELECT DATE('{safe_snap}') AS snapshot_date,
               '{safe_run}'       AS run_id,
               COALESCE(s.market_id, '{safe_market_id}') AS market_id,
               s.category, s.store_id, s.sku_id, s.series_key,
               s.model_type, s.fail_stage, s.fail_reason
        FROM _binfer_scored s
        WHERE s.record_type = 'failed'
    """)
    return output_df, decomp_df, failed_df


def write_outputs(
    spark: "SparkSession",
    output_df: "DataFrame",
    decomp_df: "DataFrame",
    catalog: str,
) -> None:
    """Idempotent MERGE into output and decomposition."""
    # Deduplicate by merge keys to prevent DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW_IN_MERGE.
    # Duplicates can arise from the series_key→champions join when multiple series map
    # to the same (market, category, store_id, sku_id) dimensional combination.
    merge_upsert(
        spark, output_df.dropDuplicates(_MERGE_KEYS_OUTPUT), catalog, "forecast", "output",
        merge_keys=_MERGE_KEYS_OUTPUT,
    )
    # Ensure 'category' exists in target — withSchemaEvolution() cannot add columns
    # referenced in the merge condition (ON clause), only in INSERT/UPDATE actions.
    try:
        spark.sql(f"ALTER TABLE {catalog}.forecast.decomposition ADD COLUMN category STRING")
    except Exception:
        pass  # column already exists
    merge_upsert(
        spark, decomp_df.dropDuplicates(_MERGE_KEYS_DECOMP), catalog, "forecast", "decomposition",
        merge_keys=_MERGE_KEYS_DECOMP,
        merge_schema=True,
    )

    # ── Post-hoc promo attribution: relabel per-feature rows with event names ────
    # DATA-DRIVEN: reads active events directly from the events table and matches
    # to decomp feature rows by normalizing tactic/subtype → feature suffix.
    # No config table dependency. The tactic→suffix mapping mirrors event_features.py.
    _logger = get_logger("inference.batch_infer", env=catalog)
    try:
        _snap_date = output_df.select("snapshot_date").first()[0]
        _cust = output_df.select("market_id").first()[0]

        # Build the event-label lookup entirely from events table data.
        # The CASE WHEN mirrors the normalization in event_features.py — no config read.
        spark.sql(f"""
            CREATE OR REPLACE TEMP VIEW _promo_event_labels AS
            WITH events_daily AS (
                SELECT
                    e.market_id, e.sku_id, e.store_id, e.event_nm,
                    EXPLODE(SEQUENCE(e.start_date, e.end_date, INTERVAL 1 DAY)) AS date_id,
                    -- tactic → feature suffix (same logic as event_features.py)
                    CASE e.tactic
                        WHEN 'Temporary Price Change' THEN 'promo_count_temp_price_change'
                        WHEN 'Buy One, Get One - Mix' THEN 'promo_count_bogo_mix'
                        WHEN 'Buy One, Get One - Match' THEN 'promo_count_bogo_match'
                        WHEN 'Mix and Match - Mix' THEN 'promo_count_mix_match_mix'
                        WHEN 'Mix and Match - Match' THEN 'promo_count_mix_match_match'
                        WHEN 'Meal Bundle' THEN 'promo_count_meal_bundle'
                        WHEN 'Free Offer' THEN 'promo_count_free_offer'
                        WHEN 'No Tactic - Advertising Only' THEN 'promo_count_advertising'
                        WHEN 'Pairings (Entree + Side)' THEN 'promo_count_pairings'
                        ELSE 'promo_count_tactic_other'
                    END AS tactic_feature,
                    -- subtype → feature suffix
                    CASE e.event_subtype
                        WHEN 'LIMITED_TIME_OFFER' THEN 'promo_count_lto'
                        WHEN 'MARKET_OFFER_DIGITAL' THEN 'promo_count_market_digital'
                        WHEN 'LOCAL_OFFER' THEN 'promo_count_local_offer'
                        ELSE 'promo_count_subtype_other'
                    END AS subtype_feature
                FROM {catalog}.gold.events e
                WHERE e.market_id = '{_cust}'
                  AND e.event_type = 'PROMOTION'
                  AND e.exclude_flag = 0
            ),
            -- Unpivot: each event contributes to BOTH its tactic feature AND subtype feature
            event_features AS (
                SELECT market_id, sku_id, store_id, date_id, event_nm,
                       tactic_feature AS feature_name
                FROM events_daily
                UNION ALL
                SELECT market_id, sku_id, store_id, date_id, event_nm,
                       subtype_feature AS feature_name
                FROM events_daily
                WHERE subtype_feature IS NOT NULL
            )
            -- Concat event names per (feature, date, sku, store)
            SELECT feature_name, date_id, sku_id, store_id, market_id,
                   CONCAT_WS(' | ', COLLECT_SET(event_nm)) AS event_label
            FROM event_features
            GROUP BY feature_name, date_id, sku_id, store_id, market_id
        """)

        # Relabel component_name from feature name → event names, AGGREGATING
        # contributions when multiple features map to the same event.
        # A plain UPDATE SET component_name = event_label creates duplicate merge
        # keys when 2 features (tactic + subtype) resolve to the same event_label.
        # Fix: stage aggregated event-level rows, DELETE feature-level, INSERT aggregated.
        spark.sql(f"""
            CREATE OR REPLACE TEMP VIEW _promo_aggregated AS
            SELECT d.snapshot_date, d.run_id, d.market_id, d.store_id, d.sku_id,
                   d.date_id, d.component_type,
                   ev.event_label AS component_name,
                   SUM(d.contribution_raw) AS contribution_raw,
                   SUM(d.contribution) AS contribution,
                   FIRST(ev.tactic) AS tactic,
                   FIRST(ev.sub_type) AS sub_type,
                   FIRST(d.model_type) AS model_type,
                   FIRST(d.category) AS category,
                   d.champion_version,
                   FIRST(d.created_ts) AS created_ts,
                   FIRST(d.created_by) AS created_by,
                   current_timestamp() AS created_date,
                   current_timestamp() AS updated_date,
                   'promo_attribution' AS updated_by
            FROM {catalog}.forecast.decomposition d
            JOIN _promo_event_labels ev
              ON  d.market_id = ev.market_id
              AND d.sku_id = ev.sku_id
              AND d.store_id = ev.store_id
              AND d.date_id = ev.date_id
              AND d.component_name = ev.feature_name
              AND d.component_type = 'tactic_subtype_lift'
            WHERE d.snapshot_date = DATE('{_snap_date}')
              AND d.market_id = '{_cust}'
            GROUP BY d.snapshot_date, d.run_id, d.market_id, d.store_id,
                     d.sku_id, d.date_id, d.component_type, ev.event_label,
                     d.champion_version
        """)
        # Materialize the staged view before deleting source rows
        spark.sql("CACHE TABLE _promo_aggregated")

        spark.sql(f"""
            DELETE FROM {catalog}.forecast.decomposition
            WHERE snapshot_date = DATE('{_snap_date}')
              AND market_id = '{_cust}'
              AND component_type = 'tactic_subtype_lift'
              AND component_name IN (SELECT DISTINCT feature_name FROM _promo_event_labels)
        """)
        spark.sql(f"""
            INSERT INTO {catalog}.forecast.decomposition
                (snapshot_date, run_id, market_id, store_id, sku_id,
                 date_id, component_type, component_name, contribution_raw,
                 contribution, tactic, sub_type, model_type, category,
                 champion_version, created_ts, created_by, created_date,
                 updated_date, updated_by)
            SELECT snapshot_date, run_id, market_id, store_id, sku_id,
                   date_id, component_type, component_name, contribution_raw,
                   contribution, tactic, sub_type, model_type, category,
                   champion_version, created_ts, created_by, created_date,
                   updated_date, updated_by
            FROM _promo_aggregated
        """)
        spark.sql("UNCACHE TABLE _promo_aggregated")
        # INSERT zero-lift rows for (sku, store, date) combos that have active
        # events but NO existing promo_lift row at all in decomposition.
        # One row per (sku, store, date) with ALL active events pipe-separated.
        # Only for horizon days (date_id > snapshot_date).
        _run_id = output_df.select("run_id").first()[0] if "run_id" in output_df.columns else "zero_fill"
        spark.sql(f"""
            INSERT INTO {catalog}.forecast.decomposition
                (snapshot_date, run_id, market_id, store_id, sku_id,
                 date_id, component_type, component_name, contribution_raw,
                 contribution, tactic, sub_type, created_ts, model_type,
                 category, champion_version, created_date, created_by,
                 updated_date, updated_by)
            WITH events_combined AS (
                -- One row per (sku, store, date) with all event names concatenated
                SELECT market_id, sku_id, store_id, date_id,
                       CONCAT_WS(' | ', COLLECT_SET(event_nm)) AS all_events
                FROM (
                    SELECT e.market_id, e.sku_id, e.store_id, e.event_nm,
                           EXPLODE(SEQUENCE(
                               GREATEST(e.start_date, DATE('{_snap_date}') + INTERVAL 1 DAY),
                               LEAST(e.end_date, DATE('{_snap_date}') + INTERVAL 140 DAY),
                               INTERVAL 1 DAY
                           )) AS date_id
                    FROM {catalog}.gold.events e
                    WHERE e.market_id = '{_cust}'
                      AND e.event_type = 'PROMOTION'
                      AND e.exclude_flag = 0
                      AND e.end_date > DATE('{_snap_date}')
                      AND e.start_date <= DATE('{_snap_date}') + INTERVAL 140 DAY
                )
                GROUP BY market_id, sku_id, store_id, date_id
            ),
            existing_promo AS (
                -- (sku, store, date) combos that already have an EVENT-LABELED
                -- promo_lift row (from relabeling or a prior zero-fill). Raw feature
                -- rows (promo_active_flag, etc.) do NOT count — they don't satisfy
                -- the guarantee of named-event attribution.
                SELECT DISTINCT sku_id, store_id, date_id
                FROM {catalog}.forecast.decomposition
                WHERE market_id = '{_cust}'
                  AND snapshot_date = DATE('{_snap_date}')
                  AND component_type = 'promo_lift'
                  AND component_name NOT RLIKE
                      '^promo_(active_flag|days_since_end|days_until_next|duration|concurrent_count|days_since_start|diversity|freshness)$'
            ),
            meta AS (
                SELECT DISTINCT champion_version, market_id, category,
                       sku_id, store_id, model_type
                FROM {catalog}.forecast.output
                WHERE market_id = '{_cust}'
                  AND snapshot_date = DATE('{_snap_date}')
                  AND horizon_day = 1
            )
            SELECT
                DATE('{_snap_date}') AS snapshot_date,
                '{_run_id}' AS run_id,
                ec.market_id,
                ec.store_id,
                ec.sku_id,
                ec.date_id,
                'promo_lift' AS component_type,
                ec.all_events AS component_name,
                0.0 AS contribution_raw,
                0.0 AS contribution,
                CAST(NULL AS STRING) AS tactic,
                CAST(NULL AS STRING) AS sub_type,
                current_timestamp() AS created_ts,
                m.model_type,
                m.category,
                m.champion_version,
                current_timestamp() AS created_date,
                'post_hoc_zero_fill' AS created_by,
                current_timestamp() AS updated_date,
                'post_hoc_zero_fill' AS updated_by
            FROM events_combined ec
            LEFT JOIN existing_promo ep
              ON  ec.sku_id = ep.sku_id
              AND ec.store_id = ep.store_id
              AND ec.date_id = ep.date_id
            INNER JOIN meta m
              ON  ec.sku_id = m.sku_id
              AND ec.store_id = m.store_id
              AND ec.market_id = m.market_id
            WHERE ep.sku_id IS NULL
        """)

        _logger.info(
            f"Promo attribution: relabeled per-feature rows with event names "
            f"and inserted zero-lift rows for snapshot {_snap_date}."
        )

    except Exception as exc:
        _logger.warning(f"Promo attribution enrichment failed (non-fatal): {exc}")