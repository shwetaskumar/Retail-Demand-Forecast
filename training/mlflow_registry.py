"""
training/mlflow_registry.py — MLflow experiment tracking + UC Model Registry.

Called once per monthly training run after champions are selected.

Design (fast registration):
  - Per-series champion state lives ONLY in Delta (champion_index). It is
    never collected to the driver and never embedded in the MLflow artifact.
  - Registration logs per-category METRICS (a small Spark aggregation) plus a
    THIN pyfunc whose artifact is just meta.json. At predict time the pyfunc
    reads champion_index from Delta for the requested series. This keeps the
    logged/registered artifact at a few KB, so UC registration is near-instant.
  - batch_infer (Delta → applyInPandas) remains the operational scorer; the
    registered pyfunc is a governed, aliased, ad-hoc/lineage entry point.

Experiment layout:
  /Shared/fcst-engine/{market}
    └── parent run: "{market}_training"
          ├── tags/metrics: market-level summary
          └── nested run per category → thin pyfunc logged + registered @champion

UC Model Registry: {catalog}.gold.{market_lower}_{cat_slug}
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from typing import TYPE_CHECKING

import mlflow
import mlflow.pyfunc
import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

# /Shared/ is a standard top-level folder present in every Databricks workspace
# regardless of who's running the job — no hardcoded username, portable across
# workspaces/deployments (unlike a specific user's home folder).
_EXPERIMENT_BASE = "/Shared/fcst-engine"
_MODEL_SCHEMA    = "gold"
_MAX_VERSIONS    = 3  # retention: keep only the latest N model versions


# ── Pyfunc model class (thin — reads champions from Delta at score time) ───────

class CategoryForecasterPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc for one market × category.

    Artifact is only meta.json — NO per-series params are embedded, so the model
    is a few KB regardless of how many series the category has. At predict time it
    reads the champions for the requested series from champion_index (Delta)
    and scores them with the shared _score_one_series routine.

    Requires an active SparkSession at predict time (batch/notebook use), which is
    the intended usage — the high-throughput path is inference.batch_infer.
    """

    def __init__(self, repo_root: str) -> None:
        # Captured once at logging time (register_champions_to_mlflow resolves it
        # from the live training notebook's own workspace path — see
        # notebooks/02_train.py) and pickled as instance state, so the loaded
        # pyfunc knows its repo root at predict time with no hardcoded username
        # baked into the source.
        if not repo_root:
            raise ValueError(
                "CategoryForecasterPyfunc requires repo_root — pass the caller's own "
                "resolved workspace path (see register_champions_to_mlflow's repo_root arg)."
            )
        self._repo_root = repo_root

    def load_context(self, context):
        import sys
        if self._repo_root not in sys.path:
            sys.path.insert(0, self._repo_root)
        with open(context.artifacts["champion_meta"]) as f:
            self._meta: dict = json.load(f)

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        import sys
        if self._repo_root not in sys.path:
            sys.path.insert(0, self._repo_root)
        from inference.batch_infer import _score_one_series  # noqa: PLC0415
        from pyspark.sql import SparkSession  # noqa: PLC0415

        meta          = self._meta
        snapshot_date = (params or {}).get("snapshot_date", meta.get("snapshot_date", ""))
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError(
                "CategoryForecasterPyfunc.predict requires an active SparkSession — it reads "
                "champions from champion_index. For high-throughput scoring use "
                "inference.batch_infer instead."
            )

        catalog  = meta["catalog"]
        market   = meta["market_id"].replace("'", "''")
        category = meta["category"].replace("'", "''")
        snap     = str(snapshot_date).replace("'", "''")

        # Only fetch champions for the series actually present in the input.
        wanted = [str(s) for s in pd.unique(model_input["series_key"])]
        if not wanted:
            return pd.DataFrame()
        in_list = ",".join("'" + w.replace("'", "''") + "'" for w in wanted)
        champ_pdf = spark.sql(f"""
            SELECT series_key, model_type, params
            FROM {catalog}.gold.champion_index
            WHERE market_id = '{market}' AND category = '{category}'
              AND snapshot_date = DATE('{snap}')
              AND series_key IN ({in_list})
        """).toPandas()
        champ_by_series = {
            str(r.series_key): (r.model_type, r.params) for r in champ_pdf.itertuples()
        }

        results = []
        for sk, grp in model_input.groupby("series_key"):
            rec = champ_by_series.get(str(sk))
            if rec is None:
                continue
            model_type, param_json = rec
            grp = grp.copy()
            grp["model_type"] = model_type
            grp["params"]     = param_json
            try:
                scored = _score_one_series(grp.reset_index(drop=True), snapshot_date)
                if not scored.empty:
                    results.append(scored)
            except Exception:
                continue

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _cat_slug(category: str) -> str:
    """Convert category name to a valid UC model-name component."""
    return re.sub(r"[^a-z0-9_]", "_", category.lower().strip())


# ── Public API ────────────────────────────────────────────────────────────────

def register_champions_to_mlflow(
    spark: "SparkSession",
    catalog: str,
    market_id: str,
    snapshot_date: str,
    champions_df: "DataFrame",
    features_df: "DataFrame | None" = None,
    training_set=None,
    repo_root: str | None = None,
) -> dict[str, str]:
    """Log training metrics and register one thin pyfunc per market × category.

    Does NOT collect per-series params to the driver — only small per-category
    metric aggregations. Champion state stays in champion_index (Delta).

    `repo_root` is the caller's own resolved workspace path (notebooks/02_train.py
    resolves it dynamically from its own notebook context — see that file's
    sys.path cell) — captured into each registered CategoryForecasterPyfunc so it
    knows where to find `inference.batch_infer` at predict time, with no hardcoded
    username in this module.
    """
    from pyspark.sql import functions as _F

    mlflow.set_registry_uri("databricks-uc")

    # Compute schema once to avoid repeated Analyze RPCs (SCPAP001)
    _champ_cols = champions_df.columns
    has_volume = "volume" in _champ_cols

    # ── Real accuracy gate result from backtest_metrics ─────────────────────
    # passed_gate in champion_index = passed_decomposition (reconstruction
    # check, always True for any champion by construction). The real gate —
    # whether the category met its WMAPE accuracy threshold — lives in
    # backtest_metrics.passed (grain='category_cohort', horizon=7).
    safe_market_id = market_id.replace("'", "''")
    safe_snap   = snapshot_date.replace("'", "''")
    gate_metrics: dict[str, dict] = {
        r["category"]: {
            "gate_passed":       bool(r["passed"]),
            "gate_wmape_pct":    round(float(r["wmape"] or 0) * 100, 4),
            "gate_accuracy_pct": round(100 - float(r["wmape"] or 0) * 100, 4),
            "gate_threshold_pct": round(float(r["threshold"] or 0) * 100, 4),
        }
        for r in spark.sql(f"""
            SELECT category, wmape, threshold, passed
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY category ORDER BY computed_ts DESC
                ) AS rn
                FROM {catalog}.gold.backtest_metrics
                WHERE market_id = '{safe_market_id}'
                  AND snapshot_date = DATE('{safe_snap}')
                  AND grain   = 'category_cohort'
                  AND horizon = 7
            ) WHERE rn = 1
        """).collect()
        if r["category"] is not None
    }

    # ── Market-level summary (single-row Spark aggregation) ───────────────────
    # wmape_pct follows wmape_by_snapshot() SAE-WMAPE logic:
    #   SUM(abs_error) / SUM(actual) × 100
    # At series level: abs_error ≈ wmape_7d × volume  (wmape_7d = abs_error/actual)
    # → market wmape_pct = SUM(wmape_7d × volume) / SUM(volume) × 100
    # This gives high-volume items their correct weight; unweighted avg(wmape_7d)
    # treats a 1-unit candy and a 10,000-unit burger equally.
    if has_volume:
        _wmape_expr = (
            _F.sum(_F.col("wmape_7d") * _F.col("volume"))
            / _F.when(_F.sum("volume") > 0, _F.sum("volume")).otherwise(None)
            * 100
        ).alias("wmape_pct")
    else:
        _wmape_expr = (_F.avg("wmape_7d") * 100).alias("wmape_pct")

    _market_aggs = [_F.count(_F.lit(1)).alias("total_champions"), _wmape_expr]
    market_row = champions_df.agg(*_market_aggs).collect()[0].asDict()

    if not market_row["total_champions"]:
        return {}

    # ── Per-category metrics (small aggregation → collect ~6 rows) ─────────────
    if has_volume:
        _cat_wmape_expr = (
            _F.sum(_F.col("wmape_7d") * _F.col("volume"))
            / _F.when(_F.sum("volume") > 0, _F.sum("volume")).otherwise(None)
            * 100
        ).alias("wmape_pct")
    else:
        _cat_wmape_expr = (_F.avg("wmape_7d") * 100).alias("wmape_pct")

    _cat_aggs = [
        _F.count(_F.lit(1)).alias("n_series"),
        _cat_wmape_expr,
        # Distribution percentiles (supplementary — not used for gate decisions)
        _F.expr("percentile_approx(wmape_7d * 100, 0.5)").alias("p50_wmape_pct"),
        _F.expr("percentile_approx(wmape_7d * 100, 0.9)").alias("p90_wmape_pct"),
    ]
    if has_volume:
        _cat_aggs.append(_F.sum("volume").alias("total_volume"))
    cat_metrics = {
        r["category"]: r.asDict()
        for r in champions_df.groupBy("category").agg(*_cat_aggs).collect()
        if r["category"] is not None
    }

    experiment_path = f"{_EXPERIMENT_BASE}/{market_id}"
    mlflow.set_experiment(experiment_path)

    run_ids: dict[str, str] = {}

    with mlflow.start_run(run_name=f"{market_id}_training") as parent_run:
        mlflow.set_tags({
            "market_id": market_id,
            "snapshot_date": snapshot_date,
            "n_categories":  str(len(cat_metrics)),
            "catalog":       catalog,
        })
        _mkt_wmape = float(market_row.get("wmape_pct") or 0.0)
        _n_passed  = sum(1 for g in gate_metrics.values() if g["gate_passed"])
        _market_metrics = {
            "total_champions":      float(market_row["total_champions"]),
            "wmape_pct":            round(_mkt_wmape, 4),
            "accuracy_pct":         round(100 - _mkt_wmape, 4),
            # Gate: categories that passed their WMAPE accuracy threshold
            "n_categories_passed":  float(_n_passed),
            "n_categories_total":   float(len(gate_metrics)),
            "pct_categories_passed": round(_n_passed / len(gate_metrics) * 100, 1) if gate_metrics else 0.0,
        }
        mlflow.log_metrics(_market_metrics)

        # ── Dataset lineage ───────────────────────────────────────────────────
        _feat_table = training_set if isinstance(training_set, str) else f"{catalog}.gold.sku_store_features"
        try:
            # Use an alias so 'mlflow' is never bound as a local variable in this
            # function — an unaliased `import mlflow.data` would shadow the
            # module-level import and cause UnboundLocalError on any earlier
            # mlflow.* call (Python marks the name local for the whole scope).
            import mlflow.data as _mlflow_data
            feature_ds = _mlflow_data.from_spark(features_df, table_name=_feat_table, version=snapshot_date)
            mlflow.log_input(feature_ds, context="training")
        except Exception:
            mlflow.log_param("feature_table",   _feat_table)
            mlflow.log_param("feature_snapshot", snapshot_date)

        # ── Feature drift stats (bounded Spark aggregation) ───────────────────
        if features_df is not None:
            # Compute schema once outside try (SCPAP001)
            _feat_cols = features_df.columns
            numeric_cols = [
                c for c in _feat_cols
                if any(c.startswith(p) for p in ("ewma_", "lag_", "rolling_", "qty"))
            ]
            if numeric_cols:
                # Build lazy transformation outside try; action inside try (SCPAP005)
                _stats_df = features_df.agg(
                    *[_F.mean(c).alias(f"mean_{c}") for c in numeric_cols[:10]],
                    *[_F.stddev(c).alias(f"std_{c}") for c in numeric_cols[:10]],
                    _F.mean("isActive").alias("pct_active"),
                    _F.count("*").alias("n_rows"),
                )
                try:
                    stats = _stats_df.collect()[0].asDict()
                    mlflow.log_metrics({k: float(v) for k, v in stats.items() if v is not None})
                except Exception:
                    pass

        for category in sorted(cat_metrics.keys()):
            run_ids[category] = _register_category(
                catalog, market_id, category, snapshot_date,
                metrics=cat_metrics[category],
                gate=gate_metrics.get(category, {}),
                parent_run_id=parent_run.info.run_id,
                repo_root=repo_root,
            )

    return run_ids


def _register_category(
    catalog: str,
    market_id: str,
    category: str,
    snapshot_date: str,
    metrics: dict,
    gate: dict,
    parent_run_id: str,
    repo_root: str | None,
) -> str:
    """Log one category nested run + a THIN pyfunc (meta-only), register @champion."""
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, ColSpec

    slug       = _cat_slug(category)
    model_name = f"{catalog}.{_MODEL_SCHEMA}.{market_id.lower()}_{slug}"
    n_series   = int(metrics.get("n_series", 0))

    with mlflow.start_run(
        run_name=f"{market_id}_{category}",
        nested=True,
        tags={"market_id": market_id, "category": category, "snapshot_date": snapshot_date, "model_type": "multi"},
    ) as run:
        _wmape = float(metrics.get("wmape_pct") or 0.0)
        _cat_metrics = {
            "n_series":      float(n_series),
            # Volume-weighted SAE-WMAPE — matches wmape_by_snapshot() definition
            "wmape_pct":     round(_wmape, 4),
            "accuracy_pct": round(100 - _wmape, 4),
            # Series-level distribution (not volume-weighted; supplementary)
            "p50_wmape_pct": float(metrics.get("p50_wmape_pct") or 0.0),
            "p90_wmape_pct": float(metrics.get("p90_wmape_pct") or 0.0),
        }
        if metrics.get("total_volume") is not None:
            _cat_metrics["total_volume"] = float(metrics["total_volume"])
        # Gate result from backtest_metrics (grain=category_cohort, horizon=7)
        # This is the real production accuracy gate, not the decomposition check.
        if gate:
            _cat_metrics["gate_passed"]        = float(gate["gate_passed"])
            _cat_metrics["gate_wmape_pct"]      = gate["gate_wmape_pct"]
            _cat_metrics["gate_accuracy_pct"]   = gate["gate_accuracy_pct"]
            _cat_metrics["gate_threshold_pct"]  = gate["gate_threshold_pct"]
        mlflow.log_metrics(_cat_metrics)
        mlflow.log_params({
            "market_id": market_id, "category": category,
            "snapshot_date": snapshot_date, "n_series": n_series,
        })

        # Thin artifact: meta only. No per-series params embedded.
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "champion_meta.json")
            with open(meta_path, "w") as f:
                json.dump({
                    "catalog":       catalog,
                    "market_id": market_id,
                    "category":      category,
                    "snapshot_date": snapshot_date,
                }, f)

            input_schema = Schema([
                ColSpec("string",  "series_key"),
                ColSpec("string",  "date_id"),
                ColSpec("double",  "qty"),
                ColSpec("integer", "isActive"),
            ])
            output_schema = Schema([
                ColSpec("string", "record_type"),
                ColSpec("string", "series_key"),
                ColSpec("string", "market_id"),
                ColSpec("string", "category"),
                ColSpec("string", "store_id"),
                ColSpec("string", "sku_id"),
                ColSpec("string", "date_id"),
                ColSpec("double", "yhat"),
            ])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=CategoryForecasterPyfunc(repo_root=repo_root),
                artifacts={"champion_meta": meta_path},
                signature=signature,
                registered_model_name=model_name,
                pip_requirements=[
                    "statsforecast==1.7.8",
                    "scikit-learn",
                    "lightgbm==4.5.0",
                ],
                metadata={
                    "market_id": market_id, "category": category,
                    "snapshot_date": snapshot_date, "n_series": n_series,
                },
            )

        run_id = run.info.run_id

    # @champion alias on the latest version + retention
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
            client.set_registered_model_alias(model_name, "champion", sorted_versions[0].version)
            for old_v in sorted_versions[_MAX_VERSIONS:]:
                try:
                    client.delete_model_version(model_name, old_v.version)
                except Exception:
                    pass
    except Exception:
        pass

    return run_id
