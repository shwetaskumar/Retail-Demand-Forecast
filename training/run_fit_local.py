"""
training/run_fit_local.py — Task 1: batch local model fitting (MMF architecture).

Uses run_engine_batch which groups by store_id (~666 Spark tasks of ~160
series each) and fits StatsForecast AutoETS + Ridge event regression per batch.

Expected: ~3 min for 106K series vs 35+ min with per-series Prophet approach.

Decomposition contract maintained: yhat = trend + seasonality + Σevent_lift.

Parameters: --market_id --env --snapshot_date --train_start_date
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path

# Unlike a notebook cell, __file__ IS available here — this only ever runs as a
# genuine local/Databricks-Connect script (never a notebook_task), so resolving
# the repo root from this file's own location is portable across machines/
# workspaces with no hardcoded username or path.
_REPO_ROOT = str(_Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
del _sys, _Path, _REPO_ROOT

import argparse


def main(market_id: str, env: str, snapshot_date: str | None = None, train_start_date: str | None = None) -> None:
    from utils.dates import resolve_snapshot_date
    from utils.io import read_table
    from utils.logging import get_logger
    from utils.spark import get_or_create_spark
    from config.loader import load_config
    from training.champion import select_champions_spark, write_champion_index
    from training.engine import run_engine_batch

    catalog = env
    spark   = get_or_create_spark(f"job2-fit-local-{market_id}")
    logger  = get_logger(__name__, market_id=market_id, env=env)

    as_of           = resolve_snapshot_date(snapshot_date)
    as_of_str       = as_of.isoformat()
    config          = load_config(spark, catalog, market_id)
    train_start     = resolve_snapshot_date(train_start_date) if train_start_date else config.market_config.history_start
    train_start_str = train_start.isoformat() if hasattr(train_start, "isoformat") else str(train_start)

    logger.info(
        f"FIT_LOCAL (batch/MMF) starting market_id={market_id} snapshot_date={as_of_str} "
        f"train_start={train_start_str} — using StatsForecast AutoETS + Ridge event regression"
    )

    features_df = read_table(spark, catalog, "gold", "sku_store_features")
    features_df.createOrReplaceTempView("_all_features")
    # Use the single most-recent snapshot <= training date.
    # Each snapshot already covers its own year; no cross-snapshot dedup needed.
    # This avoids an expensive ROW_NUMBER shuffle over multiple partitions.
    training_features_df = spark.sql(f"""
        SELECT * FROM _all_features
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date) FROM _all_features
            WHERE snapshot_date <= DATE('{as_of_str}')
        )
        AND date_id BETWEEN DATE('{train_start_str}') AND DATE('{as_of_str}')
    """)

    # Batch engine: ~666 location tasks × ~160 series each
    candidate_df = run_engine_batch(spark, training_features_df, config, as_of_str)
    candidate_df.cache()
    total = candidate_df.count()
    logger.info(f"FIT_LOCAL batch complete: {total} candidate rows")
    for row in candidate_df.groupBy("status").count().collect():
        logger.info(f"  status={row['status']} count={row['count']}")

    champions_df = select_champions_spark(spark, candidate_df)
    write_champion_index(spark, champions_df, catalog, as_of_str)
    n_champ = champions_df.count()
    logger.info(f"FIT_LOCAL: {n_champ} champions written to champion_index for market_id={market_id}")

    # ── MLflow: log metrics + register pyfunc per category in UC Model Registry ──
    if n_champ > 0:
        try:
            from pathlib import Path as _Path
            from training.mlflow_registry import register_champions_to_mlflow
            run_ids = register_champions_to_mlflow(
                spark, catalog, market_id, as_of_str, champions_df,
                features_df=training_features_df,
                repo_root=str(_Path(__file__).resolve().parent.parent),
            )
            n_cats = len(run_ids)
            logger.info(
                f"FIT_LOCAL: MLflow registered {n_cats} category pyfuncs "
                f"(market_id={market_id}, snapshot={as_of_str}) — @champion alias set"
            )
        except Exception as mlflow_exc:
            # MLflow registration is non-fatal — champions are already written to
            # champion_index; inference does not depend on MLflow.
            logger.warning(f"FIT_LOCAL: MLflow registration failed (non-fatal): {mlflow_exc}")
    else:
        logger.warning("FIT_LOCAL: 0 champions — skipping MLflow registration")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--market_id", required=True)
    p.add_argument("--env", required=True, choices=["dev", "prod"])
    p.add_argument("--snapshot_date", default=None)
    p.add_argument("--train_start_date", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(market_id=args.market_id, env=args.env, snapshot_date=args.snapshot_date, train_start_date=args.train_start_date)
