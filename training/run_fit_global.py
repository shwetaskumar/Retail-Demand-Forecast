"""
training/run_fit_global.py — Task 2: per-SKU global LightGBM (MMF pattern).

Trains one LightGBM per sku_id pooling all locations. Promotes the global model
as champion for a series only if its holdout WMAPE beats the local champion
(Prophet/SF) already registered by run_fit_local.

Parameters: --market --env --snapshot_date --train_start_date
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
    from training.champion import write_champion_index
    from training.global_lgbm import run_global_lgbm

    catalog = env
    spark   = get_or_create_spark(f"job2-fit-global-{market_id}")
    logger  = get_logger(__name__, market_id=market_id, env=env)

    as_of           = resolve_snapshot_date(snapshot_date)
    as_of_str       = as_of.isoformat()
    config          = load_config(spark, catalog, market_id)
    train_start     = resolve_snapshot_date(train_start_date) if train_start_date else config.market_config.history_start
    train_start_str = train_start.isoformat() if hasattr(train_start, "isoformat") else str(train_start)

    logger.info(f"FIT_GLOBAL starting market_id={market_id} snapshot_date={as_of_str}")

    features_df = read_table(spark, catalog, "gold", "sku_store_features")
    features_df.createOrReplaceTempView("_all_features")
    training_features_df = spark.sql(f"""
        WITH eligible AS (
            SELECT * FROM _all_features
            WHERE snapshot_date <= DATE('{as_of_str}')
              AND date_id BETWEEN DATE('{train_start_str}') AND DATE('{as_of_str}')
        ),
        latest_per_day AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY store_id, sku_id, date_id
                ORDER BY snapshot_date DESC
            ) AS rn FROM eligible
        )
        SELECT * EXCEPT (rn) FROM latest_per_day WHERE rn = 1
    """)

    # Fit global per-SKU LightGBM
    global_candidates_df = run_global_lgbm(spark, training_features_df, market_id, as_of_str,
                                            horizon=config.market_config.eval_horizon_days)
    global_candidates_df = global_candidates_df.localCheckpoint()  # .cache() unsupported on Free Edition serverless
    total_global = global_candidates_df.count()
    logger.info(f"FIT_GLOBAL: {total_global} global candidate rows produced")

    if total_global == 0:
        logger.info("FIT_GLOBAL: no global candidates — local champions unchanged")
        return

    # Compare against existing local champions; promote global only where it wins
    champion_df = read_table(spark, catalog, "gold", "champion_index")
    champion_df.createOrReplaceTempView("_local_champions")
    global_candidates_df.createOrReplaceTempView("_global_candidates")

    promoted_df = spark.sql("""
        SELECT g.*
        FROM _global_candidates g
        JOIN _local_champions c
          ON g.series_key = c.series_key
         AND g.snapshot_date = c.snapshot_date
        WHERE g.wmape_7d < c.wmape_7d   -- promote only if global is genuinely better
          AND g.status = 'ok'
    """)
    n_promoted = promoted_df.count()
    logger.info(f"FIT_GLOBAL: {n_promoted} series promoted to global LightGBM champion")

    if n_promoted > 0:
        write_champion_index(spark, promoted_df, catalog, as_of_str)

    logger.info(f"FIT_GLOBAL complete market_id={market_id}")


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
