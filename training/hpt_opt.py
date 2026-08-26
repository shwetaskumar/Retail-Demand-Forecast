"""
training/hpt.py — conditional Optuna HPT, threshold-miss only, cached (docs/instructions.md §12).

Runs only in the weekly/triggered training job (notebooks/02_train.py, and the
daily conditional-retune notebooks/05_hpt_retune.py), never in the daily
critical path otherwise. Only tunes series that missed a threshold — per
docs/instructions.md §21 acceptance criterion 3, "any level's failure routes
the affected series to diagnosis/conditional-HPT", so a series is selected for
tuning if EITHER its own market_store_sku grain failed, OR its market_sku
or category_cohort grain failed (even if that one series individually passed
— the cohort/sku aggregate still needs improving).

Caching is implicit, not a separate flag: `needs_tuning` is driven by
`training/evaluation.py::gate`'s per-run pass/fail. Once a retuned champion is
written back to `champion_index` (training/champion.py::write_champion_index,
called from notebooks/02_train.py / notebooks/05_hpt_retune.py) and it passes
on a later run, gate() will show passed=True for it and it simply stops
appearing in needs_tuning's output — no re-tune-tracking table needed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pandas as pd

if TYPE_CHECKING:
    import optuna
    from pyspark.sql import DataFrame, SparkSession

from config.loader import ForecastConfig
from training.backtest import backtest_model
from training.models.base import ForecastModel

_UPLIFT_SUFFIX = "+uplift"


def needs_tuning(gated_pdf: pd.DataFrame, champions_pdf: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of `champions_pdf` (one per series) that need conditional HPT.

    `gated_pdf` is training/evaluation.py::gate's output (all three grains);
    `champions_pdf` is training/champion.py::select_champions(_spark)'s output (one row
    per series, with market/category/sku_id/series_key). A series needs tuning if its
    own market_store_sku-grain row failed, or the market_sku or category_cohort row
    covering it failed. Rows with `passed` still null (no threshold configured) never
    trigger tuning — there's nothing to gate against.
    """
    if gated_pdf.empty or champions_pdf.empty:
        return champions_pdf.iloc[0:0]

    failing = gated_pdf[gated_pdf["passed"] == False]  # noqa: E712 -- explicit False, not NaN/None
    if failing.empty:
        return champions_pdf.iloc[0:0]

    series_grain_failures = set(failing.loc[failing["grain"] == "market_store_sku", "entity_key"])
    failing_sku_keys = set(
        zip(
            failing.loc[failing["grain"] == "market_sku", "market"],
            failing.loc[failing["grain"] == "market_sku", "category"],
            failing.loc[failing["grain"] == "market_sku", "entity_key"],
        )
    )
    failing_category_keys = set(
        zip(
            failing.loc[failing["grain"] == "category_cohort", "market_id"],
            failing.loc[failing["grain"] == "category_cohort", "category"],
        )
    )

    champions_pdf = champions_pdf.assign(
        _sku_key=list(zip(champions_pdf["market_id"], champions_pdf["category"], champions_pdf["sku_id"])),
        _category_key=list(zip(champions_pdf["market_id"], champions_pdf["category"])),
    )
    needs_it = (
        champions_pdf["series_key"].isin(series_grain_failures)
        | champions_pdf["_sku_key"].isin(failing_sku_keys)
        | champions_pdf["_category_key"].isin(failing_category_keys)
    )
    return champions_pdf.loc[needs_it].drop(columns=["_sku_key", "_category_key"]).reset_index(drop=True)


def _suggest_params(trial: "optuna.Trial", param_space: dict) -> dict:
    """`param_space` schema: {"name": {"type": "int"|"float"|"categorical", "low"/"high"
    (+ optional "step"/"log" for int/float), or "choices" for categorical}}."""
    params = {}
    for name, spec in param_space.items():
        kind = spec.get("type", "float")
        if kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unknown param_space type {kind!r} for param {name!r}")
    return params


def tune_series(
    model_factory: Callable[[], ForecastModel],
    series_pdf: pd.DataFrame,
    param_space: dict,
    n_trials: int,
    n_folds: int,
    val_weeks: int,
    eval_horizon_days: int,
    params_transform: Callable[[dict], dict] = lambda p: p,
) -> dict:
    """Run a capped-trial Optuna search (pruning on) minimizing backtest WMAPE over
    `param_space`; return the best RAW (pre-transform) params found, or `{}` if
    `param_space` is empty or every trial fails/prunes.

    `params_transform` reshapes the flat Optuna-suggested dict into whatever
    `model_factory()`'s `fit()` expects: identity for a plain Stage-1 model; nests the
    raw dict under "stage1_params" (+ the market's uplift settings) for an
    uplift-wrapped one — see tune_one_series.
    """
    if not param_space:
        return {}

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        raw_params = _suggest_params(trial, param_space)
        score = backtest_model(
            model_factory(), series_pdf, n_folds, val_weeks, eval_horizon_days, params_transform(raw_params)
        )
        if score != score:  # NaN: no usable folds / undefined WMAPE for this trial
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:  # every trial failed/pruned — study.best_trial would raise, not return None
        return {}
    return study.best_params


def tune_one_series(series_pdf: pd.DataFrame, config: ForecastConfig, snapshot_date: str) -> pd.DataFrame:
    """Per-group function (pandas in, pandas out) run inside applyInPandas
    (training/hpt.py::run_conditional_hpt): re-tune the ONE model_type that is this
    series' current (failing) champion — base type, +uplift if applicable — then
    re-evaluate with the tuned params via training.engine.evaluate_candidate (identical
    validation/row shape to the default-param pass).

    Requires a `champion_model_type` column on `series_pdf` (joined in by
    run_conditional_hpt before grouping).
    """
    from training.engine import OUTPUT_SCHEMA_COLUMNS, STAGE1_MODEL_REGISTRY, evaluate_candidate, result_row
    from training.models.uplift import UpliftModel

    pdf = series_pdf[series_pdf["isActive"] == 1].sort_values("date_id").reset_index(drop=True)
    if pdf.empty:
        return pd.DataFrame(columns=OUTPUT_SCHEMA_COLUMNS)

    series_key = pdf["series_key"].iloc[0]
    market = pdf["market_id"].iloc[0]
    category = pdf["category"].iloc[0] if "category" in pdf.columns else None
    store_id = pdf["store_id"].iloc[0]
    sku_id = pdf["sku_id"].iloc[0]
    champion_model_type = pdf["champion_model_type"].iloc[0]
    identity = (series_key, market, category, store_id, sku_id, snapshot_date)

    market_cfg = config.market_config
    is_uplift = champion_model_type.endswith(_UPLIFT_SUFFIX)
    base_model_type = champion_model_type[: -len(_UPLIFT_SUFFIX)] if is_uplift else champion_model_type

    model_cfg = next((m for m in config.models if m.model_type == base_model_type), None)
    if model_cfg is None or base_model_type not in STAGE1_MODEL_REGISTRY or not model_cfg.param_space:
        return pd.DataFrame(
            [result_row(*identity, model_type=champion_model_type, params=None, status="no_tunable_param_space")]
        )
    model_cls = STAGE1_MODEL_REGISTRY[base_model_type]

    if is_uplift:

        def model_factory() -> ForecastModel:
            return UpliftModel(model_cls())

        def params_transform(raw: dict) -> dict:
            return {
                "uplift_clip_frac": market_cfg.uplift_clip_frac,
                "uplift_learner": market_cfg.uplift_learner,
                "stage1_params": raw,
            }
    else:

        def model_factory() -> ForecastModel:
            return model_cls()

        def params_transform(raw: dict) -> dict:
            return raw

    best_raw_params = tune_series(
        model_factory,
        pdf,
        model_cfg.param_space,
        market_cfg.hpt_trials,
        market_cfg.backtest_folds,
        market_cfg.val_weeks,
        market_cfg.eval_horizon_days,
        params_transform=params_transform,
    )
    tuned_params = params_transform(best_raw_params)
    row = evaluate_candidate(model_factory(), champion_model_type, tuned_params, pdf, market_cfg, identity)
    return pd.DataFrame([row])


def run_conditional_hpt(
    spark: "SparkSession",
    features_df: "DataFrame",
    gated_pdf: pd.DataFrame,
    champions_pdf: pd.DataFrame,
    config: ForecastConfig,
    snapshot_date: str,
) -> "DataFrame":
    """Distribute conditional HPT across the cluster for exactly the series that need it
    (needs_tuning) — a second, narrower applyInPandas pass over `features_df`, joined down
    to only those series_keys. Returns the same row shape as training/engine.py's
    default-param pass; notebooks/02_train.py / notebooks/05_hpt_retune.py write the
    improved subset (training.hpt.select_improved_retunes) back into champion_index.
    """
    from training.engine import OUTPUT_SCHEMA, tag_series_key

    to_tune_pdf = needs_tuning(gated_pdf, champions_pdf)
    if to_tune_pdf.empty:
        return spark.createDataFrame([], schema=OUTPUT_SCHEMA)

    tagged_df = tag_series_key(spark, features_df, config.market_config.market_id, "_hpt_features_input")
    lookup_pdf = to_tune_pdf[["series_key", "model_type"]].rename(columns={"model_type": "champion_model_type"})
    lookup_df = spark.createDataFrame(lookup_pdf)
    tagged_df.createOrReplaceTempView("_hpt_tagged_features")
    lookup_df.createOrReplaceTempView("_hpt_series_lookup")
    to_tune_features_df = spark.sql("""
        SELECT f.*, l.champion_model_type
        FROM _hpt_tagged_features f
        JOIN _hpt_series_lookup l ON f.series_key = l.series_key
    """)
    return to_tune_features_df.groupBy("series_key").applyInPandas(
        lambda pdf: tune_one_series(pdf, config, snapshot_date), schema=OUTPUT_SCHEMA
    )


def select_improved_retunes(hpt_metrics_pdf: pd.DataFrame, original_champions_pdf: pd.DataFrame) -> pd.DataFrame:
    """Keep only retuned candidates that fit without error, pass decomposition, AND
    strictly improve (lower) wmape_7d vs. that series' current champion — never write
    back a retuned result that's no better than what's already cached."""
    if hpt_metrics_pdf.empty:
        return hpt_metrics_pdf

    valid = hpt_metrics_pdf[(hpt_metrics_pdf["status"] == "ok") & (hpt_metrics_pdf["passed_decomposition"])]
    if valid.empty:
        return valid.reset_index(drop=True)

    original_wmape = original_champions_pdf.set_index("series_key")["wmape_7d"]
    is_improved = valid.apply(
        lambda r: r["series_key"] in original_wmape.index and r["wmape_7d"] < original_wmape.loc[r["series_key"]],
        axis=1,
    )
    return valid.loc[is_improved].reset_index(drop=True)
