"""
training/engine_layered.py — JOB 2 onion-cascade engine (the engine
notebooks/02_train.py actually calls). Runs each candidate-model cost tier
(from config_model, grouped by evaluation_cost) as its OWN separate Spark
pass, rather than one applyInPandas task per location trying every candidate
in a loop. Two consequences that design avoids:
  1. Per-item log lines executing only on the EXECUTOR, invisible on the driver
     until the whole task completes (you'd only find them in Spark UI → Executors
     → stderr).
  2. A single task not returning until its hundreds of slow-candidate fits
     finish, so driver-side progress counters read 0 for the whole run.

Layers run cheapest-first (stats/ETS -> lgbm -> optional uplift -> prophet),
each restricted to the survivors the previous, cheaper layer's gate missed
(repartitioned evenly, round-robin, not by location, so passes 2+ get many
small tasks instead of a few huge per-location ones). Because each pass is its
own action, the driver logs a real summary between passes (items in / accepted
/ promoted / elapsed / avg seconds per model), and a lightweight driver-side
progress thread ticks steadily while the many-small-task passes run.

Champion selection: every layer emits its own candidate row, they are unioned,
and champion.select_champions_spark() picks the lowest-MASE row per series —
so a series that reached Prophet has up to 4 candidate rows (stats/lgbm/uplift/
prophet) and the best one wins; no "best so far" needs to be carried between
passes.

Decomposition gate (docs/instructions.md §10.1): every
candidate's decompose() is actually called and validated here (validate_decomposition
against reconstruction_tol_pct) — passed_decomposition/reconstruction_error are real
per-candidate results, not a hardcoded assumption. This also means _survivors() (the
gate deciding which series promote to the next layer) requires passed_decomposition =
true, not just status = 'ok': a candidate with a good WMAPE that fails reconstruction
can never actually become champion (champion.py::select_champions_spark filters on
passed_decomposition = true too), so treating it as "good enough" here would strand
that series with no valid champion at all.
"""
from __future__ import annotations

import threading
import time as _time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

from config.loader import ForecastConfig  # noqa: F401  (typing only at runtime import cost)
from utils.logging import get_logger
from training.engine import (
    OUTPUT_SCHEMA,
    OUTPUT_SCHEMA_COLUMNS,
    result_row,
    tag_series_key,
)
from training.models.registry import get_registry as _get_registry

_INFER_H   = 200
_SEASON    = 7
_PROPHET_TIMEOUT_S = 90

# Rolling-origin folds per candidate evaluation in the cascade (see _layer_udf_factory).
# A single holdout window means one atypical week (a weather event, a store closure, a
# data glitch) can dominate a candidate's wmape_7d/mase_7d — and with champion.py's
# WMAPE-gate preference, that noise now has real influence on champion selection, not
# just reporting. Averaging across folds smooths it out. Capped at 2 (not driven by
# market_config.backtest_folds, which stays HPT-specific and can be tuned higher for
# HPT's much smaller series subset) to bound the compute cost multiplier across every
# candidate at every cost tier for the full ~150K-series fleet, including expensive
# ones like Prophet (~2-5s/series already at ONE fold).
_CASCADE_MAX_FOLDS = 1

# Stage-1 baselines that get a bounded Stage-2 uplift wrapper when
# config_market.uplift_enabled = true. Only these interpretable, decomposable
# bases are wrapped (never lgbm/prophet, which already carry their own capacity).
# They run as conditional candidates on gate-missing survivors only — see run_engine_layered.
#
# ets_event dropped 2026-09-06: measured on the M5 recalibration run, ets_event+uplift
# (n=16, weighted WMAPE 30.1%) was statistically indistinguishable from plain ets_event
# (n=36, weighted WMAPE 29.9%) — the uplift wrapper spent a full Stage-2 fit + escalation
# tier for a ~0.2pp difference. statsforecast+uplift, by contrast, is a genuine (if
# still weak) fallback for its hardest survivors, so it stays.
_UPLIFT_BASES = ("statsforecast",)
_UPLIFT_SUFFIX = "+uplift"

# Fallback defaults for the forecast sanity cap; the live values come from
# config_market (forecast_cap_mult, cap_trailing_days) and are used unless a
# market row predates those columns. At inference each day's yhat is clipped to
# forecast_cap_mult × the trailing max daily demand (over cap_trailing_days) —
# a model-agnostic guardrail against catastrophic over-forecast (e.g. a holiday
# trend extrapolated into the post-holiday trough). Computed here at training
# from each series' own history and carried to inference via champion params
# ('forecast_cap'), so no history load is needed at scoring time.
_FORECAST_CAP_MULT = 2.5
_CAP_TRAILING_DAYS = 365


def _spread_apply(feat_df, n_partitions: int, udf, schema):
    """Run a per-series pandas UDF over an EVEN round-robin repartition.

    Uses pmod(hash(series_key), n_partitions) so every row of a series lands in
    the same bucket (series must never be split across tasks), while distributing
    distinct series evenly. The UDF still groups internally by series_key and is
    unchanged. This is the fix for the "items_done=0 for 35 min" symptom: small
    tasks complete continuously so driver accumulators flush steadily.
    """
    from pyspark.sql import functions as F

    spread = feat_df.withColumn("_spread", F.pmod(F.hash("series_key"), F.lit(int(n_partitions))))
    return (
        spread.repartition(int(n_partitions), "_spread")
        .groupBy("_spread")
        .applyInPandas(udf, schema=schema)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gate helper (driver side): decide which series get promoted to the next layer
# ─────────────────────────────────────────────────────────────────────────────
def _gate_map(config) -> dict[str, float | None]:
    """category -> gate_wmape (= 1 - accuracy_threshold), or None when unset."""
    out: dict[str, float | None] = {}
    for c in config.categories:
        out[c.category] = (1.0 - c.accuracy_threshold) if c.accuracy_threshold else None
    return out


def _skill_map(config) -> dict[str, float | None]:
    """category -> min_skill_vs_baseline (relative gate), or None for absolute-gate categories."""
    return {c.category: c.min_skill_vs_baseline for c in config.categories}


def _survivors(
    spark, layer_df: "DataFrame", gate_map: dict, margin: float = 1.0, skill_map: dict | None = None,
) -> "DataFrame":
    """Return a 1-column DataFrame [series_key] of series to promote to the next
    layer: those that did NOT produce an 'ok' AND passed_decomposition candidate
    meeting their category gate. 'stable' and 'insufficient_history' are never
    promoted.

    `margin` (config.market_config.escalation_margin, applied by the caller only
    for the cheapest/first layer) tightens the WMAPE ceiling used for THIS
    escalation decision only — e.g. 0.85 means a cost-0 candidate must beat
    85% of gate_wmape, not just gate_wmape itself, to short-circuit the cascade
    without ever trying costlier models. The actual reported gate pass/fail
    (backtest_metrics, champion_index) is untouched — this only controls whether
    other models get a chance to compete for a given series.

    `skill_map` (category -> min_skill_vs_baseline) switches a category to the relative
    gate: the ceiling becomes the series' own baseline_wmape_7d * (1 - min_skill), the
    same per-series bar as training/evaluation.py::gate_wmape_ceiling. Falls back to the
    absolute gate_wmape when the series has no baseline.

    Runs entirely in Spark (no driver collect) so it scales to full PL.
    """
    from pyspark.sql import functions as F

    # Build a small gate lookup DataFrame (category, gate_wmape), margin-adjusted.
    skill_map = skill_map or {}
    categories = sorted(set(gate_map) | set(skill_map))
    gate_rows = [
        (
            k,
            float(gate_map[k]) * margin if gate_map.get(k) is not None else None,
            float(skill_map[k]) if skill_map.get(k) is not None else None,
        )
        for k in categories
    ]
    gate_df = spark.createDataFrame(gate_rows, schema="category string, gate_wmape double, min_skill double")

    # passed_decomposition = false is excluded here too: champion.py::select_champions_spark
    # only ever selects passed_decomposition = true rows, so a low-WMAPE candidate that
    # fails reconstruction can never actually become champion — treating it as "good
    # enough" here would wrongly stop promotion and leave the series with no valid
    # champion candidate at all.
    ok = layer_df.filter("status = 'ok' AND passed_decomposition = true")
    # min ok wmape per series
    # baseline_wmape_7d is identical for every candidate of a series (it depends only on
    # the folds), so max() just picks it up.
    best = ok.groupBy("series_key", "category").agg(
        F.min("wmape_7d").alias("best_wmape"),
        F.max("baseline_wmape_7d").alias("baseline_wmape"),
    )

    # All series this layer touched, minus terminal ones (stable / insufficient).
    touched = (
        layer_df
        .filter("status NOT IN ('stable', 'insufficient_history')")
        .select("series_key", "category")
        .distinct()
    )

    joined = (
        touched.join(best, ["series_key", "category"], "left")
               .join(gate_df, "category", "left")
    )
    ceiling = F.when(
        F.col("min_skill").isNotNull() & (F.col("baseline_wmape") > 0),
        F.col("baseline_wmape") * (F.lit(1.0) - F.col("min_skill")) * F.lit(float(margin)),
    ).otherwise(F.col("gate_wmape"))
    joined = joined.withColumn("ceiling_wmape", ceiling)
    # Promote when: no ok row at all (best_wmape null), OR a gate exists and best misses it.
    promote = joined.filter(
        "best_wmape IS NULL OR (ceiling_wmape IS NOT NULL AND best_wmape > ceiling_wmape)"
    ).select("series_key").distinct()
    return promote


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG-DRIVEN ENGINE
# Reads enabled models from config_model, groups by evaluation_cost, runs each
# cost tier as a separate Spark pass. Adding a model = a wrapper file in
# training/models/ + a config row — zero changes here.
# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Generic per-series layer UDF — config-driven model evaluation
# ─────────────────────────────────────────────────────────────────────────────
def _layer_udf_factory(config, snapshot_date, model_types, timeout_s=None):
    """Generic UDF: for each series, try every model_type and emit candidate rows.

    Uses the model registry to instantiate wrappers dynamically.

    Args:
        model_types: list of model_type strings to evaluate this layer.
        timeout_s:   if set, each model fit+predict is wrapped in a thread timeout
                     (used for slow models like Prophet).
    """
    market  = config.market_config.market_id
    horizon = config.market_config.eval_horizon_days
    val_weeks = config.market_config.val_weeks
    uplift_clip_frac = config.market_config.uplift_clip_frac
    uplift_learner   = config.market_config.uplift_learner
    # Forecast-cap knobs from config (fall back to module defaults for older config rows).
    cap_mult = getattr(config.market_config, "forecast_cap_mult", _FORECAST_CAP_MULT)
    cap_days = getattr(config.market_config, "cap_trailing_days", _CAP_TRAILING_DAYS)
    recon_tol = config.market_config.reconstruction_tol_pct

    def _udf(pdf: pd.DataFrame) -> pd.DataFrame:
        from training.evaluation import wmape as _wmape
        from training.models.registry import build as _build
        from training.decomposition.decomposer import reconstruction_error, validate_decomposition
        from training.backtest import baseline_wmape as _baseline_wmape
        from training.backtest import rolling_origin_folds

        def _construct(mtype: str):
            """Build a model + its fit params, handling '+uplift' wrappers.

            A '<base>+uplift' type builds the Stage-1 base, wraps it in the bounded
            UpliftModel, and passes the market's uplift settings so fit() adds the
            clipped Stage-2 residual learner. Plain types build straight from the
            registry with empty params (default hyperparameters).
            """
            if mtype.endswith(_UPLIFT_SUFFIX):
                from training.models.uplift import UpliftModel
                base = _build(mtype[: -len(_UPLIFT_SUFFIX)])
                return UpliftModel(base), {
                    "uplift_clip_frac": uplift_clip_frac,
                    "uplift_learner":   uplift_learner,
                    "stage1_params":    {},
                }
            return _build(mtype), {}

        def _fit_eval_folds(mtype: str, folds: list, fold_naive_maes: list[float]):
            """Fit+evaluate one candidate across `folds`, mutating the SAME model
            instance fold-by-fold (identical convention to
            training/backtest.py::backtest_metrics — every model wrapper's fit()
            fully reassigns its own state each call, so refitting in place is safe).
            After the loop the model is left fitted on the LAST (most recent) fold,
            which is what get_storable_params()/the returned decomposition reflect —
            only wmape/mase/bias are averaged across folds, so a single atypical
            holdout week can't dominate the metric on its own.
            """
            mdl, _fp = _construct(mtype)
            fold_wmape, fold_mase, fold_bias = [], [], []
            decomp_last = None
            for (train_f, hold_f), naive_mae_f in zip(folds, fold_naive_maes):
                mdl.fit(train_f, hold_f, _fp)
                yhat_f = mdl.predict(hold_f).reset_index(drop=True)
                decomp_f = mdl.decompose(hold_f)
                actual_f = hold_f["qty"].astype(float).reset_index(drop=True)
                fold_wmape.append(_wmape(actual_f, yhat_f))
                mae_f = float(np.abs(actual_f.values - yhat_f.values).mean())
                fold_mase.append(
                    mae_f / naive_mae_f if naive_mae_f > 0 and naive_mae_f == naive_mae_f else float("nan")
                )
                _asum_f = float(actual_f.values.sum())
                fold_bias.append((yhat_f.values.sum() - _asum_f) / _asum_f if _asum_f != 0 else float("nan"))
                decomp_last = decomp_f
            params = mdl.get_storable_params(_fp) if hasattr(mdl, "get_storable_params") else {}
            return (
                decomp_last, params,
                float(np.nanmean(fold_wmape)), float(np.nanmean(fold_mase)), float(np.nanmean(fold_bias)),
            )

        if pdf is None or pdf.empty:
            return pd.DataFrame(columns=OUTPUT_SCHEMA_COLUMNS)
        pdf = pdf[pdf["isActive"] == 1].assign(date_id=lambda d: pd.to_datetime(d["date_id"]))
        rows: list[dict] = []

        for k, g in pdf.groupby("series_key", sort=False):
            g = g.sort_values("date_id").reset_index(drop=True)
            cat = g["category"].iloc[0] if "category" in g else None
            loc = str(g["store_id"].iloc[0])
            iid = str(g["sku_id"].iloc[0])
            ident = (k, market, cat, loc, iid, snapshot_date)

            if len(g) < horizon * 2:
                rows.append(result_row(*ident, model_type=None, params=None,
                                       status="insufficient_history"))
                continue

            folds = rolling_origin_folds(g, _CASCADE_MAX_FOLDS, val_weeks, horizon)
            if not folds:
                rows.append(result_row(*ident, model_type=None, params=None,
                                       status="insufficient_history"))
                continue

            # Per-fold naive (lag-7) MAE — shared across every candidate model this
            # series tries, since it only depends on the fold's train/holdout split,
            # not on any model's fit. Same lag-7 seasonal-naive definition as before.
            fold_naive_maes: list[float] = []
            for train_f, hold_f in folds:
                actual_f = hold_f["qty"].astype(float)
                nm = float("nan")
                if len(train_f) > 7:
                    naive_fc_f = train_f["qty"].shift(7).iloc[-len(hold_f):].fillna(0.0).values.astype(float)
                    if len(naive_fc_f) == len(hold_f):
                        nm = float(np.abs(actual_f.values - naive_fc_f).mean())
                fold_naive_maes.append(nm)
            naive_mae_avg = float(np.nanmean(fold_naive_maes)) if fold_naive_maes else float("nan")
            # Trailing-28-day-mean baseline WMAPE on the same folds (relative-gate yardstick);
            # like the naive MAE above it depends only on the folds, not on any candidate.
            _fold_base = [_baseline_wmape(tr_f, ho_f) for tr_f, ho_f in folds]
            _fold_base = [b for b in _fold_base if b == b]
            baseline_avg = float(np.mean(_fold_base)) if _fold_base else float("nan")

            # Volume + forecast cap from the LAST (most recent) fold — the same
            # window the single-fold cascade always used as its reference point.
            last_train, last_hold = folds[-1]
            vol = float(last_hold["qty"].astype(float).abs().sum())
            _cap_hist = last_train["qty"].tail(cap_days)
            _cap_max = float(_cap_hist.max()) if len(_cap_hist) else float("nan")
            forecast_cap = (
                cap_mult * _cap_max
                if (_cap_max == _cap_max and _cap_max > 0)
                else None
            )

            for mtype in model_types:
                ok_row = None
                try:
                    if timeout_s is not None:
                        # Thread-based timeout for slow models (Prophet, etc.). The whole
                        # multi-fold fit+predict+decompose loop runs inside the same
                        # timeout-guarded call so it counts against the same budget, not
                        # an unbounded extra cost after the thread joins.
                        _result_holder = [None]
                        def _fit_predict(_m=mtype):
                            try:
                                _result_holder[0] = _fit_eval_folds(_m, folds, fold_naive_maes)
                            except Exception:
                                _result_holder[0] = None
                        th = threading.Thread(target=_fit_predict, daemon=True)
                        th.start()
                        th.join(timeout_s)
                        if not th.is_alive() and _result_holder[0] is not None:
                            decomp, params, w, mase, bias = _result_holder[0]
                            if forecast_cap is not None and isinstance(params, dict):
                                params = {**params, "forecast_cap": forecast_cap}
                            passed    = validate_decomposition(decomp, recon_tol)
                            recon_err = float(reconstruction_error(decomp).max())
                            ok_row = result_row(*ident, model_type=mtype, params=params,
                                               mase_7d=mase, wmape_7d=w, bias_pct=bias, naive_mae_7d=naive_mae_avg,
                                               baseline_wmape_7d=baseline_avg,
                                               volume=vol, reconstruction_error_value=recon_err,
                                               passed_decomposition=passed, status="ok")
                        else:
                            status = f"{mtype}_timeout" if th.is_alive() else f"{mtype}_failed"
                            rows.append(result_row(*ident, model_type=mtype, params=None,
                                                   volume=vol, status=status))
                    else:
                        decomp, params, w, mase, bias = _fit_eval_folds(mtype, folds, fold_naive_maes)
                        if forecast_cap is not None:
                            params = {**params, "forecast_cap": forecast_cap}
                        passed    = validate_decomposition(decomp, recon_tol)
                        recon_err = float(reconstruction_error(decomp).max())
                        ok_row = result_row(*ident, model_type=mtype, params=params,
                                           mase_7d=mase, wmape_7d=w, bias_pct=bias, naive_mae_7d=naive_mae_avg,
                                               baseline_wmape_7d=baseline_avg,
                                           volume=vol, reconstruction_error_value=recon_err,
                                           passed_decomposition=passed, status="ok")
                except Exception:
                    rows.append(result_row(*ident, model_type=mtype, params=None,
                                           volume=vol, status=f"{mtype}_failed"))

                if ok_row is not None:
                    rows.append(ok_row)

        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=OUTPUT_SCHEMA_COLUMNS)

    return _udf


# ─────────────────────────────────────────────────────────────────────────────
# Config-driven orchestrator (shadows old hardcoded run_engine_layered above)
# ─────────────────────────────────────────────────────────────────────────────
def run_engine_layered(
    spark: "SparkSession",
    features_df: "DataFrame",
    config: "ForecastConfig",
    snapshot_date: str,
    prev_champions: "dict | None" = None,
    partitions_per_cost: "dict[int, int] | None" = None,
) -> "DataFrame":
    """Config-driven layered cascade: reads enabled models from config.models,
    groups by evaluation_cost, runs each cost tier as a separate Spark pass.

    Models are instantiated from training/models/registry.py — adding a new model
    only requires a wrapper file + a row in config_model.  Zero code changes.

    Layer grouping by cost_tier (config_model.cost_tier, falling back to each model
    class's evaluation_cost default when not set in config — see config/seed_defaults.py's
    _DEFAULT_MODEL_COST_TIERS, the config-table-driven source of truth for this order):
      - Cost 0 (fastest, alone): lgbm — evaluated first, on the full active series set.
      - Cost 2 (medium): auto_arima — survivors only.
      - Cost 3 (merged tier): statsforecast, croston, ets_event, elastic_net — survivors only.
      - Cost 4 (slow, last resort): prophet (with timeout) — survivors only.

    Adding a new model:
      1. Write training/models/<name>.py with model_type = "<name>"
      2. INSERT a row into config_model via the Databricks App.
      Zero changes to this file or any engine code.
    """
    from pyspark.sql import functions as F

    log = get_logger(__name__, market_id=config.market_config.market_id)
    gmap = _gate_map(config)
    smap = _skill_map(config)
    registry = _get_registry()

    # ── Resolve enabled models from config, grouped by cost tier ──────────────
    enabled = [m for m in config.models if m.enabled and m.model_type in registry]
    if not enabled:
        raise RuntimeError(
            "No enabled models in config_model that match the registry. "
            f"Registry has: {sorted(registry.keys())}. "
            f"Config has: {[m.model_type for m in config.models]}"
        )

    # Group by cost_tier: config table overrides class default.
    # Config param: param_group='model', param_name='cost_tier', scope=<model_type>
    # If not in config, fall back to the model CLASS's evaluation_cost attribute.
    _config_tiers = {m.model_type: m.cost_tier for m in config.models
                     if hasattr(m, 'cost_tier') and m.cost_tier is not None}
    cost_groups: dict[int, list[str]] = {}
    for m in enabled:
        cls = registry[m.model_type]
        cost = _config_tiers.get(m.model_type, cls.evaluation_cost)
        # Merge cost 0 and 1 into a single fast layer
        layer_cost = 1 if cost <= 1 else cost
        cost_groups.setdefault(layer_cost, []).append(m.model_type)

    # ── Stage-2 uplift as its OWN escalation tier (§10.2 pt 4) ────────────────
    # _UPLIFT_BASES wraps the ETS-family models (ets_event, statsforecast) — uplift
    # is a residual learner on TOP of one of those, so it must run strictly AFTER
    # its own base models' tier (escalation only kicks in once the plain stats
    # candidate has already had its shot), never sharing that tier or running before
    # it. Insert dynamically at (max base tier + 1) — relative to wherever
    # config_model.cost_tier actually placed the base models, not a hardcoded
    # position, so re-ordering the cascade in config never desyncs this insertion.
    # Every tier at or above the insertion point shifts up by one to make room.
    if config.market_config.uplift_enabled:
        enabled_types = {m.model_type for m in enabled}
        uplift_bases = [b for b in _UPLIFT_BASES if b in enabled_types]
        if uplift_bases:
            uplift_types = [f"{b}+uplift" for b in uplift_bases]
            base_tiers = [c for c, types in cost_groups.items() if any(b in types for b in uplift_bases)]
            insert_at = (max(base_tiers) if base_tiers else 0) + 1
            for c in sorted(cost_groups, reverse=True):
                if c >= insert_at:
                    cost_groups[c + 1] = cost_groups.pop(c)
            cost_groups[insert_at] = uplift_types
            log.info(f"UPLIFT enabled: uplift tier {uplift_types} at cost {insert_at} (runs on {uplift_bases}-survivors only)")

    layers = sorted(cost_groups.items())  # [(cost, [model_types]), ...]
    log.info(f"CONFIG-DRIVEN layers: {[(c, mt) for c, mt in layers]}")

    # Default partitions per cost tier (more partitions = smaller tasks = smoother progress)
    default_parts = {1: 800, 2: 2000, 3: 4000}
    parts = partitions_per_cost or default_parts

    # ── Tag series keys + cache input ────────────────────────────────────────
    # .localCheckpoint() not .cache() — .cache() raises NOT_SUPPORTED_WITH_SERVERLESS
    # on Databricks Free Edition serverless compute (Spark Connect).
    tagged = tag_series_key(spark, features_df, config.market_config.market_id, "_layered_input").localCheckpoint()
    total_active = tagged.filter("isActive = 1").select("series_key").distinct().count()
    log.info(f"LAYERED start  active_series={total_active}  snapshot={snapshot_date}  "
             f"market_id={config.market_config.market_id}  "
             f"enabled_models={[m.model_type for m in enabled]}")

    job_t0 = _time.time()
    all_layer_dfs: list = []
    survivors = None  # None means "all series" for the first layer

    # ── Run each layer ────────────────────────────────────────────────────────
    for layer_idx, (cost, model_types) in enumerate(layers):
        layer_label = f"cost{cost}({','.join(model_types)})"

        # Filter to survivors only (first layer runs on all active series)
        if survivors is not None:
            n_surv = survivors.count()
            if n_surv == 0:
                log.info(f"LAYER {layer_label} SKIPPED  (no survivors from previous layer)")
                continue
            feat_df = tagged.join(F.broadcast(survivors), "series_key", "left_semi")
        else:
            feat_df = tagged
            n_surv = total_active

        # Determine timeout (only for cost >= 3 — slow models like Prophet)
        timeout = _PROPHET_TIMEOUT_S if cost >= 3 else None

        # Run the layer via even round-robin repartition
        n_parts = max(1, min(parts.get(cost, 2000), n_surv))
        layer_df = _spread_apply(
            feat_df, n_parts,
            _layer_udf_factory(config, snapshot_date, model_types, timeout_s=timeout),
            OUTPUT_SCHEMA,
        ).localCheckpoint()
        layer_df.count()

        # Threshold-based escalation: promote a series to the next (heavier) layer
        # when its best ok candidate MISSES its category gate (WMAPE > gate_wmape),
        # not only when every model errored. This is what lets category A series that a
        # cheap stat model can't push under 5%/15% actually reach LightGBM/Prophet.
        # `_survivors` also promotes series with no ok row at all; 'stable' and
        # 'insufficient_history' are never promoted. Runs entirely in Spark.
        # escalation_margin only applies at the cheapest tier (layer_idx 0) — a
        # merely-adequate lgbm result shouldn't preempt genuinely better classical
        # models without ever letting them compete (see _survivors' docstring).
        _margin = config.market_config.escalation_margin if layer_idx == 0 else 1.0
        survivors = _survivors(spark, layer_df, gmap, margin=_margin, skill_map=smap).localCheckpoint()
        n_surv_next = survivors.count()
        # items entering this layer (n_surv) vs. still without a valid champion
        # candidate afterward (n_surv_next, promoted to the next layer) — computed
        # from already-materialized counts rather than live Spark accumulators,
        # which Databricks Free Edition serverless (Spark Connect) does not support
        # (spark.sparkContext access itself raises JVM_ATTRIBUTE_NOT_SUPPORTED).
        log.info(f"LAYER {layer_label} DONE  items={n_surv}  "
                 f"resolved={n_surv - n_surv_next}  no_valid_candidate={n_surv_next}  "
                 f"elapsed={(_time.time()-job_t0)/60:.1f}min")

        all_layer_dfs.append(layer_df)

    # ── Union all candidate rows; champion selection picks min WMAPE per series ─
    if not all_layer_dfs:
        raise RuntimeError("No layers produced candidates — check config_model.")

    all_candidates = all_layer_dfs[0]
    for df in all_layer_dfs[1:]:
        all_candidates = all_candidates.unionByName(df)

    log.info(f"LAYERED complete  total_time={(_time.time()-job_t0)/60:.1f}min  "
             f"candidate_rows={all_candidates.count()}")
    return all_candidates