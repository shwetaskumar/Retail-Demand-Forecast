# Many-Models Forecasting (MMF) Engine — Build Specification & BRD
**Project:** Retail Demand Forecast — bottom-up store×sku daily forecasting on Databricks
**Owner:** Personal portfolio project (built solo)
**Status:** Build-ready specification (instructions.md). This document makes the decisions; it is meant to be executed by an engineer or coding agent with minimal further refinement.
**Primary market:** CA (demo — Kaggle M5's state_id CA/TX/WI become market_id; see `data/kaggle_m5/loader.py`). **Design constraint:** market-agnostic — every market is onboarded by parameter/filter only.

> How to read this document. Sections 1–6 are the BRD (why, what, who, success). Sections 7–24 are the build spec (architecture, contracts, code structure, acceptance criteria). Anything marked **[ASSUMPTION]** is a default I have chosen so the build is not blocked — change it in `config/` if wrong. Anything marked **[OPEN]** is a question that does not block v1 but should be answered before prod scale-out.

> **[AMENDMENT, post-v1]** §7's diagram and §13 describe each job's deployed entrypoint as a raw `spark_python_task` running `run_build_features.py` / `run_training.py` / `run_inference.py` directly with argparse CLI params. That's superseded — **the notebook-per-job design is now authoritative**: every job (all of JOB 1/2/3, the on-demand backfill job, and the evaluate/HPT-retune/final-rescore chain appended to JOB 3) is a `notebook_task` (`notebooks/NN_*.py`, `dbutils.widgets`, UI-driven). The per-job `run_*.py` wrapper modules this document describes (`data/run_build_features.py`, `training/run_training.py`, `inference/run_inference.py`, `training/run_evaluate.py`, `training/run_hpt_retune.py`, `inference/run_infer_final.py`, `data/run_full_backfill.py`) no longer exist — their orchestration logic was moved directly into the corresponding notebook as separate, individually re-runnable cells (one per pipeline step), so a failure is traceable to the exact step in the Databricks UI rather than hidden behind one wrapper `main()`. Only genuinely shared/tested logic stays in an importable module (`data/features/*.py`, `training/engine_layered.py`, `training/champion.py`, `training/evaluation.py`, `training/hpt.py`, `training/mlflow_registry.py`, `training/models/*.py`, `inference/batch_infer.py`, `inference/future_spine.py`, `inference/dq.py`). The internal pipeline steps, table contracts, and acceptance criteria below are otherwise unchanged. Where this document's text says "run `X.py`", read that as "the notebook that now contains `X.py`'s former pipeline steps as cells."
>
> **[AMENDMENT 2, post-v1]** §6.3/§20 describe config as four typed tables. Superseded — **a single EAV config table is now authoritative**: config lives in one general-purpose EAV table, `control.forecast_config` (config/ddl/config_tables.sql), read by the same `config/loader.py::load_config` and pivoted into the identical typed dataclasses this document's contracts assume. No downstream contract changed, only the storage shape.
>
> **[AMENDMENT 3, post-v1]** The original design included three post-model adjustment notebooks (item-proxy backfill, store-proxy backfill, level-adaptation error-ratio correction) between inference and publish. Those are **removed** from this build to keep the project focused on the core modeling engine — see §7 and §14. The post-model chain is now just `infer_final → temp_closure → outbound`.
>
> **[AMENDMENT 4, post-v1]** The WMAPE gate (§3, §21) is now **relative by default**: a grain passes when its 7-day volume-weighted WMAPE beats a trailing-28-day-mean baseline's WMAPE on the same backtest folds by at least `min_skill_vs_baseline` (config param `category_skill_thresholds`, seeded at 3% for A/B/C/SEASONAL and 0% for D). The fixed per-category `accuracy_threshold` values stay in config as informational targets and remain the gate for any category without a skill threshold. Reason: at `market×store×sku` daily grain on sparse retail data, Poisson sampling noise alone exceeds the fixed targets for a typical series (median category-A series on the M5 demo sells ~13 units/week → ~18% irreducible weekly WMAPE vs the 15% an 85% target allows), so the absolute gate measured the data rather than the model. The relative gate is applied identically by the reported gate (`training/evaluation.py`), the cascade early-stop (`training/engine.py`, `training/engine_layered.py`), champion selection (`training/champion.py`) and daily breach detection (`notebooks/04_evaluate.py`); new columns are added by `config/ddl/migrate_v3_skill_gate.sql`. The same change adds an equal-weight `ensemble` candidate (`training/models/ensemble.py`, auto_arima + ets_event + lgbm, decomposition averaged per component so the §10.1 additive contract still holds exactly) and makes it the **only default candidate**: `demo/compare_selection_rules.py` (750 out-of-sample series-weeks on M5) showed that picking the backtest winner per series made forecasts worse than the 28-day-mean baseline (−7.1% skill, 90% CI −13.7%..0.0%), while always using the ensemble did not (+1.4%, CI −3.2%..+6.2%). Per-series champion selection (§10.2's cascade, champion_index, HPT) is unchanged and is re-enabled by enabling single models in config.

---

## 1. Executive Summary

This is a **bottom-up Many-Models Forecasting (MMF)** system that trains and serves a model **per `market×store×sku`**. The system runs on Databricks/Unity Catalog (targeting the free serverless tier), uses a **feature store with both historical and known-future features**, registers **one MLflow model per `market×category`** (MMF pattern) under which many per-series champions are managed, performs **daily batch inference for a 140-day horizon**, and selects champions by **7-day volume-weighted WMAPE** against per-category accuracy thresholds.

Two requirements are non-negotiable and shape every design choice:
1. **Accuracy** — beat the per-category WMAPE thresholds at the cohort level *and* report at `market×sku` and `market×store×sku`.
2. **Explainability** — every forecast must decompose **additively** into trend, seasonality, and the **per-event lift** of each named holiday, promotion, and non-promotion effect.

A layer of **agents** (monitoring, diagnosis, feature-engineering) and a **planner-override feedback loop** are designed as phase 2 extensions (see §15, §16) — deferred so this build can reach the best achievable state on modeling accuracy + explainability first.

---

## 2. Business Context & Problem Statement

**Motivating scenario.** A retail chain forecasting demand top-down (`market×sku`) and disaggregating to store level runs into the same wall every growing retailer hits: it's hard to maintain, doesn't capture store-level dynamics, and relies on a single modeling approach for every kind of product. This project builds the bottom-up alternative from scratch: one model per `market×store×sku`, decomposable, gated on accuracy, and cheap enough to run daily.

**Problem statement.** Build a production-grade, autonomous, explainable, cost-efficient bottom-up forecasting engine on Databricks that (a) beats per-category WMAPE thresholds at store×sku grain, (b) produces additive per-event decompositions, (c) onboards new markets by configuration only, and (d) runs cheaply enough that daily operation is not cost-prohibitive (training/run cost target: **well under $100/run**, not per-series).

---

## 3. Goals, Objectives & Success Metrics

| # | Objective | Metric | Target | Baseline |
|---|-----------|--------|--------|----------|
| G1 | Beat accuracy thresholds | 7-day volume-weighted WMAPE → accuracy = 1−WMAPE | Per category (demo market): A 85%, B 75%, C 65%, D 60%, SEASONAL 80% | A top-down baseline underperforms |
| G2 | Explainable forecasts | Additive decomposition present & reconstructs forecast within tolerance | 100% of served series carry full decomposition | None today |
| G3 | Autonomy | Human touchpoints / manual fixes per week | Exceptions-only | Daily firefighting |
| G4 | Onboarding | Effort to add a market | Config/filter change only, no code change | High (bespoke) |
| G5 | Cost | Cost per scheduled run | < $100/run; daily = inference-only | Unbounded / vendor cost |
| G6 | Override quality | Planner override rate | 5–10% | Unknown / high |

**Accuracy metric definition (locked).** WMAPE, **volume-weighted within each series**, evaluated over the **next 7 days**, judged at three grains: (i) **category cohort** (gate vs threshold table), (ii) **market×sku**, (iii) **market×store×sku**. Champion selection uses the 7-day WMAPE; secondary WMAPE is reported at horizons 14/28/140 for monitoring. **The threshold gate is applied at ALL three levels** — category cohort, `market×sku`, and `market×store×sku` — each against its category threshold. A run is only fully "green" when all levels pass; per-level pass/fail is recorded in `backtest_metrics`, and any level's failure routes the affected series to diagnosis/conditional-HPT (see §12, §16). The most granular level (`market×store×sku`) is the hardest to satisfy, so expect it to be the binding constraint and the main driver of HPT and feature work.

---

## 4. Stakeholders & Personas

| Persona | Type | Needs | Interaction |
|---------|------|-------|-------------|
| Data Scientist (owner/builder) | Primary owner / builder | Stable, autonomous pipeline; exception queue; ability to trigger retrain/HPT | Exceptions only |
| Demand Planners | Consumer | Accurate forecasts + decomposition; ability to override; overrides respected & learned from | Consume via a **BI dashboard**; overrides via upload |
| Leadership / reviewers | Sponsor / consumer | Accuracy, cost, and "AI & agents" story; trust & visibility | Dashboards (accuracy, cost, model health) |

**Permissions / isolation.** No market isolation in Unity Catalog. One shared schema; markets are separated by **filter** in jobs (`market_id`). **[ASSUMPTION]** BI-layer row-level filtering by market is handled in the BI layer, not by separate tables.

---

## 5. Scope

**In scope (v1) — the focus is models running with high accuracy + explainability:** feature pipeline (history + future); an **in-house many-model engine** (per-series parallel train/backtest/champion, no external accelerator package); two-stage additive decomposable modeling; conditional HPT; daily 140-day batch inference; backtest/WMAPE evaluation at three grains; BI outputs; observability/DQ; market-onboarding playbook.

**Deferred to phase 2 (build after accuracy + explainability are proven):** the **agents** (monitoring, diagnosis, feature-engineering) and the **planner override feedback loop**. Their table contracts remain defined so nothing blocks adding them later, but no code is built for them now.

**Out of scope (v1):** prescription/auto-remediation agent (specced, phase 2); weather & social-media features (interfaces defined, data source TBD — **[OPEN]**); item-proxy/store-proxy backfill and level-adaptation error-ratio correction (deliberately not built — see amendment 3 above, keeps the project scoped to the core modeling engine); **any write to / integration with a downstream order-management or replenishment system** — the build ends at the published `gold` tables + BI dashboard; downstream consumption is owned elsewhere.

---

## 6. Data Contracts (authoritative — from the source tables)

**Catalog/env:** a single Databricks Free Edition workspace catalog, e.g. `retail_forecast`. **Never hardcode the catalog**; derive from an `env`/`catalog` parameter. Schemas: `source` (read-only inputs), `gold` (our intermediate/feature outputs), `control` (config), `forecast` (final results).

### 6.1 Source tables
| Ref | Table | Purpose | Key columns used |
|-----|-------|---------|------------------|
| SALES | `source.sales` | Sales target | `store_id, sku_id, date_id, units_sold, units_promotional, units_shrinkage, market_id` |
| SKU_STORE | `source.sku_store` | Activeness window | `store_id, sku_id, active_from, active_to, market_id` |
| SKU | `source.sku` | Sku master | `sku_id, category` |
| STORE | `source.store` | Store master | `store_id, market_id` |
| EVENTS | `gold.events` | Promotions + holidays | `market_id, event_id, event_type, event_subtype, event_nm, start_date, end_date, sku_id, store_id, exclude_flag, tactic` |
| DATE | `gold.calendar_date` | Calendar/temporal | `date_id` + DOW, week-of-year, month, quarter, fiscal period, day-of-month, payday flag, weekend flag, season |

**Target:** `qty = SUM(NVL(units_sold,0) + NVL(units_promotional,0) + NVL(units_shrinkage,0))`.
**Standard keys throughout:** `store_id`, `sku_id` (**UPPER**-cased), `date_id` (DATE).
**Source filters (always):** `UPPER(market_id) = :market`.

### 6.2 Feature table — `gold.sku_store_features`
Grain `snapshot_date × store_id × sku_id × date_id`. Built by JOB 1 (DATA) from the source tables **read-only**; it does **not** modify or replace any existing table. `snapshot_date` makes it point-in-time so any as-of date can be rebuilt/replayed. See §9 for the history/future split.

### 6.3 New output tables this build creates (managed Delta, Liquid Clustering, split across `gold`/`control`/`forecast` per Medallion convention)
| Table | Grain | Purpose |
|-------|-------|---------|
| `forecast.output` | `market × store_id × sku_id × date_id` | 140-day forecast `yhat`, plus `yhat_lower/upper`, horizon day index, run_id |
| `forecast.decomposition` | same as above | Additive components: `trend, seasonality_weekly, seasonality_yearly, holiday_lift_<event>, promo_lift_<event>, temporal_features_effect, model_uplift`; plus `reconstruction_error` |
| `gold.champion_index` | `market × category × series_key` | series_key → champion `model_type`, params, artifact URI, metrics, mlflow_run_id, version, valid_from |
| `gold.backtest_metrics` | `run_id × grain × cohort/series` | WMAPE at 7/14/28/140, volume, pass/fail vs threshold, per category/sku/store×sku |
| `overrides` (phase 2) | `market × store_id × sku_id × date_id` | planner override value, system yhat, delta, reason, override source, override_rate, ingest_ts |
| `dq_audit` | `run_id × check` | row counts, null/dup keys, qty min/max, reconstruction tolerance, freshness |
| `run_registry` | `run_id` | job/task lineage, params, env, timing, cost tags |
| `forecast.downstream_feed` | `market × store_id × sku_id × date_id` | outbound publish of the latest decomposition, for any downstream consumer |

`series_key` = `concat_ws('|', market, category, store_id, sku_id)`.

---

## 7. Target Architecture — three independent jobs

All three jobs are DABs-managed, parameterized by **`market_id`, `env`, and `snapshot_date`**, read their settings from the **config Delta table**, and can be run for **any snapshot date** (point-in-time). They are decoupled — each runs on its own schedule/trigger and can be run standalone.

```
 Config Delta table (control.forecast_config), managed by the Databricks App
        │  (read filtered by market — NO YAML)
        ▼
┌─ JOB 1: DATA (feature build) ──────────────────────────────────────────────┐
│ reads SOURCE tables READ-ONLY (SALES/SKU_STORE/SKU/STORE/EVENTS/DATE)      │
│ builds features point-in-time as of snapshot_date                          │
│ writes ─► gold.sku_store_features   (snapshot_date col)                    │
└────────────────────────────────────────────────────────────────────────────┘
        ▼ (reads feature table @ snapshot_date)
┌─ JOB 2: TRAINING ──────────────────────────────────────────────────────────┐
│ per-series (applyInPandas by series_key): fit candidates ─ backtest 7d WMAPE│
│ ─ gate 3 grains ─ conditional HPT ─ decomposition check                     │
│ writes ─► champion_index, backtest_metrics, artifacts;                     │
│           registers 1 pyfunc per market×category in MLflow                  │
└────────────────────────────────────────────────────────────────────────────┘
        ▼ (reads champions + feature table @ snapshot_date)
┌─ JOB 3: INFERENCE (scoring only) ──────────────────────────────────────────┐
│ 140-day future spine ─ predict + additive decompose                        │
│ ─ evaluate ─ conditional HPT retune ─ infer_final (re-score improved)      │
│ ─ temp_closure (zero out active-closure forecasts) ─ outbound (publish)    │
│ writes ─► output, decomposition ─ DQ ─► dq_audit ─ BI dashboard           │
└────────────────────────────────────────────────────────────────────────────┘
```

**Cadence & independence:**
- **JOB 1 (Data)** — scheduled (e.g. daily) or run for a backfill `snapshot_date`. Never writes to existing tables.
- **JOB 2 (Training)** — weekly/triggered; includes conditional HPT. Runs off whatever snapshot the feature table holds.
- **JOB 3 (Inference)** — daily/cheap; scoring only, no training. Chained after scoring (self-gating, near-zero cost on a normal day): evaluate → conditional HPT retune → final re-score → temp_closure → outbound.
- Any job can be run for a **historical `snapshot_date`** to replay/backfill. Because features are point-in-time, JOB 2 and JOB 3 for a past date see only data that existed then.
- Agents + override loop are **phase 2** (not shown).

---

## 8. The 10k-model problem & the in-house engine (crux resolution)

Databricks Model Registry allows ~10k **registered models**. A single market at real-chain scale can exceed 10k series in one category. Therefore:

- **Registered MLflow model = one pyfunc per `market × category`.** A demo market → ~5 registered models (A, B, C, D, SEASONAL). Many markets × ~5 stays far under the ceiling.
- The category pyfunc is a **router**: given rows for a series, it looks up that series' champion in `champion_index`, loads the per-series artifact from a **UC Volume** (or Delta-backed blob store), and produces forecast + decomposition. Per-series artifacts are **not** individually registered.
- Per-series champion metadata (model_type, params, metrics, artifact URI, version) lives in `champion_index` (Delta). This *is* the per-series registry.
- Training is **distributed across series** with Spark `applyInPandas` grouped by `series_key` within a category (the MMF pattern). No driver-side loop over hundreds of thousands of series.

**In-house engine — recreate the MMF *pattern*, do not depend on the `mmf_sa` package.** We deliberately do **not** import the Databricks MMF accelerator: it is provided AS-IS with no support SLAs, and a core forecasting engine must not carry that dependency risk. Instead we build the same well-known pattern ourselves — it is small and standard on Databricks:

1. **Distribute per series with Spark `applyInPandas`** grouped by `series_key`. Inside each group (a single series' pandas frame) we fit each candidate model, backtest it, and pick the best — thousands of series train in parallel across the cluster. (Set `spark.sql.execution.arrow.enabled true`, `spark.sql.adaptive.enabled false` on the cluster.)
2. **Model wrappers behind one interface** (`fit`, `predict`, `decompose`) wrap **stable, mainstream libraries** — Prophet, statsforecast, LightGBM — so any single library can be swapped without touching the engine.
3. **Store champions the way MMF does — binaries/params in Delta, not per-series registration.** Per series we persist the champion's `model_type` + tuned params (+ optionally a serialized artifact in a UC Volume) into `champion_index`. Registered MLflow models = one routing pyfunc per `market×category`. This keeps us far under the ~10k registered-model ceiling at real-chain scale.
4. **Custom layers we own regardless:** additive per-event **decomposition** (§10), volume-weighted **7-day WMAPE gate at three grains** (§3), and **conditional HPT** (§12).

Dependencies are therefore only mainstream, individually-swappable forecasting libraries (Prophet / statsforecast / LightGBM / Optuna / SHAP / MLflow) — no accelerator lock-in.

---

## 9. Feature Store — history + future

Use **Databricks Feature Engineering in Unity Catalog**. `gold.sku_store_features` is the offline feature table (primary keys `store_id, sku_id, date_id`). This spec defines the **history/future contract** the model layer depends on.

**Two feature classes (tag every feature):**
- **Historical-only (known after the fact):** lags (t-1,7,14,28,365), rolling mean/std/min/max/median (7/14/28), EWMA (7,28), expanding mean/std, same-weekday-last-year, YoY ratio, trailing zero-sales ratio. These exist only up to the run date; for the future horizon they are produced **recursively/direct multi-step** (see §10.4).
- **Known-at-inference (available for the full 140-day horizon):** calendar/temporal (from `gold.calendar_date`), holidays (with pre/post ±N-day windows and bridge-day flags), promotions (active flag, type/subtype/tactic, duration, days-since-start, days-until-next, concurrent-promo count), non-promotion events. Promos are known **2 months to 1 year ahead**, so the future feature spine is reliable.

**Future feature spine.** For each active series, build a dense daily spine from `current_date+1 .. current_date+140`, join the known-at-inference features. Historical-only features for the horizon are generated by the model's multi-step strategy, never read from the future (leakage control is mandatory — see §17).

**Activeness & zero-fill:** `isActive = window_active=1 AND is_new_store=0 AND is_new_sku=0`. Zero-fill true zeros **only** where `isActive=1`. Never zero-pad new/seasonal/inactive periods into the modeled series.

---

## 10. Modeling layer — two-stage additive, decomposable (the explainability contract)

### 10.1 Decomposition contract (every served forecast must satisfy)
```
yhat(t) = trend(t)
        + seasonality_weekly(t) + seasonality_yearly(t)
        + Σ_e holiday_lift_e(t)        # one term per named holiday event
        + Σ_p promo_lift_p(t)          # one term per promo event/subtype
        + temporal_features_effect(t)          # closures/breaks/known dips
        + model_uplift(t)              # bounded residual learner contribution
```
- **Additive and reconstructing:** the components must sum to `yhat` within `reconstruction_tolerance` (**[ASSUMPTION]** ≤ 0.5% of |yhat| or 1e-6 absolute). Stored in `decomposition.reconstruction_error`; violations fail DQ.
- **Per-event attribution:** each holiday/promo named in `gold.events.event_nm` gets its own lift column (sparse; 0 when inactive).
- A model **cannot be champion unless it emits a valid decomposition.** Pure black boxes compete **only** as the bounded `model_uplift` term, never as the whole forecast.

### 10.2 Candidate models (champion competes per series)
1. **Prophet** with holiday & promo regressors (additive mode) — native decomposition. Fast, current baseline.
2. **Additive GAM / regularized linear** — trend + Fourier seasonality + event dummies (e.g., statsmodels / pyGAM / ElasticNet on engineered calendar+event features). Often beats Prophet on controllability and is fully decomposable.
3. **Explainable gradient boosting** — LightGBM on calendar+event features with **SHAP attribution grouped** into the contract's buckets (trend/seasonality/event). Strong accuracy; decomposition via grouped SHAP.
4. **Foundation/global residual learner (Chronos / LightGBM-global)** — used **only** for the bounded `model_uplift` term on top of an interpretable Stage-1 base. Buys back accuracy Prophet alone can't reach without breaking decomposition.

> **[ASSUMPTION]** Stage-1 (baseline+events) chooses among #1–#3 per series; Stage-2 (`model_uplift`) optionally adds #4 if it improves 7-day WMAPE beyond a margin and the reconstruction still holds. `model_uplift` is **clipped** to a configurable fraction of the baseline so it cannot dominate (preserves interpretability).

### 10.3 Per-series two-stage fit
1. Fit Stage-1 interpretable model → produce `trend, seasonality_*, *_lift_*, temporal_features_effect`.
2. Compute residuals; optionally fit Stage-2 learner on residuals (+ known-future features) → `model_uplift`.
3. Backtest both (Stage-1 only) and (Stage-1+Stage-2); pick the smaller-WMAPE that satisfies the contract.

### 10.4 Multi-step horizon (140 days)
**[ASSUMPTION]** Use **direct/known-future-driven** forecasting: Stage-1 is a function of trend+calendar+events (all known for 140 days), so it needs no recursion. Only the optional Stage-2 lag-based features require multi-step handling — use **direct multi-horizon** (separate quantile of lags computed as-of run date) rather than recursive feedback, to avoid error accumulation over 140 days. Lag features beyond the run date are filled from as-of-run-date values, never from predicted future.

---

## 11. Champion–challenger & MLflow

- **Backtest:** rolling-origin, **4–6 folds** over the **last 8–12 weeks**, 7-day horizon per fold. Metric = volume-weighted WMAPE.
- **Champion = lowest 7-day WMAPE** among valid (decomposing) candidates, tie-broken by (a) reconstruction error, (b) training cost.
- **Promotion:** write champion to `champion_index`; the `market×category` pyfunc is re-logged/registered in MLflow with an updated `@champion` alias and the index version it routes against. Keep previous version as challenger for rollback.
- **No per-series MLflow registration.** MLflow holds the category pyfunc + run metrics; per-series detail is in the Delta index.
- **Reproducibility:** every run logs params, data window, fold definitions, and seeds; `run_id` ties `output`, `backtest_metrics`, and `dq_audit` together.

---

## 12. Conditional hyperparameter tuning

- **Trigger:** only series (or category cohorts) **missing the threshold** after default-param backtest.
- **Engine:** Optuna, **capped budget [ASSUMPTION] 20–30 trials**, early-stopping/pruning on.
- **Cadence:** in the **weekly/triggered** retrain job only — **never** in the daily critical path.
- **Caching:** persist tuned params in `champion_index`; do not re-tune a series that already passes. Re-tune only on sustained drift (monitoring-driven) or scheduled refresh.
- **Cost guard:** HPT job has its own concurrency and a per-run trial-budget ceiling so cost stays within target.

---

## 13. Daily batch inference (critical path, cheap)

- Load each `market×category` champion pyfunc; for the active series set, build the 140-day future feature spine; produce `output` + `decomposition`.
- **Distributed** via `applyInPandas` grouped by `series_key`; **no `collect()`/`toPandas()` on full data**; autoscaling/serverless compute sized for the series volume.
- Idempotent MERGE on `(market, store_id, sku_id, date_id)`; safe to retry.
- Emit DQ (§17) and publish (§14).

---

## 14. Outputs & integrations

- **BI dashboard:** publish a semantic model over `output` + `decomposition` + `backtest_metrics`. Planners see forecast and the per-event decomposition; leaders see accuracy/cost/model-health. Market filtering via RLS in the BI layer.
- **Downstream consumption:** the 140-day forecast in `output` could feed a downstream order-management/replenishment system. **Out of scope for this build** — we publish the forecast tables via `forecast.downstream_feed`; we do not write to or integrate with any specific downstream system. The forecast tables are the contract boundary.
- **Override intake (phase 2):** overrides would flow in via silver-layer Delta tables, unioned into `overrides`. See §15.

---

## 15. Override feedback loop (closing the learning cycle) — **PHASE 2 (deferred)**

> Deferred until models hit accuracy + explainability. Table contract (`overrides`) stays defined; no code built now. Design below is the target for later.

1. **Ingest** planner overrides into `overrides` with system `yhat`, override value, delta, optional reason, `date_id`. **Source contract:** a **configurable list of silver-layer Delta tables**, unioned into `overrides` (each tagged with its source). Build the ingest to read a configurable list and normalize each to the same `overrides` schema, so adding a new source table is a `config/` change with no code change.
2. **Measure** override rate per category/market vs the **5–10%** target; surface in the BI dashboard and to the monitoring agent.
3. **Feed back** in the weekly retrain:
   - High-override series are **flagged for the diagnosis agent**.
   - Overrides become a **candidate signal**: (a) a learned **bias-correction** term applied within `model_uplift` bounds, and/or (b) an event-mining input (frequent overrides on the same dates ⇒ a missing event/feature). **[ASSUMPTION]** bias-correction is **human-in-the-loop**: applied only after the diagnosis agent proposes and a DS approves, so the system doesn't blindly learn bad overrides.
4. Override-derived adjustments are themselves **decomposed** (reported as their own component) so explainability holds.

---

## 16. Agents (monitoring, diagnosis, feature-engineering) — **PHASE 2 (deferred)**

> Deferred until models hit accuracy + explainability. Interfaces/tables stay defined; no agent code built now. Design below is the target for later.

All agents are **tools over the Delta tables + MLflow**, run as scheduled jobs or on-demand, log every action, and are **read/advisory by default** (writes gated by DS approval, except safe idempotent re-runs).

| Agent | Job | Inputs | Outputs / Actions |
|-------|-----|--------|-------------------|
| **Monitoring** | Detect drift, threshold breaches, freshness/DQ failures, override-rate spikes | `backtest_metrics, dq_audit, overrides, output` | Alerts; populate an **exception queue**; tag series for diagnosis; trigger conditional retrain/HPT |
| **Diagnosis** | Root-cause a failing/odd series | series history, decomposition, events, DQ | Structured diagnosis (data gap? new sku? missing event? model mismatch?) + recommended action (retrain, add feature, exclude, HPT) |
| **Feature-Engineering** | Profile sources, propose & (on approval) add features | source tables, `gold.events`, profiling stats | Proposed new features (new event types, cannibalization groups, price signals) with leakage tags; PR-style change to FE config |

**Phase 2 (specced, not wired):** **Prescription** agent — auto-proposes and (with guardrails) applies remediations. Define its interface now; do not enable auto-write in v1.

**Agent governance:** every agent action is logged with inputs/outputs to an audit table; destructive or model-promoting actions require DS approval; agents operate within the same UC least-privilege grants.

---

## 17. Leakage control, DQ & observability

- **Leakage:** historical-only features never computed from current/future rows; future horizon uses only known-at-inference features + as-of-run-date lags. Unit tests assert no future leakage.
- **DQ checks → `dq_audit`:** row counts, null keys, duplicate keys, negative qty, `isActive` integrity, decomposition `reconstruction_error` within tolerance, forecast freshness, horizon completeness (140 rows/series). Fail loudly; block publish on critical failures.
- **Observability:** structured Python logging to driver logs; run/cost tags (`project, market, env`); `run_registry` for lineage; accuracy/cost/health dashboards in BI. Emit metrics for each task (duration, series counts, pass rate).

---

## 18. Platform & engineering standards

- **Structure:** modular `src/`-style package (pure, testable functions returning DataFrames); orchestrating notebooks only wire tasks. Cell-1 docstring header per notebook (purpose, UC inputs/outputs, params, last-updated). PEP 8, type hints, docstrings, pytest unit tests + DQ assertions module.
- **Compute:** Databricks Free Edition **serverless** compute — no classic clusters, no node types, no cluster policies. Pin an `environment_version` + pip dependencies per job (see `databricks.yml`).
- **Spark:** Spark SQL/DataFrame API; avoid driver-side `collect()/toPandas()` on full data; avoid Python UDFs where native/Pandas exists; tune `spark.sql.shuffle.partitions`. Three-level UC names only; no `hive_metastore`, DBFS mounts, or unmanaged paths.
- **Delta:** UC **managed** tables; **Liquid Clustering** (not partition/ZORDER) on dominant keys — feature table `CLUSTER BY (date_id, store_id, sku_id)`. **Predictive Optimization** on (no hand-coded OPTIMIZE/VACUUM). Idempotent **MERGE**; explicit schema with COMMENTs/tags, NOT NULL keys, CHECK (`qty>=0`); deliberate schema evolution.
- **Governance:** secrets in Databricks Secrets / UC Volumes; automatic UC lineage via table APIs; derive catalog from `env` param.
- **CI/CD:** **Databricks Asset Bundles** define jobs/schedules/params for a single Free Edition target; GitHub Actions runs lint + tests on every push/PR; `databricks bundle deploy` is run manually by the workspace owner (no CD pipeline needed for a single-user free workspace). Restartable, idempotent end-to-end; DQ → audit table.

---

## 19. Repository structure (deliverable layout)

Clean, functional top-level folders — each stage of the pipeline is immediately identifiable.

```
repo/
  README.md
  requirements.txt             # mainstream libs only (no accelerator package)
  databricks.yml               # DABs: 3 jobs (data/training/inference), single Free Edition target

  config/                      # CONFIG = Delta tables + loader (NOT yaml per market)
    ddl/config_tables.sql      # config table DDL
    loader.py                  # reads config from Delta filtered by market -> typed object
    seed_defaults.py           # one-time bootstrap inserts (demo-market thresholds etc.)
    app/                       # Databricks App: CRUD UI over the config tables (mgmt front-end)
      README.md

  utils/                      # shared, importable everywhere
    io.py                      # Delta read/write helpers (session injected)
    spark.py                   # session helpers (never creates a session at import)
    dates.py                   # snapshot_date handling / point-in-time windows
    logging.py

  data/                        # ===== JOB 1: DATA (feature build) =====
    ddl/tables.sql             # output + feature table DDL (run in Databricks)
    features/                  # the sku_store_features pipeline (per FE spec)
      sales_base.py active_window.py lifecycle.py gap_fill.py
      stat_features.py temporal_features.py event_features.py writer.py
      halo_cannibalization_features.py category_classification.py category_features.py
    kaggle_m5/                  # Kaggle M5 dataset loader (real Walmart sales, stands in for a POS/ERP feed)
      loader.py

  training/                    # ===== JOB 2: TRAINING =====
    engine.py                  # shared building blocks + pandas-only reference cascade (test-only)
    engine_layered.py          # applyInPandas orchestrator actually called by notebooks/02_train.py
    models/                    # wrappers behind one fit/predict/decompose interface
      base.py prophet.py statsforecast.py lgbm.py uplift.py
    decomposition/decomposer.py   # additive contract + reconstruction check
    backtest.py                # rolling-origin folds (as-of snapshot_date)
    evaluation.py              # volume-weighted 7-day WMAPE gate at 3 grains
    champion.py                # best-per-series -> champion_index (+ artifacts)
    hpt.py                     # conditional Optuna, threshold-miss only, cached
    mlflow_registry.py         # one pyfunc per market x category
    run_fit_local.py           # standalone cold-start utility, not wired into databricks.yml

  inference/                   # ===== JOB 3: INFERENCE (scoring only) =====
    future_spine.py            # 140-day known-future spine from snapshot_date
    batch_infer.py             # load champions, predict + decompose, MERGE outputs
    dq.py                      # data-quality checks -> dq_audit

  agents/                      # PHASE 2 — placeholder only (README, no code yet)
  overrides/                   # PHASE 2 — placeholder only (README, no code yet)

  demo/                        # no-Spark local demo: metrics + charts for README
    run_local_demo.py

  notebooks/                   # the deployed job entrypoints AND the pipeline logic itself
                                # — one cell per pipeline step, calling
                                # into the shared modules above; no per-job wrapper .py file
    00_create_tables  00b_ingest_m5
    01_build_features  01b_backfill_features  02_train  03_score
    04_evaluate  05_hpt_retune  06_infer_final  07_full_backfill
    08_temp_closure  09_outbound

  tests/                       # local, no-Spark: pandas fixtures + mocked spark/dbutils
    conftest.py  test_features_leakage.py  test_wmape.py  test_reconstruction.py
```

---

## 20. Configuration — typed Delta tables, app-driven (NOT YAML)

Config is **stored in typed Delta tables** and managed through the **Databricks App** (CRUD front-end). The pipeline **reads config from these tables filtered by `market_id`** via `config/loader.py`. There is **no YAML file per market** — adding a market = inserting rows via the app, not adding files. This is a deliberate, established decision.

**Config table** (`control.forecast_config`, single EAV table; DDL in `config/ddl/config_tables.sql`):

| Table | Grain | Holds |
|-------|-------|-------|
| `forecast_config` | market × param_group × param_name | Single typed-EAV table storing all settings: market run settings (horizon_days, eval_horizon_days, history_start, etc.), category thresholds (accuracy_threshold, gate_grains), model config (enabled, priority, param_space), feature config (enabled, feature_class, params), and override sources. Pivoted into typed dataclasses by `config/loader.py`. |

`config/loader.py` returns a typed config object for `(market)`; every job takes `market_id`, `env`, `snapshot_date` params and pulls the rest from these tables. Seed/bootstrap defaults (e.g. demo-market thresholds A 0.85 / B 0.75 / C 0.65 / D 0.60 / SEASONAL 0.80) are inserted via the app or a one-time seed notebook — not committed as per-market YAML.

---

## 21. Acceptance criteria (Definition of Done, v1)

1. Daily job produces a **140-day** forecast for all active series in the demo market, idempotently, **< $100/run**.
2. **Decomposition present & reconstructs** within tolerance for 100% of served series (DQ-enforced).
3. **WMAPE gate at all three grains:** category-cohort, `market×sku`, and `market×store×sku` 7-day accuracy each meet/exceed thresholds (A85/B75/C65/D60/SEASONAL80); per-level pass/fail recorded in `backtest_metrics`, failures routed to diagnosis/HPT.
4. **Registration:** exactly one MLflow registered pyfunc per `market×category`; per-series champions resolved via `champion_index`; total registered models ≪ 10k at scale. **No dependency on the `mmf_sa` package** — the engine is in-house.
5. **Conditional HPT** runs only on threshold-missing series, only in the weekly/training job, cached, within budget.
6. **Onboarding test:** a second market builds end-to-end by inserting config rows for that market and changing the `market_id` param — **no code change**.
7. Platform standards met: DABs deploy (single Free Edition target), Liquid Clustering + Predictive Optimization, no hardcoded catalog, leakage tests pass, DQ audit populated, all modules import + tests pass locally with no Spark.
8. *(Phase 2)* Agents and override loop — deferred; not required for v1 done.

---

## 22. Phased plan & epics

| Phase | Epic | Key stories |
|-------|------|-------------|
| P0 (now) | Foundations | Repo (3-job layout) + DABs; `utils/` (incl. snapshot/date helpers); `data/ddl/tables.sql` + `config/ddl/config_tables.sql`; `config/loader.py` (reads Delta, not YAML) |
| P1 | JOB 1 Data | Feature pipeline (`data/features/`) → `sku_store_features`, read-only sources, `snapshot_date`-parameterized; leakage tests |
| P2 | JOB 2 Engine | `training/engine.py` (applyInPandas per-series); model wrappers (Prophet/statsforecast/LGBM) behind one interface |
| P3 | Accuracy + explainability **(the goal)** | Decomposition contract + extraction; backtest; volume-weighted WMAPE gate at 3 grains; champion → `champion_index`; conditional HPT |
| P4 | JOB 3 Serving | 140-day future spine from snapshot_date; daily scoring; `output`+`decomposition`; DQ; BI dashboard |
| P5 | Scale & onboard | Onboard market #2 by **config rows via the app** (no files, no code); cost tuning |
| **Phase 2 (later)** | Autonomy & feedback | Agents; override feedback loop; prescription; weather/social features |

---

## 23. Risks, Assumptions, Dependencies, Open Questions

**Risks.** (1) Accuracy/explainability tension — mitigated by bounded `model_uplift` + decomposition gate. (2) Cost at real-chain scale — mitigated by vectorized inference, conditional HPT off critical path, Liquid Clustering. (3) Agents learning bad overrides — mitigated by HITL. (4) Solo-maintainer bus factor — mitigated by autonomy + self-describing logs/agents (phase 2).

**Key assumptions** (all overridable in `config/`): horizon 140 / gate 7-day; registration grain `market×category`; direct multi-step for lags; uplift clip 0.25; reconstruction tol 0.5%; HPT 25 trials; bias-correction HITL. *Resolved:* threshold gate applies at **all three grains**; created tables live under Medallion-convention schemas (`source`/`gold`/`control`/`forecast`); overrides (phase 2) arrive via a **configurable list of silver-layer Delta tables**; BI market filtering is **BI-layer RLS**.

**Dependencies.** `gold.events` quality & future-promo completeness; `gold.calendar_date` calendar coverage; the override source tables (phase 2) — names/schemas TBD; UC grants.

**Open questions.** Override source table schema(s) when phase 2 is built; weather/social data sources; BI RLS ownership.

---

## 24. Next technical design activities

1. Confirm the remaining **[OPEN]** items above.
2. Stand up DABs skeleton on Databricks Free Edition; land the feature table and register it with Feature Engineering.
3. Implement the **decomposer contract + Prophet baseline** end-to-end on one category to validate the reconstruction gate and WMAPE harness before scaling to all categories.
4. Add GAM/LGBM candidates + Stage-2 uplift; wire champion index + category pyfunc.
5. Turn on daily inference + DQ + BI dashboard; then agents (phase 2); then override loop (phase 2); then onboard market #2 to prove the config-only path.
