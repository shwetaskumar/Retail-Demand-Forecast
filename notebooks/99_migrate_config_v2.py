# Databricks notebook source
# DBTITLE 1,Config Migration v1→v2
# MAGIC %md
# MAGIC ## 99_migrate_config_v2 — Collapse 270 EAV rows into ~12 compact JSON rows
# MAGIC
# MAGIC Migrates `control.forecast_config` from the legacy EAV format (one row per
# MAGIC param_group × scope × param_name) to the compact v2 format:
# MAGIC
# MAGIC | param_name | Content |
# MAGIC |---|---|
# MAGIC | `models` | `{model_type: cost_tier}` for enabled models |
# MAGIC | `param_space_<model>` | HPT search space per model |
# MAGIC | `features_historical` | JSON list of enabled historical-only features |
# MAGIC | `features_known` | JSON list of enabled known-at-inference features |
# MAGIC | `market_settings` | All scalar market settings as one JSON object |
# MAGIC | `category_thresholds` | `{category: accuracy_threshold}` |
# MAGIC
# MAGIC New key: `(market_id, category, sku_id, store_id, param_name)`
# MAGIC NULL in category/sku_id/store_id = market-wide default.

# COMMAND ----------

# DBTITLE 1,Read current v1 config for US
# Read the current EAV config for ALL markets
CATALOG = ""  # TODO: set to your Unity Catalog catalog name

v1_full_df = spark.table(f"{CATALOG}.control.forecast_config")
v1_full_pdf = v1_full_df.toPandas()
MARKETS = sorted(v1_full_pdf['market_id'].unique())
print(f"Markets found: {MARKETS}")
print(f"Total v1 rows: {len(v1_full_pdf)}")
for mkt in MARKETS:
    cnt = len(v1_full_pdf[v1_full_pdf['market_id'] == mkt])
    print(f"  {mkt}: {cnt} rows")

# COMMAND ----------

# DBTITLE 1,Transform v1 → v2 compact rows
import json
import pandas as pd
from datetime import datetime

def _migrate_one_market(v1_pdf, market):
    """Transform one market's v1 EAV rows into ~10-12 compact v2 rows."""
    rows_v2 = []

    # 1. MODELS
    model_rows = v1_pdf[v1_pdf['param_group'] == 'model']
    models_dict = {}
    param_spaces = {}
    for scope, grp in model_rows.groupby('scope'):
        params = dict(zip(grp['param_name'], grp['param_value']))
        enabled = str(params.get('enabled', 'True')).strip().lower() in ('true', '1', 'yes')
        if enabled:
            cost_tier = int(params.get('cost_tier', 0)) if params.get('cost_tier') else 0
            models_dict[scope] = cost_tier
            ps_raw = params.get('param_space', '{}')
            param_spaces[scope] = json.loads(ps_raw) if ps_raw else {}
    rows_v2.append({'param_name': 'models', 'param_value': json.dumps(models_dict, separators=(',', ':'))})
    for model_type, ps in param_spaces.items():
        rows_v2.append({'param_name': f'param_space_{model_type}', 'param_value': json.dumps(ps, separators=(',', ':'))})

    # 2. FEATURES
    feat_rows = v1_pdf[v1_pdf['param_group'] == 'feature']
    hist_features, known_features = [], []
    for scope, grp in feat_rows.groupby('scope'):
        params = dict(zip(grp['param_name'], grp['param_value']))
        enabled = str(params.get('enabled', 'True')).strip().lower() in ('true', '1', 'yes')
        if not enabled:
            continue
        if params.get('feature_class', 'historical_only') == 'historical_only':
            hist_features.append(scope)
        else:
            known_features.append(scope)
    rows_v2.append({'param_name': 'features_historical', 'param_value': json.dumps(sorted(hist_features))})
    rows_v2.append({'param_name': 'features_known', 'param_value': json.dumps(sorted(known_features))})

    # 3. MARKET SETTINGS
    market_params = v1_pdf[(v1_pdf['param_group'] == 'market') & (v1_pdf['scope'] == market)]
    ms_dict = {}
    for _, row in market_params.iterrows():
        key, val = row['param_name'], row['param_value']
        try:
            ms_dict[key] = float(val) if '.' in str(val) else int(val)
        except (ValueError, TypeError):
            if str(val).strip().lower() in ('true', 'false'):
                ms_dict[key] = str(val).strip().lower() == 'true'
            else:
                ms_dict[key] = val
    rows_v2.append({'param_name': 'market_settings', 'param_value': json.dumps(ms_dict, separators=(',', ':'))})

    # 4. CATEGORY THRESHOLDS
    cat_rows = v1_pdf[v1_pdf['param_group'] == 'category']
    thresholds = {}
    for scope, grp in cat_rows.groupby('scope'):
        params = dict(zip(grp['param_name'], grp['param_value']))
        thresholds[scope] = float(params.get('accuracy_threshold', 0.75))
    rows_v2.append({'param_name': 'category_thresholds', 'param_value': json.dumps(thresholds, separators=(',', ':'))})

    # Build rows with market key
    for r in rows_v2:
        r['market_id'] = market
        r['category'] = None
        r['sku_id'] = None
        r['store_id'] = None
        r['updated_by'] = 'migration_v2'
        r['updated_ts'] = datetime.now()

    return rows_v2, len(models_dict), len(hist_features), len(known_features)

# ── Process ALL markets ──
all_v2_rows = []
for mkt in MARKETS:
    mkt_pdf = v1_full_pdf[v1_full_pdf['market_id'] == mkt]
    rows, n_models, n_hist, n_known = _migrate_one_market(mkt_pdf, mkt)
    all_v2_rows.extend(rows)
    print(f"  {mkt}: {len(mkt_pdf)} v1 rows → {len(rows)} v2 rows ({n_models} models, {n_hist}+{n_known} features)")

v2_pdf = pd.DataFrame(all_v2_rows)
v2_pdf = v2_pdf[['market_id', 'category', 'sku_id', 'store_id', 'param_name', 'param_value', 'updated_by', 'updated_ts']]
print(f"\n=== Total: {len(v2_pdf)} v2 rows (from {len(v1_full_pdf)} v1 rows, {len(v1_full_pdf)//len(v2_pdf)}x reduction) ===")

# COMMAND ----------

# DBTITLE 1,Review v2 rows before writing
# Display the compact v2 rows for review
for mkt in MARKETS:
    mkt_rows = v2_pdf[v2_pdf['market_id'] == mkt]
    print(f"\n── {mkt} ({len(mkt_rows)} rows) ──")
    for _, row in mkt_rows.iterrows():
        val_preview = row['param_value'][:80] + ('...' if len(row['param_value']) > 80 else '')
        print(f"  {row['param_name']:<30} {val_preview}")

print(f"\n=== {len(v2_pdf)} total v2 rows (from {len(v1_full_pdf)} v1) ===")

# COMMAND ----------

# DBTITLE 1,Write v2 config to new table
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

# Convert pandas to Spark DataFrame with correct schema
schema = StructType([
    StructField("market_id", StringType(), False),
    StructField("category", StringType(), True),
    StructField("sku_id", StringType(), True),
    StructField("store_id", StringType(), True),
    StructField("param_name", StringType(), False),
    StructField("param_value", StringType(), False),
    StructField("updated_by", StringType(), True),
    StructField("updated_ts", TimestampType(), True),
])

v2_sdf = spark.createDataFrame(v2_pdf, schema=schema)

# Write directly to the original config table (overwrite schema → v2 format)
TARGET_TABLE = f"{CATALOG}.control.forecast_config"

# Drop and recreate — old table has constraints referencing removed columns
spark.sql(f"DROP TABLE IF EXISTS {TARGET_TABLE}")
v2_sdf.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)
print(f"\n✓ Written {v2_sdf.count()} rows to {TARGET_TABLE}")
print(f"  Schema replaced with v2 compact format (no more scope/param_group columns).")

# COMMAND ----------

# DBTITLE 1,Add champion_params column to champion_index
# Add champion_params column to store tuned hyperparameters per series
cols = [c.name for c in spark.table(f"{CATALOG}.gold.champion_index").schema.fields]
if 'champion_params' not in cols:
    spark.sql(f"""
        ALTER TABLE {CATALOG}.gold.champion_index
        ADD COLUMNS (champion_params STRING COMMENT 'JSON: tuned hyperparameters that won HPT for this series')
    """)
    print(f"✓ champion_params column added to {CATALOG}.gold.champion_index")
else:
    print(f"✓ champion_params column already exists")

# Verify
cols = [c.name for c in spark.table(f"{CATALOG}.gold.champion_index").schema.fields]
print(f"  Columns now: {cols}")

# COMMAND ----------

# DBTITLE 1,Validation — read back and verify
# Read back and validate all JSON values parse correctly
verify_pdf = spark.table(TARGET_TABLE).toPandas()

print(f"=== Validation: {len(verify_pdf)} rows across {verify_pdf['market_id'].nunique()} markets in {TARGET_TABLE} ===")
errors = []

for _, row in verify_pdf.iterrows():
    pn = row['param_name']
    pv = row['param_value']
    try:
        parsed = json.loads(pv)
        if pn == 'models':
            assert isinstance(parsed, dict), f"models should be dict, got {type(parsed)}"
            assert len(parsed) > 0, "models dict is empty"
            print(f"  ✓ models: {len(parsed)} enabled — {list(parsed.keys())}")
        elif pn.startswith('param_space_'):
            model = pn.replace('param_space_', '')
            assert isinstance(parsed, dict), f"param_space should be dict"
            print(f"  ✓ {pn}: {len(parsed)} hyperparams")
        elif pn == 'features_historical':
            assert isinstance(parsed, list), f"features_historical should be list"
            print(f"  ✓ features_historical: {len(parsed)} features")
        elif pn == 'features_known':
            assert isinstance(parsed, list), f"features_known should be list"
            print(f"  ✓ features_known: {len(parsed)} features")
        elif pn == 'market_settings':
            assert isinstance(parsed, dict), f"market_settings should be dict"
            assert 'horizon_days' in parsed, "missing horizon_days"
            print(f"  ✓ market_settings: {len(parsed)} keys (horizon={parsed['horizon_days']}, hpt_trials={parsed.get('hpt_trials')})")
        elif pn == 'category_thresholds':
            assert isinstance(parsed, dict), f"category_thresholds should be dict"
            assert 'A' in parsed, "missing category A"
            print(f"  ✓ category_thresholds: {parsed}")
    except Exception as e:
        errors.append(f"  ✗ {pn}: {e}")

if errors:
    print(f"\n{'='*50}")
    print(f"ERRORS ({len(errors)}):")
    for err in errors:
        print(err)
else:
    print(f"\n✓ All {len(verify_pdf)} rows validated successfully.")
    print(f"\n=== Migration complete. Next steps: ===")
    print(f"  1. Update config/loader.py to auto-detect v2 format")
    print(f"  2. Run training with new config to verify same ForecastConfig output")
    print(f"  3. Rename {TARGET_TABLE} -> control.forecast_config (or update _CONFIG_TABLE)")
