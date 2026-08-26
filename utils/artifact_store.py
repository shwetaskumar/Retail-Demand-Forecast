"""utils/artifact_store.py — UC Volume artifact storage for champion_index.

Stores heavy model blobs (serialized model weights, ETS forecast arrays, LightGBM
boosters) as individual files in the UC Volume `model_artifacts`, replacing the
6-160 KB JSON embedded in the `params` column with a lightweight URI reference.

Volume layout:
    /Volumes/{catalog}/gold/model_artifacts/{market_id}/{snapshot_date}/{series_key}_v{version}.pkl

Design decisions:
    - pickle (not joblib/cloudpickle): all stored objects are plain dicts of
      numpy arrays + python scalars — no custom classes or lambdas. pickle is
      stdlib, fast, and sufficient.
    - One file per series per version: enables O(1) lookup from artifact_uri
      without scanning directories. Average file size ~6-160 KB.
    - Compression: gzip level 1 (fast) wraps the pickle — halves ETS arrays and
      LGBM boosters at negligible CPU cost.
    - Backwards-compatible: inference checks artifact_uri first; if NULL, falls
      back to reading from params JSON (existing rows).
"""
from __future__ import annotations

import gzip
import os
import pickle


_VOLUME_BASE = "/Volumes/{catalog}/gold/model_artifacts"

# Keys in the params dict that contain heavy binary blobs and should be
# extracted to volume storage. Everything else stays in params JSON.
_HEAVY_KEYS = frozenset([
    "model_artifact",        # serialized model bytes (lgbm, prophet, elastic_net, etc.)
    "ets_forecast_values",   # precomputed 140-day forecast array (ets_event)
    "ets_state",             # ETS smoothing state for warm-start
    "arima_state",           # AutoARIMA fitted state
    "booster_bytes",         # raw LightGBM booster dump
])


def volume_path(catalog: str, market_id: str, snapshot_date: str, series_key: str, version: int) -> str:
    """Compute the deterministic artifact path for a champion."""
    base = _VOLUME_BASE.format(catalog=catalog)
    # Sanitize series_key for filesystem safety (replace / and special chars)
    safe_key = series_key.replace("/", "__").replace(" ", "_")
    return f"{base}/{market_id}/{snapshot_date}/{safe_key}_v{version}.pkl.gz"


def save_artifact(catalog: str, market_id: str, snapshot_date: str,
                  series_key: str, version: int, heavy_data: dict) -> str:
    """Serialize heavy model data to the UC Volume and return the artifact URI.

    Args:
        heavy_data: dict of key->value for all _HEAVY_KEYS extracted from params.

    Returns:
        The volume path (artifact_uri) where the data was written.
    """
    path = volume_path(catalog, market_id, snapshot_date, series_key, version)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(heavy_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_artifact(artifact_uri: str) -> dict:
    """Load heavy model data from the UC Volume.

    Returns:
        dict of key->value (same structure as was passed to save_artifact).
    """
    with gzip.open(artifact_uri, "rb") as f:
        return pickle.load(f)  # noqa: S301 — trusted internal data


def split_params(params_dict: dict) -> tuple[dict, dict]:
    """Split a params dict into (lightweight_params, heavy_data).

    lightweight_params: metadata that stays in the Delta table (hyperparams, flags).
    heavy_data: blobs that go to volume storage.

    If no heavy keys are present, heavy_data is empty and the original params
    should be stored as-is (backwards-compatible with old rows).
    """
    heavy = {}
    light = {}
    for k, v in params_dict.items():
        if k in _HEAVY_KEYS and v is not None:
            heavy[k] = v
        else:
            light[k] = v
    return light, heavy


def merge_params(lightweight_params: dict, heavy_data: dict) -> dict:
    """Reconstruct the full params dict from lightweight + heavy data.

    Used by inference to reassemble the params before passing to model.from_serialized().
    """
    merged = dict(lightweight_params)
    merged.update(heavy_data)
    return merged
