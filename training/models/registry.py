"""
training/models/registry.py — auto-discovery of all ForecastModel implementations.

Replaces the static STAGE1_MODEL_REGISTRY dict in engine.py.  Any class in
training/models/ that:
  - subclasses ForecastModel
  - is not ForecastModel itself or UpliftModel (UpliftModel is a wrapper, not
    a standalone candidate activated via config_model)
  - declares a model_type class attribute (str)

is automatically registered the first time get_registry() is called.  Adding a
new algorithm requires only:
  1. Write training/models/<name>.py with model_type = "<name>".
  2. INSERT a row into config_model via the Databricks App.
  Zero changes to engine.py, registry.py, or any other existing file.

Public API:
  get_registry() -> dict[str, type[ForecastModel]]
      Returns {model_type_string: class} for every discovered wrapper.
  build(model_type) -> ForecastModel
      Instantiates and returns the wrapper for model_type; raises KeyError if unknown.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.models.base import ForecastModel

# Modules in training/models/ that are NOT candidate wrappers and must be skipped.
_SKIP_MODULES = frozenset({"base", "uplift", "registry", "__init__"})

# "*_old" modules (e.g. lgbm_old.py) are pre-refactor snapshots kept for reference/
# rollback only — they declare the SAME model_type string as their current
# counterpart (e.g. "lgbm"), so if they were auto-discovered they'd silently win
# the registry slot on ties (dict overwrite, last-iterated module wins) since
# "_old" sorts after the base name alphabetically. Never auto-discover them.
_OLD_MODULE_SUFFIX = "_old"

# Module-level cache — populated once on first call, then reused for the lifetime
# of the Spark task / driver process (safe because wrappers are stateless classes).
_REGISTRY: dict[str, type] | None = None


def _discover() -> dict[str, type]:
    """Scan training/models/ and register every concrete ForecastModel subclass.

    Uses pkgutil.iter_modules so that adding a new .py file to training/models/
    is sufficient — no explicit import list to maintain anywhere.
    """
    from training.models.base import ForecastModel as _Base  # lazy: avoids circular at module init
    import training.models as _pkg

    registry: dict[str, type] = {}
    for _, module_name, _ in pkgutil.iter_modules(_pkg.__path__):
        if module_name in _SKIP_MODULES or module_name.endswith(_OLD_MODULE_SUFFIX):
            continue
        try:
            mod = importlib.import_module(f"training.models.{module_name}")
        except Exception:
            # Broken module: skip silently so one bad wrapper doesn't kill the engine.
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if (
                obj is None
                or not isinstance(obj, type)
                or obj is _Base
                or not issubclass(obj, _Base)
            ):
                continue
            mtype = getattr(obj, "model_type", None)
            if mtype and isinstance(mtype, str):
                registry[mtype] = obj
    return registry


def get_registry() -> dict[str, type]:
    """Return {model_type: class} for every discovered ForecastModel subclass.

    Cached after the first call.  Thread-safe for read-only access within
    a single Spark task (applyInPandas workers are single-threaded per task).
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _discover()
    return _REGISTRY


def build(model_type: str) -> "ForecastModel":
    """Instantiate a ForecastModel by model_type string.

    Args:
        model_type: The model_type string declared on the wrapper class
                    (e.g. "ets_event", "auto_arima", "prophet", "lgbm").

    Returns:
        A fresh (unfitted) instance of the corresponding ForecastModel subclass.

    Raises:
        KeyError: If model_type is not found in the registry.
    """
    reg = get_registry()
    if model_type not in reg:
        raise KeyError(
            f"Unknown model_type {model_type!r}. "
            f"Registered types: {sorted(reg)}. "
            f"Add a ForecastModel subclass with model_type={model_type!r} to training/models/."
        )
    return reg[model_type]()
