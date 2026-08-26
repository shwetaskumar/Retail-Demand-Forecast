"""
utils/logging.py — structured logging, never `print` (docs/instructions.md §17 Observability).

Every job task logs via a logger tagged with (market_id, env, run_id) so
duration/series-counts/pass-rate metrics can be correlated across driver logs.
"""
from __future__ import annotations

import logging


def get_logger(
    name: str,
    market_id: str | None = None,
    env: str | None = None,
    run_id: str | None = None,
) -> logging.LoggerAdapter:
    """Return a logger tagged with run context (market_id/env/run_id) for structured job logs."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s]%(context)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    tags = [f"{k}={v}" for k, v in (("market_id", market_id), ("env", env), ("run_id", run_id)) if v is not None]
    context = f" ({', '.join(tags)})" if tags else ""
    return logging.LoggerAdapter(logger, {"context": context})
