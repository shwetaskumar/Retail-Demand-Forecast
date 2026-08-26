"""
utils/dates.py — snapshot_date handling / point-in-time windows (docs/instructions.md §4 lock).

Every source-read window and lifecycle window is anchored on `snapshot_date`,
never `current_date()` directly, so any historical as-of date can be replayed.
Pure Python / stdlib `datetime` only — no Spark dependency, so these are
unit-tested directly (tests/test_dates.py).
"""
from __future__ import annotations

import datetime as dt

_MIN_INCREMENTAL_READ_DAYS = 400


def resolve_snapshot_date(snapshot_date: str | dt.date | None) -> dt.date:
    """Return the effective as-of date: the given value, or today() if None (default)."""
    if snapshot_date is None or snapshot_date == "":
        return dt.date.today()
    if isinstance(snapshot_date, str):
        return dt.date.fromisoformat(snapshot_date)
    return snapshot_date


def read_window(snapshot_date: dt.date, incremental_read_days: int) -> tuple[dt.date, dt.date]:
    """Return (start, end) source-read bounds: [snapshot_date - incremental_read_days, snapshot_date].

    incremental_read_days must be >= 400 so t-365 lags and YoY features compute
    (feature_engineering_spec.md Parameters).
    """
    if incremental_read_days < _MIN_INCREMENTAL_READ_DAYS:
        raise ValueError(
            f"incremental_read_days must be >= {_MIN_INCREMENTAL_READ_DAYS} "
            f"(t-365 lag + YoY features need it), got {incremental_read_days}"
        )
    start = snapshot_date - dt.timedelta(days=incremental_read_days)
    return start, snapshot_date


def write_window(snapshot_date: dt.date, incremental_write_days: int) -> tuple[dt.date, dt.date]:
    """Return (start, end) bounds for the incremental write (last `incremental_write_days` days)."""
    start = snapshot_date - dt.timedelta(days=incremental_write_days)
    return start, snapshot_date


def future_horizon(snapshot_date: dt.date, horizon_days: int) -> tuple[dt.date, dt.date]:
    """Return (start, end) for the future spine: [snapshot_date + 1, snapshot_date + horizon_days].

    Feeds inference/future_spine.py (140-day horizon, docs/instructions.md §9/§13).
    """
    start = snapshot_date + dt.timedelta(days=1)
    end = snapshot_date + dt.timedelta(days=horizon_days)
    return start, end


def eval_window(snapshot_date: dt.date, eval_horizon_days: int) -> tuple[dt.date, dt.date]:
    """Return (start, end) for the champion evaluation horizon (7-day WMAPE, docs/instructions.md §3).

    Same shape as `future_horizon` but sized to the evaluation horizon; callers
    pass a fold's origin date (not necessarily the run's overall snapshot_date)
    when scoring a rolling-origin backtest fold (training/backtest.py).
    """
    start = snapshot_date + dt.timedelta(days=1)
    end = snapshot_date + dt.timedelta(days=eval_horizon_days)
    return start, end
