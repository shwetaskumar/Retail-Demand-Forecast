"""
tests/test_dates.py — point-in-time window unit tests for utils/dates.py (Step 2).

Pure stdlib `datetime` — no pandas/Spark fixtures needed.
"""
from __future__ import annotations

import datetime as dt

import pytest

from utils.dates import (
    eval_window,
    future_horizon,
    read_window,
    resolve_snapshot_date,
    write_window,
)


def test_resolve_snapshot_date_defaults_to_today_when_none():
    assert resolve_snapshot_date(None) == dt.date.today()


def test_resolve_snapshot_date_parses_iso_string():
    assert resolve_snapshot_date("2026-07-01") == dt.date(2026, 7, 1)


def test_resolve_snapshot_date_passes_through_date():
    d = dt.date(2026, 7, 1)
    assert resolve_snapshot_date(d) is d


def test_read_window_bounds():
    snapshot_date = dt.date(2026, 7, 1)
    start, end = read_window(snapshot_date, incremental_read_days=400)
    assert end == snapshot_date
    assert start == snapshot_date - dt.timedelta(days=400)
    assert (end - start).days == 400


def test_read_window_rejects_short_lookback():
    snapshot_date = dt.date(2026, 7, 1)
    with pytest.raises(ValueError, match="incremental_read_days must be >= 400"):
        read_window(snapshot_date, incremental_read_days=30)


def test_write_window_bounds():
    snapshot_date = dt.date(2026, 7, 1)
    start, end = write_window(snapshot_date, incremental_write_days=365)
    assert end == snapshot_date
    assert start == snapshot_date - dt.timedelta(days=365)


def test_future_horizon_is_strictly_after_snapshot_date():
    snapshot_date = dt.date(2026, 7, 1)
    start, end = future_horizon(snapshot_date, horizon_days=140)
    assert start == snapshot_date + dt.timedelta(days=1)
    assert end == snapshot_date + dt.timedelta(days=140)
    assert start > snapshot_date
    # 140 inclusive days: end - start + 1 == 140
    assert (end - start).days + 1 == 140


def test_eval_window_is_strictly_after_snapshot_date():
    snapshot_date = dt.date(2026, 7, 1)
    start, end = eval_window(snapshot_date, eval_horizon_days=7)
    assert start == snapshot_date + dt.timedelta(days=1)
    assert end == snapshot_date + dt.timedelta(days=7)
    assert (end - start).days + 1 == 7


def test_future_horizon_and_eval_window_never_include_snapshot_date_itself():
    # leakage control (docs/instructions.md §17): the horizon must start AFTER snapshot_date,
    # never on or before it.
    snapshot_date = dt.date(2026, 7, 1)
    future_start, _ = future_horizon(snapshot_date, horizon_days=140)
    eval_start, _ = eval_window(snapshot_date, eval_horizon_days=7)
    assert future_start > snapshot_date
    assert eval_start > snapshot_date
