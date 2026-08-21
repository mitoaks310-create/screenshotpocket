"""TSE trading calendar derived from free official sources."""

from __future__ import annotations

import pandas as pd
import pytest

from screener.calendar import (
    EXCEPTIONAL_CLOSURES,
    closed_days,
    is_trading_day,
    load_holidays,
    trading_days,
)


def _holidays():
    df = load_holidays()
    if df.empty:
        pytest.skip("holiday list not downloaded in this checkout")
    return df


def test_holiday_list_covers_a_long_history():
    df = _holidays()
    assert df["date"].min().year <= 1960
    assert df["date"].max().year >= pd.Timestamp.today().year


def test_weekends_are_never_trading_days():
    for day in ("2026-08-22", "2026-08-23"):  # Sat, Sun
        assert not is_trading_day(day, _holidays())


def test_national_holidays_are_not_trading_days():
    # 2026-01-12 成人の日 (Monday), 2026-05-05 こどもの日 (Tuesday)
    for day in ("2026-01-12", "2026-05-05"):
        assert not is_trading_day(day, _holidays())


def test_the_new_year_break_is_closed_beyond_the_national_holiday():
    """Jan 1 is a national holiday; Jan 2-3 and Dec 31 are exchange-specific."""
    holidays = _holidays()
    # 2026-01-02 is a Friday, so only the TSE rule can close it.
    assert not is_trading_day("2026-01-02", holidays)
    # 2025-12-31 is a Wednesday.
    assert not is_trading_day("2025-12-31", holidays)


def test_ordinary_weekdays_are_trading_days():
    for day in ("2026-08-20", "2026-06-10", "2025-03-05"):
        assert is_trading_day(day, _holidays())


def test_known_exchange_outage_is_closed():
    """2020-10-01 was a full-day outage; no calendar rule predicts it."""
    assert pd.Timestamp("2020-10-01") in EXCEPTIONAL_CLOSURES
    assert not is_trading_day("2020-10-01", _holidays())


def test_trading_day_count_per_year_is_plausible():
    """The TSE runs roughly 245 sessions a year."""
    days = trading_days("2023-01-01", "2023-12-31", _holidays())
    assert 235 <= len(days) <= 250


def test_trading_days_are_sorted_and_unique():
    days = trading_days("2024-01-01", "2024-12-31", _holidays())
    assert days.is_monotonic_increasing
    assert days.is_unique


def test_empty_range_returns_empty_index():
    assert len(trading_days("2026-08-20", "2026-08-01", _holidays())) == 0


def test_single_closed_day_range():
    assert len(trading_days("2026-01-01", "2026-01-01", _holidays())) == 0


def test_closed_days_stay_inside_the_requested_range():
    closed = closed_days("2026-03-01", "2026-03-31", _holidays())
    assert all(pd.Timestamp("2026-03-01") <= d <= pd.Timestamp("2026-03-31") for d in closed)
