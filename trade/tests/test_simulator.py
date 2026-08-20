"""The trade simulator defines the objective, so its edge cases are pinned here.

Each test builds bars by hand and checks the exact R multiple, because a
subtle error in exit resolution would silently shift every expected value the
system reports.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from screener.config import DEFAULT_CONFIG, TradePlan
from screener.simulator import simulate_panel

CODE = "1234"
ATR = 10.0
# entry 100, stop_atr_mult 1.5 -> risk 15, stop 85; target_atr_mult 3 -> 130
ENTRY, RISK, STOP, TARGET = 100.0, 15.0, 85.0, 130.0


def make_config(**plan_kwargs):
    plan = TradePlan(
        stop_atr_mult=1.5, target_atr_mult=3.0, max_hold_bars=5, cost_pct=0.0,
        **plan_kwargs,
    )
    return dataclasses.replace(DEFAULT_CONFIG, trade=plan)


def build(bars: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bars: list of dicts with open/high/low/close, one per trading day."""
    dates = pd.bdate_range("2026-01-05", periods=len(bars))
    panel = pd.DataFrame(
        {
            "code": CODE,
            "date": dates,
            "open": [b["open"] for b in bars],
            "high": [b["high"] for b in bars],
            "low": [b["low"] for b in bars],
            "close": [b["close"] for b in bars],
            "volume": 100_000.0,
        }
    )
    atr = pd.DataFrame({CODE: [ATR] * len(bars)}, index=dates)
    return panel, atr


def flat(price: float) -> dict:
    return {"open": price, "high": price, "low": price, "close": price}


def r_of(panel, atr, config, signal_index=0) -> float:
    trades = simulate_panel(panel, config=config, atr=atr)
    row = trades[trades["date"] == panel["date"].iloc[signal_index]]
    assert len(row) == 1, f"expected one trade, got {len(row)}"
    return float(row["r_multiple"].iloc[0])


def test_target_hit_gives_reward_risk_ratio():
    bars = [flat(100.0)] * 8
    bars[1] = {"open": ENTRY, "high": 101.0, "low": 99.0, "close": 100.0}
    bars[3] = {"open": 120.0, "high": 135.0, "low": 119.0, "close": 132.0}
    panel, atr = build(bars)
    assert r_of(panel, atr, make_config()) == pytest.approx(2.0)


def test_stop_hit_gives_minus_one_r():
    bars = [flat(100.0)] * 8
    bars[1] = {"open": ENTRY, "high": 101.0, "low": 99.0, "close": 100.0}
    bars[2] = {"open": 90.0, "high": 92.0, "low": 80.0, "close": 88.0}
    panel, atr = build(bars)
    assert r_of(panel, atr, make_config()) == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_the_open_not_the_stop():
    """A stop is not a guaranteed price; a gap-down fill must hurt more than 1R."""
    bars = [flat(100.0)] * 8
    bars[1] = {"open": ENTRY, "high": 101.0, "low": 99.0, "close": 100.0}
    bars[2] = {"open": 75.0, "high": 78.0, "low": 70.0, "close": 76.0}
    panel, atr = build(bars)
    # fill at 75, not at the 85 stop: (75 - 100) / 15
    assert r_of(panel, atr, make_config()) == pytest.approx(-25.0 / 15.0)


def test_gap_above_target_fills_at_the_open():
    bars = [flat(100.0)] * 8
    bars[1] = {"open": ENTRY, "high": 101.0, "low": 99.0, "close": 100.0}
    bars[2] = {"open": 140.0, "high": 145.0, "low": 139.0, "close": 142.0}
    panel, atr = build(bars)
    assert r_of(panel, atr, make_config()) == pytest.approx(40.0 / 15.0)


def test_same_bar_touching_both_levels_books_the_stop():
    """Daily bars cannot prove which came first, so the pessimistic side wins."""
    bars = [flat(100.0)] * 8
    bars[1] = {"open": ENTRY, "high": 101.0, "low": 99.0, "close": 100.0}
    bars[2] = {"open": 100.0, "high": 135.0, "low": 84.0, "close": 120.0}
    panel, atr = build(bars)
    assert r_of(panel, atr, make_config()) == pytest.approx(-1.0)


def test_time_exit_uses_the_close_of_the_last_held_bar():
    bars = [flat(100.0)] * 9
    bars[1] = {"open": ENTRY, "high": 101.0, "low": 99.0, "close": 100.0}
    bars[5] = {"open": 104.0, "high": 106.0, "low": 103.0, "close": 106.0}
    panel, atr = build(bars)
    trades = simulate_panel(panel, config=make_config(), atr=atr)
    row = trades[trades["date"] == panel["date"].iloc[0]].iloc[0]
    assert row["exit_reason"] == "time"
    assert row["bars_held"] == 5
    # exit at close of bar index 5 = 106
    assert float(row["r_multiple"]) == pytest.approx(6.0 / 15.0)


def test_entry_is_the_next_open_never_the_signal_close():
    """Guards against the most damaging form of look-ahead in the system."""
    bars = [flat(100.0)] * 8
    bars[0] = {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}
    bars[1] = {"open": 200.0, "high": 205.0, "low": 199.0, "close": 200.0}
    panel, atr = build(bars)
    trades = simulate_panel(panel, config=make_config(), atr=atr)
    row = trades[trades["date"] == panel["date"].iloc[0]].iloc[0]
    assert float(row["entry"]) == pytest.approx(200.0)
    assert row["entry_date"] == panel["date"].iloc[1]


def test_costs_reduce_the_r_multiple():
    bars = [flat(100.0)] * 8
    bars[1] = {"open": ENTRY, "high": 101.0, "low": 99.0, "close": 100.0}
    bars[3] = {"open": 120.0, "high": 135.0, "low": 119.0, "close": 132.0}
    panel, atr = build(bars)
    plan = TradePlan(stop_atr_mult=1.5, target_atr_mult=3.0, max_hold_bars=5, cost_pct=0.003)
    config = dataclasses.replace(DEFAULT_CONFIG, trade=plan)
    # 2R gross, minus 0.3% of the 100 entry = 0.30 yen, over a 15 yen risk unit
    assert r_of(panel, atr, config) == pytest.approx(2.0 - 0.3 / 15.0)


def test_unresolved_trades_at_the_end_of_history_are_dropped():
    """A half-finished trade would bias calibration toward the recent past."""
    bars = [flat(100.0)] * 4
    panel, atr = build(bars)
    trades = simulate_panel(panel, config=make_config(), atr=atr)
    # With 4 bars and a 5-bar hold, no signal can resolve.
    assert trades.empty


def test_exit_date_matches_bars_held():
    bars = [flat(100.0)] * 9
    bars[1] = {"open": ENTRY, "high": 101.0, "low": 99.0, "close": 100.0}
    bars[3] = {"open": 120.0, "high": 135.0, "low": 119.0, "close": 132.0}
    panel, atr = build(bars)
    trades = simulate_panel(panel, config=make_config(), atr=atr)
    row = trades[trades["date"] == panel["date"].iloc[0]].iloc[0]
    assert row["bars_held"] == 3
    assert row["exit_date"] == panel["date"].iloc[3]
