"""TSE daily price limits, and the simulator's use of them."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from screener.config import DEFAULT_CONFIG, TradePlan
from screener.limits import limit_width, locked_limit_down, locked_limit_up
from screener.simulator import simulate_panel
from tests.test_simulator import build, flat, make_config


# ------------------------------------------------------------ limit table


@pytest.mark.parametrize(
    "base,width",
    [
        (99.0, 30.0),
        (100.0, 50.0),      # boundary: 100 is no longer "under 100"
        (150.0, 50.0),
        (500.0, 100.0),
        (999.0, 150.0),
        (1_000.0, 300.0),
        (2_500.0, 500.0),
        (9_999.0, 1_500.0),
        (25_000.0, 5_000.0),
    ],
)
def test_limit_width_matches_the_published_table(base, width):
    assert limit_width(np.array([base]))[0] == pytest.approx(width)


def test_limit_width_is_monotone_in_price():
    prices = np.array([50, 150, 400, 900, 4_000, 40_000, 900_000], dtype=float)
    widths = limit_width(prices)
    assert list(widths) == sorted(widths)


def test_limit_width_is_nan_for_missing_prices():
    assert np.isnan(limit_width(np.array([np.nan]))[0])


def test_limit_width_handles_the_open_ended_top_band():
    assert limit_width(np.array([99_000_000.0]))[0] == pytest.approx(10_000_000.0)


# ------------------------------------------------------------ lock detection


def test_locked_limit_up_requires_the_bar_never_to_trade_below_the_limit():
    upper = np.array([130.0])
    # Opens at the limit and never trades lower: no seller, no fill.
    assert locked_limit_up(np.array([130.0]), np.array([130.0]), np.array([130.0]), upper)[0]
    # Touches the limit but trades below it during the day: a fill was available.
    assert not locked_limit_up(np.array([120.0]), np.array([130.0]), np.array([118.0]), upper)[0]


def test_locked_limit_down_requires_the_bar_never_to_trade_above_the_limit():
    lower = np.array([70.0])
    assert locked_limit_down(np.array([70.0]), np.array([70.0]), np.array([70.0]), lower)[0]
    assert not locked_limit_down(np.array([80.0]), np.array([85.0]), np.array([70.0]), lower)[0]


# --------------------------------------------------------------- simulator


def test_signal_is_dropped_when_the_entry_bar_is_locked_limit_up():
    """No fill was available, so there was never a trade to record."""
    bars = [flat(100.0)] * 8
    # prev close 100 -> upper limit 150; the bar opens and stays pinned there.
    bars[1] = {"open": 150.0, "high": 150.0, "low": 150.0, "close": 150.0}
    panel, atr = build(bars)
    trades = simulate_panel(panel, config=make_config(), atr=atr)
    assert panel["date"].iloc[0] not in set(trades["date"])


def test_the_same_signal_is_kept_when_price_limits_are_ignored():
    """Confirms the drop above comes from the limit rule, not another filter."""
    bars = [flat(100.0)] * 8
    bars[1] = {"open": 150.0, "high": 150.0, "low": 150.0, "close": 150.0}
    panel, atr = build(bars)
    plan = TradePlan(
        stop_atr_mult=1.5, target_atr_mult=3.0, max_hold_bars=5,
        cost_pct=0.0, respect_price_limits=False,
    )
    config = dataclasses.replace(DEFAULT_CONFIG, trade=plan)
    trades = simulate_panel(panel, config=config, atr=atr)
    assert panel["date"].iloc[0] in set(trades["date"])


def test_entry_bar_that_merely_touches_the_limit_is_still_tradable():
    bars = [flat(100.0)] * 8
    # Opens below the 150 limit, touches it, so a fill existed at the open.
    bars[1] = {"open": 110.0, "high": 150.0, "low": 108.0, "close": 149.0}
    panel, atr = build(bars)
    trades = simulate_panel(panel, config=make_config(), atr=atr)
    row = trades[trades["date"] == panel["date"].iloc[0]]
    assert len(row) == 1
    assert float(row["entry"].iloc[0]) == pytest.approx(110.0)


def test_stop_on_a_limit_locked_bar_fills_worse_than_one_r():
    """Being locked limit-down means the holder could not sell at the stop."""
    # entry 100 (atr 10, stop_atr_mult 1.5) -> stop 85.
    bars = [flat(100.0)] * 9
    bars[1] = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
    # prev close 100 -> lower limit 50; the bar is pinned there all session,
    # so no sale is possible even though the stop at 85 was breached.
    bars[2] = {"open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0}
    # Next session is where the holder actually gets out.
    bars[3] = {"open": 45.0, "high": 48.0, "low": 44.0, "close": 46.0}
    panel, atr = build(bars)
    trades = simulate_panel(panel, config=make_config(), atr=atr)
    row = trades[trades["date"] == panel["date"].iloc[0]].iloc[0]
    assert row["exit_reason"] == "stop"
    # Filled at 45 on the following open — not at the 50 lock, and nowhere
    # near the 85 stop the naive model would have booked.
    assert float(row["r_multiple"]) == pytest.approx((45.0 - 100.0) / 15.0)


def test_price_limits_do_not_change_an_ordinary_trade():
    bars = [flat(1000.0)] * 8
    bars[1] = {"open": 1000.0, "high": 1010.0, "low": 990.0, "close": 1000.0}
    bars[3] = {"open": 1020.0, "high": 1035.0, "low": 1015.0, "close": 1032.0}
    panel, atr = build(bars)
    with_limits = simulate_panel(panel, config=make_config(), atr=atr)
    plan = TradePlan(
        stop_atr_mult=1.5, target_atr_mult=3.0, max_hold_bars=5,
        cost_pct=0.0, respect_price_limits=False,
    )
    without = simulate_panel(
        panel, config=dataclasses.replace(DEFAULT_CONFIG, trade=plan), atr=atr
    )
    a = with_limits[with_limits["date"] == panel["date"].iloc[0]]["r_multiple"].iloc[0]
    b = without[without["date"] == panel["date"].iloc[0]]["r_multiple"].iloc[0]
    assert float(a) == pytest.approx(float(b))
