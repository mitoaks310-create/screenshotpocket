from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener import indicators as ind


def frame(values, cols=("A",)):
    idx = pd.bdate_range("2026-01-05", periods=len(values))
    return pd.DataFrame({c: values for c in cols}, index=idx)


def test_sma_matches_plain_mean_and_warms_up():
    df = frame([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ind.sma(df, 3)
    assert out["A"].iloc[:2].isna().all()
    assert out["A"].iloc[2] == pytest.approx(2.0)
    assert out["A"].iloc[4] == pytest.approx(4.0)


def test_true_range_first_bar_is_the_bar_span():
    high = frame([10.0, 12.0])
    low = frame([8.0, 9.0])
    close = frame([9.0, 11.0])
    tr = ind.true_range(high, low, close)
    # No previous close exists on bar 0, so TR is high-low rather than NaN.
    assert tr["A"].iloc[0] == pytest.approx(2.0)
    # Bar 1: max(12-9, |12-9|, |9-9|) = 3
    assert tr["A"].iloc[1] == pytest.approx(3.0)


def test_true_range_uses_previous_close_on_gaps():
    high = frame([10.0, 20.0])
    low = frame([9.0, 19.0])
    close = frame([9.5, 19.5])
    tr = ind.true_range(high, low, close)
    # Gap up: |20 - 9.5| = 10.5 dominates the 1.0 bar span.
    assert tr["A"].iloc[1] == pytest.approx(10.5)


def test_atr_uses_wilder_smoothing():
    n = 14
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.standard_normal(60))
    high = frame(base + 1.0)
    low = frame(base - 1.0)
    close = frame(base)
    atr = ind.atr(high, low, close, n)
    tr = ind.true_range(high, low, close)
    expected = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pd.testing.assert_frame_equal(atr, expected)


def test_rsi_is_100_when_every_bar_rises():
    df = frame(list(np.arange(1.0, 40.0)))
    out = ind.rsi(df, 14)
    assert out["A"].iloc[-1] == pytest.approx(100.0)


def test_rsi_stays_within_bounds_on_noisy_data():
    rng = np.random.default_rng(3)
    df = frame(list(100 + np.cumsum(rng.standard_normal(200))))
    out = ind.rsi(df, 14).dropna()
    assert out["A"].between(0.0, 100.0).all()


def test_bollinger_percent_b_is_zero_at_lower_and_one_at_upper():
    rng = np.random.default_rng(7)
    df = frame(list(100 + rng.standard_normal(80) * 3))
    pb, bw, mid = ind.bollinger(df, 25, 2.0)
    sd = df.rolling(25, min_periods=25).std(ddof=0)
    upper = mid + 2 * sd
    lower = mid - 2 * sd
    at_upper = (upper - lower).where(lambda x: x > 0)
    # Reconstruct %B from the definition and compare.
    expected = (df - lower) / at_upper
    pd.testing.assert_frame_equal(pb.dropna(), expected.dropna())
    assert (bw.dropna() > 0).all().all()


def test_adx_is_bounded_and_high_for_a_clean_trend():
    trend = np.arange(1.0, 80.0)
    high = frame(list(trend + 0.5))
    low = frame(list(trend - 0.5))
    close = frame(list(trend))
    adx = ind.adx(high, low, close, 14).dropna()
    assert adx["A"].between(0.0, 100.0).all()
    # A monotone ramp is as trending as it gets.
    assert adx["A"].iloc[-1] > 60.0


def test_obv_accumulates_signed_volume():
    close = frame([10.0, 11.0, 10.5, 12.0])
    volume = frame([100.0, 200.0, 300.0, 400.0])
    out = ind.obv(close, volume)
    # first bar has no direction (diff NaN -> 0), then +200, -300, +400
    assert list(out["A"]) == pytest.approx([0.0, 200.0, -100.0, 300.0])


def test_slope_pct_is_per_bar_growth():
    df = frame([100.0] * 10 + [100.0 * (1.01**10)])
    out = ind.slope_pct(df, 10)
    assert out["A"].iloc[-1] == pytest.approx(0.01, rel=1e-6)


def test_indicators_handle_multiple_columns_independently():
    idx = pd.bdate_range("2026-01-05", periods=30)
    df = pd.DataFrame({"A": np.arange(30.0), "B": np.arange(30.0) * 2}, index=idx)
    out = ind.sma(df, 5)
    assert out["B"].iloc[-1] == pytest.approx(out["A"].iloc[-1] * 2)
