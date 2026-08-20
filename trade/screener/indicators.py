"""Technical indicators, vectorised over a whole cross-section at once.

Every function takes and returns a *wide* frame indexed by date with one
column per issue code.  Computing 3,700 stocks column-wise in a handful of
pandas calls is roughly two orders of magnitude faster than looping per stock,
which is what makes a full-universe walk-forward backtest practical.

Smoothing follows Wilder's convention (``alpha = 1/n``) wherever the classic
definition of the indicator calls for it — ATR, ADX and RSI — so values line
up with what charting packages display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def sma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).mean()


def ema(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(span=n, adjust=False, min_periods=n).mean()


def roc(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rate of change over ``n`` bars, as a fraction."""
    return df / df.shift(n) - 1.0


def slope_pct(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Average per-bar fractional change of a series over ``n`` bars.

    Used on moving averages, where the *direction* of the average matters more
    than its level.
    """
    return (df / df.shift(n)) ** (1.0 / n) - 1.0


def true_range(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame
) -> pd.DataFrame:
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    # fmax rather than maximum: on the first bar there is no previous close, so
    # b and c are NaN and the true range is simply the bar's own span.
    values = np.fmax(np.fmax(a.to_numpy(), b.to_numpy()), c.to_numpy())
    return pd.DataFrame(values, index=a.index, columns=a.columns)


def atr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14
) -> pd.DataFrame:
    return _wilder(true_range(high, low, close), n)


def adx(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14
) -> pd.DataFrame:
    """Average Directional Index — trend *strength*, direction-agnostic."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr_n = _wilder(true_range(high, low, close), n)
    plus_di = 100.0 * _wilder(plus_dm, n) / tr_n
    minus_di = 100.0 * _wilder(minus_dm, n) / tr_n

    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return _wilder(dx, n)


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder(gain, n)
    avg_loss = _wilder(loss, n)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # A window with no losses at all is RSI 100 by definition, not NaN.
    return out.where(avg_loss.ne(0.0) | avg_gain.isna(), 100.0)


def bollinger(
    close: pd.DataFrame, n: int = 25, k: float = 2.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(percent_b, bandwidth, middle)``.

    ``percent_b`` is 0 at the lower band and 1 at the upper band; ``bandwidth``
    is the band span divided by the middle band, which is the standard measure
    of volatility compression.
    """
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    span = (upper - lower).replace(0.0, np.nan)
    percent_b = (close - lower) / span
    bandwidth = span / mid
    return percent_b, bandwidth, mid


def stochastic(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    n: int = 14,
    d: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    span = (hh - ll).replace(0.0, np.nan)
    k = 100.0 * (close - ll) / span
    return k, k.rolling(d, min_periods=d).mean()


def obv(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """On-balance volume: cumulative signed volume."""
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0.0).cumsum()


def rolling_max(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=max(2, n // 2)).max()


def rolling_min(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=max(2, n // 2)).min()


def rolling_percentile(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Percentile rank of the latest value within its own trailing window.

    Answers "is this the tightest this stock's bands have been in months?",
    which a raw level cannot, because band width is not comparable across
    stocks of different volatility.
    """
    return df.rolling(n, min_periods=max(20, n // 4)).rank(pct=True)


def zscore(df: pd.DataFrame, n: int) -> pd.DataFrame:
    mean = df.rolling(n, min_periods=max(20, n // 4)).mean()
    sd = df.rolling(n, min_periods=max(20, n // 4)).std(ddof=0)
    return (df - mean) / sd.replace(0.0, np.nan)


def realised_vol(close: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Annualised close-to-close volatility."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(n, min_periods=n).std(ddof=0) * np.sqrt(245.0)
