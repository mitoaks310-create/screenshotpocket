"""Tokyo Stock Exchange daily price limits (値幅制限).

A backtest that fills every order at the open quietly assumes those fills were
available.  On the TSE they often are not: a stock that gaps to its daily
limit-up (ストップ高) trades with buyers queued and sellers absent, and a retail
market order will not be filled at the open — if at all.

This matters more than it sounds, because the bias is concentrated exactly
where the screener is most confident.  Breakout and momentum names are the ones
that gap; treating an unfillable limit-up open as a clean entry hands the
backtest its best trades for free.

The limit width is a step function of the previous close (the 基準値段),
published by JPX.  The same table drives the lower limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: (upper bound of base price, limit width in yen).  A base price strictly
#: below the bound takes that width; the final entry is the open-ended top.
#: Source: JPX 内国株の売買制度 / 制限値幅.
PRICE_LIMIT_TABLE: tuple[tuple[float, float], ...] = (
    (100, 30),
    (200, 50),
    (500, 80),
    (700, 100),
    (1_000, 150),
    (1_500, 300),
    (2_000, 400),
    (3_000, 500),
    (5_000, 700),
    (7_000, 1_000),
    (10_000, 1_500),
    (15_000, 3_000),
    (20_000, 4_000),
    (30_000, 5_000),
    (50_000, 7_000),
    (70_000, 10_000),
    (100_000, 15_000),
    (150_000, 30_000),
    (200_000, 40_000),
    (300_000, 50_000),
    (500_000, 70_000),
    (700_000, 100_000),
    (1_000_000, 150_000),
    (1_500_000, 300_000),
    (2_000_000, 400_000),
    (3_000_000, 500_000),
    (5_000_000, 700_000),
    (7_000_000, 1_000_000),
    (10_000_000, 1_500_000),
    (15_000_000, 3_000_000),
    (20_000_000, 4_000_000),
    (30_000_000, 5_000_000),
    (50_000_000, 7_000_000),
    (float("inf"), 10_000_000),
)

_BOUNDS = np.array([b for b, _ in PRICE_LIMIT_TABLE], dtype="float64")
_WIDTHS = np.array([w for _, w in PRICE_LIMIT_TABLE], dtype="float64")


def limit_width(base_price: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Daily limit width in yen for each base price.

    JPX widens the limit for stocks that hit it repeatedly, so this is the
    ordinary-case width — a floor on how far a stock may move, not a ceiling.
    Using the base width keeps the model conservative in the right direction:
    it flags *at least* the bars that were genuinely locked.
    """
    values = np.asarray(
        base_price.to_numpy() if isinstance(base_price, (pd.DataFrame, pd.Series)) else base_price,
        dtype="float64",
    )
    # The table reads "100円未満 -> 30円", so a base price of exactly 100 belongs
    # to the *next* band. side="right" picks the first bound strictly greater
    # than the price, which is what "未満" means.
    idx = np.searchsorted(_BOUNDS, values, side="right")
    idx = np.clip(idx, 0, len(_WIDTHS) - 1)
    out = _WIDTHS[idx]
    return np.where(np.isfinite(values), out, np.nan)


def limit_prices(prev_close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Upper and lower limit price for each (date, code), from the prior close."""
    width = limit_width(prev_close)
    upper = pd.DataFrame(
        prev_close.to_numpy(dtype="float64") + width,
        index=prev_close.index,
        columns=prev_close.columns,
    )
    lower = pd.DataFrame(
        prev_close.to_numpy(dtype="float64") - width,
        index=prev_close.index,
        columns=prev_close.columns,
    )
    return upper, lower


def locked_limit_up(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Bars a buyer could not realistically get filled on.

    The strict case is 寄らずのストップ高 — the bar never trades below the limit,
    so ``low >= upper``.  A bar that merely *touches* the limit intraday still
    offered a fill at the open, so it is not flagged.
    """
    tol = 1e-9
    return np.isfinite(upper) & (low >= upper - tol) & (open_ >= upper - tol)


def locked_limit_down(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, lower: np.ndarray
) -> np.ndarray:
    """Bars a holder could not sell into: the bar never trades above the limit."""
    tol = 1e-9
    return np.isfinite(lower) & (high <= lower + tol) & (open_ <= lower + tol)
