"""Yahoo Finance provider — the free source with usable history.

Requires network access to ``query*.finance.yahoo.com``.  Yahoo rate-limits
aggressively per IP, so requests are batched and retried with backoff.

**Adjustment matters more here than anywhere else in the package.**  The
obvious call, ``auto_adjust=True``, back-adjusts prices for dividends as well
as splits.  That is right for measuring total return and wrong for technical
analysis: it shifts every historical price away from what actually traded, so a
"52-week high", a moving average, or a Bollinger band computed on that series
is not the level anyone was looking at.  With a ~2% dividend yield the drift
compounds to several percent over a multi-year window — comparable to the
edges being measured.

Yahoo's underlying OHLC is split-adjusted but *not* dividend-adjusted, so
``auto_adjust=False`` returns exactly what this system wants: a continuous
series across splits, at the prices that actually printed.

``repair=True`` asks yfinance to fix known Yahoo defects — splits Yahoo failed
to apply to history, and the 100x unit errors it occasionally serves for
non-US listings.  Both are silent corruptions that would otherwise reach the
indicators, and both are exactly what :mod:`screener.quality` would later have
to quarantine, so it is cheaper to repair them at the source.
"""

from __future__ import annotations

import time
from typing import Sequence

import pandas as pd

from .base import empty_panel, normalise_panel


class YFinanceProvider:
    name = "yfinance"

    def __init__(
        self,
        batch_size: int = 50,
        pause: float = 1.0,
        retries: int = 3,
        adjust_dividends: bool = False,
        repair: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.pause = pause
        self.retries = retries
        #: Leave this off for technical analysis; see the module docstring.
        self.adjust_dividends = adjust_dividends
        self.repair = repair

    def fetch(
        self,
        codes: Sequence[str],
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        codes = list(dict.fromkeys(codes))
        frames: list[pd.DataFrame] = []
        for i in range(0, len(codes), self.batch_size):
            batch = codes[i : i + self.batch_size]
            symbols = [f"{c}.T" for c in batch]
            raw = self._download_with_retry(yf, symbols, start, end)
            if raw is None or raw.empty:
                continue
            frames.append(self._unstack(raw, batch))
            if i + self.batch_size < len(codes):
                time.sleep(self.pause)
        if not frames:
            return empty_panel()
        return normalise_panel(pd.concat(frames, ignore_index=True))

    def _download_with_retry(self, yf, symbols, start, end):
        delay = self.pause
        supports_repair = True
        for attempt in range(self.retries):
            try:
                kwargs = dict(
                    start=start,
                    end=end,
                    interval="1d",
                    # Split-adjusted, dividend-unadjusted — see module docstring.
                    auto_adjust=self.adjust_dividends,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
                if self.repair and supports_repair:
                    kwargs["repair"] = True
                raw = yf.download(symbols, **kwargs)
                if raw is not None and not raw.empty:
                    return raw
            except TypeError:
                # Older yfinance builds lack `repair`; drop it and try again
                # rather than losing the fetch over an optional flag.
                if supports_repair:
                    supports_repair = False
                    continue
            except Exception:  # noqa: BLE001 - provider errors are retried, then skipped
                pass
            if attempt < self.retries - 1:
                time.sleep(delay)
                delay *= 2
        return None

    @staticmethod
    def _unstack(raw: pd.DataFrame, codes: Sequence[str]) -> pd.DataFrame:
        """Flatten yfinance's per-ticker column blocks into tidy rows."""
        rows: list[pd.DataFrame] = []
        for code in codes:
            symbol = f"{code}.T"
            if isinstance(raw.columns, pd.MultiIndex):
                if symbol not in raw.columns.get_level_values(0):
                    continue
                sub = raw[symbol]
            else:
                # Single-symbol download returns flat columns.
                sub = raw
            sub = sub.dropna(how="all")
            if sub.empty:
                continue
            frame = pd.DataFrame(
                {
                    "code": code,
                    "date": sub.index,
                    "open": sub.get("Open"),
                    "high": sub.get("High"),
                    "low": sub.get("Low"),
                    "close": sub.get("Close"),
                    "volume": sub.get("Volume"),
                }
            )
            rows.append(frame.reset_index(drop=True))
        if not rows:
            return empty_panel()
        return pd.concat(rows, ignore_index=True)
