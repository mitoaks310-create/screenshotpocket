"""Yahoo Finance provider (primary source for daily JP bars).

Requires network access to ``query*.finance.yahoo.com``.  Yahoo rate-limits
aggressively per IP, so requests are batched and retried with backoff.
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
    ) -> None:
        self.batch_size = batch_size
        self.pause = pause
        self.retries = retries

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
        for attempt in range(self.retries):
            try:
                raw = yf.download(
                    symbols,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
                if raw is not None and not raw.empty:
                    return raw
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
