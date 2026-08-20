"""Stooq CSV provider (fallback source).

Stooq serves free daily OHLCV for JP issues at ``/q/d/l/?s=<code>.jp&i=d``.
It is one request per symbol and it sometimes answers with a JavaScript
browser challenge instead of CSV; such responses are detected and skipped so a
partial fetch never silently poisons the cache with HTML.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from typing import Sequence

import pandas as pd

from .base import empty_panel, normalise_panel

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


class StooqProvider:
    name = "stooq"

    def __init__(self, pause: float = 0.3, retries: int = 2, timeout: int = 30) -> None:
        self.pause = pause
        self.retries = retries
        self.timeout = timeout

    def fetch(
        self,
        codes: Sequence[str],
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for code in dict.fromkeys(codes):
            text = self._get(f"{code}.jp")
            if text is None:
                continue
            frame = self._parse(text, code)
            if frame is not None:
                frames.append(frame)
            time.sleep(self.pause)
        if not frames:
            return empty_panel()
        panel = normalise_panel(pd.concat(frames, ignore_index=True))
        if start is not None:
            panel = panel[panel["date"] >= pd.Timestamp(start)]
        if end is not None:
            panel = panel[panel["date"] <= pd.Timestamp(end)]
        return panel.reset_index(drop=True)

    def _get(self, symbol: str) -> str | None:
        url = STOOQ_URL.format(symbol=symbol)
        delay = self.pause
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": _UA})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                    body = resp.read().decode("utf-8", errors="replace")
                # Stooq answers a bot challenge with HTML rather than an error.
                if body.lstrip().lower().startswith(("<!doctype", "<html")):
                    return None
                if not body.startswith("Date,"):
                    return None
                return body
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt < self.retries - 1:
                    time.sleep(delay)
                    delay *= 2
        return None

    @staticmethod
    def _parse(text: str, code: str) -> pd.DataFrame | None:
        df = pd.read_csv(io.StringIO(text))
        expected = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not expected.issubset(df.columns):
            return None
        return pd.DataFrame(
            {
                "code": code,
                "date": df["Date"],
                "open": df["Open"],
                "high": df["High"],
                "low": df["Low"],
                "close": df["Close"],
                "volume": df["Volume"],
            }
        )
