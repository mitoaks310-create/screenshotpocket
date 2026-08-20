"""Price-provider interface.

Every provider returns the same tidy panel so the rest of the system never
learns where the bars came from:

===========  =========================================================
``code``     4-character JPX issue code, e.g. ``"7203"``
``date``     timezone-naive ``datetime64[ns]``, normalised to midnight
``open``     float, split-adjusted
``high``     float, split-adjusted
``low``      float, split-adjusted
``close``    float, split-adjusted
``volume``   float, shares
===========  =========================================================
"""

from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd

PANEL_COLUMNS = ["code", "date", "open", "high", "low", "close", "volume"]


class PriceProvider(Protocol):
    """Fetches daily split-adjusted OHLCV bars for JPX issue codes."""

    name: str

    def fetch(
        self,
        codes: Sequence[str],
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a tidy panel with :data:`PANEL_COLUMNS`.

        Codes with no data are omitted rather than returned as empty frames.
        """
        ...


def empty_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": pd.Series(dtype="object"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )


def normalise_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a provider's raw output into the canonical panel shape."""
    if df.empty:
        return empty_panel()
    out = df.loc[:, PANEL_COLUMNS].copy()
    out["code"] = out["code"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    # A zero-volume bar is a non-trading placeholder some feeds emit; a bar
    # where the high/low do not bracket the open/close is corrupt.
    out = out[(out["volume"].fillna(0) > 0) & (out["close"] > 0)]
    bracket = (
        (out["high"] >= out[["open", "close", "low"]].max(axis=1) - 1e-9)
        & (out["low"] <= out[["open", "close", "high"]].min(axis=1) + 1e-9)
    )
    out = out[bracket]
    out = out.drop_duplicates(subset=["code", "date"], keep="last")
    return out.sort_values(["code", "date"]).reset_index(drop=True)
