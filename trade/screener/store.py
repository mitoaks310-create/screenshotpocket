"""Local OHLCV cache.

The whole panel lives in one Parquet file.  A per-code layout would make
incremental writes marginally cheaper, but every stage after ``update`` reads
the *entire* panel to compute cross-sectional ranks, and one columnar file is
far faster for that than several thousand small ones.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CACHE_DIR
from .providers.base import PANEL_COLUMNS, empty_panel, normalise_panel


class PriceStore:
    def __init__(self, cache_dir: Path = CACHE_DIR, filename: str = "ohlcv.parquet") -> None:
        self.cache_dir = Path(cache_dir)
        self.path = self.cache_dir / filename
        self._fallback = self.cache_dir / filename.replace(".parquet", ".csv.gz")

    # ------------------------------------------------------------------ read

    def load(self, codes: list[str] | None = None) -> pd.DataFrame:
        if self.path.exists():
            panel = pd.read_parquet(self.path)
        elif self._fallback.exists():
            panel = pd.read_csv(self._fallback, dtype={"code": str}, parse_dates=["date"])
        else:
            return empty_panel()
        panel["code"] = panel["code"].astype(str)
        if codes is not None:
            panel = panel[panel["code"].isin(set(codes))]
        return panel.sort_values(["code", "date"]).reset_index(drop=True)

    def last_date(self, code: str | None = None) -> pd.Timestamp | None:
        panel = self.load([code] if code else None)
        if panel.empty:
            return None
        return pd.Timestamp(panel["date"].max())

    def coverage(self) -> pd.DataFrame:
        """Per-code first/last bar and bar count — used by ``status``."""
        panel = self.load()
        if panel.empty:
            return pd.DataFrame(columns=["code", "first", "last", "bars"])
        agg = panel.groupby("code")["date"].agg(["min", "max", "count"])
        agg.columns = ["first", "last", "bars"]
        return agg.reset_index()

    # ----------------------------------------------------------------- write

    def save(self, panel: pd.DataFrame) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        panel = panel.loc[:, PANEL_COLUMNS]
        try:
            panel.to_parquet(self.path, index=False)
            return self.path
        except (ImportError, ValueError):
            # No pyarrow/fastparquet available — keep working via gzip CSV.
            panel.to_csv(self._fallback, index=False)
            return self._fallback

    def upsert(self, incoming: pd.DataFrame) -> pd.DataFrame:
        """Merge new bars over existing ones, newest wins on collision."""
        incoming = normalise_panel(incoming)
        if incoming.empty:
            return self.load()
        existing = self.load()
        merged = (
            pd.concat([existing, incoming], ignore_index=True)
            .drop_duplicates(subset=["code", "date"], keep="last")
            .sort_values(["code", "date"])
            .reset_index(drop=True)
        )
        self.save(merged)
        return merged

    def clear(self) -> None:
        for p in (self.path, self._fallback):
            if p.exists():
                p.unlink()


def to_wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivot one OHLCV field to a ``date x code`` matrix."""
    return panel.pivot(index="date", columns="code", values=field).sort_index()
