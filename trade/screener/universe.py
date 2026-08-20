"""JPX listed-issue universe.

The screener needs a stable list of tradable Japanese equities together with
sector and size metadata.  JPX publishes the authoritative list as an Excel
file; :func:`refresh_universe` downloads and normalises it, and the result is
committed to ``trade/data/universe/jpx_universe.csv`` so the package works
without network access.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import UNIVERSE_CSV

JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)

#: Only these JPX market segments contain ordinary domestic common stock.
DOMESTIC_SEGMENTS = {
    "プライム（内国株式）": "prime",
    "スタンダード（内国株式）": "standard",
    "グロース（内国株式）": "growth",
}

#: TOPIX size buckets, normalised to short slugs.
SIZE_SLUGS = {
    "TOPIX Core30": "core30",
    "TOPIX Large70": "large70",
    "TOPIX Mid400": "mid400",
    "TOPIX Small 1": "small1",
    "TOPIX Small 2": "small2",
}

COLUMNS = ["code", "name", "market", "sector33", "sector17", "size"]


@dataclass(frozen=True)
class Stock:
    """One listed issue."""

    code: str
    name: str
    market: str
    sector33: str
    sector17: str
    size: str

    @property
    def yahoo_symbol(self) -> str:
        return f"{self.code}.T"

    @property
    def stooq_symbol(self) -> str:
        return f"{self.code}.jp"


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(
        columns={
            "コード": "code",
            "銘柄名": "name",
            "市場・商品区分": "market",
            "33業種区分": "sector33",
            "17業種区分": "sector17",
            "規模区分": "size",
        }
    )
    df = df[df["market"].isin(DOMESTIC_SEGMENTS)].copy()
    df["market"] = df["market"].map(DOMESTIC_SEGMENTS)
    df["size"] = df["size"].map(SIZE_SLUGS).fillna("none")
    df["code"] = df["code"].astype(str).str.strip()
    for col in ("name", "sector33", "sector17"):
        df[col] = df[col].astype(str).str.strip()
    # Codes containing letters exist (e.g. some ETNs) but domestic common stock
    # is always 4 digits; guard anyway so downstream symbol building is safe.
    df = df[df["code"].str.fullmatch(r"[0-9A-Z]{4}")]
    return df[COLUMNS].sort_values("code").reset_index(drop=True)


def refresh_universe(dest: Path = UNIVERSE_CSV, url: str = JPX_LIST_URL) -> pd.DataFrame:
    """Download the current JPX issue list and rewrite the bundled CSV."""
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed JPX URL
        payload = resp.read()
    raw = pd.read_excel(io.BytesIO(payload))
    df = _normalise(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def load_universe(
    path: Path = UNIVERSE_CSV,
    markets: tuple[str, ...] | None = None,
    sizes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load the bundled universe, optionally filtered by market and size."""
    if not path.exists():
        raise FileNotFoundError(
            f"universe file not found: {path}. Run `python -m screener.cli universe --refresh`."
        )
    df = pd.read_csv(path, dtype={"code": str})
    if markets:
        df = df[df["market"].isin(markets)]
    if sizes:
        df = df[df["size"].isin(sizes)]
    return df.reset_index(drop=True)


def to_stocks(df: pd.DataFrame) -> list[Stock]:
    return [Stock(**row) for row in df[COLUMNS].to_dict("records")]
