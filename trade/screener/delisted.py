"""Delisted issues, scraped from JPX's public listing.

The bundled universe holds only *currently* listed companies, so a backtest
over several years silently drops every issue that left the market in between.
JPX publishes the delisting list as plain HTML with one page per year, which is
enough to reconstruct a point-in-time universe without a paid data plan.

A finding worth stating plainly, because it contradicts the usual advice:
**Japanese delistings are overwhelmingly M&A, not failure.**  Measured over
2019-2026, about 96% of delistings were takeovers, squeeze-outs, MBOs or group
reorganisations, and under 3% were genuine business failures.  The familiar
US-equity intuition — "you are missing the losers, so your backtest is
inflated" — does not transfer.  Here the missing names are mostly companies
that left at a takeover premium, and a long-only technical strategy that
excludes them is, if anything, understating its right tail.

Fixing this is still worth doing for correctness.  It is just not the first
thing to spend money on.
"""

from __future__ import annotations

import io
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

BASE = "https://www.jpx.co.jp"
CURRENT_PAGE = "/listing/stocks/delisted/index.html"
ARCHIVE_TEMPLATE = "/listing/stocks/delisted/archives-{n:02d}.html"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

DELISTED_CSV = DATA_DIR / "universe" / "jpx_delisted.csv"

#: Reason substrings that mark a delisting as a corporate action rather than a
#: failure.  Used only for reporting — nothing downstream branches on it.
ACQUISITION_PATTERNS = (
    "買収", "売渡請求", "ＭＢＯ", "MBO", "完全子会社", "合併",
    "株式移転", "公開買付", "株式交換", "株式の併合",
)
FAILURE_PATTERNS = (
    "不適合", "債務超過", "破産", "民事再生", "会社更生",
    "銀行取引停止", "業績基準", "内部管理", "虚偽記載",
)


def _get(path: str, timeout: int = 60) -> str:
    req = urllib.request.Request(BASE + path, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed JPX host
        return resp.read().decode("utf-8", errors="replace")


def _archive_paths(html: str) -> list[str]:
    """Read the year drop-down rather than guessing how many archives exist."""
    block = re.search(r'<select[^>]*class="backnumber".*?</select>', html, re.S)
    if not block:
        return []
    return re.findall(r'value="([^"]+)"', block.group(0))


def refresh_delisted(dest: Path = DELISTED_CSV, pause: float = 0.3) -> pd.DataFrame:
    """Scrape every available year and write a tidy CSV."""
    current = _get(CURRENT_PAGE)
    paths = list(dict.fromkeys([CURRENT_PAGE, *_archive_paths(current)]))

    frames: list[pd.DataFrame] = []
    for i, path in enumerate(paths):
        html = current if path == CURRENT_PAGE else _get(path)
        frames.extend(_tables_from(html))
        if i < len(paths) - 1:
            time.sleep(pause)

    if not frames:
        raise RuntimeError("no delisting tables found — JPX page layout may have changed")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={"上場廃止日": "delist_date", "銘柄名": "name",
                 "コード": "code", "市場区分": "market_name", "上場廃止理由": "reason"}
    )
    df["delist_date"] = pd.to_datetime(df["delist_date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.strip()
    df = (
        df.dropna(subset=["delist_date"])
        .loc[df["code"].str.fullmatch(r"[0-9A-Z]{4}", na=False)]
        .drop_duplicates(subset=["code", "delist_date"])
        .sort_values("delist_date")
        .reset_index(drop=True)
    )
    df["category"] = df["reason"].map(classify_reason)

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def _tables_from(html: str) -> list[pd.DataFrame]:
    out = []
    for table in pd.read_html(io.StringIO(html)):
        cols = [str(c).replace(" ", "") for c in table.columns]
        if any("上場廃止日" in c for c in cols) and any("コード" in c for c in cols):
            table.columns = cols
            out.append(table)
    return out


def classify_reason(reason: str | float) -> str:
    """``acquisition`` / ``failure`` / ``other``."""
    text = "" if not isinstance(reason, str) else reason
    if any(p in text for p in FAILURE_PATTERNS):
        return "failure"
    if any(p in text for p in ACQUISITION_PATTERNS):
        return "acquisition"
    return "other"


def load_delisted(path: Path = DELISTED_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["code", "name", "delist_date", "reason", "category"])
    return pd.read_csv(path, dtype={"code": str}, parse_dates=["delist_date"])


def point_in_time_universe(
    universe: pd.DataFrame,
    delisted: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """The universe as it stood on ``as_of``.

    Currently-listed names are assumed to have been listed then too — which is
    wrong for recent IPOs, but the ``min_history_bars`` filter already excludes
    anything without a year of bars, so an IPO cannot reach the ranking anyway.
    """
    as_of = pd.Timestamp(as_of)
    if delisted.empty:
        return universe
    still_listed = delisted[delisted["delist_date"] > as_of]
    extra = still_listed[["code", "name"]].copy()
    extra["market"] = "delisted"
    extra["sector33"] = pd.NA
    extra["sector17"] = pd.NA
    extra["size"] = "none"
    keep = [c for c in universe.columns if c in extra.columns]
    return pd.concat([universe, extra[keep]], ignore_index=True).drop_duplicates("code")


def summarise(delisted: pd.DataFrame, start=None, end=None) -> dict:
    """Counts by category over a window — what the docstring above reports."""
    df = delisted
    if start is not None:
        df = df[df["delist_date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["delist_date"] <= pd.Timestamp(end)]
    if df.empty:
        return {"total": 0}
    counts = df["category"].value_counts().to_dict()
    total = int(len(df))
    return {
        "total": total,
        "by_category": {k: int(v) for k, v in counts.items()},
        "failure_share": float(counts.get("failure", 0) / total),
        "acquisition_share": float(counts.get("acquisition", 0) / total),
    }
