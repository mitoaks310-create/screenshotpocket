"""Tokyo Stock Exchange trading calendar, built from free official sources.

Knowing which days *should* have a bar is what separates "this stock did not
trade" from "my data feed dropped a day".  Without it, a gap-detection check
either flags every public holiday or nothing at all.

Two sources, both free and authoritative:

* National holidays come from the Cabinet Office's published CSV, which covers
  1955 onward — far more history than JPX's own page, which only lists the
  current and next year.
* The exchange-specific closures on top of that (the year-end/new-year break,
  and the handful of one-off outages) are encoded here.

``validate_against_jpx`` cross-checks the derived calendar against JPX's own
published list for the years JPX still publishes, so a drift in either source
shows up as a test failure rather than as silently wrong gap counts.
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

CABINET_HOLIDAY_CSV = "https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv"
JPX_CALENDAR_URL = "https://www.jpx.co.jp/corporate/about-jpx/calendar/index.html"
HOLIDAY_CSV = DATA_DIR / "universe" / "jp_holidays.csv"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: The TSE is shut for the new-year break on top of the national holidays.
#: January 1 is already a national holiday; 2, 3 and December 31 are not.
_YEAR_END_CLOSURES = ((1, 2), (1, 3), (12, 31))

#: One-off full-day closures that no rule predicts.  2020-10-01 is the
#: arrowhead hardware failure, when the exchange did not trade at all.
EXCEPTIONAL_CLOSURES: frozenset[pd.Timestamp] = frozenset(
    {pd.Timestamp("2020-10-01")}
)


def refresh_holidays(dest: Path = HOLIDAY_CSV, url: str = CABINET_HOLIDAY_CSV) -> pd.DataFrame:
    """Download the Cabinet Office holiday list and cache it."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed gov URL
        raw = resp.read()
    # The file is Shift-JIS; fall back to UTF-8 in case that ever changes.
    for encoding in ("shift_jis", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError("could not decode the Cabinet Office holiday CSV")

    df.columns = ["date", "name"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def load_holidays(path: Path = HOLIDAY_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "name"])
    return pd.read_csv(path, parse_dates=["date"])


def closed_days(start, end, holidays: pd.DataFrame | None = None) -> set[pd.Timestamp]:
    """Every weekday in the range on which the TSE does not trade."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    holidays = load_holidays() if holidays is None else holidays

    closed: set[pd.Timestamp] = set()
    if not holidays.empty:
        in_range = holidays[(holidays["date"] >= start) & (holidays["date"] <= end)]
        closed.update(pd.Timestamp(d).normalize() for d in in_range["date"])

    for year in range(start.year, end.year + 1):
        for month, day in _YEAR_END_CLOSURES:
            try:
                closed.add(pd.Timestamp(year=year, month=month, day=day))
            except ValueError:  # pragma: no cover - fixed valid dates
                continue

    closed.update(EXCEPTIONAL_CLOSURES)
    return {d for d in closed if start <= d <= end}


def trading_days(start, end, holidays: pd.DataFrame | None = None) -> pd.DatetimeIndex:
    """Sessions the TSE was open, as a DatetimeIndex."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end < start:
        return pd.DatetimeIndex([])
    weekdays = pd.bdate_range(start, end)
    closed = closed_days(start, end, holidays)
    return pd.DatetimeIndex([d for d in weekdays if d not in closed])


def is_trading_day(day, holidays: pd.DataFrame | None = None) -> bool:
    day = pd.Timestamp(day).normalize()
    if day.weekday() >= 5:
        return False
    return day not in closed_days(day, day, holidays)


# --------------------------------------------------------------------------
# Cross-check against JPX's own published closure list
# --------------------------------------------------------------------------


def fetch_jpx_closures(url: str = JPX_CALENDAR_URL) -> pd.DataFrame:
    """JPX's published closure list — only the current and next year."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed JPX URL
        html = resp.read().decode("utf-8", errors="replace")

    frames = []
    for table in pd.read_html(io.StringIO(html)):
        cols = [str(c).replace(" ", "") for c in table.columns]
        if any("日付" in c for c in cols) and any("名称" in c for c in cols):
            table.columns = cols
            frames.append(table)
    if not frames:
        return pd.DataFrame(columns=["date", "name"])

    df = pd.concat(frames, ignore_index=True).rename(columns={"日付": "raw", "名称": "name"})
    # Dates arrive as "2026/01/01（木）" — drop the weekday in brackets.
    df["date"] = pd.to_datetime(
        df["raw"].astype(str).map(lambda s: re.sub(r"（.*?）", "", s).strip()),
        errors="coerce",
    )
    return df.dropna(subset=["date"])[["date", "name"]].reset_index(drop=True)


def validate_against_jpx(holidays: pd.DataFrame | None = None) -> dict:
    """Compare the derived calendar with JPX's list for the years it covers.

    Weekend entries in JPX's list are ignored: the derived calendar excludes
    weekends structurally, so they are not disagreements.
    """
    jpx = fetch_jpx_closures()
    if jpx.empty:
        return {"checked": 0, "missing": [], "extra": []}

    jpx_weekday = {
        pd.Timestamp(d).normalize()
        for d in jpx["date"]
        if pd.Timestamp(d).weekday() < 5
    }
    start, end = min(jpx_weekday), max(jpx_weekday)
    # Compare weekdays only on both sides. The derived set lists weekend
    # closures too, but those are inert — trading_days starts from a weekday
    # range — and counting them as disagreements would make this check noisy
    # enough to ignore.
    derived = {d for d in closed_days(start, end, holidays) if d.weekday() < 5}

    return {
        "checked": len(jpx_weekday),
        "range": (str(start.date()), str(end.date())),
        # In JPX's list but not derived: the derived calendar would wrongly
        # expect a bar on that day.
        "missing": sorted(str(d.date()) for d in jpx_weekday - derived),
        # Derived but not in JPX's list: the derived calendar would wrongly
        # excuse a genuinely missing bar.
        "extra": sorted(str(d.date()) for d in derived - jpx_weekday),
    }
