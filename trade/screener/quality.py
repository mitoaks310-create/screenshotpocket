"""Data quality checks.

Free price data is usable — but only if you verify it, because the failure
modes are silent.  A missed stock split does not look like an error; it looks
like a 50% crash, and every indicator downstream cheerfully computes a value
from it.  A dropped week of bars does not raise; it just shortens a moving
average window.  These checks turn silent corruption into a reported number.

The strongest check here is specific to the Tokyo market.  The TSE enforces a
daily price limit, so a close-to-close move larger than that limit is not
merely unusual — it is **impossible**.  Any bar that shows one is a data error
rather than a market event, which makes it a far sharper detector of missed
corporate actions than any percentage threshold could be.

Severity:

``error``
    Near-certainly corrupt.  Exclude the issue until it is fixed.
``warning``
    Suspicious; worth looking at before trusting the name.
``info``
    Worth knowing, not worth acting on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import trading_days
from .limits import limit_width

ISSUE_COLUMNS = ["code", "date", "check", "severity", "detail", "value"]

#: Ratios a missed split or consolidation typically produces.  Both directions
#: are listed because reverse splits (株式併合) are common in Japan.
COMMON_SPLIT_RATIOS = (
    1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 100.0,
    1 / 1.5, 0.5, 0.4, 1 / 3, 0.25, 0.2, 0.1, 0.01,
)


def check_panel(
    panel: pd.DataFrame,
    limit_multiple: float = 2.0,
    split_tolerance: float = 0.02,
    max_missing_pct: float = 0.02,
    stale_days: int = 7,
    holidays: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run every check over a tidy OHLCV panel and return the issues found.

    ``limit_multiple`` sets how far past the statutory price limit a move must
    go before it is called an error rather than a warning.  JPX widens the
    limit for a stock that keeps hitting it, so a move of 1-2x the base width
    can be legitimate; beyond that it effectively cannot be.
    """
    if panel.empty:
        return pd.DataFrame(columns=ISSUE_COLUMNS)

    issues: list[pd.DataFrame] = []
    issues.append(_check_duplicates(panel))
    issues.append(_check_ohlc_consistency(panel))
    issues.append(_check_price_limit_violations(panel, limit_multiple, split_tolerance))
    issues.append(_check_flat_bars(panel))
    issues.append(_check_zero_volume(panel))
    issues.append(_check_missing_bars(panel, max_missing_pct, holidays))
    issues.append(_check_staleness(panel, stale_days, holidays))

    out = pd.concat([df for df in issues if not df.empty], ignore_index=True) if any(
        not df.empty for df in issues
    ) else pd.DataFrame(columns=ISSUE_COLUMNS)
    if out.empty:
        return pd.DataFrame(columns=ISSUE_COLUMNS)
    return out.loc[:, ISSUE_COLUMNS].sort_values(["severity", "code", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------- checks


def _issue(code, date, check, severity, detail, value) -> dict:
    return {
        "code": code,
        "date": date,
        "check": check,
        "severity": severity,
        "detail": detail,
        "value": value,
    }


def _check_duplicates(panel: pd.DataFrame) -> pd.DataFrame:
    dup = panel.duplicated(subset=["code", "date"], keep=False)
    if not dup.any():
        return pd.DataFrame(columns=ISSUE_COLUMNS)
    rows = [
        _issue(code, None, "duplicate_dates", "error",
               f"{n} duplicated (code, date) rows", float(n))
        for code, n in panel[dup].groupby("code").size().items()
    ]
    return pd.DataFrame(rows)


def _check_ohlc_consistency(panel: pd.DataFrame) -> pd.DataFrame:
    hi = panel[["open", "close", "low"]].max(axis=1)
    lo = panel[["open", "close", "high"]].min(axis=1)
    bad = (panel["high"] < hi - 1e-9) | (panel["low"] > lo + 1e-9)
    if not bad.any():
        return pd.DataFrame(columns=ISSUE_COLUMNS)
    rows = [
        _issue(code, None, "ohlc_inconsistent", "error",
               f"{n} bars where high/low do not bracket open/close", float(n))
        for code, n in panel[bad].groupby("code").size().items()
    ]
    return pd.DataFrame(rows)


def _check_price_limit_violations(
    panel: pd.DataFrame, limit_multiple: float, split_tolerance: float
) -> pd.DataFrame:
    """Close-to-close moves the exchange could not have permitted.

    This is the sharpest signal of a missed corporate action available for
    Japanese equities: the move is not just large, it is outside what the
    exchange's own rules allow, so it did not happen as recorded.
    """
    df = panel.sort_values(["code", "date"])
    prev = df.groupby("code")["close"].shift(1)
    move = (df["close"] - prev).abs()
    width = limit_width(prev.to_numpy())

    with np.errstate(invalid="ignore", divide="ignore"):
        excess = move / width
        ratio = df["close"] / prev

    # Closing exactly at the limit is ordinary (ストップ高／安引け), so the
    # threshold needs slack for that plus tick rounding in the feed. Without
    # it every limit-down close is reported as corrupt.
    flagged = np.isfinite(excess) & (excess > 1.001)
    if not flagged.any():
        return pd.DataFrame(columns=ISSUE_COLUMNS)

    sub = df[flagged].copy()
    sub["excess"] = excess[flagged]
    sub["ratio"] = ratio[flagged]

    rows = []
    for r in sub.itertuples():
        near = _nearest_split_ratio(r.ratio, split_tolerance)
        if near is not None:
            rows.append(
                _issue(r.code, r.date, "unadjusted_split", "error",
                       f"close moved x{r.ratio:.4f} — matches an unadjusted "
                       f"{_describe_ratio(near)} and breaches the daily price limit",
                       float(r.ratio))
            )
        elif r.excess > limit_multiple:
            rows.append(
                _issue(r.code, r.date, "price_limit_violation", "error",
                       f"close moved {r.excess:.1f}x the daily price limit — "
                       "impossible on the TSE, so the bar is wrong",
                       float(r.excess))
            )
        else:
            rows.append(
                _issue(r.code, r.date, "price_limit_violation", "warning",
                       f"close moved {r.excess:.1f}x the base price limit — "
                       "possible if JPX had widened the limit that day",
                       float(r.excess))
            )
    return pd.DataFrame(rows)


def _nearest_split_ratio(ratio: float, tolerance: float) -> float | None:
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    for candidate in COMMON_SPLIT_RATIOS:
        if abs(ratio / candidate - 1.0) <= tolerance:
            return candidate
    return None


def _describe_ratio(ratio: float) -> str:
    if ratio >= 1:
        return f"{ratio:g}:1 split"
    inverse = 1.0 / ratio
    return f"1:{inverse:g} consolidation"


def _check_flat_bars(panel: pd.DataFrame, min_run: int = 5) -> pd.DataFrame:
    """Runs of bars with no intraday range at all.

    One flat bar is an illiquid day.  A run of them usually means the feed
    forward-filled a price the stock never traded at.
    """
    flat = (
        (panel["open"] == panel["high"])
        & (panel["high"] == panel["low"])
        & (panel["low"] == panel["close"])
    )
    if not flat.any():
        return pd.DataFrame(columns=ISSUE_COLUMNS)

    df = panel.assign(_flat=flat).sort_values(["code", "date"])
    rows = []
    for code, g in df.groupby("code"):
        runs = (g["_flat"] != g["_flat"].shift()).cumsum()
        longest = g[g["_flat"]].groupby(runs).size()
        if longest.empty:
            continue
        n = int(longest.max())
        if n >= min_run:
            rows.append(
                _issue(code, None, "flat_bar_run", "warning",
                       f"{n} consecutive bars with zero intraday range — "
                       "likely forward-filled rather than traded", float(n))
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=ISSUE_COLUMNS)


def _check_zero_volume(panel: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    zero = panel["volume"].fillna(0) <= 0
    if not zero.any():
        return pd.DataFrame(columns=ISSUE_COLUMNS)
    rows = []
    for code, g in panel.assign(_z=zero).groupby("code"):
        share = float(g["_z"].mean())
        if share > threshold:
            rows.append(
                _issue(code, None, "zero_volume", "warning",
                       f"{share:.1%} of bars report no volume", share)
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=ISSUE_COLUMNS)


def _check_missing_bars(
    panel: pd.DataFrame, max_missing_pct: float, holidays: pd.DataFrame | None
) -> pd.DataFrame:
    """Sessions the exchange was open but the feed has no bar for.

    Compared against the real TSE calendar, so public holidays are not
    mistaken for gaps — the reason a naive gap check is useless here.
    """
    rows = []
    for code, g in panel.groupby("code"):
        first, last = g["date"].min(), g["date"].max()
        expected = trading_days(first, last, holidays)
        if len(expected) == 0:
            continue
        have = set(pd.DatetimeIndex(g["date"]).normalize())
        missing = [d for d in expected if d not in have]
        if not missing:
            continue
        share = len(missing) / len(expected)
        longest = _longest_run(missing)
        severity = "error" if share > max_missing_pct * 5 else (
            "warning" if share > max_missing_pct else "info"
        )
        rows.append(
            _issue(code, None, "missing_bars", severity,
                   f"{len(missing)} of {len(expected)} sessions missing "
                   f"({share:.1%}), longest gap {longest} sessions", share)
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=ISSUE_COLUMNS)


def _longest_run(days: list[pd.Timestamp]) -> int:
    if not days:
        return 0
    best = run = 1
    for a, b in zip(days, days[1:]):
        # Consecutive *sessions*, not consecutive calendar days.
        run = run + 1 if (b - a).days <= 4 else 1
        best = max(best, run)
    return best


def _check_staleness(
    panel: pd.DataFrame, stale_days: int, holidays: pd.DataFrame | None
) -> pd.DataFrame:
    """Codes whose last bar lags the rest of the panel."""
    panel_last = panel["date"].max()
    sessions = trading_days(panel_last - pd.Timedelta(days=60), panel_last, holidays)
    if len(sessions) == 0:
        return pd.DataFrame(columns=ISSUE_COLUMNS)

    rows = []
    for code, last in panel.groupby("code")["date"].max().items():
        behind = int((sessions > last).sum())
        if behind > stale_days:
            rows.append(
                _issue(code, last, "stale", "warning",
                       f"last bar is {behind} sessions behind the panel — "
                       "delisted, halted, or the fetch failed for this code",
                       float(behind))
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=ISSUE_COLUMNS)


# -------------------------------------------------------------- aggregation


def summarise(issues: pd.DataFrame, panel: pd.DataFrame | None = None) -> dict:
    """Headline counts, for the CLI and the dashboard."""
    out: dict = {
        "issues": int(len(issues)),
        "by_severity": {},
        "by_check": {},
        "codes_with_errors": 0,
    }
    if panel is not None and not panel.empty:
        out["codes"] = int(panel["code"].nunique())
        out["bars"] = int(len(panel))
        out["date_range"] = [str(panel["date"].min().date()), str(panel["date"].max().date())]
    if issues.empty:
        return out

    out["by_severity"] = {k: int(v) for k, v in issues["severity"].value_counts().items()}
    out["by_check"] = {k: int(v) for k, v in issues["check"].value_counts().items()}
    errors = issues[issues["severity"] == "error"]
    out["codes_with_errors"] = int(errors["code"].nunique())
    if panel is not None and not panel.empty and out.get("codes"):
        out["clean_share"] = 1.0 - out["codes_with_errors"] / out["codes"]
    return out


def quarantine(panel: pd.DataFrame, issues: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop every code that has an ``error``-level issue.

    A stock with a missed split is not slightly wrong — every moving average,
    ATR and 52-week high computed from it is wrong, and it will usually rank
    *well* because the artefact looks like a huge move.  Excluding it is the
    only safe response.
    """
    if issues.empty or panel.empty:
        return panel, []
    bad = sorted(set(issues.loc[issues["severity"] == "error", "code"].dropna()))
    if not bad:
        return panel, []
    return panel[~panel["code"].isin(bad)].reset_index(drop=True), bad


# ------------------------------------------------------------ reconciliation


def reconcile(
    a: pd.DataFrame, b: pd.DataFrame, tolerance: float = 0.005
) -> pd.DataFrame:
    """Compare two providers' bars for the same (code, date).

    Where two independent free sources agree, confidence is high.  Where they
    disagree by more than a rounding difference, at least one is wrong — and
    which one is not knowable from the data alone, so this reports rather than
    resolves.
    """
    if a.empty or b.empty:
        return pd.DataFrame(columns=["code", "bars_compared", "disagree", "disagree_pct", "max_diff"])

    merged = a.merge(b, on=["code", "date"], suffixes=("_a", "_b"))
    if merged.empty:
        return pd.DataFrame(columns=["code", "bars_compared", "disagree", "disagree_pct", "max_diff"])

    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (merged["close_a"] / merged["close_b"] - 1.0).abs()
    merged = merged.assign(_diff=diff, _bad=diff > tolerance)

    grouped = merged.groupby("code")
    out = grouped.agg(
        bars_compared=("_diff", "size"),
        disagree=("_bad", "sum"),
        max_diff=("_diff", "max"),
    ).reset_index()
    out["disagree_pct"] = out["disagree"] / out["bars_compared"]
    return out.sort_values("disagree_pct", ascending=False).reset_index(drop=True)
