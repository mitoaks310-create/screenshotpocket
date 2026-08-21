"""Data quality checks — what makes free data trustworthy.

Each test injects one realistic corruption into a clean panel and asserts it
is caught. The clean-baseline test matters just as much: a check that fires on
good data gets switched off, and then catches nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.providers import SyntheticProvider
from screener.quality import check_panel, quarantine, reconcile, summarise

CODES = [f"{7000 + i}" for i in range(6)]


@pytest.fixture(scope="module")
def clean():
    return SyntheticProvider().fetch(CODES, start="2024-01-01", end="2026-06-30")


def issues_for(panel, check=None):
    out = check_panel(panel)
    return out if check is None else out[out["check"] == check]


# ------------------------------------------------------------- clean data


def test_clean_synthetic_data_raises_no_issues(clean):
    """The baseline must be silent or every other test is meaningless."""
    assert check_panel(clean).empty


def test_summarise_of_a_clean_panel(clean):
    stats = summarise(check_panel(clean), clean)
    assert stats["issues"] == 0
    assert stats["codes"] == len(CODES)
    assert stats["codes_with_errors"] == 0


def test_empty_panel_is_handled():
    assert check_panel(pd.DataFrame()).empty


# --------------------------------------------------- corporate actions


def test_unadjusted_split_is_detected(clean):
    bad = clean.copy()
    m = (bad["code"] == CODES[0]) & (bad["date"] >= "2025-06-02")
    for col in ("open", "high", "low", "close"):
        bad.loc[m, col] /= 3.0
    found = issues_for(bad, "unadjusted_split")
    assert len(found) == 1
    assert found["code"].iloc[0] == CODES[0]
    assert found["severity"].iloc[0] == "error"


def test_unadjusted_reverse_split_is_detected(clean):
    bad = clean.copy()
    m = (bad["code"] == CODES[1]) & (bad["date"] >= "2025-06-02")
    for col in ("open", "high", "low", "close"):
        bad.loc[m, col] *= 5.0
    found = issues_for(bad, "unadjusted_split")
    assert len(found) == 1
    assert found["code"].iloc[0] == CODES[1]


def test_hundredfold_unit_error_is_detected(clean):
    bad = clean.copy()
    idx = bad.index[(bad["code"] == CODES[2]) & (bad["date"] == "2025-11-04")]
    assert len(idx) == 1
    bad.loc[idx, ["open", "high", "low", "close"]] *= 100
    errors = check_panel(bad)
    errors = errors[errors["severity"] == "error"]
    assert CODES[2] in set(errors["code"])


def test_a_move_beyond_the_price_limit_is_an_error_even_without_a_round_ratio(clean):
    """The TSE limit makes this impossible, so it is a data fault by definition."""
    bad = clean.copy()
    idx = bad.index[(bad["code"] == CODES[3]) & (bad["date"] == "2025-07-01")]
    bad.loc[idx, ["open", "high", "low", "close"]] *= 7.3  # not a round split ratio
    found = issues_for(bad, "price_limit_violation")
    assert not found.empty
    assert "error" in set(found["severity"])


def test_a_close_exactly_at_the_limit_is_not_flagged():
    """ストップ安引け is ordinary; flagging it would make the check unusable."""
    dates = pd.bdate_range("2026-01-05", periods=6)
    close = [1000.0, 700.0, 700.0, 700.0, 700.0, 700.0]  # 1000 -> limit width 300
    panel = pd.DataFrame(
        {
            "code": "9999",
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 100_000.0,
        }
    )
    assert issues_for(panel, "price_limit_violation").empty


# ------------------------------------------------------------- coverage


def test_missing_bars_are_detected_against_the_trading_calendar(clean):
    bad = clean[
        ~((clean["code"] == CODES[0]) & clean["date"].between("2025-03-03", "2025-03-21"))
    ]
    found = issues_for(bad, "missing_bars")
    assert CODES[0] in set(found["code"])


def test_public_holidays_are_not_reported_as_missing(clean):
    """The reason a naive gap check is useless for Japanese equities."""
    assert issues_for(clean, "missing_bars").empty


def test_stale_code_is_detected(clean):
    bad = clean[~((clean["code"] == CODES[1]) & (clean["date"] > "2026-02-01"))]
    found = issues_for(bad, "stale")
    assert CODES[1] in set(found["code"])


def test_flat_forward_filled_run_is_detected(clean):
    bad = clean.copy()
    m = (bad["code"] == CODES[2]) & bad["date"].between("2025-09-01", "2025-09-20")
    price = float(bad.loc[m, "close"].iloc[0])
    for col in ("open", "high", "low", "close"):
        bad.loc[m, col] = price
    found = issues_for(bad, "flat_bar_run")
    assert CODES[2] in set(found["code"])


def test_a_single_flat_bar_is_not_flagged(clean):
    """One flat bar is an illiquid day, not a data fault."""
    bad = clean.copy()
    idx = bad.index[(bad["code"] == CODES[3]) & (bad["date"] == "2025-05-07")]
    price = float(bad.loc[idx, "close"].iloc[0])
    bad.loc[idx, ["open", "high", "low", "close"]] = price
    assert issues_for(bad, "flat_bar_run").empty


def test_duplicate_rows_are_detected(clean):
    bad = pd.concat([clean, clean[clean["code"] == CODES[0]].head(3)], ignore_index=True)
    found = issues_for(bad, "duplicate_dates")
    assert CODES[0] in set(found["code"])


def test_zero_volume_share_is_detected(clean):
    bad = clean.copy()
    m = bad["code"] == CODES[4]
    n = int(m.sum())
    zero_idx = bad.index[m][: n // 5]
    bad.loc[zero_idx, "volume"] = 0.0
    found = issues_for(bad, "zero_volume")
    assert CODES[4] in set(found["code"])


# ----------------------------------------------------------- quarantine


def test_quarantine_drops_only_codes_with_errors(clean):
    bad = clean.copy()
    m = (bad["code"] == CODES[0]) & (bad["date"] >= "2025-06-02")
    for col in ("open", "high", "low", "close"):
        bad.loc[m, col] /= 3.0
    issues = check_panel(bad)
    kept, dropped = quarantine(bad, issues)
    assert dropped == [CODES[0]]
    assert CODES[0] not in set(kept["code"])
    assert set(kept["code"]) == set(CODES) - {CODES[0]}


def test_quarantine_is_a_no_op_on_clean_data(clean):
    kept, dropped = quarantine(clean, check_panel(clean))
    assert dropped == []
    assert len(kept) == len(clean)


# --------------------------------------------------------- reconciliation


def test_reconcile_reports_no_disagreement_for_identical_sources(clean):
    out = reconcile(clean, clean)
    assert (out["disagree"] == 0).all()


def test_reconcile_detects_a_diverging_source(clean):
    other = clean.copy()
    m = other["code"] == CODES[0]
    other.loc[m, "close"] *= 1.05
    out = reconcile(clean, other)
    row = out[out["code"] == CODES[0]].iloc[0]
    assert row["disagree_pct"] > 0.9
    assert out[out["code"] == CODES[1]]["disagree_pct"].iloc[0] == 0.0


def test_reconcile_tolerates_rounding_differences(clean):
    other = clean.copy()
    other["close"] = other["close"] * 1.0001
    out = reconcile(clean, other, tolerance=0.005)
    assert (out["disagree"] == 0).all()


def test_reconcile_handles_empty_input(clean):
    assert reconcile(clean, pd.DataFrame()).empty
    assert reconcile(pd.DataFrame(), clean).empty
