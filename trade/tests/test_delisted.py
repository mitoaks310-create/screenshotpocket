from __future__ import annotations

import pandas as pd
import pytest

from screener.delisted import (
    classify_reason,
    load_delisted,
    point_in_time_universe,
    summarise,
)


def _delisted():
    return pd.DataFrame(
        {
            "code": ["1111", "2222", "3333"],
            "name": ["A", "B", "C"],
            "delist_date": pd.to_datetime(["2021-06-01", "2024-03-01", "2026-01-01"]),
            "reason": ["株式等売渡請求による取得", "上場維持基準への不適合", "申請による上場廃止"],
            "category": ["acquisition", "failure", "other"],
        }
    )


def _universe():
    return pd.DataFrame(
        {
            "code": ["7203", "6758"],
            "name": ["トヨタ", "ソニーＧ"],
            "market": ["prime", "prime"],
            "sector33": ["輸送用機器", "電気機器"],
            "sector17": ["自動車", "電機"],
            "size": ["core30", "core30"],
        }
    )


# ------------------------------------------------------------ classification


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("株式等売渡請求による取得", "acquisition"),
        ("ＭＢＯ（公開買付け、株式併合）", "acquisition"),
        ("他社による買収（公開買付け、株式併合）", "acquisition"),
        ("株式の併合", "acquisition"),
        ("上場維持基準への不適合", "failure"),
        ("民事再生手続き", "failure"),
        ("有価証券報告書等の虚偽記載", "failure"),
        ("申請による上場廃止", "other"),
    ],
)
def test_classify_reason(reason, expected):
    assert classify_reason(reason) == expected


def test_classify_reason_handles_missing_values():
    assert classify_reason(float("nan")) == "other"
    assert classify_reason(None) == "other"


def test_failure_patterns_win_over_acquisition_patterns():
    """A squeeze-out following a listing breach is still a failure."""
    assert classify_reason("上場維持基準への不適合による株式の併合") == "failure"


# ------------------------------------------------------ point-in-time universe


def test_point_in_time_universe_adds_names_still_listed_on_that_date():
    out = point_in_time_universe(_universe(), _delisted(), pd.Timestamp("2022-01-01"))
    codes = set(out["code"])
    assert {"7203", "6758"} <= codes
    assert "2222" in codes  # delisted 2024, so listed in 2022
    assert "3333" in codes  # delisted 2026
    assert "1111" not in codes  # already gone by 2022


def test_point_in_time_universe_at_a_late_date_adds_nothing():
    out = point_in_time_universe(_universe(), _delisted(), pd.Timestamp("2026-08-01"))
    assert set(out["code"]) == {"7203", "6758"}


def test_point_in_time_universe_preserves_the_original_columns():
    out = point_in_time_universe(_universe(), _delisted(), pd.Timestamp("2020-01-01"))
    assert list(out.columns) == list(_universe().columns)


def test_point_in_time_universe_without_delisting_data_is_a_no_op():
    out = point_in_time_universe(_universe(), pd.DataFrame(), pd.Timestamp("2020-01-01"))
    pd.testing.assert_frame_equal(out, _universe())


def test_point_in_time_universe_does_not_duplicate_existing_codes():
    delisted = _delisted()
    delisted.loc[0, "code"] = "7203"
    out = point_in_time_universe(_universe(), delisted, pd.Timestamp("2019-01-01"))
    assert out["code"].is_unique


# ---------------------------------------------------------------- summary


def test_summarise_reports_category_shares():
    s = summarise(_delisted(), "2020-01-01", "2027-01-01")
    assert s["total"] == 3
    assert s["by_category"]["acquisition"] == 1
    assert s["failure_share"] == pytest.approx(1 / 3)


def test_summarise_respects_the_window():
    s = summarise(_delisted(), "2023-01-01", "2025-01-01")
    assert s["total"] == 1


def test_summarise_of_an_empty_window():
    assert summarise(_delisted(), "2030-01-01", "2031-01-01") == {"total": 0}


def test_load_delisted_returns_an_empty_frame_when_absent(tmp_path):
    out = load_delisted(tmp_path / "nope.csv")
    assert out.empty
    assert "delist_date" in out.columns


# ------------------------------------------------- the bundled real dataset


def test_bundled_delisting_data_shows_japanese_delistings_are_mostly_m_and_a():
    """Documents the finding that reframes survivorship bias for JP equities.

    If this ever flips, the guidance in the README needs revisiting — so it is
    worth failing the build over.
    """
    df = load_delisted()
    if df.empty:
        pytest.skip("delisting data not scraped in this checkout")
    stats = summarise(df, "2019-08-01", "2026-08-20")
    assert stats["total"] > 400
    assert stats["acquisition_share"] > 0.85
    assert stats["failure_share"] < 0.10
