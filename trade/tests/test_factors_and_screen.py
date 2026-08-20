from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from screener.config import DEFAULT_CONFIG, Filters
from screener.factors import CONTEXT_COLUMNS, FACTOR_NAMES, compute_factor_panel
from screener.providers import SyntheticProvider
from screener.providers.base import normalise_panel
from screener.screen import LOT_SIZE, Account, market_regime, position_size

BENCH = DEFAULT_CONFIG.benchmark_code


def synthetic(codes, start="2020-01-01", end="2026-08-19"):
    return SyntheticProvider().fetch(list(codes), start=start, end=end)


def loose_config(**filter_kwargs):
    base = dict(
        min_turnover_yen=0.0, min_price=0.0, max_price=1e9,
        min_history_bars=260, min_atr_pct=0.0, max_atr_pct=1.0,
    )
    base.update(filter_kwargs)
    return dataclasses.replace(DEFAULT_CONFIG, filters=Filters(**base))


# ---------------------------------------------------------------- factors


def test_factor_panel_produces_every_declared_factor():
    codes = [BENCH] + [f"{7000+i}" for i in range(40)]
    fp = compute_factor_panel(synthetic(codes), config=loose_config())
    assert not fp.empty
    for name in FACTOR_NAMES:
        assert name in fp.columns
    for name in CONTEXT_COLUMNS:
        assert name in fp.columns


def test_benchmark_is_never_a_candidate():
    """It is an ETF used as an input; ranking it would be a data leak into the
    universe, and it is not in the JPX common-stock list anyway."""
    codes = [BENCH] + [f"{7000+i}" for i in range(40)]
    fp = compute_factor_panel(synthetic(codes), config=loose_config())
    assert BENCH not in set(fp["code"])


def test_turnover_filter_removes_illiquid_names():
    codes = [BENCH] + [f"{7000+i}" for i in range(60)]
    panel = synthetic(codes)
    permissive = compute_factor_panel(panel, config=loose_config())
    strict = compute_factor_panel(
        panel, config=loose_config(min_turnover_yen=5e9)
    )
    assert len(strict) < len(permissive)
    if not strict.empty:
        assert (strict["turnover_ma25"] >= 5e9).all()


def test_price_filters_are_enforced():
    codes = [BENCH] + [f"{7000+i}" for i in range(60)]
    fp = compute_factor_panel(
        synthetic(codes), config=loose_config(min_price=800.0, max_price=3000.0)
    )
    if not fp.empty:
        assert fp["close"].between(800.0, 3000.0).all()


def test_atr_percent_filter_is_enforced():
    codes = [BENCH] + [f"{7000+i}" for i in range(60)]
    fp = compute_factor_panel(
        synthetic(codes), config=loose_config(min_atr_pct=0.02, max_atr_pct=0.04)
    )
    if not fp.empty:
        assert fp["atr_pct"].between(0.02, 0.04).all()


def test_sector_relative_momentum_is_centred_within_a_sector():
    codes = [BENCH] + [f"{7000+i}" for i in range(40)]
    sectors = {f"{7000+i}": ("A" if i % 2 else "B") for i in range(40)}
    fp = compute_factor_panel(synthetic(codes), config=loose_config(), sectors=sectors)
    last = fp[fp["date"] == fp["date"].max()]
    for _, g in last.groupby("sector33"):
        if len(g) >= 3:
            # It is a deviation from the sector median, so the median is ~0.
            assert g["sector_rs_60"].median() == pytest.approx(0.0, abs=1e-6)


def test_sector_factor_is_absent_without_a_sector_map():
    codes = [BENCH] + [f"{7000+i}" for i in range(40)]
    fp = compute_factor_panel(synthetic(codes), config=loose_config(), sectors=None)
    assert fp["sector_rs_60"].isna().all()


def test_trend_stack_is_bounded_zero_to_four():
    codes = [BENCH] + [f"{7000+i}" for i in range(30)]
    fp = compute_factor_panel(synthetic(codes), config=loose_config())
    assert fp["trend_stack"].between(0, 4).all()


def test_empty_panel_returns_empty_factor_frame():
    assert compute_factor_panel(pd.DataFrame()).empty


# ---------------------------------------------------------- position size


def test_position_size_rounds_down_to_whole_lots():
    a = Account(equity_yen=1_000_000, risk_pct=0.01)  # 10,000 yen of risk
    out = position_size(1000.0, 37.0, a)
    # 10000/37 = 270.3 shares -> 200 after lot rounding
    assert out["shares"] == 200
    assert out["shares"] % LOT_SIZE == 0
    assert out["risk_yen"] == pytest.approx(200 * 37.0)


def test_position_size_never_exceeds_the_risk_budget():
    a = Account(equity_yen=5_000_000, risk_pct=0.005)
    out = position_size(2500.0, 61.0, a)
    assert out["risk_yen"] <= a.risk_yen + 1e-9


def test_position_size_is_capped_by_the_notional_limit():
    """A very tight stop must not translate into an oversized position."""
    a = Account(equity_yen=1_000_000, risk_pct=0.02, max_position_pct=0.2)
    out = position_size(100.0, 0.5, a)
    assert out["capped_by"] == "position_cap"
    assert out["cost_yen"] <= 1_000_000 * 0.2 + 1e-9


def test_position_size_reports_when_one_lot_already_exceeds_the_budget():
    a = Account(equity_yen=1_000_000, risk_pct=0.01)  # 10,000 yen
    out = position_size(20_000.0, 500.0, a)  # 1 lot = 50,000
    assert out["shares"] == 0
    assert out["capped_by"] == "min_lot_exceeds_risk"
    assert out["min_lot_risk_yen"] == pytest.approx(50_000.0)


def test_position_size_rejects_nonsense_inputs():
    a = Account()
    assert position_size(float("nan"), 10.0, a)["shares"] == 0
    assert position_size(100.0, 0.0, a)["shares"] == 0
    assert position_size(100.0, -5.0, a)["shares"] == 0


# --------------------------------------------------------------- regime


def test_market_regime_reads_the_benchmark_trend():
    codes = [BENCH] + [f"{7000+i}" for i in range(30)]
    panel = synthetic(codes)
    fp = compute_factor_panel(panel, config=loose_config())
    r = market_regime(panel, fp, loose_config())
    assert r["trend"] in {"up", "down"}
    assert r["above_sma75"] == (r["benchmark_close"] > r["benchmark_sma75"])
    assert 0.0 <= r["breadth_above_sma200"] <= 1.0


def test_market_regime_survives_a_missing_benchmark():
    codes = [f"{7000+i}" for i in range(30)]
    panel = synthetic(codes)
    fp = compute_factor_panel(panel, config=loose_config())
    r = market_regime(panel, fp, loose_config())
    assert "trend" not in r  # reported as unknown rather than guessed


# ------------------------------------------------------- panel hygiene


def test_normalise_panel_rejects_bars_where_high_low_do_not_bracket():
    bad = pd.DataFrame(
        {
            "code": ["A", "A"],
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "open": [100.0, 100.0],
            "high": [101.0, 99.0],   # second bar's high is below its open
            "low": [99.0, 98.0],
            "close": [100.0, 100.0],
            "volume": [1000.0, 1000.0],
        }
    )
    out = normalise_panel(bad)
    assert len(out) == 1


def test_normalise_panel_drops_zero_volume_placeholder_bars():
    df = pd.DataFrame(
        {
            "code": ["A", "A"],
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "open": [100.0, 100.0], "high": [101.0, 101.0],
            "low": [99.0, 99.0], "close": [100.0, 100.0],
            "volume": [1000.0, 0.0],
        }
    )
    assert len(normalise_panel(df)) == 1


def test_normalise_panel_deduplicates_keeping_the_latest():
    df = pd.DataFrame(
        {
            "code": ["A", "A"],
            "date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
            "open": [100.0, 100.0], "high": [110.0, 110.0],
            "low": [99.0, 99.0], "close": [100.0, 105.0],
            "volume": [1000.0, 1000.0],
        }
    )
    out = normalise_panel(df)
    assert len(out) == 1 and out["close"].iloc[0] == 105.0


def test_synthetic_provider_is_deterministic():
    a = synthetic(["7203", "6758"], end="2026-06-30")
    b = synthetic(["7203", "6758"], end="2026-06-30")
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_bars_are_internally_consistent():
    p = synthetic([f"{7000+i}" for i in range(20)])
    assert (p["high"] >= p[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (p["low"] <= p[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (p["close"] > 0).all() and (p["volume"] > 0).all()


def test_synthetic_benchmark_is_less_volatile_than_typical_stocks():
    """The benchmark stands in for an index tracker, not another single name."""
    p = synthetic([BENCH] + [f"{7000+i}" for i in range(25)])
    vols = {}
    for code, g in p.groupby("code"):
        vols[code] = float(np.log(g["close"]).diff().std())
    others = [v for c, v in vols.items() if c != BENCH]
    assert vols[BENCH] < np.median(others)
