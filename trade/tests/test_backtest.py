from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from screener.backtest import (
    decile_table,
    fit_final_model,
    make_folds,
    simulate_portfolio,
    spearman_monotonicity,
    walk_forward,
)
from screener.config import BacktestConfig, DEFAULT_CONFIG


def test_folds_leave_an_embargo_between_training_and_testing():
    """Without the gap, trades opened at the end of training are still running
    inside the test window, and the result is not out-of-sample at all."""
    dates = pd.bdate_range("2020-01-01", periods=1200)
    folds = make_folds(dates, DEFAULT_CONFIG)
    assert folds
    for _, _, train_end, test_start, _ in folds:
        gap = np.busday_count(train_end.date(), test_start.date())
        assert gap >= DEFAULT_CONFIG.backtest.embargo_bars


def test_folds_never_test_before_they_train():
    dates = pd.bdate_range("2020-01-01", periods=1200)
    for _, train_start, train_end, test_start, test_end in make_folds(dates, DEFAULT_CONFIG):
        assert train_start < train_end < test_start < test_end


def test_folds_use_expanding_training_windows():
    dates = pd.bdate_range("2020-01-01", periods=1200)
    folds = make_folds(dates, DEFAULT_CONFIG)
    ends = [f[2] for f in folds]
    assert ends == sorted(ends)
    assert len({f[1] for f in folds}) == 1  # all share one start


def test_too_little_history_yields_no_folds():
    assert make_folds(pd.bdate_range("2026-01-01", periods=50), DEFAULT_CONFIG) == []


# ------------------------------------------------------------ decile table


def _oos(scores, rs, dates=None):
    n = len(scores)
    dates = dates if dates is not None else pd.bdate_range("2026-01-05", periods=n)
    return pd.DataFrame(
        {
            "fold": 0,
            "date": dates,
            "code": [f"{1000+i}" for i in range(n)],
            "score": scores,
            "ev_r": np.zeros(n),
            "bucket": 0,
            "r_multiple": rs,
            "entry_date": dates,
            "exit_date": dates,
            "exit_reason": "time",
        }
    )


def test_decile_table_splits_into_equal_buckets():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=5000)
    table = decile_table(_oos(scores, rng.normal(size=5000)), 10)
    assert len(table) == 10
    assert table["n"].sum() == 5000
    # Percentile-rank bucketing lands within a couple of rows of even, not exact.
    assert table["n"].max() - table["n"].min() <= 5


def test_monotonicity_is_one_when_score_perfectly_orders_outcomes():
    scores = np.linspace(-3, 3, 4000)
    table = decile_table(_oos(scores, scores.copy()), 10)
    assert spearman_monotonicity(table) == pytest.approx(1.0)


def test_monotonicity_is_negative_when_the_score_is_inverted():
    scores = np.linspace(-3, 3, 4000)
    table = decile_table(_oos(scores, -scores), 10)
    assert spearman_monotonicity(table) == pytest.approx(-1.0)


# -------------------------------------------------------------- portfolio


def test_portfolio_respects_the_max_open_position_limit():
    n = 400
    dates = np.repeat(pd.bdate_range("2026-01-05", periods=40), 10)
    oos = _oos(np.linspace(3, -3, n), np.zeros(n), dates=pd.DatetimeIndex(dates))
    oos["ev_r"] = 0.1
    # Every trade stays open for a long time, so the cap has to bind.
    oos["exit_date"] = oos["date"] + pd.Timedelta(days=200)
    cfg = dataclasses.replace(
        DEFAULT_CONFIG, backtest=BacktestConfig(max_open_positions=5, top_n_per_day=10)
    )
    curve, _ = simulate_portfolio(oos, cfg)
    assert curve["open_positions"].max() <= 5


def test_portfolio_skips_negative_ev_signals_when_asked():
    n = 60
    oos = _oos(np.linspace(-1, 1, n), np.full(n, 1.0))
    oos["ev_r"] = -0.05
    _curve, stats = simulate_portfolio(oos, DEFAULT_CONFIG, require_positive_ev=True)
    assert stats["trades"] == 0


def test_portfolio_compounds_winning_trades():
    n = 30
    dates = pd.bdate_range("2026-01-05", periods=n)
    oos = _oos(np.ones(n), np.full(n, 1.0), dates=dates)
    oos["ev_r"] = 0.2
    oos["exit_date"] = oos["date"] + pd.Timedelta(days=1)
    _curve, stats = simulate_portfolio(oos, DEFAULT_CONFIG)
    assert stats["trades"] > 0
    assert stats["total_return"] > 0
    assert stats["win_rate"] == pytest.approx(1.0)


def test_portfolio_never_takes_the_same_name_twice_at_once():
    dates = pd.DatetimeIndex(np.repeat(pd.bdate_range("2026-01-05", periods=10), 3))
    n = len(dates)
    oos = _oos(np.ones(n), np.zeros(n), dates=dates)
    oos["code"] = "SAME"
    oos["ev_r"] = 0.1
    oos["exit_date"] = oos["date"] + pd.Timedelta(days=60)
    curve, _ = simulate_portfolio(oos, DEFAULT_CONFIG)
    assert curve["open_positions"].max() == 1


# ------------------------------------------------------- final calibration


def test_final_model_calibrates_ev_on_out_of_sample_pairs():
    """In-sample calibration reports the expectancy the fit was chosen to
    reproduce; the live EV must come from the walk-forward pairs instead."""
    from screener.scoring import fit_model

    rng = np.random.default_rng(2)
    n = 4000
    dates = pd.DatetimeIndex(np.repeat(pd.bdate_range("2024-01-01", periods=80), 50))
    z = pd.DataFrame(
        {
            "date": dates,
            "code": [f"{1000 + i % 50}" for i in range(n)],
            "z_roc60": rng.normal(size=n),
        }
    )
    trades = z[["date", "code"]].copy()
    trades["r_multiple"] = rng.normal(size=n)

    # Out-of-sample pairs that say the edge is much weaker than in-sample.
    oos = pd.DataFrame({"score": rng.normal(size=n), "r_multiple": np.full(n, -0.05)})

    in_sample = fit_model(z, trades, DEFAULT_CONFIG)
    final = fit_final_model(z, trades, oos, DEFAULT_CONFIG)

    assert final.ev_map.global_ev == pytest.approx(-0.05)
    assert final.ev_map.to_dict() != in_sample.ev_map.to_dict()
    # Weights still come from the full history.
    assert final.weights == in_sample.weights


def test_walk_forward_on_empty_input_is_a_no_op():
    folds, oos = walk_forward(pd.DataFrame(), pd.DataFrame(), DEFAULT_CONFIG)
    assert folds == [] and oos.empty
