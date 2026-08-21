"""Overlap-corrected IC statistics and the multiple-testing gate.

These guard the correction that changed which factors get weight: with a
15-bar holding period, consecutive days' ICs score overlapping outcomes, so
treating them as independent overstates every t-statistic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from screener.config import DEFAULT_CONFIG, ScoringConfig
from screener.scoring import (
    _autocorr1,
    _newey_west_t,
    factor_ic,
    normal_ppf,
    significance_threshold,
    weights_from_ic,
)


# ------------------------------------------------------------ Newey-West


def test_newey_west_matches_the_naive_t_on_independent_data():
    rng = np.random.default_rng(0)
    x = rng.normal(0.1, 1.0, size=4000)
    naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    nw = _newey_west_t(x, 15)
    # No autocorrelation to correct for, so the two should be close.
    assert nw == pytest.approx(naive, rel=0.25)


def test_newey_west_shrinks_the_t_stat_on_autocorrelated_data():
    """The whole point: overlapping observations are not independent."""
    rng = np.random.default_rng(1)
    n, phi = 4000, 0.8
    e = rng.normal(0, 1, size=n)
    x = np.empty(n)
    x[0] = 0.2
    for i in range(1, n):
        x[i] = 0.2 * (1 - phi) + phi * x[i - 1] + e[i] * 0.3
    naive = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    nw = _newey_west_t(x, 15)
    assert abs(nw) < abs(naive) * 0.6


def test_newey_west_is_zero_for_degenerate_input():
    assert _newey_west_t(np.array([1.0]), 5) == 0.0
    assert _newey_west_t(np.zeros(50), 5) == 0.0


def test_autocorr1_recovers_a_known_ar1_coefficient():
    rng = np.random.default_rng(2)
    n, phi = 20000, 0.7
    x = np.empty(n)
    x[0] = 0.0
    e = rng.normal(0, 1, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    assert _autocorr1(x) == pytest.approx(phi, abs=0.05)


# --------------------------------------------------------- thresholds


def test_normal_ppf_matches_known_quantiles():
    assert normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert normal_ppf(0.995) == pytest.approx(2.575829, abs=1e-4)
    assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-6)


def test_normal_ppf_rejects_values_outside_the_unit_interval():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            normal_ppf(bad)


def test_significance_threshold_rises_with_the_number_of_factors_tested():
    one = significance_threshold(1)
    many = significance_threshold(16)
    assert one == pytest.approx(1.96, abs=0.01)
    assert many > one
    assert many == pytest.approx(2.96, abs=0.02)


# --------------------------------------------------------------- gating


def _ic_table(rows):
    return pd.DataFrame(
        rows, columns=["factor", "ic_mean", "ic_t"]
    ).assign(ic_std=0.1, ic_t_naive=0.0, ic_autocorr1=0.0, n_dates=500, n_trades=10000)


def test_insignificant_factors_get_no_weight_even_with_a_positive_ic():
    """A positive IC that fails the t-test is exactly what the gate is for."""
    table = _ic_table([("strong", 0.05, 6.0), ("lucky", 0.04, 1.5)])
    w = weights_from_ic(table, DEFAULT_CONFIG)
    assert w["lucky"] == 0.0
    assert w["strong"] == pytest.approx(1.0)


def test_gate_can_be_disabled():
    table = _ic_table([("strong", 0.05, 6.0), ("lucky", 0.04, 1.5)])
    config = dataclasses.replace(
        DEFAULT_CONFIG, scoring=ScoringConfig(require_significance=False)
    )
    w = weights_from_ic(table, config)
    assert w["lucky"] > 0.0


def test_explicit_threshold_overrides_the_bonferroni_default():
    table = _ic_table([("a", 0.05, 6.0), ("b", 0.04, 2.2)])
    config = dataclasses.replace(DEFAULT_CONFIG, scoring=ScoringConfig(min_ic_t=2.0))
    w = weights_from_ic(table, config)
    assert w["b"] > 0.0


def test_negative_t_stats_never_earn_weight():
    table = _ic_table([("good", 0.05, 6.0), ("inverted", 0.05, -6.0)])
    w = weights_from_ic(table, DEFAULT_CONFIG)
    assert w["inverted"] == 0.0


def test_weights_still_sum_to_one_after_gating():
    table = _ic_table([("a", 0.05, 6.0), ("b", 0.03, 5.0), ("c", 0.04, 0.5)])
    w = weights_from_ic(table, DEFAULT_CONFIG)
    assert sum(w.values()) == pytest.approx(1.0)


# ------------------------------------------------------ end-to-end shape


def test_factor_ic_reports_both_t_statistics_and_the_autocorrelation():
    rng = np.random.default_rng(3)
    dates = pd.DatetimeIndex(np.repeat(pd.bdate_range("2024-01-01", periods=120), 40))
    n = len(dates)
    z = pd.DataFrame(
        {
            "date": dates,
            "code": [f"{1000 + i % 40}" for i in range(n)],
            "z_roc60": rng.normal(size=n),
        }
    )
    trades = z[["date", "code"]].copy()
    trades["r_multiple"] = rng.normal(size=n)

    table = factor_ic(z, trades, factor_names=["roc60"], nw_lags=15)
    assert set(table.columns) >= {"ic_mean", "ic_t", "ic_t_naive", "ic_autocorr1"}
    assert len(table) == 1
    assert np.isfinite(table["ic_t"].iloc[0])


def test_factor_ic_lag_choice_changes_the_corrected_t_but_not_the_naive_one():
    rng = np.random.default_rng(4)
    dates = pd.DatetimeIndex(np.repeat(pd.bdate_range("2024-01-01", periods=200), 40))
    n = len(dates)
    z = pd.DataFrame(
        {
            "date": dates,
            "code": [f"{1000 + i % 40}" for i in range(n)],
            "z_roc60": rng.normal(size=n),
        }
    )
    trades = z[["date", "code"]].copy()
    trades["r_multiple"] = z["z_roc60"] * 0.3 + rng.normal(size=n)

    a = factor_ic(z, trades, factor_names=["roc60"], nw_lags=1)
    b = factor_ic(z, trades, factor_names=["roc60"], nw_lags=30)
    assert a["ic_t_naive"].iloc[0] == pytest.approx(b["ic_t_naive"].iloc[0])
    assert a["ic_t"].iloc[0] != pytest.approx(b["ic_t"].iloc[0])
