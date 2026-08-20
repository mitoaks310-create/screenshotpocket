from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.config import DEFAULT_CONFIG
from screener.factors import FactorSpec, directional_value
from screener.scoring import (
    EVMap,
    composite_score,
    fit_ev_map,
    weights_from_ic,
    zscore_factors,
)


# ------------------------------------------------------------- directions


def test_higher_direction_passes_values_through():
    spec = FactorSpec("f", "c", "higher", "l", "r")
    s = pd.Series([1.0, 2.0, 3.0])
    pd.testing.assert_series_equal(directional_value(s, spec), s)


def test_lower_direction_flips_the_sign():
    spec = FactorSpec("f", "c", "lower", "l", "r")
    s = pd.Series([1.0, 2.0, 3.0])
    assert list(directional_value(s, spec)) == [-1.0, -2.0, -3.0]


def test_band_direction_scores_everything_inside_the_band_equally():
    spec = FactorSpec("f", "c", "band", "l", "r", band=(40.0, 60.0))
    s = pd.Series([40.0, 50.0, 60.0])
    out = directional_value(s, spec)
    assert list(out) == pytest.approx([0.0, 0.0, 0.0])


def test_band_direction_penalises_distance_outside_the_band():
    spec = FactorSpec("f", "c", "band", "l", "r", band=(40.0, 60.0))
    s = pd.Series([30.0, 70.0])
    out = directional_value(s, spec)
    # Ten units outside a twenty-wide band is -0.5 on either side.
    assert list(out) == pytest.approx([-0.5, -0.5])


def test_band_direction_is_symmetric():
    spec = FactorSpec("f", "c", "band", "l", "r", band=(0.0, 1.5))
    below = directional_value(pd.Series([-1.5]), spec).iloc[0]
    above = directional_value(pd.Series([3.0]), spec).iloc[0]
    assert below == pytest.approx(above)


# ---------------------------------------------------------------- z-scores


def make_factor_panel(n_codes=60, n_dates=3, seed=0):
    from screener.factors import FACTOR_NAMES

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=n_dates)
    rows = []
    for d in dates:
        for i in range(n_codes):
            row = {"date": d, "code": f"{1000+i}"}
            for name in FACTOR_NAMES:
                row[name] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


def test_zscores_are_standardised_within_each_date():
    fp = make_factor_panel()
    z = zscore_factors(fp)
    for _, g in z.groupby("date"):
        col = g["z_roc60"]
        assert col.mean() == pytest.approx(0.0, abs=1e-5)
        assert col.std(ddof=1) == pytest.approx(1.0, abs=0.05)


def test_zscores_are_clipped_to_four_sigma():
    fp = make_factor_panel()
    fp.loc[0, "roc60"] = 1e9
    z = zscore_factors(fp)
    assert z["z_roc60"].abs().max() <= 4.0 + 1e-6


def test_thin_cross_sections_are_dropped():
    fp = make_factor_panel(n_codes=5, n_dates=2)
    assert zscore_factors(fp).empty


# ---------------------------------------------------------------- weights


def test_weights_drop_factors_below_the_noise_floor():
    ic = pd.DataFrame(
        {"factor": ["a", "b", "c"], "ic_mean": [0.05, 0.001, -0.03]}
    )
    w = weights_from_ic(ic, DEFAULT_CONFIG)
    assert w["b"] == 0.0 and w["c"] == 0.0
    assert w["a"] == pytest.approx(1.0)


def test_weights_are_normalised_to_one():
    ic = pd.DataFrame({"factor": ["a", "b"], "ic_mean": [0.05, 0.015]})
    w = weights_from_ic(ic, DEFAULT_CONFIG)
    assert sum(w.values()) == pytest.approx(1.0)


def test_all_noise_falls_back_to_equal_weights_not_zeros():
    ic = pd.DataFrame({"factor": ["a", "b"], "ic_mean": [-0.1, -0.2]})
    w = weights_from_ic(ic, DEFAULT_CONFIG)
    assert w == {"a": 0.5, "b": 0.5}


# ------------------------------------------------------------- composite


def test_composite_ignores_missing_factors_instead_of_treating_them_as_zero():
    z = pd.DataFrame(
        {"date": [pd.Timestamp("2026-01-05")] * 2, "code": ["A", "B"],
         "z_roc60": [2.0, 2.0], "z_adx14": [np.nan, 2.0]}
    )
    score = composite_score(z, {"roc60": 0.5, "adx14": 0.5})
    # B has both factors at 2.0; A has one, and should also score 2.0 rather
    # than being dragged to 1.0 by an absent factor.
    assert score.iloc[0] == pytest.approx(2.0)
    assert score.iloc[1] == pytest.approx(2.0)


def test_composite_is_nan_when_no_weighted_factor_is_present():
    z = pd.DataFrame(
        {"date": [pd.Timestamp("2026-01-05")], "code": ["A"], "z_roc60": [np.nan]}
    )
    assert np.isnan(composite_score(z, {"roc60": 1.0}).iloc[0])


# ----------------------------------------------------------------- EV map


def test_ev_map_is_monotone_when_score_predicts_outcome():
    rng = np.random.default_rng(1)
    scores = pd.Series(rng.normal(size=6000))
    r = pd.Series(scores * 0.4 + rng.normal(size=6000) * 0.5)
    ev = fit_ev_map(scores, r, DEFAULT_CONFIG)
    assert ev.ev == sorted(ev.ev)


def test_ev_map_prediction_interpolates_and_clamps():
    ev = EVMap(centers=[-1.0, 0.0, 1.0], ev=[-0.2, 0.0, 0.2],
               win_rate=[0.3, 0.4, 0.5], counts=[10, 10, 10],
               edges=[-np.inf, -0.5, 0.5, np.inf], global_ev=0.0)
    out = ev.predict(pd.Series([-5.0, -0.5, 0.5, 5.0]))
    assert out[0] == pytest.approx(-0.2)  # clamped at the low end
    assert out[1] == pytest.approx(-0.1)  # interpolated
    assert out[2] == pytest.approx(0.1)
    assert out[3] == pytest.approx(0.2)  # clamped at the high end


def test_ev_map_shrinks_thin_buckets_toward_the_global_mean():
    # One extreme bucket with few observations should not assert a wild EV.
    scores = pd.Series(list(np.linspace(-1, 1, 400)))
    r = pd.Series([0.0] * 399 + [50.0])
    ev = fit_ev_map(scores, r, DEFAULT_CONFIG)
    assert max(ev.ev) < 1.0


def test_ev_map_round_trips_through_dict():
    ev = EVMap([0.0], [0.1], [0.5], [100], [-np.inf, np.inf], 0.05)
    again = EVMap.from_dict(ev.to_dict())
    assert again.ev == ev.ev and again.global_ev == ev.global_ev


def test_bucket_of_assigns_indices_within_range():
    ev = EVMap(centers=[-1.0, 0.0, 1.0], ev=[-0.2, 0.0, 0.2],
               win_rate=[0.3, 0.4, 0.5], counts=[10, 10, 10],
               edges=[-np.inf, -0.5, 0.5, np.inf], global_ev=0.0)
    b = ev.bucket_of(pd.Series([-9.0, 0.0, 9.0]))
    assert list(b) == [0, 1, 2]
