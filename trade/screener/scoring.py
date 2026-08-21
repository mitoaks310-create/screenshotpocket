"""Turning factors into an expected-value estimate.

Three stages, deliberately kept separate so each can be inspected on its own:

1. **Normalise.**  Each factor is mapped so that higher is better, winsorised,
   and z-scored *within each date's cross-section*.  Cross-sectional
   normalisation is what makes the score answer "which of today's stocks is
   best" rather than "is today a good day", and it makes the score immune to
   market-wide drift in the raw factor levels.

2. **Weight.**  Weights come from each factor's historical information
   coefficient — the rank correlation between the factor and the realised R
   multiple, measured per date and averaged.  Weights are shrunk and floored at
   zero, so a factor whose edge is indistinguishable from noise contributes
   nothing instead of adding variance.  A factor is never allowed to flip sign;
   the direction declared in :mod:`screener.factors` is a prior that a single
   training window is not permitted to overturn.

3. **Calibrate.**  The composite score is mapped to an expected value in R by
   bucketing training trades by score and measuring what each bucket actually
   produced.  This is the step that makes the output a number with units
   ("+0.18R per trade") rather than an uninterpretable ranking score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .factors import FACTOR_NAMES, SPEC_BY_NAME, directional_value

IC_COLUMNS = [
    "factor",
    "ic_mean",
    "ic_std",
    "ic_t_naive",
    "ic_t",
    "ic_autocorr1",
    "n_dates",
    "n_trades",
]


def zscore_factors(
    factor_panel: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    factor_names: Iterable[str] = FACTOR_NAMES,
) -> pd.DataFrame:
    """Winsorised, direction-adjusted cross-sectional z-scores.

    Returns ``date``, ``code`` and one ``z_<factor>`` column per factor.  Dates
    with too few names to form a meaningful cross-section are dropped.
    """
    factor_names = list(factor_names)
    if factor_panel.empty:
        return pd.DataFrame(columns=["date", "code", *(f"z_{n}" for n in factor_names)])

    df = factor_panel
    counts = df.groupby("date")["code"].transform("size")
    df = df[counts >= config.scoring.min_cross_section]
    if df.empty:
        return pd.DataFrame(columns=["date", "code", *(f"z_{n}" for n in factor_names)])

    directional = pd.DataFrame(
        {
            name: directional_value(df[name].astype("float64"), SPEC_BY_NAME[name])
            for name in factor_names
        },
        index=df.index,
    )
    directional["date"] = df["date"].to_numpy()

    grouped = directional.groupby("date", sort=False)
    p = config.scoring.winsor_pct
    lo = grouped[factor_names].quantile(p).reindex(directional["date"]).to_numpy()
    hi = grouped[factor_names].quantile(1.0 - p).reindex(directional["date"]).to_numpy()
    clipped = pd.DataFrame(
        np.clip(directional[factor_names].to_numpy(), lo, hi),
        index=directional.index,
        columns=factor_names,
    )
    clipped["date"] = directional["date"].to_numpy()

    cg = clipped.groupby("date", sort=False)[factor_names]
    mean = cg.transform("mean")
    std = cg.transform("std").replace(0.0, np.nan)
    z = (clipped[factor_names] - mean) / std

    out = pd.DataFrame({"date": df["date"].to_numpy(), "code": df["code"].to_numpy()})
    for name in factor_names:
        # A z-score is only meaningful within roughly +/-4 sigma; clipping keeps
        # one freak value from dominating a weighted sum.
        out[f"z_{name}"] = z[name].clip(-4.0, 4.0).to_numpy().astype("float32")
    return out.reset_index(drop=True)


def composite_score(
    z_panel: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Weighted mean of available z-scores.

    Dividing by the weight of the factors that are actually present — rather
    than by the total weight — stops a stock with one missing factor from being
    penalised as though that factor had scored zero.
    """
    names = [n for n in weights if f"z_{n}" in z_panel.columns and weights[n] > 0]
    if not names:
        return pd.Series(np.nan, index=z_panel.index, dtype="float64")
    z = z_panel[[f"z_{n}" for n in names]].to_numpy(dtype="float64")
    w = np.array([weights[n] for n in names], dtype="float64")
    present = np.isfinite(z)
    weighted = np.nansum(np.where(present, z, 0.0) * w, axis=1)
    denom = (present * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(denom > 0, weighted / denom, np.nan)
    return pd.Series(score, index=z_panel.index, dtype="float64")


# --------------------------------------------------------------------------
# Weight fitting
# --------------------------------------------------------------------------


def factor_ic(
    z_panel: pd.DataFrame,
    trades: pd.DataFrame,
    factor_names: Iterable[str] = FACTOR_NAMES,
    min_names_per_date: int = 20,
    nw_lags: int | None = None,
) -> pd.DataFrame:
    """Per-factor information coefficient statistics.

    IC is computed as a Spearman correlation *within each date* and then
    averaged across dates.  Pooling all dates into one correlation instead
    would let a few unusually wide cross-sections dominate, and would confound
    cross-sectional skill with the market's own time-series swings.
    """
    # Only score factors the panel actually carries, so a partial z-panel
    # (a subset of factors, or a factor that was never computable) narrows the
    # table instead of raising.
    factor_names = [n for n in factor_names if f"z_{n}" in z_panel.columns]
    if not factor_names:
        return pd.DataFrame(
            columns=IC_COLUMNS
        )
    merged = z_panel.merge(
        trades[["date", "code", "r_multiple"]], on=["date", "code"], how="inner"
    )
    if merged.empty:
        return pd.DataFrame(
            columns=IC_COLUMNS
        )

    xcols = [f"z_{n}" for n in factor_names]
    # Require every factor present so one Pearson-on-ranks formula covers the
    # whole block; the eligibility filter already demands more history than any
    # factor needs, so this discards very little.
    merged = merged.dropna(subset=xcols + ["r_multiple"])
    sizes = merged.groupby("date")["code"].transform("size")
    merged = merged[sizes >= min_names_per_date]
    if merged.empty:
        return pd.DataFrame(
            columns=IC_COLUMNS
        )

    # Rank within each date, then correlate: Pearson on ranks is Spearman.
    ranked = merged.groupby("date", sort=True)[xcols + ["r_multiple"]].rank()
    ranked["date"] = merged["date"].to_numpy()

    grouped = ranked.groupby("date", sort=True)
    cx = ranked[xcols] - grouped[xcols].transform("mean")
    cy = ranked["r_multiple"] - grouped["r_multiple"].transform("mean")
    date_key = ranked["date"]

    num = cx.mul(cy, axis=0).groupby(date_key).sum()
    sxx = cx.pow(2).groupby(date_key).sum()
    syy = cy.pow(2).groupby(date_key).sum()
    den = np.sqrt(sxx.mul(syy, axis=0))
    ic = (num / den.replace(0.0, np.nan)).dropna(how="all")

    n_dates = ic.notna().sum()
    mean = ic.mean()
    std = ic.std(ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_naive = mean / (std / np.sqrt(n_dates.replace(0, np.nan)))

    # Consecutive days' ICs are NOT independent: with a holding period of H
    # bars, two ICs one day apart score outcomes that share (H-1)/H of their
    # window.  Measured autocorrelation of the daily IC series runs 0.6-0.8,
    # and treating the days as independent overstates t by roughly 2-3x.
    # Newey-West with H lags corrects the standard error for that overlap.
    lags = nw_lags if nw_lags is not None else 15
    t_nw = {c: _newey_west_t(ic[c].dropna().to_numpy(), lags) for c in xcols}
    ac1 = {c: _autocorr1(ic[c].dropna().to_numpy()) for c in xcols}

    return pd.DataFrame(
        {
            "factor": factor_names,
            "ic_mean": [float(mean.get(c, 0.0) or 0.0) for c in xcols],
            "ic_std": [float(std.get(c, 0.0) or 0.0) for c in xcols],
            "ic_t_naive": [float(np.nan_to_num(t_naive.get(c, 0.0))) for c in xcols],
            # ``ic_t`` stays the name the rest of the system reads, but it now
            # carries the overlap-corrected statistic.
            "ic_t": [float(np.nan_to_num(t_nw[c])) for c in xcols],
            "ic_autocorr1": [float(np.nan_to_num(ac1[c])) for c in xcols],
            "n_dates": [int(n_dates.get(c, 0)) for c in xcols],
            "n_trades": int(len(merged)),
        }
    )


def _newey_west_t(x: np.ndarray, lags: int) -> float:
    """t-statistic for the mean of a serially correlated series.

    Uses Bartlett weights, so the long-run variance stays positive
    semi-definite however strong the autocorrelation is.
    """
    n = len(x)
    if n < 3:
        return 0.0
    mean = float(x.mean())
    e = x - mean
    var = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        var += 2.0 * weight * float(e[lag:] @ e[:-lag]) / n
    if var <= 0:
        return 0.0
    return mean / np.sqrt(var / n)


def _autocorr1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation — the diagnostic that motivates the correction."""
    if len(x) < 3:
        return 0.0
    a, b = x[:-1], x[1:]
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = np.sqrt((a_c @ a_c) * (b_c @ b_c))
    return float(a_c @ b_c / denom) if denom > 0 else 0.0


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF by bisection on ``erfc``.

    A dependency on scipy for one quantile is not worth it, and bisection over
    40 iterations is exact to well past the precision this threshold needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        cdf = 0.5 * math.erfc(-mid / math.sqrt(2.0))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def significance_threshold(n_tested: int, alpha: float = 0.05) -> float:
    """Two-sided |t| a factor must clear, Bonferroni-corrected.

    Sixteen factors are screened against the same outcomes, so the largest of
    sixteen noise draws clears an uncorrected 5% bar routinely.  Correcting for
    the family keeps the weighting from being handed to whichever factor got
    lucky in this particular sample.
    """
    n_tested = max(1, int(n_tested))
    return abs(normal_ppf(alpha / (2.0 * n_tested)))


def weights_from_ic(ic_table: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> dict[str, float]:
    """Shrunk, non-negative, normalised weights, gated on significance.

    Two gates, and both matter:

    *Magnitude* — the IC has to beat a shrinkage floor, so a factor whose edge
    rounds to nothing contributes nothing rather than adding variance.

    *Significance* — the overlap-corrected t-statistic has to clear a
    Bonferroni threshold for the number of factors screened.  Without this,
    whichever of sixteen factors got luckiest in the training window collects
    weight on the strength of that luck.  This gate is why a factor can show a
    positive IC and still be dropped.
    """
    if ic_table.empty:
        return {name: 1.0 / len(FACTOR_NAMES) for name in FACTOR_NAMES}

    raw = (ic_table["ic_mean"] - config.scoring.ic_shrink).clip(lower=0.0)

    if config.scoring.require_significance and "ic_t" in ic_table.columns:
        threshold = (
            config.scoring.min_ic_t
            if config.scoring.min_ic_t is not None
            else significance_threshold(len(ic_table), config.scoring.alpha)
        )
        raw = raw.where(ic_table["ic_t"].fillna(0.0) >= threshold, 0.0)

    total = float(raw.sum())
    if total <= 0:
        # Nothing cleared the bar.  Equal weights are a poor model, but an
        # all-zero one scores every stock identically and hides the failure;
        # the backtest's monotonicity check is what surfaces it.
        return {name: 1.0 / len(ic_table) for name in ic_table["factor"]}
    return {
        str(row.factor): float(w / total)
        for row, w in zip(ic_table.itertuples(), raw)
    }


# --------------------------------------------------------------------------
# Score -> expected value calibration
# --------------------------------------------------------------------------


@dataclass
class EVMap:
    """Monotone-ish lookup from composite score to expected R."""

    centers: list[float]
    ev: list[float]
    win_rate: list[float]
    counts: list[int]
    edges: list[float]
    global_ev: float

    def predict(self, scores: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype="float64")
        if not self.centers:
            return np.full(arr.shape, self.global_ev)
        out = np.interp(arr, self.centers, self.ev, left=self.ev[0], right=self.ev[-1])
        return np.where(np.isfinite(arr), out, np.nan)

    def bucket_of(self, scores: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype="float64")
        return np.clip(
            np.searchsorted(np.asarray(self.edges[1:-1]), arr, side="right"),
            0,
            max(0, len(self.centers) - 1),
        )

    def to_dict(self) -> dict:
        return {
            "centers": self.centers,
            "ev": self.ev,
            "win_rate": self.win_rate,
            "counts": self.counts,
            "edges": self.edges,
            "global_ev": self.global_ev,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EVMap":
        return cls(
            centers=list(d["centers"]),
            ev=list(d["ev"]),
            win_rate=list(d["win_rate"]),
            counts=[int(c) for c in d["counts"]],
            edges=list(d["edges"]),
            global_ev=float(d["global_ev"]),
        )


def fit_ev_map(
    scores: pd.Series,
    r_multiples: pd.Series,
    config: Config = DEFAULT_CONFIG,
) -> EVMap:
    """Bucket training trades by score and measure what each bucket produced."""
    df = pd.DataFrame({"score": scores.to_numpy(), "r": r_multiples.to_numpy()}).dropna()
    global_ev = float(df["r"].mean()) if not df.empty else 0.0
    n_buckets = config.scoring.ev_buckets
    if len(df) < n_buckets * 10:
        return EVMap([], [], [], [], [], global_ev)

    edges = np.unique(np.quantile(df["score"], np.linspace(0.0, 1.0, n_buckets + 1)))
    if len(edges) < 3:
        return EVMap([], [], [], [], [], global_ev)
    edges[0], edges[-1] = -np.inf, np.inf

    bucket = pd.cut(df["score"], bins=edges, labels=False, include_lowest=True)
    grouped = df.groupby(bucket)
    prior = config.scoring.min_bucket_trades

    centers, evs, wins, counts = [], [], [], []
    for _, g in grouped:
        n = len(g)
        raw_ev = float(g["r"].mean())
        # Shrink thin buckets toward the overall mean: a bucket holding 30
        # trades has no business asserting a wildly different expectancy.
        shrunk = (n * raw_ev + prior * global_ev) / (n + prior)
        centers.append(float(g["score"].mean()))
        evs.append(shrunk)
        wins.append(float((g["r"] > 0).mean()))
        counts.append(int(n))

    order = np.argsort(centers)
    finite_edges = [float(e) for e in edges]
    return EVMap(
        centers=[centers[i] for i in order],
        ev=[evs[i] for i in order],
        win_rate=[wins[i] for i in order],
        counts=[counts[i] for i in order],
        edges=finite_edges,
        global_ev=global_ev,
    )


@dataclass
class ScoringModel:
    """Everything needed to score a fresh cross-section."""

    weights: dict[str, float]
    ev_map: EVMap
    ic_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    trained_from: str | None = None
    trained_to: str | None = None
    n_train_trades: int = 0

    def score(self, z_panel: pd.DataFrame) -> pd.DataFrame:
        out = z_panel[["date", "code"]].copy()
        out["score"] = composite_score(z_panel, self.weights).to_numpy()
        out["ev_r"] = self.ev_map.predict(out["score"])
        return out

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "ev_map": self.ev_map.to_dict(),
            "ic_table": self.ic_table.to_dict("records") if not self.ic_table.empty else [],
            "trained_from": self.trained_from,
            "trained_to": self.trained_to,
            "n_train_trades": self.n_train_trades,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScoringModel":
        return cls(
            weights={str(k): float(v) for k, v in d["weights"].items()},
            ev_map=EVMap.from_dict(d["ev_map"]),
            ic_table=pd.DataFrame(d.get("ic_table") or []),
            trained_from=d.get("trained_from"),
            trained_to=d.get("trained_to"),
            n_train_trades=int(d.get("n_train_trades", 0)),
        )


def fit_model(
    z_panel: pd.DataFrame,
    trades: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> ScoringModel:
    """Fit weights and the EV calibration on one training window."""
    # The overlap that inflates the naive t-statistic is exactly the holding
    # period, so that is the lag count the standard error has to span.
    lags = config.scoring.nw_lags
    if lags is None:
        lags = config.trade.max_hold_bars
    ic_table = factor_ic(z_panel, trades, nw_lags=lags)
    weights = weights_from_ic(ic_table, config)

    merged = z_panel.merge(
        trades[["date", "code", "r_multiple"]], on=["date", "code"], how="inner"
    )
    scores = composite_score(merged, weights)
    ev_map = fit_ev_map(scores, merged["r_multiple"], config)

    return ScoringModel(
        weights=weights,
        ev_map=ev_map,
        ic_table=ic_table,
        trained_from=str(merged["date"].min().date()) if not merged.empty else None,
        trained_to=str(merged["date"].max().date()) if not merged.empty else None,
        n_train_trades=int(len(merged)),
    )
