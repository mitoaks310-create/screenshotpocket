"""Deterministic synthetic market generator.

This provider exists so the whole pipeline — factors, trade simulation,
walk-forward calibration, dashboard export — can be exercised and unit-tested
without network access, and so the dashboard has something to render on a
machine that has never run ``update``.

The generator reproduces the *statistical texture* real screeners have to cope
with: a common market factor with regime switching, sector co-movement, fat
tails, GARCH-style volatility clustering, slow-moving idiosyncratic drift, and
volume that spikes with absolute returns.  It does **not** encode the strategy
the screener looks for, so out-of-sample statistics measured on synthetic bars
say nothing about real-world edge — they only prove the plumbing is correct.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import normalise_panel

# Regime transition matrix over (bull, chop, bear); rows sum to 1.  The strong
# diagonal produces multi-month regimes rather than day-to-day flicker.
_REGIME_P = np.array(
    [
        [0.985, 0.012, 0.003],
        [0.020, 0.965, 0.015],
        [0.010, 0.030, 0.960],
    ]
)
_REGIME_DRIFT = np.array([0.0007, 0.0000, -0.0011])
_REGIME_VOL = np.array([0.0085, 0.0105, 0.0180])


class SyntheticProvider:
    name = "synthetic"

    def __init__(
        self,
        seed: int = 20260820,
        n_sectors: int = 12,
        benchmark_code: str = "1306",
    ) -> None:
        self.seed = seed
        self.n_sectors = n_sectors
        self.benchmark_code = benchmark_code

    def fetch(
        self,
        codes: Sequence[str],
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        codes = list(dict.fromkeys(codes))
        end = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp("2026-08-19")
        start = (
            pd.Timestamp(start).normalize()
            if start is not None
            else end - pd.Timedelta(days=int(365 * 6))
        )
        dates = pd.bdate_range(start, end)
        if len(dates) == 0 or not codes:
            return normalise_panel(pd.DataFrame())

        n = len(dates)
        market, sectors = self._common_factors(n)

        frames = [
            self._one_stock(code, dates, market, sectors[hash_sector(code, self.n_sectors)])
            for code in codes
        ]
        return normalise_panel(pd.concat(frames, ignore_index=True))

    def _common_factors(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        regime = np.empty(n, dtype=int)
        regime[0] = 0
        for t in range(1, n):
            regime[t] = rng.choice(3, p=_REGIME_P[regime[t - 1]])
        market = _REGIME_DRIFT[regime] + _REGIME_VOL[regime] * rng.standard_normal(n)
        # Sector factors: mean-reverting rotation on top of the market.
        sectors = np.empty((self.n_sectors, n))
        for s in range(self.n_sectors):
            shock = rng.standard_normal(n) * 0.006
            level = np.zeros(n)
            for t in range(1, n):
                level[t] = 0.97 * level[t - 1] + shock[t]
            sectors[s] = level - np.roll(level, 1)
            sectors[s][0] = 0.0
        return market, sectors

    def _one_stock(
        self,
        code: str,
        dates: pd.DatetimeIndex,
        market: np.ndarray,
        sector: np.ndarray,
    ) -> pd.DataFrame:
        n = len(dates)
        rng = np.random.default_rng(self.seed + hash_code(code))

        is_benchmark = code == self.benchmark_code
        if is_benchmark:
            # The benchmark is an index tracker: it *is* the market factor plus
            # a little tracking noise, not another idiosyncratic name.
            beta, sector_beta, base_vol = 1.0, 0.0, 0.0015
        else:
            beta = float(np.clip(rng.normal(1.0, 0.35), 0.2, 2.2))
            sector_beta = float(np.clip(rng.normal(1.0, 0.4), 0.0, 2.5))
            base_vol = float(np.clip(rng.lognormal(np.log(0.013), 0.42), 0.005, 0.060))

        # GARCH(1,1)-style conditional variance gives volatility clustering.
        omega, alpha, gamma = base_vol**2 * 0.05, 0.08, 0.87
        var = np.full(n, base_vol**2)
        eps = np.empty(n)
        shocks = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)  # unit variance
        for t in range(n):
            if t > 0:
                var[t] = omega + alpha * eps[t - 1] ** 2 + gamma * var[t - 1]
            eps[t] = np.sqrt(var[t]) * shocks[t]

        # Slow AR(1) idiosyncratic drift: firms go through multi-month runs.
        # The shock is scaled so the stationary drift std is ~8 bp/day (~13%
        # annualised) — enough to create real winners and losers without the
        # 10x dispersion an unscaled random walk produces over six years.
        phi = 0.985
        drift = np.zeros(n)
        dshock = rng.standard_normal(n) * (0.0008 * np.sqrt(1 - phi**2))
        for t in range(1, n):
            drift[t] = phi * drift[t - 1] + dshock[t]

        if is_benchmark:
            ret = beta * market + eps
        else:
            ret = beta * market + sector_beta * sector + drift + eps

        start_price = float(np.exp(rng.uniform(np.log(320), np.log(9000))))
        close = start_price * np.exp(np.cumsum(ret))

        prev_close = np.concatenate([[start_price], close[:-1]])
        gap = rng.standard_normal(n) * np.sqrt(var) * 0.45
        open_ = prev_close * np.exp(gap)

        # Bracket high/low around open/close so the bar is always internally
        # consistent, with the intraday range scaled by the day's volatility.
        span = np.sqrt(var) * 0.9
        hi_body = np.maximum(open_, close)
        lo_body = np.minimum(open_, close)
        high = hi_body * np.exp(np.abs(rng.standard_normal(n)) * span)
        low = lo_body * np.exp(-np.abs(rng.standard_normal(n)) * span)

        base_turnover = float(np.exp(rng.uniform(np.log(2e7), np.log(9e9))))
        vol_z = np.abs(ret) / np.sqrt(var)
        volume = (base_turnover / close) * np.exp(
            0.35 * rng.standard_normal(n) + 0.30 * (vol_z - 0.8)
        )

        return pd.DataFrame(
            {
                "code": code,
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.maximum(np.round(volume, -2), 100.0),
            }
        )


def hash_code(code: str) -> int:
    """Stable (cross-process) integer hash — ``hash()`` is salted per run."""
    h = 2166136261
    for ch in str(code):
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def hash_sector(code: str, n_sectors: int) -> int:
    return hash_code(code) % n_sectors
