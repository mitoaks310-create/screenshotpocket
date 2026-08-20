"""Central configuration for the swing-trade screening system.

Everything that defines *what a trade is* lives here.  The expected value the
screener reports is only meaningful relative to a fixed, mechanical trade plan,
so these numbers are part of the model — changing them invalidates a previously
fitted model and requires a re-run of ``backtest``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
UNIVERSE_CSV = DATA_DIR / "universe" / "jpx_universe.csv"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "data"

#: Benchmark used for relative-strength factors.  TOPIX ETF 1306 tracks TOPIX
#: and, unlike the ``^TOPX`` index symbol, is available from every provider.
BENCHMARK_CODE = "1306"

TRADING_DAYS_PER_YEAR = 245


@dataclass(frozen=True)
class TradePlan:
    """The mechanical trade the screener is trying to find good instances of.

    Expected value is measured in *R multiples*, where 1R is the initial risk
    (the distance from entry to the initial stop).  Expressing outcomes in R
    makes trades on a 500-yen stock and a 9,000-yen stock directly comparable,
    which is what lets a single cross-sectional ranking be meaningful.
    """

    #: Entry happens at the open of the bar after the signal bar.  Using the
    #: signal bar's own close would leak information no live trader has.
    entry_delay_bars: int = 1
    #: Initial stop distance, in ATR(14) units below the entry price.
    stop_atr_mult: float = 1.5
    #: Profit target distance, in ATR(14) units above the entry price.
    target_atr_mult: float = 3.0
    #: Exit at the close of this bar if neither stop nor target was hit.
    max_hold_bars: int = 15
    #: Round-trip cost (commission + spread + slippage) as a fraction of the
    #: entry price.  0.15% is a realistic retail figure for liquid JP equities.
    cost_pct: float = 0.0015

    @property
    def reward_risk(self) -> float:
        return self.target_atr_mult / self.stop_atr_mult


@dataclass(frozen=True)
class Filters:
    """Hard tradability gates applied before any scoring happens.

    These are not scored factors — a stock that fails any of them is removed
    from the universe entirely.  Illiquid names dominate naive backtests with
    returns that cannot be realised at size, so this is the single most
    important guard in the system.
    """

    #: Minimum 25-day average daily turnover (price x volume) in yen.
    min_turnover_yen: float = 300_000_000
    #: Minimum close price in yen.
    min_price: float = 300.0
    #: Maximum close price in yen — a single 100-share lot must stay affordable.
    max_price: float = 30_000.0
    #: Bars of history required before a stock can be scored.
    min_history_bars: int = 260
    #: Reject stocks whose ATR% is above this — the stop would be absurdly wide.
    max_atr_pct: float = 0.12
    #: Reject stocks whose ATR% is below this — no room for the target to fill.
    min_atr_pct: float = 0.010
    markets: tuple[str, ...] = ("prime", "standard", "growth")
    sizes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ScoringConfig:
    """How raw factors become a single expected-value estimate."""

    #: Cross-sectional winsorisation percentile for raw factor values.
    winsor_pct: float = 0.02
    #: Number of score buckets used to build the score -> EV lookup table.
    ev_buckets: int = 10
    #: Minimum trades in a bucket before its EV estimate is trusted.
    min_bucket_trades: int = 200
    #: Information-coefficient shrinkage.  A factor's weight is proportional to
    #: max(0, IC - shrink), so factors whose edge is indistinguishable from
    #: noise drop out instead of contributing random variance.
    ic_shrink: float = 0.005
    #: Floor on the number of names scored on a given date.
    min_cross_section: int = 30


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward validation layout."""

    #: Number of walk-forward folds.
    n_folds: int = 4
    #: Bars of training history before the first out-of-sample fold.
    initial_train_bars: int = 500
    #: Gap between the end of training and the start of testing, in bars.  Must
    #: be at least ``max_hold_bars`` so a training trade cannot still be open
    #: when the test window begins.
    embargo_bars: int = 20
    #: Only signals ranked in the top N on a date become simulated trades.
    top_n_per_day: int = 5
    #: Maximum simultaneous open positions in the equity-curve simulation.
    max_open_positions: int = 8
    #: Fraction of equity risked per trade in the equity-curve simulation.
    risk_per_trade: float = 0.01


@dataclass(frozen=True)
class Config:
    trade: TradePlan = field(default_factory=TradePlan)
    filters: Filters = field(default_factory=Filters)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    benchmark_code: str = BENCHMARK_CODE

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = Config()
