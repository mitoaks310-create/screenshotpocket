"""Technical factors and the hard tradability filter.

A *factor* is one measurable property of a chart that we believe carries
information about the expected value of a swing long.  Each one is declared
with an explicit :class:`FactorSpec` stating which direction is supposed to be
good, because that a-priori direction is what stops the calibration step from
flipping a factor's sign to chase noise in a single training fold.

Three directions exist:

``higher``
    More is better (momentum, trend alignment, accumulation).
``lower``
    Less is better (volatility compression before expansion).
``band``
    A middle range is best.  Pullback depth is the archetype: no pullback means
    chasing an extended move, too deep a pullback means the trend has broken.
    Values are scored by their distance *outside* the band, so everything
    inside is equally good.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import Config, DEFAULT_CONFIG

Direction = Literal["higher", "lower", "band"]


@dataclass(frozen=True)
class FactorSpec:
    name: str
    category: str
    direction: Direction
    label_ja: str
    rationale_ja: str
    band: tuple[float, float] | None = None


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    # ---------------------------------------------------------------- trend
    FactorSpec(
        "trend_stack",
        "trend",
        "higher",
        "移動平均の並び",
        "終値>25日>75日>200日の順に並んでいるほど上昇トレンドが明確。0〜4点。",
    ),
    FactorSpec(
        "sma75_slope",
        "trend",
        "higher",
        "75日線の傾き",
        "中期移動平均が上向きかどうか。トレンドの持続性を測る。",
    ),
    FactorSpec(
        "adx14",
        "trend",
        "higher",
        "ADX(14)",
        "トレンドの強さ。方向を問わない強度指標で、値動きが方向性を持つほど高い。",
    ),
    # ------------------------------------------------------------- momentum
    FactorSpec(
        "roc60",
        "momentum",
        "higher",
        "60日騰落率",
        "中期の値上がり率。順張りの基本となる素直なモメンタム。",
    ),
    FactorSpec(
        "resid_mom_60",
        "momentum",
        "higher",
        "残差モメンタム(60日)",
        "60日騰落率からβ×TOPIX寄与を差し引いた銘柄固有の上昇分。地合いに乗っただけの上昇を除外する。",
    ),
    FactorSpec(
        "sector_rs_60",
        "momentum",
        "higher",
        "同業種内相対強度(60日)",
        "同じ33業種の銘柄群の中央値に対する超過。業種循環の中で先導しているかを見る。",
    ),
    FactorSpec(
        "mom_120_20",
        "momentum",
        "higher",
        "中期モメンタム(120-20日)",
        "直近20日を除いた120日モメンタム。短期反転の影響を除いた本質的な強さ。",
    ),
    # --------------------------------------------------------------- timing
    FactorSpec(
        "pullback_atr",
        "timing",
        "band",
        "25日線からの押し幅(ATR)",
        "上昇トレンド中の適度な押し目を狙う。伸びきった場面も崩れた場面も避ける。",
        band=(0.0, 1.5),
    ),
    FactorSpec(
        "rsi14",
        "timing",
        "band",
        "RSI(14)",
        "過熱でも売られ過ぎでもない中庸ゾーンが、上昇再開前の待機として最も期待値が高い。",
        band=(40.0, 60.0),
    ),
    FactorSpec(
        "percent_b",
        "timing",
        "band",
        "ボリンジャー%B",
        "バンド内での位置。下限付近は崩壊リスク、上限付近は過熱。中央やや下を好む。",
        band=(0.2, 0.6),
    ),
    # ----------------------------------------------------------- volatility
    FactorSpec(
        "bbw_position",
        "volatility",
        "lower",
        "バンド幅の収縮度",
        "直近120日のレンジ内でバンド幅が下位にあるほどスクイズ。収縮の後には拡大が来やすい。",
    ),
    FactorSpec(
        "vol_ratio_short_long",
        "volatility",
        "lower",
        "短期/長期ボラ比",
        "10日ボラ÷60日ボラ。エントリー直前に値動きが落ち着いている方がストップが機能しやすい。",
    ),
    # --------------------------------------------------------------- volume
    FactorSpec(
        "obv_slope",
        "volume",
        "higher",
        "OBVの傾き",
        "出来高を売買方向で符号付き累積したもの。上向きなら継続的な買い集めを示唆。",
    ),
    FactorSpec(
        "vol_ratio25",
        "volume",
        "higher",
        "出来高比(25日平均比)",
        "平常時より出来高を伴っているか。注目度の高まりを捉える。",
    ),
    # ------------------------------------------------------------ structure
    FactorSpec(
        "near_52w_high",
        "structure",
        "higher",
        "52週高値までの距離",
        "年初来高値圏は上値のしこりが少ない。テクニカルで最も頑健な要素のひとつ。",
    ),
    FactorSpec(
        "breakout_20",
        "structure",
        "higher",
        "20日高値ブレイク幅",
        "直近20日の高値をどれだけ上抜けたか。短期的な需給転換を捉える。",
    ),
)

FACTOR_NAMES: tuple[str, ...] = tuple(spec.name for spec in FACTOR_SPECS)
SPEC_BY_NAME: dict[str, FactorSpec] = {spec.name: spec for spec in FACTOR_SPECS}

#: Columns carried alongside the factors: needed to simulate a trade, to size a
#: position, and to render a chart, but never scored.
CONTEXT_COLUMNS = (
    "close",
    "atr14",
    "atr_pct",
    "turnover_ma25",
    "sma25",
    "sma75",
    "sma200",
)


def _wide(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        field: panel.pivot(index="date", columns="code", values=field).sort_index()
        for field in ("open", "high", "low", "close", "volume")
    }


def compute_factor_panel(
    panel: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    sectors: pd.Series | dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute every factor plus context for every (date, code) that is tradable.

    Rows failing the hard filters in :class:`~screener.config.Filters` are
    dropped here rather than scored and ranked low.  An illiquid stock is not a
    bad trade, it is an *unavailable* one, and leaving such rows in the
    cross-section would distort every z-score computed from it.

    ``sectors`` maps issue code to its 33-sector name and enables the
    sector-relative momentum factor; without it that factor is NaN and simply
    drops out of scoring.
    """
    if panel.empty:
        return pd.DataFrame(columns=["date", "code", *FACTOR_NAMES, *CONTEXT_COLUMNS])

    w = _wide(panel)
    close, high, low, volume = w["close"], w["high"], w["low"], w["volume"]

    bench = _benchmark_series(close, config.benchmark_code)

    sma25 = ind.sma(close, 25)
    sma75 = ind.sma(close, 75)
    sma200 = ind.sma(close, 200)
    atr14 = ind.atr(high, low, close, 14)
    atr_pct = atr14 / close
    turnover = close * volume
    turnover_ma25 = ind.sma(turnover, 25)

    percent_b, bandwidth, _ = ind.bollinger(close, 25, 2.0)
    bbw_min = ind.rolling_min(bandwidth, 120)
    bbw_max = ind.rolling_max(bandwidth, 120)
    bbw_position = (bandwidth - bbw_min) / (bbw_max - bbw_min).replace(0.0, np.nan)

    vol10 = ind.realised_vol(close, 10)
    vol60 = ind.realised_vol(close, 60)

    obv = ind.obv(close, volume)
    vol_ma25 = ind.sma(volume, 25)
    # Scale the OBV change by typical daily volume so the slope is comparable
    # across a 500-share-a-day stock and a 5-million-share-a-day stock.
    obv_slope = (obv - obv.shift(25)) / (vol_ma25 * 25.0).replace(0.0, np.nan)

    high_52w = ind.rolling_max(high, 240)
    high_20 = ind.rolling_max(high, 20).shift(1)

    roc60 = ind.roc(close, 60)
    bench_roc60 = ind.roc(bench, 60)
    beta = _rolling_beta(close, bench, 120)

    factors = {
        "trend_stack": (
            (close > sma25).astype("float64")
            + (sma25 > sma75).astype("float64")
            + (sma75 > sma200).astype("float64")
            + (close > sma200).astype("float64")
        ).where(sma200.notna()),
        "sma75_slope": ind.slope_pct(sma75, 20),
        "adx14": ind.adx(high, low, close, 14),
        "roc60": roc60,
        "resid_mom_60": roc60 - beta * bench_roc60,
        # Filled in after the melt, once each row knows its sector.
        "sector_rs_60": pd.DataFrame(np.nan, index=close.index, columns=close.columns),
        "mom_120_20": close.shift(20) / close.shift(120) - 1.0,
        "pullback_atr": (sma25 - close) / atr14.replace(0.0, np.nan),
        "rsi14": ind.rsi(close, 14),
        "percent_b": percent_b,
        "bbw_position": bbw_position,
        "vol_ratio_short_long": vol10 / vol60.replace(0.0, np.nan),
        "obv_slope": obv_slope,
        "vol_ratio25": volume / vol_ma25.replace(0.0, np.nan),
        "near_52w_high": close / high_52w.replace(0.0, np.nan),
        "breakout_20": close / high_20.replace(0.0, np.nan) - 1.0,
    }
    context = {
        "close": close,
        "atr14": atr14,
        "atr_pct": atr_pct,
        "turnover_ma25": turnover_ma25,
        "sma25": sma25,
        "sma75": sma75,
        "sma200": sma200,
    }

    eligible = _eligibility_mask(close, atr_pct, turnover_ma25, config)
    # The benchmark is a data input, not a candidate: it is an ETF, so it is
    # absent from the JPX common-stock universe and must not be ranked.
    if config.benchmark_code in eligible.columns:
        eligible[config.benchmark_code] = False

    out = _melt(factors, context, eligible)
    return _add_sector_relative(out, sectors)


def _rolling_beta(close: pd.DataFrame, bench: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling OLS beta of each stock's daily return against the benchmark."""
    ret = close.pct_change()
    bret = bench.pct_change()
    cov = (ret * bret).rolling(n, min_periods=n // 2).mean() - (
        ret.rolling(n, min_periods=n // 2).mean() * bret.rolling(n, min_periods=n // 2).mean()
    )
    var = bret.rolling(n, min_periods=n // 2).var(ddof=0).replace(0.0, np.nan)
    # Clip to a plausible range so a near-zero benchmark variance window cannot
    # manufacture a beta of 40 and blow up the residual.
    return (cov / var).clip(-3.0, 3.0)


def _add_sector_relative(
    df: pd.DataFrame, sectors: pd.Series | dict[str, str] | None
) -> pd.DataFrame:
    """Momentum relative to the median of the stock's own 33-sector peers."""
    if df.empty:
        return df
    if sectors is None:
        df["sector33"] = pd.NA
        return df
    mapping = dict(sectors) if not isinstance(sectors, dict) else sectors
    df["sector33"] = df["code"].map(mapping)
    peer_median = df.groupby(["date", "sector33"], observed=True)["roc60"].transform("median")
    df["sector_rs_60"] = (df["roc60"] - peer_median).astype("float32")
    return df


def _benchmark_series(close: pd.DataFrame, benchmark_code: str) -> pd.DataFrame:
    """Benchmark close, broadcast to every column so factor maths stays wide.

    If the benchmark is missing from the cache the relative-strength factor
    degrades to NaN and is simply dropped from scoring, rather than silently
    becoming an absolute-momentum duplicate.
    """
    if benchmark_code in close.columns:
        series = close[benchmark_code]
    else:
        series = pd.Series(np.nan, index=close.index)
    return pd.DataFrame(
        np.repeat(series.to_numpy()[:, None], close.shape[1], axis=1),
        index=close.index,
        columns=close.columns,
    )


def _eligibility_mask(
    close: pd.DataFrame,
    atr_pct: pd.DataFrame,
    turnover_ma25: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    f = config.filters
    bars_seen = close.notna().cumsum()
    return (
        close.notna()
        & (close >= f.min_price)
        & (close <= f.max_price)
        & (turnover_ma25 >= f.min_turnover_yen)
        & (atr_pct >= f.min_atr_pct)
        & (atr_pct <= f.max_atr_pct)
        & (bars_seen >= f.min_history_bars)
    )


def _melt(
    factors: dict[str, pd.DataFrame],
    context: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    """Stack the wide matrices into tidy rows, keeping only eligible cells.

    Filtering before the stack matters: the full universe over six years is
    tens of millions of cells, most of them untradable, and materialising all
    of them as rows is what would make this step run out of memory.
    """
    mask = eligible.to_numpy()
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return pd.DataFrame(columns=["date", "code", *FACTOR_NAMES, *CONTEXT_COLUMNS])

    out = {
        "date": eligible.index.to_numpy()[rows],
        "code": eligible.columns.to_numpy()[cols],
    }
    for name, frame in {**factors, **context}.items():
        out[name] = frame.to_numpy(dtype="float64")[rows, cols]

    df = pd.DataFrame(out)
    for name in FACTOR_NAMES:
        df[name] = df[name].astype("float32")
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def directional_value(values: pd.Series, spec: FactorSpec) -> pd.Series:
    """Map a raw factor onto a scale where higher always means better."""
    if spec.direction == "higher":
        return values
    if spec.direction == "lower":
        return -values
    if spec.band is None:
        raise ValueError(f"factor {spec.name!r} declares direction 'band' without a band")
    lo, hi = spec.band
    scale = (hi - lo) or 1.0
    below = (lo - values).clip(lower=0.0)
    above = (values - hi).clip(lower=0.0)
    return -(below + above) / scale
