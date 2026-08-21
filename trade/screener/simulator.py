"""Mechanical trade simulation — the definition of the objective.

The screener does not predict returns; it predicts the outcome of one specific,
fully-specified trade:

* enter at the **open of the next bar** after the signal (never the signal
  bar's own close — that price is not available to anyone acting on the signal),
* place the initial stop ``stop_atr_mult`` ATRs below the entry,
* place the target ``target_atr_mult`` ATRs above the entry,
* otherwise exit at the close after ``max_hold_bars`` bars.

The outcome is recorded as an **R multiple**: profit divided by the initial
risk.  That normalisation is what makes a 400-yen stock and a 9,000-yen stock
comparable in a single ranking, and it is what "expected value" refers to
everywhere else in this package.

Two realism details matter more than they look:

*Gaps.*  A stop is not a guaranteed fill.  If a bar opens through the stop, the
fill is the open, not the stop price.  Modelling the gap is what keeps the
measured EV honest — assuming clean stop fills quietly inflates every result.

*Same-bar ambiguity.*  Daily bars cannot say whether the low or the high came
first.  When one bar touches both stop and target, this simulator always books
the stop.  That is pessimistic by construction, and deliberately so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .limits import limit_width, locked_limit_down, locked_limit_up

EXIT_STOP, EXIT_TARGET, EXIT_TIME = 0, 1, 2
EXIT_LABELS = {EXIT_STOP: "stop", EXIT_TARGET: "target", EXIT_TIME: "time"}

_SENTINEL = 1 << 30


def simulate_panel(
    panel: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    atr: pd.DataFrame | None = None,
    chunk_size: int = 250,
) -> pd.DataFrame:
    """Simulate the trade plan from every (date, code) as a signal bar.

    Returns tidy rows with one record per signal that produced a *completed*
    trade.  Signals too close to the end of history for the trade to resolve
    are dropped rather than truncated, since a half-finished trade would bias
    the calibration toward whatever the market did most recently.
    """
    from . import indicators as ind

    if panel.empty:
        return _empty_result()

    wide = {
        field: panel.pivot(index="date", columns="code", values=field).sort_index()
        for field in ("open", "high", "low", "close")
    }
    if atr is None:
        highs = wide["high"]
        lows = wide["low"]
        atr = ind.atr(highs, lows, wide["close"], 14)
    atr = atr.reindex(index=wide["close"].index, columns=wide["close"].columns)

    dates = wide["close"].index
    codes = wide["close"].columns
    frames: list[pd.DataFrame] = []
    for start in range(0, len(codes), chunk_size):
        block = codes[start : start + chunk_size]
        frames.append(
            _simulate_block(
                {k: v[block].to_numpy(dtype="float64") for k, v in wide.items()},
                atr[block].to_numpy(dtype="float64"),
                dates,
                block,
                config,
            )
        )
    result = pd.concat(frames, ignore_index=True) if frames else _empty_result()
    return result.sort_values(["date", "code"]).reset_index(drop=True)


def _simulate_block(
    w: dict[str, np.ndarray],
    atr: np.ndarray,
    dates: pd.DatetimeIndex,
    codes: pd.Index,
    config: Config,
) -> pd.DataFrame:
    plan = config.trade
    o, h, l, c = w["open"], w["high"], w["low"], w["close"]
    n_dates, n_codes = c.shape
    delay = plan.entry_delay_bars
    hold = plan.max_hold_bars

    entry = _shift_back(o, delay)
    risk = plan.stop_atr_mult * atr
    with np.errstate(invalid="ignore"):
        stop = entry - risk
        target = entry + plan.target_atr_mult * atr

    # Daily limit prices are set off the previous close, so the entry bar's
    # limits key off the bar before it.
    prev_close_at_entry = _shift_back(c, delay - 1)
    width = limit_width(prev_close_at_entry)
    upper_at_entry = prev_close_at_entry + width
    entry_high = _shift_back(h, delay)
    entry_low = _shift_back(l, delay)
    unfillable = (
        locked_limit_up(entry, entry_high, entry_low, upper_at_entry)
        if plan.respect_price_limits
        else np.zeros_like(entry, dtype=bool)
    )

    # Bars the position is exposed to: the entry bar itself through the last
    # bar of the holding window.
    offsets = np.arange(delay, delay + hold)
    highs = np.stack([_shift_back(h, k) for k in offsets])
    lows = np.stack([_shift_back(l, k) for k in offsets])
    opens = np.stack([_shift_back(o, k) for k in offsets])
    closes = np.stack([_shift_back(c, k) for k in offsets])

    stop_hit = lows <= stop[None, :, :]
    target_hit = highs >= target[None, :, :]

    first_stop = np.where(stop_hit.any(axis=0), stop_hit.argmax(axis=0), _SENTINEL)
    first_target = np.where(target_hit.any(axis=0), target_hit.argmax(axis=0), _SENTINEL)

    # Ties go to the stop: a daily bar cannot prove the high came first.
    stop_first = first_stop <= first_target
    resolved = np.minimum(first_stop, first_target)
    timed_out = resolved == _SENTINEL
    exit_idx = np.where(timed_out, hold - 1, resolved)

    rows = np.arange(n_dates)[:, None]
    cols = np.arange(n_codes)[None, :]
    idx = np.clip(exit_idx, 0, hold - 1)
    exit_open = opens[idx, rows, cols]
    exit_close = closes[idx, rows, cols]

    # A gap through the level fills at the open, not at the level.
    stop_fill = np.minimum(stop, exit_open)
    target_fill = np.maximum(target, exit_open)

    if plan.respect_price_limits:
        # A bar locked limit-down offers no exit at all: the holder is stuck
        # and sells into the next session instead. Ignoring this books a clean
        # -1R on precisely the days when the loss was worst.
        prev_closes = np.stack([_shift_back(c, k - 1) for k in offsets])
        lower_bounds = prev_closes - limit_width(prev_closes)
        locked_down = locked_limit_down(opens, highs, lows, lower_bounds)

        exit_locked = locked_down[idx, rows, cols]
        next_idx = np.clip(idx + 1, 0, hold - 1)
        next_open = opens[next_idx, rows, cols]
        # Only defer when a later bar actually exists to sell into.
        can_defer = exit_locked & (idx + 1 <= hold - 1) & np.isfinite(next_open)
        stop_fill = np.where(can_defer, np.minimum(stop_fill, next_open), stop_fill)

    exit_price = np.where(
        timed_out,
        exit_close,
        np.where(stop_first, stop_fill, target_fill),
    )
    reason = np.where(
        timed_out, EXIT_TIME, np.where(stop_first, EXIT_STOP, EXIT_TARGET)
    ).astype("int8")

    with np.errstate(invalid="ignore", divide="ignore"):
        gross_r = (exit_price - entry) / risk
        cost_r = plan.cost_pct * entry / risk
    net_r = gross_r - cost_r

    valid = (
        np.isfinite(entry)
        & np.isfinite(exit_price)
        & np.isfinite(atr)
        & (risk > 0)
        & (entry > 0)
        # A signal whose entry bar was locked limit-up never became a trade.
        # Dropping it is not the same as dropping a losing trade: there was no
        # position to lose on, and keeping it would credit the strategy with a
        # fill nobody could get.
        & ~unfillable
    )
    # Drop trades whose holding window runs past the end of the data: the
    # window is complete only if a resolution happened, or the final bar exists.
    window_complete = np.isfinite(closes[hold - 1]) | (~timed_out)
    valid &= window_complete

    r_idx, c_idx = np.nonzero(valid)
    if len(r_idx) == 0:
        return _empty_result()

    bars_held = (exit_idx + delay)[r_idx, c_idx].astype("int16")
    exit_pos = np.minimum(r_idx + bars_held.astype("int64"), n_dates - 1)
    return pd.DataFrame(
        {
            "date": dates.to_numpy()[r_idx],
            "code": codes.to_numpy()[c_idx],
            "entry_date": dates.to_numpy()[np.minimum(r_idx + delay, n_dates - 1)],
            "exit_date": dates.to_numpy()[exit_pos],
            "entry": entry[r_idx, c_idx],
            "stop": stop[r_idx, c_idx],
            "target": target[r_idx, c_idx],
            "risk_yen": risk[r_idx, c_idx],
            "exit_price": exit_price[r_idx, c_idx],
            "exit_reason": pd.Categorical(
                [EXIT_LABELS[v] for v in reason[r_idx, c_idx]],
                categories=list(EXIT_LABELS.values()),
            ),
            "bars_held": bars_held,
            "r_multiple": net_r[r_idx, c_idx].astype("float32"),
        }
    )


def _shift_back(arr: np.ndarray, k: int) -> np.ndarray:
    """``out[t] = arr[t + k]``, NaN-padded at the tail."""
    if k == 0:
        return arr
    out = np.full_like(arr, np.nan)
    if k < arr.shape[0]:
        out[:-k] = arr[k:]
    return out


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "code": pd.Series(dtype="object"),
            "entry_date": pd.Series(dtype="datetime64[ns]"),
            "exit_date": pd.Series(dtype="datetime64[ns]"),
            "entry": pd.Series(dtype="float64"),
            "stop": pd.Series(dtype="float64"),
            "target": pd.Series(dtype="float64"),
            "risk_yen": pd.Series(dtype="float64"),
            "exit_price": pd.Series(dtype="float64"),
            "exit_reason": pd.Series(dtype="object"),
            "bars_held": pd.Series(dtype="int16"),
            "r_multiple": pd.Series(dtype="float32"),
        }
    )


def summarise(trades: pd.DataFrame) -> dict:
    """Headline statistics for a set of simulated trades."""
    if trades.empty:
        return {
            "trades": 0,
            "expectancy_r": None,
            "win_rate": None,
            "avg_win_r": None,
            "avg_loss_r": None,
            "payoff": None,
            "profit_factor": None,
        }
    r = trades["r_multiple"].astype("float64")
    wins = r[r > 0]
    losses = r[r <= 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "trades": int(len(r)),
        "expectancy_r": float(r.mean()),
        "win_rate": float(len(wins) / len(r)),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "payoff": float(wins.mean() / -losses.mean()) if len(losses) and losses.mean() != 0 else None,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
        "exit_mix": {
            str(k): int(v) for k, v in trades["exit_reason"].value_counts().items()
        },
    }
