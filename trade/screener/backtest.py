"""Walk-forward validation.

The point of this module is to answer one question honestly: *does a high score
actually precede a better trade, on data the model never saw?*

In-sample, any factor combination looks good — the weights were chosen to make
it look good.  So the model is refitted from scratch on each training window
and judged only on the window that follows, with an embargo gap between them.
The embargo matters more than it appears: a trade signalled on the last day of
training stays open for up to ``max_hold_bars`` afterwards, so without a gap
the training labels and the test period would overlap and the "out-of-sample"
result would be partly in-sample.

The headline output is the out-of-sample decile table: average realised R by
score bucket.  If that table is not monotone, the score is not measuring what
it claims to, however good the equity curve looks.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .scoring import ScoringModel, composite_score, fit_model
from .simulator import summarise


@dataclass
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    model: ScoringModel
    scored: pd.DataFrame = field(default_factory=pd.DataFrame)

    def describe(self) -> dict:
        return {
            "fold": self.index,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "train_trades": self.model.n_train_trades,
            "test_signals": int(len(self.scored)),
        }


def make_folds(dates: pd.DatetimeIndex, config: Config = DEFAULT_CONFIG) -> list[tuple]:
    """Expanding-window splits: training always starts at the beginning."""
    bt = config.backtest
    dates = pd.DatetimeIndex(sorted(set(dates)))
    n = len(dates)
    usable = n - bt.initial_train_bars - bt.embargo_bars
    if usable <= 0:
        return []
    test_len = usable // bt.n_folds
    if test_len < 20:
        return []

    folds = []
    for i in range(bt.n_folds):
        train_end_idx = bt.initial_train_bars + i * test_len
        test_start_idx = train_end_idx + bt.embargo_bars
        test_end_idx = min(test_start_idx + test_len, n) - 1
        if test_start_idx >= n or test_end_idx <= test_start_idx:
            break
        folds.append(
            (
                i,
                dates[0],
                dates[train_end_idx - 1],
                dates[test_start_idx],
                dates[test_end_idx],
            )
        )
    return folds


def walk_forward(
    z_panel: pd.DataFrame,
    trades: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> tuple[list[Fold], pd.DataFrame]:
    """Fit and score each fold.  Returns the folds and all out-of-sample rows."""
    if z_panel.empty or trades.empty:
        return [], pd.DataFrame()

    labelled = z_panel.merge(
        trades[["date", "code", "r_multiple", "entry_date", "exit_date", "exit_reason"]],
        on=["date", "code"],
        how="inner",
    )
    if labelled.empty:
        return [], pd.DataFrame()

    z_cols = ["date", "code", *[c for c in labelled.columns if c.startswith("z_")]]

    folds: list[Fold] = []
    oos_parts: list[pd.DataFrame] = []
    for idx, tr_start, tr_end, te_start, te_end in make_folds(
        pd.DatetimeIndex(labelled["date"].unique()), config
    ):
        train = labelled[(labelled["date"] >= tr_start) & (labelled["date"] <= tr_end)]
        test = labelled[(labelled["date"] >= te_start) & (labelled["date"] <= te_end)]
        if train.empty or test.empty:
            continue

        model = fit_model(train[z_cols], train[["date", "code", "r_multiple"]], config)
        scored = test.copy()
        scored["score"] = composite_score(test, model.weights).to_numpy()
        scored["ev_r"] = model.ev_map.predict(scored["score"])
        scored["bucket"] = model.ev_map.bucket_of(scored["score"])
        scored["fold"] = idx
        scored = scored.dropna(subset=["score"])

        fold = Fold(idx, tr_start, tr_end, te_start, te_end, model, scored)
        folds.append(fold)
        oos_parts.append(
            scored[
                [
                    "fold",
                    "date",
                    "code",
                    "score",
                    "ev_r",
                    "bucket",
                    "r_multiple",
                    "entry_date",
                    "exit_date",
                    "exit_reason",
                ]
            ]
        )

    oos = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
    return folds, oos


def fit_final_model(
    z_panel: pd.DataFrame,
    trades: pd.DataFrame,
    oos: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> ScoringModel:
    """The model used for live screening.

    Weights are refitted on the full history — more data makes the information
    coefficients steadier, and the weights are a slow-moving quantity.

    The EV calibration, however, is taken from the **pooled out-of-sample**
    score/outcome pairs produced by the walk-forward.  Calibrating it in-sample
    would report the expectancy the model was fitted to reproduce, which is
    reliably too optimistic; the walk-forward pairs are the only ones in the
    system that were never used to choose a weight.
    """
    from .scoring import fit_ev_map

    model = fit_model(z_panel, trades, config)
    if oos is None or oos.empty:
        return model
    model.ev_map = fit_ev_map(oos["score"], oos["r_multiple"], config)
    return model


def decile_table(oos: pd.DataFrame, n_buckets: int = 10) -> pd.DataFrame:
    """Out-of-sample realised R by score bucket — the core validation output."""
    if oos.empty:
        return pd.DataFrame(
            columns=["bucket", "n", "score_mean", "ev_pred", "ev_real", "win_rate"]
        )
    # Re-bucket on the pooled out-of-sample scores so each bucket holds an
    # equal share of test trades regardless of per-fold score drift.
    ranks = oos["score"].rank(pct=True, method="first")
    bucket = np.clip((ranks * n_buckets).astype(int), 0, n_buckets - 1)
    grouped = oos.assign(_b=bucket).groupby("_b")
    out = grouped.agg(
        n=("r_multiple", "size"),
        score_mean=("score", "mean"),
        ev_pred=("ev_r", "mean"),
        ev_real=("r_multiple", "mean"),
        win_rate=("r_multiple", lambda s: float((s > 0).mean())),
    ).reset_index(names="bucket")
    return out


def spearman_monotonicity(table: pd.DataFrame) -> float:
    """Rank correlation between bucket index and realised EV.

    1.0 means every step up in score produced a better average trade.
    """
    if len(table) < 3:
        return float("nan")
    # Pearson on ranks — avoids depending on scipy for a single number.
    a = pd.Series(table["bucket"]).rank()
    b = pd.Series(table["ev_real"]).rank()
    return float(a.corr(b))


# --------------------------------------------------------------------------
# Portfolio simulation
# --------------------------------------------------------------------------


def simulate_portfolio(
    oos: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    require_positive_ev: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Event-driven equity curve over the out-of-sample signals.

    Constraints that a decile table cannot express are applied here: only so
    many positions can be open at once, and capital compounds, so the order in
    which trades arrive changes the result.  Each position risks a fixed
    fraction of equity, which is what makes an R multiple translate directly
    into a percentage of the account.
    """
    bt = config.backtest
    if oos.empty:
        return pd.DataFrame(columns=["date", "equity", "open_positions"]), _empty_stats()

    pool = oos.dropna(subset=["score"]).copy()
    if require_positive_ev:
        pool = pool[pool["ev_r"] > 0]
    if pool.empty:
        return pd.DataFrame(columns=["date", "equity", "open_positions"]), _empty_stats()

    pool = pool.sort_values(["date", "score"], ascending=[True, False])
    equity = 1.0
    # (exit_date, seq, risk_amount, r_multiple, code); seq breaks ties so the
    # heap never has to compare the trailing fields.
    open_heap: list[tuple] = []
    seq = 0
    curve: list[tuple] = []
    realised: list[float] = []

    all_dates = pd.DatetimeIndex(sorted(pool["date"].unique()))
    by_date = dict(list(pool.groupby("date")))

    for today in all_dates:
        # Close everything that resolved on or before today, oldest first.
        while open_heap and open_heap[0][0] <= today:
            _, _, risk_amount, r, _code = heapq.heappop(open_heap)
            equity += risk_amount * r
            realised.append(r)

        slots = bt.max_open_positions - len(open_heap)
        if slots > 0:
            held = {p[4] for p in open_heap}
            todays = by_date.get(today)
            if todays is not None:
                taken = 0
                for row in todays.itertuples():
                    if taken >= min(slots, bt.top_n_per_day):
                        break
                    # One position per name at a time: pyramiding the same
                    # stock would concentrate risk the sizing rule assumes away.
                    if row.code in held:
                        continue
                    risk_amount = equity * bt.risk_per_trade
                    seq += 1
                    heapq.heappush(
                        open_heap,
                        (row.exit_date, seq, risk_amount, float(row.r_multiple), row.code),
                    )
                    held.add(row.code)
                    taken += 1
        curve.append((today, equity, len(open_heap)))

    # Settle anything still open at the end of the test period.
    while open_heap:
        _, _, risk_amount, r, _ = heapq.heappop(open_heap)
        equity += risk_amount * r
        realised.append(r)
    if curve:
        curve.append((all_dates[-1], equity, 0))

    curve_df = pd.DataFrame(curve, columns=["date", "equity", "open_positions"])
    return curve_df, _portfolio_stats(curve_df, realised, config)


def _portfolio_stats(curve: pd.DataFrame, realised: list[float], config: Config) -> dict:
    if curve.empty or not realised:
        return _empty_stats()
    eq = curve["equity"].to_numpy(dtype="float64")
    r = np.asarray(realised, dtype="float64")

    running_max = np.maximum.accumulate(eq)
    drawdown = eq / running_max - 1.0
    days = max((curve["date"].iloc[-1] - curve["date"].iloc[0]).days, 1)
    years = days / 365.25
    total_return = float(eq[-1] / eq[0] - 1.0)

    daily = pd.Series(eq, index=pd.DatetimeIndex(curve["date"])).pct_change().dropna()
    sharpe = (
        float(daily.mean() / daily.std(ddof=1) * np.sqrt(245.0))
        if len(daily) > 2 and daily.std(ddof=1) > 0
        else None
    )
    wins = r[r > 0]
    losses = r[r <= 0]
    gross_loss = float(-losses.sum())
    return {
        "trades": int(len(r)),
        "total_return": total_return,
        "cagr": float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0) if years > 0.1 else None,
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "expectancy_r": float(r.mean()),
        "win_rate": float(len(wins) / len(r)),
        "payoff": float(wins.mean() / -losses.mean()) if len(losses) and len(wins) else None,
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else None,
        "risk_per_trade": config.backtest.risk_per_trade,
        "max_open_positions": config.backtest.max_open_positions,
    }


def _empty_stats() -> dict:
    return {
        "trades": 0,
        "total_return": None,
        "cagr": None,
        "max_drawdown": None,
        "sharpe": None,
        "expectancy_r": None,
        "win_rate": None,
        "payoff": None,
        "profit_factor": None,
    }


def run_backtest(
    z_panel: pd.DataFrame,
    trades: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> dict:
    """Full walk-forward run, packaged for the report and the dashboard."""
    folds, oos = walk_forward(z_panel, trades, config)
    if not folds:
        return {
            "folds": [],
            "deciles": [],
            "monotonicity": None,
            "baseline": summarise(trades),
            "oos_summary": _empty_stats(),
            "equity_curve": [],
            "top_bucket": None,
        }

    table = decile_table(oos, config.scoring.ev_buckets)
    curve, stats = simulate_portfolio(oos, config)

    top = table.iloc[-1] if not table.empty else None
    return {
        "folds": [f.describe() for f in folds],
        "deciles": table.to_dict("records"),
        "monotonicity": spearman_monotonicity(table),
        "baseline": summarise(trades),
        "oos_all_signals": {
            "trades": int(len(oos)),
            "expectancy_r": float(oos["r_multiple"].mean()),
            "win_rate": float((oos["r_multiple"] > 0).mean()),
        },
        "oos_summary": stats,
        "equity_curve": [
            {"date": str(pd.Timestamp(d).date()), "equity": float(e)}
            for d, e in zip(curve["date"], curve["equity"])
        ],
        "top_bucket": None
        if top is None
        else {
            "n": int(top["n"]),
            "ev_real": float(top["ev_real"]),
            "ev_pred": float(top["ev_pred"]),
            "win_rate": float(top["win_rate"]),
        },
        "fold_models": [
            {"fold": f.index, "weights": f.model.weights, "ev_map": f.model.ev_map.to_dict()}
            for f in folds
        ],
    }
