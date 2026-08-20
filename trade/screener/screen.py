"""Today's screen: rank the cross-section and turn each name into a trade plan.

A ranking on its own is not actionable.  What comes out of here is, for each
candidate: the entry reference, the stop, the target, the share count that
risks a fixed fraction of the account, and the factor contributions that put
the name where it is — so a discretionary trader can disagree with the machine
for a stated reason rather than a vague one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .factors import FACTOR_NAMES, SPEC_BY_NAME
from .scoring import ScoringModel

#: Japanese equities trade in units of 100 shares for almost every issue.
LOT_SIZE = 100


@dataclass(frozen=True)
class Account:
    """Sizing inputs.  ``risk_pct`` is fraction of equity risked per trade."""

    equity_yen: float = 3_000_000.0
    risk_pct: float = 0.01
    #: A single position may not exceed this fraction of the account, however
    #: tight its stop.  Without this cap a very low-ATR name would be sized to
    #: an absurd notional just because its 1R happens to be small.
    max_position_pct: float = 0.25

    @property
    def risk_yen(self) -> float:
        return self.equity_yen * self.risk_pct


def market_regime(panel: pd.DataFrame, factor_panel: pd.DataFrame, config: Config) -> dict:
    """Benchmark trend and universe breadth on the screening date.

    A long-only swing system has no edge to harvest when the whole market is
    below its own trend; reporting this alongside the ranking is what stops the
    top of the list from being read as "buy these" on a day when nothing should
    be bought.
    """
    out: dict = {"benchmark_code": config.benchmark_code}
    bench = panel[panel["code"] == config.benchmark_code].sort_values("date")
    if not bench.empty and len(bench) >= 75:
        close = bench["close"].to_numpy(dtype="float64")
        sma25 = float(np.mean(close[-25:]))
        sma75 = float(np.mean(close[-75:]))
        last = float(close[-1])
        out.update(
            {
                "benchmark_close": last,
                "benchmark_sma25": sma25,
                "benchmark_sma75": sma75,
                "above_sma75": bool(last > sma75),
                "trend": "up" if last > sma75 else "down",
            }
        )
    if not factor_panel.empty:
        latest = factor_panel[factor_panel["date"] == factor_panel["date"].max()]
        if not latest.empty:
            above200 = (latest["close"] > latest["sma200"]).mean()
            out["breadth_above_sma200"] = float(above200)
            out["universe_size"] = int(len(latest))
    return out


def position_size(entry: float, risk_per_share: float, account: Account) -> dict:
    """Lot-aware share count for a fixed-fractional risk budget."""
    if not np.isfinite(entry) or not np.isfinite(risk_per_share) or risk_per_share <= 0:
        return {"shares": 0, "cost_yen": 0.0, "risk_yen": 0.0, "capped_by": "invalid"}

    raw_shares = account.risk_yen / risk_per_share
    shares = int(np.floor(raw_shares / LOT_SIZE) * LOT_SIZE)
    capped_by = "risk"

    max_notional = account.equity_yen * account.max_position_pct
    if shares * entry > max_notional:
        shares = int(np.floor(max_notional / entry / LOT_SIZE) * LOT_SIZE)
        capped_by = "position_cap"

    if shares <= 0:
        # One lot already risks more than the budget allows.
        return {
            "shares": 0,
            "cost_yen": 0.0,
            "risk_yen": 0.0,
            "capped_by": "min_lot_exceeds_risk",
            "min_lot_risk_yen": float(risk_per_share * LOT_SIZE),
        }

    return {
        "shares": shares,
        "cost_yen": float(shares * entry),
        "risk_yen": float(shares * risk_per_share),
        "capped_by": capped_by,
    }


def run_screen(
    model: ScoringModel,
    z_panel: pd.DataFrame,
    factor_panel: pd.DataFrame,
    universe: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    account: Account = Account(),
    as_of: pd.Timestamp | None = None,
    top_n: int = 30,
) -> pd.DataFrame:
    """Rank one date's cross-section and attach a full trade plan to each name."""
    if z_panel.empty or factor_panel.empty:
        return pd.DataFrame()

    as_of = pd.Timestamp(as_of) if as_of is not None else z_panel["date"].max()
    z_today = z_panel[z_panel["date"] == as_of]
    f_today = factor_panel[factor_panel["date"] == as_of]
    if z_today.empty or f_today.empty:
        return pd.DataFrame()

    # Carry the z-scores through: the per-factor contribution breakdown is the
    # main reason a human can audit the ranking, and model.score() returns only
    # the composite.
    scored = model.score(z_today).merge(z_today, on=["date", "code"], how="left")
    merged = scored.merge(f_today, on=["date", "code"], how="inner")
    merged = merged.dropna(subset=["score"]).sort_values("score", ascending=False)
    merged = merged.head(top_n).reset_index(drop=True)
    if merged.empty:
        return merged

    names = universe.set_index("code")["name"].to_dict() if not universe.empty else {}
    plan = config.trade

    # The signal fires on today's close; the entry is tomorrow's open, which is
    # unknown. Today's close is the honest reference price for planning, and the
    # dashboard labels it as such rather than pretending it is a fill.
    entry_ref = merged["close"].to_numpy(dtype="float64")
    atr = merged["atr14"].to_numpy(dtype="float64")
    risk_per_share = plan.stop_atr_mult * atr

    merged["rank"] = np.arange(1, len(merged) + 1)
    merged["name"] = merged["code"].map(names).fillna("")
    merged["entry_ref"] = entry_ref
    merged["stop"] = entry_ref - risk_per_share
    merged["target"] = entry_ref + plan.target_atr_mult * atr
    merged["risk_per_share"] = risk_per_share
    merged["stop_pct"] = risk_per_share / entry_ref
    merged["target_pct"] = (plan.target_atr_mult * atr) / entry_ref
    merged["reward_risk"] = plan.reward_risk

    sizing = [
        position_size(e, r, account) for e, r in zip(entry_ref, risk_per_share)
    ]
    merged["shares"] = [s["shares"] for s in sizing]
    merged["cost_yen"] = [s["cost_yen"] for s in sizing]
    merged["position_risk_yen"] = [s["risk_yen"] for s in sizing]
    merged["sizing_note"] = [s["capped_by"] for s in sizing]

    merged["ev_yen"] = merged["ev_r"] * merged["position_risk_yen"]
    merged["contributions"] = _contributions(merged, model.weights)
    merged["bucket"] = model.ev_map.bucket_of(merged["score"])
    merged["est_win_rate"] = _bucket_lookup(model, merged["bucket"], "win_rate")
    return merged


def _contributions(df: pd.DataFrame, weights: dict[str, float]) -> list[list[dict]]:
    """Per-name factor contributions, largest absolute effect first."""
    active = [n for n in FACTOR_NAMES if weights.get(n, 0.0) > 0 and f"z_{n}" in df.columns]
    rows: list[list[dict]] = []
    for _, row in df.iterrows():
        items = []
        for name in active:
            z = row.get(f"z_{name}")
            if z is None or not np.isfinite(z):
                continue
            spec = SPEC_BY_NAME[name]
            items.append(
                {
                    "factor": name,
                    "label": spec.label_ja,
                    "category": spec.category,
                    "z": round(float(z), 3),
                    "weight": round(float(weights[name]), 4),
                    "contribution": round(float(z) * float(weights[name]), 4),
                    "raw": _round_or_none(row.get(name)),
                }
            )
        items.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        rows.append(items)
    return rows


def _bucket_lookup(model: ScoringModel, buckets: pd.Series, field: str) -> list[float | None]:
    values = getattr(model.ev_map, field, [])
    out: list[float | None] = []
    for b in np.asarray(buckets, dtype="int64"):
        out.append(float(values[b]) if 0 <= b < len(values) else None)
    return out


def _round_or_none(v) -> float | None:
    if v is None:
        return None
    f = float(v)
    return None if not np.isfinite(f) else round(f, 4)
