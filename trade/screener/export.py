"""JSON export for the dashboard.

The dashboard is a static page with no build step and no backend, so every
number it displays has to arrive in one of these three files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, DASHBOARD_DATA_DIR, DEFAULT_CONFIG
from .factors import FACTOR_NAMES, SPEC_BY_NAME
from .pipeline import Dataset, _json_safe
from .screen import Account
from .scoring import ScoringModel

CHART_BARS = 140


def write_all(
    out_dir: Path,
    dataset: Dataset,
    model: ScoringModel,
    report: dict,
    candidates: pd.DataFrame,
    regime: dict,
    account: Account,
    config: Config = DEFAULT_CONFIG,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _write(out_dir / "screen.json", screen_payload(dataset, model, candidates, regime, account, config)),
        _write(out_dir / "backtest.json", backtest_payload(report, model, config)),
        _write(out_dir / "charts.json", charts_payload(dataset, candidates)),
    ]
    return written


def screen_payload(
    dataset: Dataset,
    model: ScoringModel,
    candidates: pd.DataFrame,
    regime: dict,
    account: Account,
    config: Config,
) -> dict:
    rows = []
    for row in candidates.to_dict("records"):
        rows.append(
            {
                "rank": int(row["rank"]),
                "code": row["code"],
                "name": row.get("name", ""),
                "sector33": _clean(row.get("sector33")),
                "score": _num(row.get("score")),
                "ev_r": _num(row.get("ev_r")),
                "ev_yen": _num(row.get("ev_yen")),
                "est_win_rate": _num(row.get("est_win_rate")),
                "bucket": int(row["bucket"]) if row.get("bucket") is not None else None,
                "close": _num(row.get("close")),
                "entry_ref": _num(row.get("entry_ref")),
                "stop": _num(row.get("stop")),
                "target": _num(row.get("target")),
                "stop_pct": _num(row.get("stop_pct")),
                "target_pct": _num(row.get("target_pct")),
                "reward_risk": _num(row.get("reward_risk")),
                "atr14": _num(row.get("atr14")),
                "atr_pct": _num(row.get("atr_pct")),
                "turnover_ma25": _num(row.get("turnover_ma25")),
                "shares": int(row.get("shares") or 0),
                "cost_yen": _num(row.get("cost_yen")),
                "position_risk_yen": _num(row.get("position_risk_yen")),
                "sizing_note": row.get("sizing_note"),
                "contributions": row.get("contributions") or [],
            }
        )

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of": str(dataset.as_of.date()) if dataset.as_of is not None else None,
        "provider": dataset.provider,
        "is_demo": dataset.provider == "synthetic",
        "universe_size": int(dataset.factor_panel["code"].nunique())
        if not dataset.factor_panel.empty
        else 0,
        "data_quality": _quality_block(dataset),
        "regime": _json_safe(regime),
        "account": {
            "equity_yen": account.equity_yen,
            "risk_pct": account.risk_pct,
            "risk_yen": account.risk_yen,
            "max_position_pct": account.max_position_pct,
        },
        "trade_plan": {
            "stop_atr_mult": config.trade.stop_atr_mult,
            "target_atr_mult": config.trade.target_atr_mult,
            "max_hold_bars": config.trade.max_hold_bars,
            "cost_pct": config.trade.cost_pct,
            "reward_risk": config.trade.reward_risk,
        },
        "filters": {
            "min_turnover_yen": config.filters.min_turnover_yen,
            "min_price": config.filters.min_price,
            "max_price": config.filters.max_price,
            "min_atr_pct": config.filters.min_atr_pct,
            "max_atr_pct": config.filters.max_atr_pct,
        },
        "factors": [
            {
                "name": spec.name,
                "label": spec.label_ja,
                "category": spec.category,
                "direction": spec.direction,
                "band": list(spec.band) if spec.band else None,
                "rationale": spec.rationale_ja,
                "weight": round(float(model.weights.get(spec.name, 0.0)), 4),
            }
            for spec in (SPEC_BY_NAME[n] for n in FACTOR_NAMES)
        ],
        "candidates": rows,
    }


def _quality_block(dataset) -> dict:
    """What the quality gate removed, so the dashboard can say so out loud."""
    from .quality import summarise as quality_summary

    issues = getattr(dataset, "quality_issues", None)
    quarantined = list(getattr(dataset, "quarantined", []) or [])
    block: dict = {"quarantined": quarantined, "quarantined_count": len(quarantined)}
    if issues is not None and not issues.empty:
        stats = quality_summary(issues, dataset.panel)
        block["by_severity"] = stats.get("by_severity", {})
        block["by_check"] = stats.get("by_check", {})
    return block


def backtest_payload(report: dict, model: ScoringModel, config: Config) -> dict:
    ic = model.ic_table
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "folds": _json_safe(report.get("folds", [])),
        "deciles": _json_safe(report.get("deciles", [])),
        "monotonicity": _num(report.get("monotonicity")),
        "baseline": _json_safe(report.get("baseline", {})),
        "oos_all_signals": _json_safe(report.get("oos_all_signals", {})),
        "oos_summary": _json_safe(report.get("oos_summary", {})),
        "equity_curve": report.get("equity_curve", []),
        "top_bucket": _json_safe(report.get("top_bucket")),
        "ic_table": [
            {
                "factor": r["factor"],
                "label": SPEC_BY_NAME[r["factor"]].label_ja
                if r["factor"] in SPEC_BY_NAME
                else r["factor"],
                "category": SPEC_BY_NAME[r["factor"]].category
                if r["factor"] in SPEC_BY_NAME
                else "",
                "ic_mean": _num(r.get("ic_mean")),
                "ic_t": _num(r.get("ic_t")),
                "weight": round(float(model.weights.get(r["factor"], 0.0)), 4),
            }
            for r in (ic.to_dict("records") if not ic.empty else [])
        ],
        "config": _json_safe(config.to_dict()),
    }


def charts_payload(dataset: Dataset, candidates: pd.DataFrame) -> dict:
    """Recent bars and moving averages for each candidate, for the detail view."""
    if candidates.empty or dataset.panel.empty:
        return {"bars": CHART_BARS, "series": {}}

    codes = set(candidates["code"])
    panel = dataset.panel[dataset.panel["code"].isin(codes)]
    series: dict[str, dict] = {}
    for code, g in panel.groupby("code"):
        g = g.sort_values("date").tail(CHART_BARS)
        if g.empty:
            continue
        close = g["close"]
        series[str(code)] = {
            "date": [str(pd.Timestamp(d).date()) for d in g["date"]],
            "open": _floats(g["open"]),
            "high": _floats(g["high"]),
            "low": _floats(g["low"]),
            "close": _floats(close),
            "volume": _floats(g["volume"]),
            # Recomputed on the trimmed window would be wrong for the first
            # bars, so take the averages from the full history then trim.
            "sma25": _floats(_trailing_sma(dataset.panel, code, 25).tail(len(g))),
            "sma75": _floats(_trailing_sma(dataset.panel, code, 75).tail(len(g))),
        }
    return {"bars": CHART_BARS, "series": series}


def _trailing_sma(panel: pd.DataFrame, code: str, n: int) -> pd.Series:
    g = panel[panel["code"] == code].sort_values("date")
    return g["close"].rolling(n, min_periods=n).mean()


def _floats(s: pd.Series) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), 2) for v in s.to_numpy(dtype="float64")]


def _num(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    return str(v)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


DEFAULT_OUT_DIR = DASHBOARD_DATA_DIR
