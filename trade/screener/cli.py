"""Command line interface.

    python -m screener.cli universe --refresh
    python -m screener.cli update --provider yfinance --years 7
    python -m screener.cli backtest
    python -m screener.cli screen --equity 3000000 --risk 0.01
    python -m screener.cli serve

``run`` chains update, backtest and screen, which is the normal daily job.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

from .config import DASHBOARD_DATA_DIR, DEFAULT_CONFIG, MODEL_DIR, Config
from .export import write_all
from .pipeline import build_dataset, load_model, save_model, train, update_prices
from .screen import Account, market_regime, run_screen
from .store import PriceStore
from .universe import load_universe, refresh_universe


def _config_from_args(args) -> Config:
    """Apply CLI overrides to the frozen default config."""
    config = DEFAULT_CONFIG
    filt = config.filters
    changes = {}
    if getattr(args, "min_turnover", None) is not None:
        changes["min_turnover_yen"] = args.min_turnover
    if getattr(args, "markets", None):
        changes["markets"] = tuple(args.markets)
    if changes:
        config = dataclasses.replace(config, filters=dataclasses.replace(filt, **changes))
    return config


# ---------------------------------------------------------------- commands


def cmd_universe(args) -> int:
    if args.refresh:
        df = refresh_universe()
        print(f"universe refreshed: {len(df)} issues -> data/universe/jpx_universe.csv")
    df = load_universe()
    print(f"{len(df)} issues")
    print(df["market"].value_counts().to_string())
    print(df["size"].value_counts().to_string())
    return 0


def cmd_update(args) -> int:
    config = _config_from_args(args)
    universe = load_universe(markets=config.filters.markets)
    if args.limit:
        universe = universe.head(args.limit)
    codes = universe["code"].tolist()

    print(f"fetching {len(codes)} codes via {args.provider} ({args.years}y)…")
    store = PriceStore()
    if args.rebuild:
        store.clear()
    panel = update_prices(
        codes,
        provider_name=args.provider,
        years=args.years,
        store=store,
        config=config,
        incremental=not args.rebuild,
    )
    if panel.empty:
        print("no data fetched — check network access or try --provider synthetic", file=sys.stderr)
        return 1
    print(
        f"cache now holds {len(panel):,} bars for {panel['code'].nunique():,} codes "
        f"({panel['date'].min().date()} … {panel['date'].max().date()})"
    )
    return 0


def cmd_backtest(args) -> int:
    config = _config_from_args(args)
    dataset = build_dataset(provider_name=args.provider, config=config, limit=args.limit)
    print(
        f"dataset: {len(dataset.factor_panel):,} scoreable rows, "
        f"{dataset.factor_panel['code'].nunique():,} codes, "
        f"{len(dataset.trades):,} simulated trades"
    )
    model, report = train(dataset, config)
    path = save_model(model, report, dataset, MODEL_DIR, config)
    _print_backtest(report, model)
    print(f"\nmodel saved -> {path}")
    return 0


def cmd_screen(args) -> int:
    config = _config_from_args(args)
    model, payload = load_model()
    dataset = build_dataset(provider_name=args.provider, config=config, limit=args.limit)
    account = Account(
        equity_yen=args.equity, risk_pct=args.risk, max_position_pct=args.max_position
    )

    candidates = run_screen(
        model, dataset.z_panel, dataset.factor_panel, dataset.universe,
        config=config, account=account, top_n=args.top,
    )
    if candidates.empty:
        print("no candidates passed the filters on the latest date", file=sys.stderr)
        return 1

    regime = market_regime(dataset.panel, dataset.factor_panel, config)
    _print_screen(candidates, regime, dataset, account)

    out_dir = Path(args.out) if args.out else DASHBOARD_DATA_DIR
    written = write_all(
        out_dir, dataset, model, payload.get("backtest", {}), candidates, regime, account, config
    )
    print("\nwrote " + ", ".join(str(p) for p in written))
    if args.csv:
        cols = [
            "rank", "code", "name", "score", "ev_r", "est_win_rate", "close",
            "entry_ref", "stop", "target", "shares", "cost_yen", "position_risk_yen",
        ]
        candidates[cols].to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    return 0


def cmd_run(args) -> int:
    for step in (cmd_update, cmd_backtest, cmd_screen):
        rc = step(args)
        if rc != 0:
            return rc
    return 0


def cmd_status(args) -> int:
    store = PriceStore()
    cov = store.coverage()
    if cov.empty:
        print("price cache is empty")
    else:
        print(f"cache: {len(cov):,} codes, {int(cov['bars'].sum()):,} bars")
        print(f"  dates: {cov['first'].min().date()} … {cov['last'].max().date()}")
        print(f"  median bars/code: {int(cov['bars'].median()):,}")
    model_path = MODEL_DIR / "model.json"
    if model_path.exists():
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        print(f"model: trained {payload.get('trained_at')} on provider={payload.get('provider')}")
        print(f"  as_of={payload.get('as_of')} codes={payload.get('n_codes')}")
    else:
        print("model: not trained")
    return 0


def cmd_serve(args) -> int:
    import functools
    import http.server
    import socketserver

    root = Path(__file__).resolve().parent.parent / "dashboard"
    if not (root / "index.html").exists():
        print(f"dashboard not found at {root}", file=sys.stderr)
        return 1
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"serving {root} at http://localhost:{args.port}/  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


# ------------------------------------------------------------------ output


def _print_backtest(report: dict, model) -> None:
    print("\n=== walk-forward folds ===")
    for f in report.get("folds", []):
        print(
            f"  fold {f['fold']}: train {f['train_start']}…{f['train_end']} "
            f"({f['train_trades']:,} trades) -> test {f['test_start']}…{f['test_end']} "
            f"({f['test_signals']:,} signals)"
        )

    base = report.get("baseline", {})
    if base.get("trades"):
        print(
            f"\nbaseline (every signal, all dates): {base['trades']:,} trades  "
            f"EV {base['expectancy_r']:+.4f}R  win {base['win_rate']:.1%}  "
            f"payoff {base['payoff']:.2f}"
        )

    print("\n=== out-of-sample expected value by score decile ===")
    print(f"{'bucket':>6} {'n':>8} {'score':>8} {'EV pred':>9} {'EV real':>9} {'win':>7}")
    for d in report.get("deciles", []):
        print(
            f"{int(d['bucket']):>6} {int(d['n']):>8,} {d['score_mean']:>8.3f} "
            f"{d['ev_pred']:>+9.4f} {d['ev_real']:>+9.4f} {d['win_rate']:>7.1%}"
        )
    mono = report.get("monotonicity")
    if mono is not None:
        print(f"\nmonotonicity (bucket vs realised EV, Spearman): {mono:+.3f}")
        if mono < 0.6:
            print(
                "  ⚠ weak: a higher score did not reliably precede a better trade "
                "out of sample. Treat the EV numbers as unvalidated."
            )

    stats = report.get("oos_summary", {})
    if stats.get("trades"):
        print(
            f"\nportfolio (top-N, {stats['risk_per_trade']:.1%} risk/trade, "
            f"max {stats['max_open_positions']} open): {stats['trades']:,} trades"
        )
        print(
            f"  total {stats['total_return']:+.1%}  CAGR {_pct(stats.get('cagr'))}  "
            f"maxDD {stats['max_drawdown']:.1%}  Sharpe {_f(stats.get('sharpe'))}  "
            f"EV {stats['expectancy_r']:+.4f}R  win {stats['win_rate']:.1%}"
        )

    print("\n=== factor information coefficients (full history) ===")
    ic = model.ic_table
    if not ic.empty:
        merged = ic.copy()
        merged["weight"] = merged["factor"].map(model.weights).fillna(0.0)
        merged = merged.sort_values("weight", ascending=False)
        print(f"{'factor':<22} {'IC':>8} {'t':>8} {'weight':>8}")
        for r in merged.itertuples():
            print(f"{r.factor:<22} {r.ic_mean:>+8.4f} {r.ic_t:>+8.2f} {r.weight:>8.3f}")


def _print_screen(candidates: pd.DataFrame, regime: dict, dataset, account: Account) -> None:
    as_of = dataset.as_of
    print(f"\n=== screen {as_of.date() if as_of is not None else '?'} ===")
    if dataset.provider == "synthetic":
        print("  ** SYNTHETIC DEMO DATA — not real prices, not tradable **")
    trend = regime.get("trend")
    breadth = regime.get("breadth_above_sma200")
    if trend:
        line = f"  market: benchmark {trend} vs 75MA"
        if breadth is not None:
            line += f", breadth {breadth:.0%} above 200MA"
        print(line)
        if trend == "down":
            print("  ⚠ benchmark below its 75-day average — long swing setups are lower odds here.")
    print(
        f"  account ¥{account.equity_yen:,.0f}, risk {account.risk_pct:.1%} "
        f"(¥{account.risk_yen:,.0f}/trade)"
    )

    best_ev = float(candidates["ev_r"].max())
    if best_ev <= 0:
        print(
            f"\n  ⚠ every candidate's calibrated EV is negative (best {best_ev:+.3f}R). "
            "The ranking still shows the relatively strongest charts, but the model "
            "is saying there is no edge to take today."
        )

    head = candidates.head(20)
    print(
        f"\n{'#':>3} {'code':<6} {'name':<14} {'score':>7} {'EV(R)':>7} {'win':>6} "
        f"{'close':>9} {'stop':>9} {'target':>9} {'shares':>7} {'cost':>11}  note"
    )
    for r in head.itertuples():
        name = str(r.name)[:13]
        win = f"{r.est_win_rate:.0%}" if pd.notna(r.est_win_rate) else "  -"
        note = ""
        if r.sizing_note == "min_lot_exceeds_risk":
            note = "  1 lot exceeds risk budget"
        elif r.sizing_note == "position_cap":
            note = "  capped by position limit"
        print(
            f"{r.rank:>3} {r.code:<6} {name:<14} {r.score:>+7.3f} {r.ev_r:>+7.3f} {win:>6} "
            f"{r.entry_ref:>9,.1f} {r.stop:>9,.1f} {r.target:>9,.1f} "
            f"{r.shares:>7,} {r.cost_yen:>11,.0f}{note}"
        )

    untradable = int((candidates["shares"] == 0).sum())
    if untradable:
        print(
            f"\n  {untradable} of {len(candidates)} candidates cannot be sized at "
            f"¥{account.equity_yen:,.0f} with {account.risk_pct:.1%} risk: "
            "one 100-share lot already risks more than the budget."
        )


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:+.1%}"


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="screener",
        description="Technical swing-trade screening system for Japanese equities.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def shared(sp, *, provider_default="yfinance"):
        sp.add_argument("--provider", default=provider_default,
                        choices=["yfinance", "stooq", "synthetic"])
        sp.add_argument("--limit", type=int, default=None,
                        help="only use the first N issues (faster trial runs)")
        sp.add_argument("--min-turnover", type=float, default=None,
                        help="override the minimum 25-day average turnover in yen")
        sp.add_argument("--markets", nargs="*", default=None,
                        choices=["prime", "standard", "growth"])

    sp = sub.add_parser("universe", help="inspect or refresh the JPX issue list")
    sp.add_argument("--refresh", action="store_true", help="re-download from JPX")
    sp.set_defaults(func=cmd_universe)

    sp = sub.add_parser("update", help="fetch price bars into the local cache")
    shared(sp)
    sp.add_argument("--years", type=float, default=7.0)
    sp.add_argument("--rebuild", action="store_true", help="discard the cache and refetch")
    sp.set_defaults(func=cmd_update)

    sp = sub.add_parser("backtest", help="walk-forward validate and fit the live model")
    shared(sp)
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("screen", help="rank today's cross-section and export the dashboard")
    shared(sp)
    sp.add_argument("--top", type=int, default=30)
    sp.add_argument("--equity", type=float, default=3_000_000.0)
    sp.add_argument("--risk", type=float, default=0.01)
    sp.add_argument("--max-position", type=float, default=0.25)
    sp.add_argument("--out", default=None, help="output directory for dashboard JSON")
    sp.add_argument("--csv", default=None, help="also write the ranking to this CSV path")
    sp.set_defaults(func=cmd_screen)

    sp = sub.add_parser("run", help="update + backtest + screen")
    shared(sp)
    sp.add_argument("--years", type=float, default=7.0)
    sp.add_argument("--rebuild", action="store_true")
    sp.add_argument("--top", type=int, default=30)
    sp.add_argument("--equity", type=float, default=3_000_000.0)
    sp.add_argument("--risk", type=float, default=0.01)
    sp.add_argument("--max-position", type=float, default=0.25)
    sp.add_argument("--out", default=None)
    sp.add_argument("--csv", default=None)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("status", help="show cache and model state")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("serve", help="serve the dashboard over HTTP")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
