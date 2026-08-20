"""Orchestration: cache -> factors -> labels -> model -> screen.

Kept separate from the CLI so the same sequence can be driven from a notebook
or a scheduled job without going through argument parsing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import fit_final_model, run_backtest, walk_forward
from .config import Config, DEFAULT_CONFIG, MODEL_DIR
from .factors import compute_factor_panel
from .providers import get_provider
from .scoring import ScoringModel, zscore_factors
from .simulator import simulate_panel
from .store import PriceStore
from .universe import load_universe

MODEL_FILE = "model.json"
META_FILE = "dataset.json"


@dataclass
class Dataset:
    """Everything derived from the price cache, computed once and reused."""

    panel: pd.DataFrame
    factor_panel: pd.DataFrame
    trades: pd.DataFrame
    z_panel: pd.DataFrame
    universe: pd.DataFrame
    provider: str

    @property
    def as_of(self) -> pd.Timestamp | None:
        if self.factor_panel.empty:
            return None
        return pd.Timestamp(self.factor_panel["date"].max())


def update_prices(
    codes: list[str],
    provider_name: str = "yfinance",
    years: float = 7.0,
    store: PriceStore | None = None,
    config: Config = DEFAULT_CONFIG,
    incremental: bool = True,
) -> pd.DataFrame:
    """Fetch bars into the cache.

    The benchmark is always fetched alongside the requested codes; the
    relative-strength and residual-momentum factors are silently useless
    without it.
    """
    store = store or PriceStore()
    provider = get_provider(provider_name)

    wanted = list(dict.fromkeys([config.benchmark_code, *codes]))
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(365.25 * years))

    if incremental:
        last = store.last_date()
        if last is not None:
            # Re-fetch a short overlap so late corrections and any bars missed
            # during a partial run get repaired rather than left as a hole.
            start = max(start, last - pd.Timedelta(days=10))

    fetched = provider.fetch(wanted, start=start, end=end)
    return store.upsert(fetched)


def build_dataset(
    provider_name: str = "yfinance",
    config: Config = DEFAULT_CONFIG,
    store: PriceStore | None = None,
    limit: int | None = None,
) -> Dataset:
    """Load the cache and compute factors, labels and z-scores."""
    store = store or PriceStore()
    universe = load_universe(markets=config.filters.markets, sizes=config.filters.sizes)
    if limit:
        universe = universe.head(limit)

    codes = set(universe["code"]) | {config.benchmark_code}
    panel = store.load()
    if panel.empty:
        raise RuntimeError(
            "price cache is empty — run `python -m screener.cli update` first "
            "(or `--provider synthetic` for an offline demo)."
        )
    panel = panel[panel["code"].isin(codes)].reset_index(drop=True)

    sectors = dict(zip(universe["code"], universe["sector33"]))
    factor_panel = compute_factor_panel(panel, config=config, sectors=sectors)
    trades = simulate_panel(panel, config=config)
    z_panel = zscore_factors(factor_panel, config=config)

    return Dataset(
        panel=panel,
        factor_panel=factor_panel,
        trades=trades,
        z_panel=z_panel,
        universe=universe,
        provider=provider_name,
    )


def train(dataset: Dataset, config: Config = DEFAULT_CONFIG) -> tuple[ScoringModel, dict]:
    """Walk-forward validate, then fit the live model on out-of-sample calibration."""
    report = run_backtest(dataset.z_panel, dataset.trades, config)
    _folds, oos = walk_forward(dataset.z_panel, dataset.trades, config)
    model = fit_final_model(dataset.z_panel, dataset.trades, oos, config)
    return model, report


def save_model(
    model: ScoringModel,
    report: dict,
    dataset: Dataset,
    model_dir: Path = MODEL_DIR,
    config: Config = DEFAULT_CONFIG,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.to_dict(),
        "config": config.to_dict(),
        "provider": dataset.provider,
        "as_of": str(dataset.as_of.date()) if dataset.as_of is not None else None,
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_codes": int(dataset.factor_panel["code"].nunique())
        if not dataset.factor_panel.empty
        else 0,
        "backtest": _json_safe(report),
    }
    path = model_dir / MODEL_FILE
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_model(model_dir: Path = MODEL_DIR) -> tuple[ScoringModel, dict]:
    path = model_dir / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model at {path}. Run `python -m screener.cli backtest` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ScoringModel.from_dict(payload["model"]), payload


def _json_safe(obj):
    """Recursively convert numpy/pandas scalars so ``json.dumps`` accepts them."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if pd.isna(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, float) and pd.isna(obj):
        return None
    return obj
