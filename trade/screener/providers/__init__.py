"""Price providers."""

from __future__ import annotations

from .base import PANEL_COLUMNS, PriceProvider, empty_panel, normalise_panel
from .jquants_provider import JQuantsError, JQuantsProvider
from .stooq_provider import StooqProvider
from .synthetic_provider import SyntheticProvider
from .yfinance_provider import YFinanceProvider

PROVIDERS = {
    "jquants": JQuantsProvider,
    "yfinance": YFinanceProvider,
    "stooq": StooqProvider,
    "synthetic": SyntheticProvider,
}


def get_provider(name: str, **kwargs) -> PriceProvider:
    try:
        factory = PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown provider {name!r}; available: {', '.join(sorted(PROVIDERS))}"
        ) from None
    return factory(**kwargs)


__all__ = [
    "PANEL_COLUMNS",
    "PROVIDERS",
    "JQuantsError",
    "JQuantsProvider",
    "PriceProvider",
    "StooqProvider",
    "SyntheticProvider",
    "YFinanceProvider",
    "empty_panel",
    "get_provider",
    "normalise_panel",
]
