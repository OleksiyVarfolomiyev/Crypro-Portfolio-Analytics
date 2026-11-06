"""Venue-neutral account and position snapshots, all valued in USD."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

ASSET_CLASSES = ("crypto", "equity", "futures", "options", "fx", "bonds", "tradfi perp", "other")


@dataclass(frozen=True)
class VenuePosition:
    """One open position. ``quantity`` is signed (negative is short) in contract units."""

    venue: str
    account: str
    symbol: str
    asset_class: str
    quantity: float
    avg_price: float
    mark_price: float
    # Units of the underlying per contract: 1 for stocks and perps, e.g. 50 for ES futures.
    multiplier: float = 1.0
    currency: str = "USD"
    # USD per unit of ``currency``.
    fx_to_usd: float = 1.0
    unrealized_usd: float = 0.0
    margin_usd: float = float("nan")
    # Venue-specific identifier used to re-request prices (IB conId, HL DEX).
    ref: str = ""

    @property
    def side(self) -> str:
        return "LONG" if self.quantity > 0 else "SHORT"

    @property
    def notional_usd(self) -> float:
        """Signed USD exposure: positive for longs, negative for shorts."""
        return self.quantity * self.mark_price * self.multiplier * self.fx_to_usd


@dataclass(frozen=True)
class VenueAccount:
    """Balances and open positions for one venue account."""

    venue: str
    account: str
    label: str
    equity_usd: float
    cash_usd: float
    margin_used_usd: float
    available_usd: float
    unrealized_usd: float
    positions: tuple[VenuePosition, ...] = ()
    base_currency: str = "USD"
    as_of: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="UTC"))
    notes: tuple[str, ...] = ()

    @property
    def gross_usd(self) -> float:
        return sum(abs(position.notional_usd) for position in self.positions)

    @property
    def net_usd(self) -> float:
        return sum(position.notional_usd for position in self.positions)
