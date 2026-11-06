"""HL wallets as desk venue accounts (read-only, public API)."""

from __future__ import annotations

from desk.accounts import Account
from desk.hl.account_analytics import (
    as_float,
    fetch_perp_states,
    live_account_value,
    perp_market_group,
    spot_balance_usd,
)
from desk.hl.trade_analytics import is_unified_account
from desk.venues.base import VenueAccount, VenuePosition

VENUE = "hl"


def positions_from_states(wallet: str, states: dict[str, dict]) -> list[VenuePosition]:
    """Convert per-DEX clearinghouse states into venue positions."""
    positions = []
    for dex, state in states.items():
        for item in state.get("assetPositions", []):
            position = item.get("position", {})
            quantity = as_float(position.get("szi"))
            if not quantity:
                continue
            coin = str(position.get("coin", ""))
            value = abs(as_float(position.get("positionValue")))
            positions.append(VenuePosition(
                venue=VENUE,
                account=wallet,
                symbol=coin,
                asset_class="crypto" if perp_market_group(coin) == "Perp Crypto" else "tradfi perp",
                quantity=quantity,
                avg_price=as_float(position.get("entryPx")),
                mark_price=value / abs(quantity),
                unrealized_usd=as_float(position.get("unrealizedPnl")),
                margin_usd=as_float(position.get("marginUsed")),
                ref=dex,
            ))
    return positions


def snapshot(account: Account) -> VenueAccount:
    """Read one wallet's balances and perp positions from HL."""
    wallet = account.id
    states = fetch_perp_states(wallet)
    spot_total, _, unpriced = spot_balance_usd(wallet)
    perp_equity = sum(as_float((state.get("marginSummary") or {}).get("accountValue"))
                      for state in states.values())
    margin_used = sum(as_float((state.get("marginSummary") or {}).get("totalMarginUsed"))
                      for state in states.values())
    unified = is_unified_account(wallet.lower())
    equity = live_account_value(spot_total, perp_equity, unified)
    positions = positions_from_states(wallet, states)
    notes = tuple(f"{coin} spot balance has no USD price and is excluded" for coin, _ in unpriced)
    return VenueAccount(
        venue=VENUE,
        account=wallet,
        label=account.name,
        equity_usd=equity,
        # Unified accounts hold perp equity inside USDC, so cash is the spot total there.
        cash_usd=spot_total if unified else spot_total + perp_equity - sum(
            position.unrealized_usd for position in positions),
        margin_used_usd=margin_used,
        available_usd=max(0.0, equity - margin_used),
        unrealized_usd=sum(position.unrealized_usd for position in positions),
        positions=tuple(positions),
        notes=notes,
    )
