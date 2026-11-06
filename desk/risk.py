"""Desk-wide sizing rules and the open-risk book across every venue.

Both executors size new trades against the combined equity of all desk accounts and
refuse new risk once combined open risk reaches the aggregate cap. Each position's risk
is its ATR stop distance plus round-trip costs (and, on HL, a 5-day adverse
funding reserve), valued in USD.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from desk.accounts import Account, accounts_for, load_accounts
from desk.venues.base import VenueAccount, VenuePosition

# Sizing rules shared by every venue's executor.
RESERVE_PCT = 0.20
RISK_PCT_OF_USABLE_MARGIN = 0.05
MAX_TOTAL_RISK_PCT = 0.20
MAX_RISK_PCT_OVERRIDE_LIMIT = 50.0
STOP_ATR_MULTIPLE = 1.25
REWARD_RISK_RATIO = 1.0
ATR_PERIOD = 14
# Without usable price history, a position is assumed able to lose this share of its value.
FALLBACK_RISK_PCT = 0.10
EXECUTION_DIR = Path(__file__).resolve().parent.parent / "execution"


def per_position_budget(equity: float) -> float:
    """Risk budget for one new position: 5% of the 80% usable part of desk equity."""
    return equity * (1 - RESERVE_PCT) * RISK_PCT_OF_USABLE_MARGIN


def estimated_commission_usd(asset_class: str, quantity: float, price: float,
                             multiplier: float = 1.0, fx_to_usd: float = 1.0) -> float:
    """Rough IB Pro commission for one side of a trade, in USD.

    Stocks: $0.005/share, at least $1 and at most 1% of value. Futures: $2.50/contract.
    Others: 2 bp of notional, at least $2. The executor replaces this with IB's own
    what-if estimate when TWS provides one.
    """
    quantity = abs(quantity)
    notional = quantity * price * multiplier * fx_to_usd
    if asset_class == "equity":
        return min(max(1.0, 0.005 * quantity * fx_to_usd), 0.01 * notional) if notional else 0.0
    if asset_class in {"futures", "options"}:
        return 2.5 * quantity * fx_to_usd
    return max(2.0, 0.0002 * notional) if notional else 0.0


def ib_position_risk(position: VenuePosition, atr: float | None) -> tuple[float, str]:
    """USD risk of an IB position: ATR stop plus a round-trip commission reserve."""
    units = abs(position.quantity) * position.multiplier * position.fx_to_usd
    price = position.mark_price if math.isfinite(position.mark_price) else position.avg_price
    commissions = 2 * estimated_commission_usd(position.asset_class, position.quantity, price,
                                               position.multiplier, position.fx_to_usd)
    if atr is not None and math.isfinite(atr) and atr > 0:
        return units * atr * STOP_ATR_MULTIPLE + commissions, f"{STOP_ATR_MULTIPLE:g} x ATR({ATR_PERIOD})"
    return units * price * FALLBACK_RISK_PCT + commissions, \
        f"no price history: assumed {FALLBACK_RISK_PCT:.0%} move"


@dataclass(frozen=True)
class BookEntry:
    """Equity and estimated open risk for one venue account (or a group of HL wallets)."""

    venue: str
    label: str
    equity_usd: float
    risk_usd: float
    margin_used_usd: float = 0.0
    positions: int = 0
    notes: tuple[str, ...] = ()
    position_risks: tuple[tuple[str, float, str], ...] = ()


@dataclass
class DeskBook:
    """Open risk across the desk; ``unreachable`` lists accounts that could not be read."""

    entries: list[BookEntry] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)

    @property
    def equity_usd(self) -> float:
        return sum(entry.equity_usd for entry in self.entries)

    @property
    def risk_usd(self) -> float:
        return sum(entry.risk_usd for entry in self.entries)


def ib_book_entry(account: Account, ib=None) -> tuple[BookEntry, VenueAccount]:
    """Read an IB account and estimate each position's risk from IB daily bars."""
    from desk.hl.trade_analytics import wilder_atr
    from desk.venues import ib as ib_venue

    own = ib is None
    ib = ib_venue.connect(account, "risk", readonly=True) if own else ib
    try:
        snapshot = ib_venue.snapshot_from_ib(ib, account)
        from ib_async import Contract

        position_risks = []
        for position in snapshot.positions:
            atr, note = None, ""
            try:
                contract = Contract(conId=int(position.ref))
                ib.qualifyContracts(contract)
                atr = wilder_atr(ib_venue.daily_bars(ib, contract, "3 M"), ATR_PERIOD)
            except Exception as exc:  # no history or a request timeout: fall back, never fail
                note = str(exc) or type(exc).__name__
            risk, basis = ib_position_risk(position, atr)
            position_risks.append((position.symbol, risk, basis if atr else f"{basis} ({note})"))
    finally:
        if own:
            ib.disconnect()
    return BookEntry(
        venue="ib", label=snapshot.label, equity_usd=snapshot.equity_usd,
        risk_usd=sum(risk for _, risk, _ in position_risks),
        margin_used_usd=snapshot.margin_used_usd, positions=len(snapshot.positions),
        notes=snapshot.notes, position_risks=tuple(position_risks),
    ), snapshot


def _hl_executor():
    """Import the HL executor module, which lives in the execution scripts folder."""
    if str(EXECUTION_DIR) not in sys.path:
        sys.path.insert(0, str(EXECUTION_DIR))
    import hl_trade

    return hl_trade


def hl_book_entry(wallets: list[str], label: str = "") -> BookEntry:
    """Collateral equity and executor-consistent open risk across HL wallets."""
    executor = _hl_executor()
    dexes = executor.get_dexes([])
    markets = executor.fetch_markets(dexes)
    snapshot = executor.collect_snapshot(wallets, dexes, markets)
    _, risk = executor.plan_trades([], markets, snapshot, include_external=False)
    return BookEntry(
        venue="hl",
        label=label or f"HL ({len(snapshot.risk_wallets)} wallet{'s' if len(snapshot.risk_wallets) != 1 else ''})",
        equity_usd=snapshot.equity, risk_usd=risk, margin_used_usd=snapshot.margin_used,
        positions=len(snapshot.positions),
    )


def other_venue_book(venue: str, accounts: list[Account] | None = None,
                     already_counted: tuple[str, ...] = ()) -> DeskBook:
    """Risk book of every configured account the calling executor does not already count.

    That is every account on another venue, plus any account on ``venue`` itself whose ID
    is not in ``already_counted`` (such as an HL wallet left out of ``--risk-wallets``).

    Unreachable accounts are listed in ``unreachable`` rather than raising, so the
    caller decides whether trading may continue without them.
    """
    book = DeskBook()
    if accounts is None:
        try:
            accounts = load_accounts()
        except (OSError, ValueError) as exc:  # a broken accounts.json blocks live trading, not dry runs
            book.unreachable.append(f"accounts.json could not be read: {exc}")
            return book
    # Each account is read on its own, so one failure never hides another account's risk.
    # Any error marks the account unreachable; the caller decides whether to trade on.
    counted = {item.lower() for item in already_counted}
    if venue != "ib" or counted:
        for account in accounts_for("ib", accounts):
            if venue == "ib" and account.id.lower() in counted:
                continue
            try:
                entry, _ = ib_book_entry(account)
                book.entries.append(entry)
            except Exception as exc:
                book.unreachable.append(f"{account.name}: {str(exc) or type(exc).__name__}")
    if venue != "hl" or counted:
        for account in accounts_for("hl", accounts):
            if venue == "hl" and account.id.lower() in counted:
                continue
            try:
                book.entries.append(hl_book_entry([account.id], account.name))
            except Exception as exc:
                book.unreachable.append(f"{account.name}: {type(exc).__name__}: {exc}")
    return book


def print_book(book: DeskBook, title: str = "Other desk accounts") -> None:
    """Print the external accounts that count toward the aggregate risk cap."""
    if not book.entries and not book.unreachable:
        return
    print(f"{title} (included in equity and the aggregate risk cap):")
    for entry in book.entries:
        print(f"  {entry.label}: equity ${entry.equity_usd:,.2f}, open risk ${entry.risk_usd:,.2f} "
              f"across {entry.positions} position{'s' if entry.positions != 1 else ''}")
        for note in entry.notes:
            print(f"    note: {note}")
        for symbol, risk, basis in entry.position_risks:
            if "no price history" in basis:
                print(f"    {symbol}: ${risk:,.2f} ({basis})")
    for problem in book.unreachable:
        print(f"  UNREACHABLE {problem}")
