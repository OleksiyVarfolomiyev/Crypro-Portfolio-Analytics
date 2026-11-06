"""Guarded IB executor: dry-run by default, live requires typed confirmation.

Signals use a ``symbol,side`` CSV with optional ``sec_type,exchange,currency,expiry``
columns (defaults: STK on SMART in USD; futures need an exchange and use the front month
when no expiry is given). Each entry is a marketable limit order bracketed by a GTC
stop-loss and take-profit (one cancels the other), all held by IB, so no local process has
to keep running. New risk is sized against desk equity (IB plus every other account in
accounts.json) and capped by desk-wide open risk, like the HL executor.

TWS must be logged in with the API enabled and "Read-Only API" off for live orders.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desk import risk as desk_risk  # noqa: E402
from desk.accounts import Account, accounts_for, load_accounts  # noqa: E402
from desk.hl.trade_analytics import wilder_atr  # noqa: E402
from desk.risk import (  # noqa: E402
    ATR_PERIOD,
    MAX_RISK_PCT_OVERRIDE_LIMIT,
    RESERVE_PCT,
    REWARD_RISK_RATIO,
    STOP_ATR_MULTIPLE,
    DeskBook,
    estimated_commission_usd,
    other_venue_book,
    per_position_budget,
    print_book,
)
from desk.venues import ib as ib_venue  # noqa: E402
from desk.venues.ib import ContractSpec  # noqa: E402

# Shared with the HL executor: one kill switch halts new entries on every venue.
KILL_SWITCH_FILE = Path(__file__).resolve().parent / "KILL_SWITCH"
KILL_SWITCH_PHRASE = "SHUT IT ALL DOWN"
ENTRY_TIMEOUT_SECONDS = 60
# A marketable limit crosses the spread by this many ticks, capped at this share of price.
ENTRY_CROSS_TICKS = 2
MAX_ENTRY_SLIPPAGE_PCT = 0.005
MIN_ORDER_VALUE_USD = 10.0
# Orders may work in IB's extended sessions (US stocks: 04:00-09:30 and 16:00-20:00 ET).
# Stops then become stop-limits, the IB order type that can trigger outside regular hours,
# with the limit this many ATRs beyond the stop so a fast move can still fill.
STOP_LIMIT_ATR_FRACTION = 0.5
# IB rejects a stop-limit whose limit is "too far through" the stop: about 2% was accepted
# and 3.8% rejected on US stocks, so the gap is capped at 1.5% of the stop price.
STOP_LIMIT_MAX_PCT = 0.015
# Kill-switch closes: marketable limits this far through the price, widened each attempt.
KILL_CLOSE_SLIPPAGES = (0.02, 0.05, 0.10)
KILL_ATTEMPT_SECONDS = 20
CANCEL_WAIT_SECONDS = 10
KILL_ORDER_REF = "desk-kill"
# Re-protection stop distance when a position has no ATR.
FALLBACK_STOP_PCT = 0.10
DONE_STATUSES = {"Cancelled", "ApiCancelled", "Inactive", "Filled"}
# Extended-hours orders on by default; --rth-only turns them off for the run.
extended_hours = True
# Aggregate cap for this run; --max-risk-pct overrides it.
max_total_risk_pct = desk_risk.MAX_TOTAL_RISK_PCT


@dataclass(frozen=True)
class Signal:
    spec: ContractSpec
    side: str


@dataclass
class IBPlan:
    """A sized, risk-checked bracket entry for one contract."""

    symbol: str
    side: str
    asset_class: str
    reference: float
    stop_price: float
    target_price: float
    atr: float
    quantity: float = 0.0
    entry_limit: float = 0.0
    risk_usd: float = 0.0
    commission_usd: float = 0.0
    margin_usd: float = float("nan")
    notional_usd: float = 0.0
    multiplier: float = 1.0
    fx_to_usd: float = 1.0
    currency: str = "USD"
    tick: float = 0.01
    live_quote: bool = False
    block_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    contract: object = None


@dataclass(frozen=True)
class Sizing:
    quantity: float
    stop_price: float
    target_price: float
    entry_limit: float
    risk_usd: float
    block_reason: str = ""


def commission_per_unit_usd(asset_class: str, price: float, multiplier: float, fx_to_usd: float) -> float:
    """Marginal commission per unit (the per-order minimum is a fixed cost, not per unit)."""
    if asset_class == "equity":
        return 0.005 * fx_to_usd
    if asset_class in {"futures", "options"}:
        return 2.5 * fx_to_usd
    return 0.0002 * price * multiplier * fx_to_usd


def side_direction(side: str) -> int:
    side = side.strip().upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError(f"Side must be LONG or SHORT, not {side!r}.")
    return 1 if side == "LONG" else -1


def read_signals(path: Path) -> list[Signal]:
    """Load unique ``symbol,side`` rows (plus optional contract columns) from a CSV."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{path} has no signals.")
    signals, seen = [], set()
    for number, row in enumerate(rows, start=2):
        row = {str(key).strip().lower(): str(value or "").strip() for key, value in row.items()}
        if not row.get("symbol") or not row.get("side"):
            raise ValueError(f"{path} line {number}: symbol and side are required.")
        side_direction(row["side"])
        spec = ContractSpec(
            symbol=row["symbol"].upper(),
            sec_type=(row.get("sec_type") or "STK").upper(),
            exchange=row.get("exchange", "").upper(),
            currency=(row.get("currency") or "USD").upper(),
            expiry=row.get("expiry", ""),
        )
        key = (spec.symbol, spec.sec_type, spec.exchange, spec.expiry)
        if key in seen:
            raise ValueError(f"{path} line {number}: {spec.symbol} is listed twice.")
        seen.add(key)
        signals.append(Signal(spec, row["side"].upper()))
    return signals


def size_bracket(
    side: str,
    reference: float,
    bid: float,
    ask: float,
    atr: float,
    tick: float,
    step: float,
    multiplier: float,
    fx_to_usd: float,
    risk_budget: float,
    risk_headroom: float,
    margin_room_usd: float,
    margin_per_unit_usd: float,
    commission_per_unit_usd: float,
) -> Sizing:
    """Size a bracket entry from ATR risk, the remaining desk risk room, and margin room.

    Risk per unit is the stop distance plus round-trip commission, in USD. The stop sits
    1.25 ATR from the reference price and the target the same distance the other way.
    """
    direction = side_direction(side)
    if not (math.isfinite(atr) and atr > 0 and reference > 0):
        raise ValueError("ATR and reference price must be positive.")
    distance = atr * STOP_ATR_MULTIPLE
    stop = ib_venue.round_to_tick(reference - direction * distance, tick, "down" if direction > 0 else "up")
    target = ib_venue.round_to_tick(reference + direction * distance * REWARD_RISK_RATIO, tick,
                                "up" if direction > 0 else "down")
    if stop <= 0:
        raise ValueError("The ATR stop would be at or below zero.")
    if direction > 0:
        far_side = ask if ask > 0 else reference
        limit = min(far_side + ENTRY_CROSS_TICKS * tick, reference * (1 + MAX_ENTRY_SLIPPAGE_PCT))
        limit = ib_venue.round_to_tick(limit, tick, "up")
    else:
        far_side = bid if bid > 0 else reference
        limit = max(far_side - ENTRY_CROSS_TICKS * tick, reference * (1 - MAX_ENTRY_SLIPPAGE_PCT))
        limit = ib_venue.round_to_tick(limit, tick, "down")
    # The fill can be as far as the limit, so risk is measured from there to the stop.
    risk_per_unit = abs(limit - stop) * multiplier * fx_to_usd + 2 * commission_per_unit_usd
    room = max(0.0, min(risk_budget, risk_headroom))
    by_risk = room / risk_per_unit
    by_margin = margin_room_usd / margin_per_unit_usd if margin_per_unit_usd > 0 else by_risk
    quantity = math.floor(max(0.0, min(by_risk, by_margin)) / step + 1e-9) * step
    quantity = round(quantity, 8)
    reason = ""
    if risk_headroom <= 0:
        reason = f"desk open risk already meets or exceeds the {max_total_risk_pct * 100:g}% equity cap"
    elif margin_room_usd <= 0:
        reason = f"no IB margin remains above the {RESERVE_PCT:.0%} equity reserve"
    elif quantity <= 0:
        reason = "risk/margin-limited quantity is below one size step"
    elif quantity * reference * multiplier * fx_to_usd < MIN_ORDER_VALUE_USD:
        reason = f"order value is below ${MIN_ORDER_VALUE_USD:g}"
    return Sizing(quantity, stop, target, limit, quantity * risk_per_unit, reason)


def _what_if(ib, contract, side: str, quantity: float, limit: float, account: str):
    """IB's margin and commission estimate for an order, without sending it."""
    from ib_async import LimitOrder

    order = LimitOrder("BUY" if side == "LONG" else "SELL", quantity, limit, account=account, tif="DAY")
    return ib.whatIfOrder(contract, order)


def _usd(value, rate: float) -> float:
    return float(value) * rate if ib_venue.is_set(value) else float("nan")


def plan_ib_trades(ib, account: Account, signals: list[Signal], external: DeskBook) -> tuple[list[IBPlan], dict]:
    """Price, size, and IB-check every signal against desk risk and IB margin."""
    entry, snapshot = desk_risk.ib_book_entry(account, ib)
    acct = snapshot.account
    base, usd_per = ib_venue.fx_rates_to_usd(ib.accountValues(acct))
    desk_equity = snapshot.equity_usd + external.equity_usd
    current_risk = entry.risk_usd + external.risk_usd
    headroom = desk_equity * max_total_risk_pct - current_risk
    margin_room = max(0.0, min(snapshot.available_usd,
                               snapshot.equity_usd * (1 - RESERVE_PCT) - snapshot.margin_used_usd))
    budget = per_position_budget(desk_equity)
    summary = {
        "account": acct, "base": base, "ib_equity": snapshot.equity_usd, "ib_risk": entry.risk_usd,
        "ib_margin_used": snapshot.margin_used_usd, "desk_equity": desk_equity,
        "current_risk": current_risk, "risk_cap": desk_equity * max_total_risk_pct,
        "margin_room": margin_room, "budget": budget, "position_risks": entry.position_risks,
        "external": external,
    }
    held = {position.ref: position for position in snapshot.positions}
    plans: list[IBPlan] = []
    for signal in signals:
        try:
            contract, details = ib_venue.resolve_contract(ib, signal.spec)
            klass = ib_venue.asset_class(contract)
            mult = ib_venue.multiplier(contract)
            fx = usd_per.get(contract.currency)
            if fx is None:
                raise ValueError(f"no {contract.currency} exchange rate in the IB account")
            bars = ib_venue.daily_bars(ib, contract)
            atr = wilder_atr(bars, ATR_PERIOD)
            quote = ib_venue.quote(ib, contract)
            reference = quote.reference
            closed_price = not reference > 0
            if closed_price:
                # No quote at all (market closed or no data subscription): plan from the last
                # completed daily close. A live entry still requires a real-time quote.
                reference = float(bars["close"].iloc[-1]) if not bars.empty else 0.0
            if not reference > 0:
                raise ValueError("IB returned no price (no market data and no prior close)")
        except (ValueError, TypeError) as exc:
            plans.append(IBPlan(signal.spec.symbol, signal.side, "", 0, 0, 0, 0, block_reason=str(exc)))
            continue
        tick = float(details.minTick) if ib_venue.is_set(details.minTick) and details.minTick > 0 else 0.01
        # IB's API rejects fractional stock orders even where the desktop app allows them.
        step = max(1.0, ib_venue.size_step(details)) if klass == "equity" else ib_venue.size_step(details)
        position = held.get(str(contract.conId))
        plan = IBPlan(
            symbol=ib_venue.contract_symbol(contract), side=signal.side, asset_class=klass,
            reference=reference, stop_price=0, target_price=0, atr=atr, multiplier=mult,
            fx_to_usd=fx, currency=contract.currency, tick=tick, live_quote=quote.live, contract=contract,
        )
        if position is not None and (position.quantity > 0) != (signal.side == "LONG"):
            plan.block_reason = f"held {position.side.lower()} {abs(position.quantity):g}; close it before reversing"
            plans.append(plan)
            continue
        if closed_price:
            plan.warnings.append(f"no quote from IB; planned from the last daily close "
                                 f"({bars['date'].iloc[-1]:%Y-%m-%d}). A live run needs a real-time quote")
        elif not quote.live:
            plan.warnings.append("quote is delayed or from the prior close; a live run needs real-time data")
        commission_unit = commission_per_unit_usd(klass, reference, mult, fx)
        # First pass: size by risk alone, then let IB's what-if price margin and commission.
        sizing = size_bracket(signal.side, reference, quote.bid, quote.ask, atr, tick, step, mult, fx,
                              budget, headroom, margin_room, 0.0, commission_unit)
        margin_unit = float("nan")
        if sizing.quantity > 0:
            state = _what_if(ib, contract, signal.side, sizing.quantity, sizing.entry_limit, acct)
            init_change = _usd(getattr(state, "initMarginChange", None), usd_per[base])
            commission = getattr(state, "commission", None)
            if math.isfinite(init_change):
                margin_unit = max(0.0, init_change) / sizing.quantity
            else:
                errors = "; ".join(getattr(ib, "desk_errors", [])[-2:])
                plan.warnings.append("IB what-if returned no margin estimate" + (f" ({errors})" if errors else ""))
            if ib_venue.is_set(commission):
                rate = usd_per.get(getattr(state, "commissionCurrency", "") or base, usd_per[base])
                commission_unit = max(commission_unit, float(commission) * rate / sizing.quantity)
            if getattr(state, "warningText", ""):
                plan.warnings.append(f"IB: {state.warningText}")
            if not math.isfinite(margin_unit) or margin_unit <= 0:
                # No usable estimate (cash accounts report a zero margin change): assume the
                # full purchase cost, which is what a cash account must pay.
                margin_unit = reference * mult * fx
                plan.warnings.append("IB gave no margin estimate; full position value assumed")
            sizing = size_bracket(signal.side, reference, quote.bid, quote.ask, atr, tick, step, mult, fx,
                                  budget, headroom, margin_room, margin_unit, commission_unit)
        plan.quantity = sizing.quantity
        plan.stop_price, plan.target_price, plan.entry_limit = sizing.stop_price, sizing.target_price, sizing.entry_limit
        plan.risk_usd = sizing.risk_usd
        plan.commission_usd = 2 * commission_unit * sizing.quantity
        plan.margin_usd = margin_unit * sizing.quantity if math.isfinite(margin_unit) else float("nan")
        plan.notional_usd = sizing.quantity * reference * mult * fx
        plan.block_reason = sizing.block_reason
        plans.append(plan)
        if plan.quantity > 0 and not plan.block_reason:
            headroom -= plan.risk_usd
            margin_room -= plan.margin_usd if math.isfinite(plan.margin_usd) else 0.0
    return plans, summary


def print_plans(plans: list[IBPlan], summary: dict) -> None:
    print(f"IB account: {summary['account']} (base currency {summary['base']})")
    print(f"IB net liquidation: ${summary['ib_equity']:,.2f}; initial margin used ${summary['ib_margin_used']:,.2f}")
    print_book(summary["external"])
    print(f"Desk equity (all venues): ${summary['desk_equity']:,.2f}")
    print(f"Estimated desk open risk: ${summary['current_risk']:,.2f} "
          f"({summary['current_risk'] / summary['desk_equity']:.2%} of desk equity; IB ${summary['ib_risk']:,.2f})")
    for symbol, risk, basis in summary["position_risks"]:
        if "no price history" in basis:
            print(f"  IB {symbol}: ${risk:,.2f} ({basis})")
    print(f"Aggregate risk cap: ${summary['risk_cap']:,.2f} ({max_total_risk_pct * 100:g}% of desk equity)")
    print(f"Per-position risk budget: ${summary['budget']:,.2f} (5% of the 80% usable-equity amount)")
    print(f"IB margin room above its {RESERVE_PCT:.0%} reserve: ${summary['margin_room']:,.2f}")
    for plan in plans:
        if not plan.reference:
            print(f"{plan.symbol} {plan.side}: BLOCKED: {plan.block_reason}")
            continue
        money = "" if plan.currency == "USD" else f" {plan.currency}"
        print(
            f"{plan.symbol} {plan.side} ({plan.asset_class}): quantity={plan.quantity:g}, "
            f"reference={plan.reference:,.6g}{money}, entry limit={plan.entry_limit:,.6g}, "
            f"stop={plan.stop_price:,.6g}, target={plan.target_price:,.6g}, ATR={plan.atr:,.4g}, "
            f"value=${plan.notional_usd:,.2f}, risk=${plan.risk_usd:,.2f}, "
            f"est. round-trip commission=${plan.commission_usd:,.2f}, "
            f"initial margin={'n/a' if not math.isfinite(plan.margin_usd) else f'${plan.margin_usd:,.2f}'}"
            + (f" — BLOCKED: {plan.block_reason}" if plan.block_reason else "")
        )
        for warning in plan.warnings:
            print(f"  note: {warning}")


def kill_switch_engaged() -> bool:
    return KILL_SWITCH_FILE.exists()


def stop_limit_price(exit_action: str, stop: float, atr: float, tick: float) -> float:
    """Limit for an extended-hours stop-limit: half an ATR beyond the stop, at most 1.5% of it."""
    offset = atr * STOP_LIMIT_ATR_FRACTION if math.isfinite(atr) and atr > 0 else stop * 0.01
    offset = max(min(offset, stop * STOP_LIMIT_MAX_PCT), tick)
    if exit_action == "SELL":
        return ib_venue.round_to_tick(max(tick, stop - offset), tick, "down")
    return ib_venue.round_to_tick(stop + offset, tick, "up")


def exit_stop_order(exit_action: str, quantity: float, stop: float, atr: float, tick: float, account: str):
    """Protective GTC stop: a stop-limit that also works outside regular hours, or a plain
    stop that triggers only in regular hours under --rth-only."""
    from ib_async import StopLimitOrder, StopOrder

    if not extended_hours:
        return StopOrder(exit_action, quantity, stop, account=account, tif="GTC", orderRef=PROTECT_ORDER_REF)
    return StopLimitOrder(exit_action, quantity, stop_limit_price(exit_action, stop, atr, tick), stop,
                          account=account, tif="GTC", outsideRth=True, orderRef=PROTECT_ORDER_REF)


def exit_target_order(exit_action: str, quantity: float, price: float, account: str):
    """GTC take-profit limit, working in extended sessions unless --rth-only."""
    from ib_async import LimitOrder

    return LimitOrder(exit_action, quantity, price, account=account, tif="GTC",
                      outsideRth=extended_hours, orderRef=PROTECT_ORDER_REF)


def _wait_for_parent(ib, trade, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not trade.isDone():
        if kill_switch_engaged():
            print("Kill switch engaged while waiting; cancelling the entry.")
            break
        ib.sleep(1)


def submit_bracket(ib, plan: IBPlan, account: str) -> str:
    """Send one bracket, wait for the entry, and tidy partial fills. Returns a status line."""
    fresh = ib_venue.quote(ib, plan.contract)
    if not fresh.live:
        return f"{plan.symbol}: skipped, no real-time quote (market closed or no live data subscription)."
    direction = side_direction(plan.side)
    if (fresh.reference - plan.stop_price) * direction <= 0:
        return f"{plan.symbol}: skipped, the price {fresh.reference:,.6g} is already through the stop."
    if direction * (fresh.reference - plan.reference) > plan.atr * 0.5:
        return f"{plan.symbol}: skipped, the price moved more than half an ATR since the plan."
    action = "BUY" if direction > 0 else "SELL"
    far = fresh.ask if direction > 0 else fresh.bid
    limit = plan.entry_limit if far <= 0 else ib_venue.round_to_tick(
        far + direction * ENTRY_CROSS_TICKS * plan.tick, plan.tick, "up" if direction > 0 else "down")
    # Never pay through the planned limit: it is what the risk was sized from.
    limit = min(limit, plan.entry_limit) if direction > 0 else max(limit, plan.entry_limit)
    from ib_async import LimitOrder

    exit_action = "SELL" if direction > 0 else "BUY"
    # Entry (DAY) bracketed by a GTC target and stop that cancel each other, all able to work
    # in extended sessions. Only the last child transmits, so IB activates the three together.
    parent = LimitOrder(action, plan.quantity, limit, orderId=ib.client.getReqId(), transmit=False,
                        tif="DAY", outsideRth=extended_hours, account=account)
    take_profit = exit_target_order(exit_action, plan.quantity, plan.target_price, account)
    stop_loss = exit_stop_order(exit_action, plan.quantity, plan.stop_price, plan.atr, plan.tick, account)
    for child in (take_profit, stop_loss):
        child.orderId, child.parentId, child.transmit = ib.client.getReqId(), parent.orderId, False
    stop_loss.transmit = True
    trades = [ib.placeOrder(plan.contract, order) for order in (parent, take_profit, stop_loss)]
    parent, take_profit, stop_loss = trades
    _wait_for_parent(ib, parent, ENTRY_TIMEOUT_SECONDS)
    filled = parent.orderStatus.filled
    if not parent.isDone():
        ib.cancelOrder(parent.order)
        ib.sleep(2)
        filled = parent.orderStatus.filled
    if filled <= 0:
        reasons = "; ".join(entry.message for entry in parent.log if entry.message)
        for child in (take_profit, stop_loss):
            if not child.isDone():
                ib.cancelOrder(child.order)
        return f"{plan.symbol}: entry not filled ({parent.orderStatus.status}){': ' + reasons if reasons else ''}."
    if filled < plan.quantity:
        for child in (take_profit, stop_loss):
            child.order.totalQuantity = filled
            ib.placeOrder(plan.contract, child.order)
        ib.sleep(1)
    price = parent.orderStatus.avgFillPrice
    return (f"{plan.symbol}: filled {filled:g} at {price:,.6g}; GTC stop {plan.stop_price:,.6g} and "
            f"take-profit {plan.target_price:,.6g} are held by IB as one-cancels-other.")


# ---------------------------------------------------------------------------
# Protection for positions opened outside the executor (--protect-open)
# ---------------------------------------------------------------------------

STOP_ORDER_TYPES = {"STP", "STP LMT", "TRAIL", "TRAIL LIMIT"}
PROTECT_ORDER_REF = "desk-protect"


@dataclass
class ProtectionPlan:
    """Missing stop/target quantities and levels for one open IB position."""

    symbol: str
    con_id: int
    side: str
    position_quantity: float
    protectable_quantity: float
    mark: float
    atr: float = float("nan")
    tick: float = 0.01
    stop_price: float = float("nan")
    target_price: float = float("nan")
    existing_stop_quantity: float = 0.0
    existing_target_quantity: float = 0.0
    stop_quantity_to_add: float = 0.0
    target_quantity_to_add: float = 0.0
    risk_usd: float = 0.0
    fx_to_usd: float = 1.0
    multiplier: float = 1.0
    block_reason: str = ""
    notes: list[str] = field(default_factory=list)
    contract: object = None
    # Desk-placed exit orders to cancel before placing the new ones (--replace).
    replace_trades: list = field(default_factory=list)
    # Ladder steps (quantity, stop, target), nearest first; one step is a single stop/target.
    ladder: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def exit_action(self) -> str:
        return "SELL" if self.side == "LONG" else "BUY"


def _remaining(trade) -> float:
    return max(0.0, float(trade.order.totalQuantity) - float(trade.orderStatus.filled or 0))


def exit_trades(trades, con_id: int, exit_action: str, mark: float, direction: int) -> tuple[list, list]:
    """Active resting stops and profit-side limits that would close this position."""
    stops, targets = [], []
    for trade in trades:
        order = trade.order
        if getattr(trade.contract, "conId", None) != con_id or order.action != exit_action:
            continue
        if trade.orderStatus.status in DONE_STATUSES:
            continue
        if order.orderType in STOP_ORDER_TYPES:
            stops.append(trade)
        elif order.orderType == "LMT" and ib_venue.is_set(order.lmtPrice) \
                and direction * (float(order.lmtPrice) - mark) > 0:
            targets.append(trade)
    return stops, targets


def existing_protection(trades, con_id: int, exit_action: str, mark: float, direction: int) -> tuple[float, float]:
    """Remaining quantity of resting stops and profit-side limits that close this position."""
    stops, targets = exit_trades(trades, con_id, exit_action, mark, direction)
    return sum(map(_remaining, stops)), sum(map(_remaining, targets))


def ladder_steps(quantity: float, step: float, mark: float, distance: float, direction: int, tick: float,
                 steps: int, multiplier: float = 1.0, fx_to_usd: float = 1.0) -> list[tuple[float, float, float]]:
    """Split ``quantity`` into up to ``steps`` (quantity, stop, target) levels, nearest first.

    Level k of n sits k/n of the full ATR distance from the mark, like the HL ladder. Steps
    are reduced until every part is a whole size step worth at least the minimum order
    value and every level is a distinct price on the right side of the mark.
    """
    units = int(round(quantity / step))
    for count in range(min(steps, units), 0, -1):
        base, extra = divmod(units, count)
        sizes = [(base + (index < extra)) * step for index in range(count)]
        stops = [ib_venue.round_to_tick(mark - direction * distance * (index + 1) / count, tick,
                                        "down" if direction > 0 else "up") for index in range(count)]
        targets = [ib_venue.round_to_tick(mark + direction * distance * (index + 1) / count, tick,
                                          "up" if direction > 0 else "down") for index in range(count)]
        distinct = len(set(stops)) == count and len(set(targets)) == count
        sides_ok = all(direction * (price - mark) < 0 and price > 0 for price in stops) and \
            all(direction * (price - mark) > 0 for price in targets)
        big_enough = all(size * price * multiplier * fx_to_usd >= MIN_ORDER_VALUE_USD
                         for size, price in zip(sizes, stops))
        if count == 1 or (distinct and sides_ok and big_enough):
            return [(round(size, 8), stop, target) for size, stop, target in zip(sizes, stops, targets)]
    return []


def plan_protection(
    symbol: str, con_id: int, quantity: float, mark: float, atr: float, tick: float, step: float,
    trades, with_targets: bool = True, multiplier: float = 1.0, fx_to_usd: float = 1.0, steps: int = 1,
    replace: bool = False,
) -> ProtectionPlan:
    """Stop (and target) orders that would cover the uncovered part of one position.

    Levels sit 1.25 x ATR from the current mark, the same rule as new entries. Only whole
    size steps can be protected: IB's API rejects fractional orders, so a fractional
    remainder is left uncovered and reported.
    """
    side = "LONG" if quantity > 0 else "SHORT"
    direction = 1 if quantity > 0 else -1
    protectable = math.floor(abs(quantity) / step + 1e-9) * step
    plan = ProtectionPlan(symbol, con_id, side, quantity, round(protectable, 8), mark, atr, tick,
                          multiplier=multiplier, fx_to_usd=fx_to_usd)
    if abs(quantity) - protectable > 1e-9:
        plan.notes.append(f"{abs(quantity) - protectable:g} fractional share(s) cannot be protected through "
                          "IB's API; set that part in TWS")
    if protectable <= 0:
        plan.block_reason = "whole position is fractional; IB's API cannot place fractional orders"
        return plan
    if not (math.isfinite(mark) and mark > 0):
        plan.block_reason = "no current mark price"
        return plan
    if not (math.isfinite(atr) and atr > 0):
        plan.block_reason = "no ATR (no daily price history)"
        return plan
    distance = atr * STOP_ATR_MULTIPLE
    plan.stop_price = ib_venue.round_to_tick(mark - direction * distance, tick, "down" if direction > 0 else "up")
    plan.target_price = ib_venue.round_to_tick(mark + direction * distance * REWARD_RISK_RATIO, tick,
                                               "up" if direction > 0 else "down")
    if plan.stop_price <= 0:
        plan.block_reason = "the ATR stop would be at or below zero"
        return plan
    stops, targets = exit_trades(trades, con_id, plan.exit_action, mark, direction)
    if replace and (stops or targets):
        if all(getattr(trade.order, "orderRef", "") == PROTECT_ORDER_REF for trade in stops + targets):
            plan.replace_trades = stops + targets  # cancelled first, then re-placed
            trades = [trade for trade in trades if trade not in plan.replace_trades]
        else:
            plan.notes.append("--replace leaves it alone: some exit orders were not placed by the desk")
    plan.existing_stop_quantity, plan.existing_target_quantity = existing_protection(
        trades, con_id, plan.exit_action, mark, direction)
    plan.risk_usd = abs(mark - plan.stop_price) * protectable * multiplier * fx_to_usd
    existing = plan.existing_stop_quantity + plan.existing_target_quantity
    if plan.existing_stop_quantity >= protectable and (not with_targets or plan.existing_target_quantity >= protectable):
        return plan  # already protected; nothing to add
    if existing > 0:
        # IB stock orders have no reduce-only flag: a new order not linked to the existing
        # ones could fill alongside them and sell more than is held.
        plan.block_reason = (f"existing exit orders (stop {plan.existing_stop_quantity:g}, target "
                             f"{plan.existing_target_quantity:g}) cover part of it; adding unlinked orders could "
                             "sell more than is held, so adjust it in TWS")
        return plan
    plan.ladder = ladder_steps(protectable, step, mark, distance, direction, tick, steps, multiplier, fx_to_usd)
    if steps > 1 and len(plan.ladder) < steps:
        plan.notes.append(f"{len(plan.ladder)} step(s) instead of {steps}: the position is too small to split further")
    plan.stop_quantity_to_add = protectable
    if with_targets:
        plan.target_quantity_to_add = protectable
    # Risk to the ladder: each part to its own stop.
    plan.risk_usd = sum(abs(mark - stop) * size for size, stop, _ in plan.ladder) * multiplier * fx_to_usd
    return plan


def collect_protection_plans(ib, account: str, with_targets: bool, steps: int = 1,
                             replace: bool = False, symbols: set[str] | None = None) -> list[ProtectionPlan]:
    """Read positions and open orders, then plan the missing protection for each position."""
    from ib_async import Contract

    ib.reqAllOpenOrders()
    ib.sleep(1)
    trades = ib.openTrades()
    _, usd_per = ib_venue.fx_rates_to_usd(ib.accountValues(account))
    plans = []
    for item in ib.portfolio(account):
        if not item.position:
            continue
        if symbols and ib_venue.contract_symbol(item.contract).upper() not in symbols:
            continue
        contract = Contract(conId=item.contract.conId)
        atr, tick, step = float("nan"), 0.01, 1.0
        try:
            ib.qualifyContracts(contract)
            details = ib.reqContractDetails(contract)[0]
            tick = float(details.minTick) if ib_venue.is_set(details.minTick) and details.minTick > 0 else 0.01
            klass = ib_venue.asset_class(contract)
            step = max(1.0, ib_venue.size_step(details)) if klass == "equity" else ib_venue.size_step(details)
            atr = wilder_atr(ib_venue.daily_bars(ib, contract, "3 M"), ATR_PERIOD)
        except Exception as exc:  # the plan below reports the missing ATR; keep the others going
            print(f"{ib_venue.contract_symbol(item.contract)}: {str(exc) or type(exc).__name__}")
        if contract.secType == "STK":
            contract.exchange = "SMART"  # route exits through SMART, not the listing exchange
        mark = float(item.marketPrice) if ib_venue.is_set(item.marketPrice) else float("nan")
        plan = plan_protection(
            ib_venue.contract_symbol(item.contract), item.contract.conId, float(item.position), mark, atr,
            tick, step, trades, with_targets, ib_venue.multiplier(item.contract),
            usd_per.get(item.contract.currency, 1.0), steps, replace,
        )
        plan.contract = contract
        plans.append(plan)
    return plans


def print_protection_plans(plans: list[ProtectionPlan], with_targets: bool) -> None:
    if not plans:
        print("No open IB positions.")
        return
    print(f"Stops sit {STOP_ATR_MULTIPLE:g} x ATR({ATR_PERIOD}) from the current mark"
          + (", targets the same distance the other way (one cancels the other)." if with_targets else "."))
    for plan in plans:
        head = f"{plan.symbol} {plan.side} {abs(plan.position_quantity):g}"
        if plan.block_reason:
            print(f"{head}: BLOCKED: {plan.block_reason}")
        else:
            parts = [f"mark {plan.mark:,.4g}", f"ATR {plan.atr:,.4g}"]
            def stop_text(stop):
                if not extended_hours:
                    return f"stop {stop:,.4g}"
                return f"stop {stop:,.4g} (limit {stop_limit_price(plan.exit_action, stop, plan.atr, plan.tick):,.4g})"
            if plan.ladder and len(plan.ladder) > 1:
                parts.append("ladder " + " | ".join(
                    f"{size:g} @ {stop_text(stop)}" + (f" / target {target:,.4g}" if with_targets else "")
                    for size, stop, target in plan.ladder))
            else:
                parts.append(f"{stop_text(plan.stop_price)} {(plan.stop_price / plan.mark - 1):+.1%}")
                if with_targets:
                    parts.append(f"target {plan.target_price:,.4g} ({(plan.target_price / plan.mark - 1):+.1%})")
            parts.append(f"risk to stop ${plan.risk_usd:,.2f}")
            covered = f"already covered: stop {plan.existing_stop_quantity:g}, target {plan.existing_target_quantity:g}"
            adds = f"add stop {plan.stop_quantity_to_add:g}" + (
                f", target {plan.target_quantity_to_add:g}" if with_targets else "")
            replacing = f"; replace {len(plan.replace_trades)} desk order(s)" if plan.replace_trades else ""
            print(f"{head}: {', '.join(parts)}; {covered}; {adds}{replacing}")
        for note in plan.notes:
            print(f"  note: {note}")
    if extended_hours:
        print("Stops are GTC stop-limits that also work in pre-market and after-hours (limit half an ATR, at most 1.5%, "
              "past the stop); a gap past the limit leaves the order resting unfilled. Targets work in "
              "extended hours too.")
    else:
        print("--rth-only: stops trigger in regular hours only, so an opening gap can fill below the stop.")


def has_protection_to_add(plans: list[ProtectionPlan]) -> bool:
    return any(not plan.block_reason and (plan.stop_quantity_to_add > 0 or plan.target_quantity_to_add > 0)
               for plan in plans)


def _settle(ib, trades, seconds: int = 10) -> None:
    """Give IB a few seconds to accept or reject newly placed orders."""
    for _ in range(seconds):
        ib.sleep(1)
        if all(trade.orderStatus.status in {"Submitted", "PreSubmitted", "Filled", "Cancelled", "ApiCancelled", "Inactive"}
               for trade in trades):
            return


def _wait_until_done(ib, trades, seconds: float) -> bool:
    """Wait for orders to reach a final state; return whether all did."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if all(trade.orderStatus.status in DONE_STATUSES for trade in trades):
            return True
        ib.sleep(0.5)
    return all(trade.orderStatus.status in DONE_STATUSES for trade in trades)


def submit_protection(ib, plan: ProtectionPlan, account: str) -> str:
    """Place each ladder step's GTC stop and target, linked one-cancels-other per step.

    Linking each step's pair means at most one of them fills, so the stops and targets
    together can never sell more than the position (IB stocks have no reduce-only flag).
    With --replace, the desk's old exit orders are cancelled first and the new ones are
    sent only once IB confirms the cancellations (a cash account treats overlapping sell
    orders as a short sale).
    """
    if plan.replace_trades:
        for trade in plan.replace_trades:
            ib.cancelOrder(trade.order)
        if not _wait_until_done(ib, plan.replace_trades, CANCEL_WAIT_SECONDS):
            return f"{plan.symbol}: the old exit orders did not confirm cancelled; nothing new was placed."
    from ib_async import StopOrder

    trades = []
    pairs = []  # (stop trade, target trade or None, group)
    stamp = int(time.time())
    with_targets = plan.target_quantity_to_add > 0
    for index, (size, stop, target) in enumerate(plan.ladder or [(plan.protectable_quantity, plan.stop_price,
                                                                   plan.target_price)]):
        group = f"{PROTECT_ORDER_REF}-{plan.con_id}-{stamp}-{index + 1}"
        orders = [exit_stop_order(plan.exit_action, size, stop, plan.atr, plan.tick, account)]
        if with_targets:
            orders.append(exit_target_order(plan.exit_action, size, target, account))
            for order in orders:
                order.ocaGroup, order.ocaType = group, 1  # a fill cancels its partner
        placed = [ib.placeOrder(plan.contract, order) for order in orders]
        trades.extend(placed)
        pairs.append((placed[0], placed[1] if len(placed) > 1 else None, group))
    _settle(ib, trades)
    # Safety net: a rejected stop must never leave its target (or the position) without one.
    lines = []
    for stop_trade, target_trade, group in pairs:
        if stop_trade.orderStatus.status not in {"Cancelled", "ApiCancelled", "Inactive"}:
            continue
        old = stop_trade.order
        retry = StopOrder(old.action, old.totalQuantity, old.auxPrice, account=account, tif="GTC",
                          orderRef=PROTECT_ORDER_REF)
        if target_trade is not None:
            retry.ocaGroup, retry.ocaType = group, 1
        retry_trade = ib.placeOrder(plan.contract, retry)
        trades.append(retry_trade)
        _settle(ib, [retry_trade])
        if retry_trade.orderStatus.status in {"Cancelled", "ApiCancelled", "Inactive"}:
            if target_trade is not None and target_trade.orderStatus.status not in DONE_STATUSES:
                ib.cancelOrder(target_trade.order)
                _wait_until_done(ib, [target_trade], CANCEL_WAIT_SECONDS)
            lines.append(f"{plan.symbol}: WARNING: IB rejected both stop types for {old.totalQuantity:g}; its target "
                         "was cancelled too. Set this stop in TWS.")
        else:
            lines.append(f"{plan.symbol}: stop-limit rejected, so a plain regular-hours stop at "
                         f"{old.auxPrice:,.4g} replaced it.")
    for trade in trades:
        order, status = trade.order, trade.orderStatus.status
        if order.orderType in {"STP", "STP LMT"}:
            price = f"stop {order.auxPrice:,.4g}" + (f" limit {order.lmtPrice:,.4g}" if order.orderType == "STP LMT" else "")
        else:
            price = f"{order.lmtPrice:,.4g}"
        hours = " ext hours" if getattr(order, "outsideRth", False) else ""
        problem = "; ".join(entry.message for entry in trade.log if entry.message and status in {"Cancelled", "Inactive"})
        lines.append(f"{plan.symbol}: {order.orderType} {order.action} {order.totalQuantity:g} at {price}{hours} "
                     f"-> {status}" + (f" ({problem})" if problem else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

@dataclass
class KillTarget:
    """One open position the kill switch will close, with the orders that protect it."""

    symbol: str
    contract: object
    quantity: float
    closable: float
    mark: float
    tick: float
    atr: float
    session: str
    exits: list = field(default_factory=list)

    @property
    def exit_action(self) -> str:
        return "SELL" if self.quantity > 0 else "BUY"

    @property
    def fractional(self) -> float:
        return round(abs(self.quantity) - self.closable, 8)


def close_limit_price(exit_action: str, reference: float, slippage: float, tick: float) -> float:
    """A marketable limit ``slippage`` through the reference price."""
    if exit_action == "SELL":
        return ib_venue.round_to_tick(max(tick, reference * (1 - slippage)), tick, "down")
    return ib_venue.round_to_tick(reference * (1 + slippage), tick, "up")


def kill_targets(ib, account: str) -> tuple[list[KillTarget], list]:
    """Positions to close and every other open order (entries and orphans) to cancel."""
    from ib_async import Contract

    ib.reqAllOpenOrders()
    ib.sleep(1)
    trades = [trade for trade in ib.openTrades() if trade.orderStatus.status not in DONE_STATUSES
              and getattr(trade.order, "account", "") in ("", account)]
    targets, claimed = [], set()
    for item in ib.portfolio(account):
        if not item.position:
            continue
        contract = Contract(conId=item.contract.conId)
        tick, step, atr, session = 0.01, 1.0, float("nan"), "unknown"
        try:
            ib.qualifyContracts(contract)
            details = ib.reqContractDetails(contract)[0]
            tick = float(details.minTick) if ib_venue.is_set(details.minTick) and details.minTick > 0 else 0.01
            klass = ib_venue.asset_class(contract)
            step = max(1.0, ib_venue.size_step(details)) if klass == "equity" else ib_venue.size_step(details)
            session = ib_venue.session_state(details)
            atr = wilder_atr(ib_venue.daily_bars(ib, contract, "3 M"), ATR_PERIOD)
        except Exception as exc:
            print(f"{ib_venue.contract_symbol(item.contract)}: {str(exc) or type(exc).__name__}")
        if contract.secType == "STK":
            contract.exchange = "SMART"
        quantity = float(item.position)
        exit_action = "SELL" if quantity > 0 else "BUY"
        exits = [trade for trade in trades if getattr(trade.contract, "conId", None) == item.contract.conId
                 and trade.order.action == exit_action]
        claimed.update(id(trade) for trade in exits)
        targets.append(KillTarget(
            ib_venue.contract_symbol(item.contract), contract, quantity,
            round(math.floor(abs(quantity) / step + 1e-9) * step, 8),
            float(item.marketPrice) if ib_venue.is_set(item.marketPrice) else float("nan"),
            tick, atr, session, exits,
        ))
    others = [trade for trade in trades if id(trade) not in claimed]
    return targets, others


def print_kill_switch_plan(ib, account: str) -> None:
    targets, others = kill_targets(ib, account)
    print(f"IB {account}: {len(targets)} open position(s); {len(others)} other open order(s) to cancel first.")
    for target in targets:
        if target.closable <= 0:
            print(f"  {target.symbol}: {abs(target.quantity):g} share(s), all fractional: IB's API cannot sell "
                  "them, so close it in TWS")
            continue
        line = (f"  {target.symbol}: close {target.closable:g} of {abs(target.quantity):g} with a limit "
                f"{KILL_CLOSE_SLIPPAGES[0]:.0%} through {target.mark:,.4g}, widening to "
                f"{KILL_CLOSE_SLIPPAGES[-1]:.0%}; session {target.session}; "
                f"{len(target.exits)} exit order(s) cancelled just before")
        if target.session == "closed":
            line += "; no session is open, so the close waits for the next one (04:00 ET for US stocks)"
        print(line)
        if target.fractional:
            print(f"    {target.fractional:g} fractional share(s) cannot be sold through IB's API; close them in TWS")
    print("Closes are limit orders that work in extended hours. A position whose close does not fill "
          "gets its protective stop back.")


def _reprotect(ib, target: KillTarget, quantity: float, account: str) -> str:
    """Put a protective stop back on whatever the kill switch could not close."""
    distance = target.atr * STOP_ATR_MULTIPLE if math.isfinite(target.atr) and target.atr > 0 \
        else target.mark * FALLBACK_STOP_PCT
    direction = 1 if target.quantity > 0 else -1
    stop = ib_venue.round_to_tick(target.mark - direction * distance, target.tick, "down" if direction > 0 else "up")
    trade = ib.placeOrder(target.contract, exit_stop_order(target.exit_action, quantity, stop, target.atr,
                                                           target.tick, account))
    ib.sleep(2)
    return f"re-protected {quantity:g} with a stop at {stop:,.4g} ({trade.orderStatus.status})"


def run_kill_switch(ib, account: str) -> bool:
    """Halt entries, cancel other orders, then close every whole-share position.

    A cash account rejects a sell while resting sells already claim the shares, so each
    position's exit orders are cancelled just before its close is placed. Closes are
    marketable limits that work in extended hours, widened 2% -> 5% -> 10%. A close that
    still does not fill is cancelled and the position re-protected. With no session open
    the closes are left queued (GTC) at the widest limit for the next session.
    """
    from ib_async import LimitOrder

    KILL_SWITCH_FILE.write_text(f"Engaged {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    print(f"New entries are halted on every venue until --clear-kill-switch ({KILL_SWITCH_FILE}).")
    targets, others = kill_targets(ib, account)
    ok = True
    for trade in others:
        ib.cancelOrder(trade.order)
    if others and not _wait_until_done(ib, others, CANCEL_WAIT_SECONDS):
        print("WARNING: some non-exit orders did not confirm cancelled.")
        ok = False
    closes = []
    for target in targets:
        if target.fractional:
            print(f"{target.symbol}: {target.fractional:g} fractional share(s) must be closed in TWS.")
            ok = False
        if target.closable <= 0:
            continue
        for trade in target.exits:
            ib.cancelOrder(trade.order)
        if target.exits and not _wait_until_done(ib, target.exits, CANCEL_WAIT_SECONDS):
            print(f"{target.symbol}: its exit orders did not confirm cancelled; not closing (they still protect it).")
            ok = False
            continue
        reference = target.mark
        try:
            quote = ib_venue.quote(ib, target.contract)
            if quote.live and quote.reference > 0:
                reference = quote.reference
        except Exception:
            pass
        slippage = KILL_CLOSE_SLIPPAGES[-1] if target.session == "closed" else KILL_CLOSE_SLIPPAGES[0]
        order = LimitOrder(target.exit_action, target.closable,
                           close_limit_price(target.exit_action, reference, slippage, target.tick),
                           account=account, tif="GTC", outsideRth=True, orderRef=KILL_ORDER_REF)
        closes.append((target, ib.placeOrder(target.contract, order), reference))
    live = [entry for entry in closes if entry[0].session != "closed"]
    for attempt, slippage in enumerate(KILL_CLOSE_SLIPPAGES):
        pending = [entry for entry in live if entry[1].orderStatus.status not in DONE_STATUSES]
        if not pending:
            break
        if attempt:
            for target, trade, reference in pending:
                trade.order.lmtPrice = close_limit_price(target.exit_action, reference, slippage, target.tick)
                ib.placeOrder(target.contract, trade.order)  # same order ID: a price change
        _wait_until_done(ib, [trade for _, trade, _ in pending], KILL_ATTEMPT_SECONDS)
    for target, trade, _ in closes:
        status, filled = trade.orderStatus.status, float(trade.orderStatus.filled or 0)
        if status == "Filled":
            print(f"{target.symbol}: closed {filled:g} at {trade.orderStatus.avgFillPrice:,.4g}.")
            continue
        ok = False
        if target.session == "closed":
            print(f"{target.symbol}: no session open; close for {target.closable:g} queued at "
                  f"{trade.order.lmtPrice:,.4g} for the next session (it has no stop meanwhile).")
            continue
        if status not in DONE_STATUSES:
            ib.cancelOrder(trade.order)
            _wait_until_done(ib, [trade], CANCEL_WAIT_SECONDS)
        filled = float(trade.orderStatus.filled or 0)
        remaining = round(target.closable - filled, 8)
        reason = "; ".join(entry.message for entry in trade.log if entry.message)
        message = f"{target.symbol}: close {status}, filled {filled:g} of {target.closable:g}"
        if reason:
            message += f" ({reason})"
        if remaining > 0:
            message += "; " + _reprotect(ib, target, remaining, account)
        print(message)
    return ok


def resolve_account(accounts: list[Account], wanted: str | None) -> Account:
    ib_accounts = accounts_for("ib", accounts)
    if wanted:
        for account in ib_accounts:
            if account.id == wanted or account.label == wanted:
                return account
        return Account("ib", wanted)
    if len(ib_accounts) > 1:
        raise ValueError("Several IB accounts are configured; choose one with --account.")
    return ib_accounts[0] if ib_accounts else Account("ib", "")


def main(argv: list[str] | None = None) -> int:
    global max_total_risk_pct
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--account", help="IB account number or accounts.json label (default: the configured one).")
    parser.add_argument("--signals", type=Path, help="CSV with symbol,side and optional sec_type,exchange,currency,expiry.")
    parser.add_argument("--port", type=int, help="TWS socket port (7496 live, 7497 paper); overrides accounts.json.")
    parser.add_argument("--max-risk-pct", type=float, help="Aggregate risk cap for this run only (default 20).")
    parser.add_argument("--ignore-unreachable", action="store_true",
                        help="Trade even if another desk account cannot be read; its risk is then left out.")
    parser.add_argument("--kill-switch", action="store_true",
                        help="Show (or with --live, execute) cancelling every IB order and closing every IB position.")
    parser.add_argument("--clear-kill-switch", action="store_true", help="Allow new entries again after the kill switch.")
    parser.add_argument("--protect-open", action="store_true",
                        help="Show (or with --live, place) GTC ATR stops and targets for open positions that have none.")
    parser.add_argument("--stops-only", action="store_true", help="With --protect-open: place stops without targets.")
    parser.add_argument("--ladder", type=int, default=1, metavar="STEPS",
                        help="With --protect-open: split each position into up to STEPS stop/target pairs at "
                        "k/STEPS of the ATR distance (default 1, a single pair).")
    parser.add_argument("--replace", action="store_true",
                        help="With --protect-open: cancel the desk's own existing exit orders and place fresh ones "
                        "(for example to move them to extended hours or a new ladder).")
    parser.add_argument("--symbols", help="With --protect-open: only these comma-separated symbols, e.g. NET,SWMR.")
    parser.add_argument("--rth-only", action="store_true",
                        help="Keep orders to regular trading hours (default: they also work in pre-market and "
                        "after-hours, with stops as stop-limits).")
    parser.add_argument("--live", action="store_true", help="Allow live orders after typing the confirmation phrases.")
    args = parser.parse_args(argv)
    if (args.stops_only or args.ladder != 1 or args.replace or args.symbols) and not args.protect_open:
        parser.error("--stops-only, --ladder, --replace and --symbols apply only to --protect-open.")
    symbols = {item.strip().upper() for item in args.symbols.split(",") if item.strip()} if args.symbols else None
    global extended_hours
    extended_hours = not args.rth_only
    if not 1 <= args.ladder <= 10:
        parser.error("--ladder must be between 1 and 10.")
    if args.protect_open and (args.signals or args.kill_switch):
        parser.error("--protect-open cannot be combined with --signals or --kill-switch.")
    if args.max_risk_pct is not None:
        if not 0 < args.max_risk_pct <= MAX_RISK_PCT_OVERRIDE_LIMIT:
            parser.error(f"--max-risk-pct must be above 0 and at most {MAX_RISK_PCT_OVERRIDE_LIMIT:g}.")
        max_total_risk_pct = args.max_risk_pct / 100
        print(f"NOTE: aggregate risk cap for this run is {args.max_risk_pct:g}%, not the default 20%.")
    if args.clear_kill_switch:
        KILL_SWITCH_FILE.unlink(missing_ok=True)
        print("Kill switch cleared; new entries are allowed again.")
        return 0
    try:
        accounts = load_accounts()
        account = resolve_account(accounts, args.account)
        if args.port:
            account = Account(account.venue, account.id, account.label, account.host, args.port)
        print(f"Connecting to TWS at {account.host}:{account.port} "
              f"({'LIVE account port' if account.port in (7496, 4001) else 'paper/other port'}).")
        ib = ib_venue.connect(account, "execution", readonly=not args.live)
        try:
            acct = ib_venue.account_id(ib, account)
            if args.kill_switch:
                print_kill_switch_plan(ib, acct)
                if not args.live:
                    print("Dry run only. No order was sent.")
                    return 0
                if input(f"Type {KILL_SWITCH_PHRASE} to cancel all IB orders and market-close every "
                         "IB position now: ").strip() != KILL_SWITCH_PHRASE:
                    print("Confirmation did not match; no order was sent.")
                    return 0
                return 0 if run_kill_switch(ib, acct) else 2
            if args.protect_open:
                # Protection only reduces risk, so it works even while the kill switch is engaged.
                with_targets = not args.stops_only
                plans = collect_protection_plans(ib, acct, with_targets, args.ladder, args.replace, symbols)
                print_protection_plans(plans, with_targets)
                if not has_protection_to_add(plans):
                    print("No missing protection that can be placed safely.")
                    if not args.live:
                        print("Dry run only. No order was sent.")
                    return 0
                if not args.live:
                    print("Dry run only. No order was sent.")
                    return 0
                if input("Type LIVE TRADING to review protection orders for open IB positions: ").strip() != "LIVE TRADING":
                    print("Confirmation did not match; no order was sent.")
                    return 0
                plans = collect_protection_plans(ib, acct, with_targets, args.ladder, args.replace, symbols)
                print("Refreshed protection check immediately before submission:")
                print_protection_plans(plans, with_targets)
                if not has_protection_to_add(plans):
                    print("The refreshed check found nothing to add; no order was sent.")
                    return 0
                if input("Type SUBMIT PROTECTION ORDERS to place them: ").strip() != "SUBMIT PROTECTION ORDERS":
                    print("Confirmation did not match; no order was sent.")
                    return 0
                for plan in plans:
                    if not plan.block_reason and (plan.stop_quantity_to_add or plan.target_quantity_to_add):
                        print(submit_protection(ib, plan, acct))
                return 0
            if args.signals is None:
                parser.error("--signals is required unless --protect-open, --kill-switch or --clear-kill-switch is used.")
            if args.live and kill_switch_engaged():
                print("Kill switch is engaged; no order was sent. Use --clear-kill-switch to trade again.")
                return 2
            signals = read_signals(args.signals)
            external = other_venue_book("ib", accounts)
            if external.unreachable and args.live and not args.ignore_unreachable:
                print_book(external)
                raise RuntimeError("a configured desk account could not be read, so its risk is unknown. "
                                   "Fix it, or rerun with --ignore-unreachable.")
            plans, summary = plan_ib_trades(ib, account, signals, external)
            print_plans(plans, summary)
            if not args.live:
                print("Dry run only. No order was sent.")
                return 0
            if not any(plan.quantity > 0 and not plan.block_reason for plan in plans):
                print("No live order can be sent because every plan is blocked or zero.")
                return 2
            if input("Type LIVE TRADING to review these IB orders: ").strip() != "LIVE TRADING":
                print("Confirmation did not match; no order was sent.")
                return 0
            external = other_venue_book("ib", accounts)
            plans, summary = plan_ib_trades(ib, account, signals, external)
            print("Refreshed preflight immediately before submission:")
            print_plans(plans, summary)
            sendable = [plan for plan in plans if plan.quantity > 0 and not plan.block_reason]
            if not sendable:
                print("The refreshed check blocks every order; no order was sent.")
                return 2
            if input("Type SUBMIT ORDERS to send the refreshed plan: ").strip() != "SUBMIT ORDERS":
                print("Confirmation did not match; no order was sent.")
                return 0
            for plan in sendable:
                if kill_switch_engaged():
                    print("Kill switch engaged; remaining orders were not sent.")
                    return 2
                print(submit_bracket(ib, plan, acct))
            return 0
        finally:
            ib.disconnect()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"IB execution stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
