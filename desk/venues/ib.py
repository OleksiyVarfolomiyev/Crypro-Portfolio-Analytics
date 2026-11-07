"""IB accounts through a running TWS or IB Gateway session (ib_async).

TWS must be logged in with the API enabled: File > Global Configuration > API >
Settings > "Enable ActiveX and Socket Clients". No credential passes through this code;
the TWS session is the authentication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from desk.accounts import IB_CLIENT_IDS, Account
from desk.venues.base import VenueAccount, VenuePosition

VENUE = "ib"
ACCOUNTING_TIMEZONE = "America/New_York"
# IB marks unset numeric fields with the largest double.
UNSET = 1.7976931348623157e308
# Informational TWS codes: market-data farm status and similar, not request failures.
INFO_CODES = {2103, 2104, 2105, 2106, 2107, 2108, 2119, 2158}
ASSET_CLASS_BY_SEC_TYPE = {
    "STK": "equity", "ETF": "equity", "FUT": "futures", "CONTFUT": "futures",
    "OPT": "options", "FOP": "options", "WAR": "options", "CASH": "fx",
    "CRYPTO": "crypto", "BOND": "bonds", "BILL": "bonds", "CFD": "equity",
}
# Live (1) or, without a live subscription, delayed (3) data; IB returns live data
# on type 3 whenever the account is subscribed.
MARKET_DATA_DELAYED = 3
LIVE_MARKET_DATA_TYPE = 1


class IBConnectionError(RuntimeError):
    """TWS or IB Gateway could not be reached or refused the API connection."""


def is_set(value: Any) -> bool:
    """Return whether an IB numeric field holds a real value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and abs(number) < UNSET / 2


def run_off_event_loop(function, *args, **kwargs):
    """Call ``function``, moving it to a worker thread when an event loop is running here.

    Jupyter runs each cell inside an asyncio loop, and ib_async's blocking API then fails
    with "This event loop is already running". The worker thread gets its own loop, so the
    TWS connection opens, works and closes there. Scripts (no running loop) call directly.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return function(*args, **kwargs)

    def work():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return function(*args, **kwargs)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ib") as pool:
        return pool.submit(work).result()


def client_ids(purpose: str, attempts: int = 3) -> list[int]:
    """Client IDs to try, in order.

    Order-placing connections keep their fixed ID, so their orders stay tied to one client.
    Read-only connections get fresh random IDs, so a connection a crashed notebook never
    closed cannot block the next run (TWS keeps such an ID taken until that process ends).
    """
    import random

    base = IB_CLIENT_IDS[purpose]
    if purpose == "execution":
        return [base]
    return [base * 1000 + random.randint(1, 999) for _ in range(attempts)]


def connect(account: Account, purpose: str = "overview", readonly: bool = True, timeout: float = 10):
    """Connect to TWS for one account and return the ``ib_async.IB`` client.

    Read-only connections cannot place orders. An ID TWS reports as in use (error 326)
    is retried with another, so the overview, a risk check, a notebook and the executor
    can all be connected at once.
    """
    from ib_async import IB

    ids = client_ids(purpose)
    for attempt, client_id in enumerate(ids):
        ib = IB()
        errors: list[str] = []

        def record_error(req_id, code, message, contract=None, errors=errors):
            if code not in INFO_CODES:
                errors.append(f"{code}: {message}")

        ib.errorEvent += record_error
        ib.desk_errors = errors
        try:
            ib.connect(account.host, account.port, clientId=client_id,
                       timeout=timeout, readonly=readonly, account=account.id)
            break
        except (ConnectionRefusedError, OSError, TimeoutError) as exc:
            ib.disconnect()
            in_use = any(message.startswith("326:") for message in errors)
            if in_use and attempt < len(ids) - 1:
                continue
            told = "; ".join(errors[-3:])
            if in_use:
                hint = (f"Client ID {client_id} is already connected to TWS, often from an earlier run "
                        "that did not disconnect: restart that notebook kernel or script.")
            else:
                hint = ("Check that TWS is logged in and API > Settings > 'Enable ActiveX and Socket "
                        "Clients' is on with this socket port.")
            raise IBConnectionError(
                f"Could not connect to TWS at {account.host}:{account.port} "
                f"({str(exc) or type(exc).__name__}{'; TWS said: ' + told if told else ''}). {hint}"
            ) from exc
    managed = ib.managedAccounts()
    if account.id and account.id not in managed:
        ib.disconnect()
        raise IBConnectionError(f"TWS does not manage account {account.id}; it manages {managed}.")
    ib.reqMarketDataType(MARKET_DATA_DELAYED)
    return ib


def account_id(ib, account: Account) -> str:
    """Return the configured account number, or the session's only/first account."""
    if account.id:
        return account.id
    managed = ib.managedAccounts()
    if not managed:
        raise IBConnectionError("TWS reported no managed accounts.")
    return managed[0]


def fx_rates_to_usd(account_values) -> tuple[str, dict[str, float]]:
    """Return the base currency and USD per unit of each currency IB reports.

    IB's ``ExchangeRate`` tag gives units of base currency per unit of each currency.
    """
    to_base = {}
    base = "USD"
    for item in account_values:
        if item.tag == "ExchangeRate" and is_set(item.value) and float(item.value) > 0:
            to_base[item.currency] = float(item.value)
            if float(item.value) == 1.0 and item.currency != "BASE":
                base = item.currency
    to_base.setdefault(base, 1.0)
    usd_in_base = to_base.get("USD")
    if usd_in_base is None:
        raise ValueError(f"IB reported no USD exchange rate for base currency {base}.")
    return base, {currency: rate / usd_in_base for currency, rate in to_base.items()}


def _summary_value(summary, tag: str) -> float:
    for item in summary:
        if item.tag == tag and is_set(item.value):
            return float(item.value)
    return float("nan")


def asset_class(contract) -> str:
    return ASSET_CLASS_BY_SEC_TYPE.get(getattr(contract, "secType", ""), "other")


def contract_symbol(contract) -> str:
    """Readable symbol: local symbol for futures/options, ``EUR.USD`` for FX pairs."""
    sec_type = getattr(contract, "secType", "")
    if sec_type in {"FUT", "OPT", "FOP"} and getattr(contract, "localSymbol", ""):
        return contract.localSymbol
    if sec_type == "CASH":
        return f"{contract.symbol}.{contract.currency}"
    return contract.symbol


def multiplier(contract) -> float:
    value = getattr(contract, "multiplier", "") or ""
    try:
        return float(value) if float(value) > 0 else 1.0
    except ValueError:
        return 1.0


def snapshot_from_ib(ib, account: Account) -> VenueAccount:
    """Build a venue snapshot from an already connected IB client."""
    acct = account_id(ib, account)
    values = ib.accountValues(acct)
    base, usd_per = fx_rates_to_usd(values)
    to_usd = usd_per[base]
    summary = ib.accountSummary(acct)
    equity = _summary_value(summary, "NetLiquidation")
    if not math.isfinite(equity):
        raise ValueError(f"IB account {acct} returned no NetLiquidation value.")
    positions = []
    notes = []
    for item in ib.portfolio(acct):
        if not item.position:
            continue
        contract = item.contract
        mult = multiplier(contract)
        currency = contract.currency or base
        if currency not in usd_per:
            notes.append(f"{contract_symbol(contract)}: no {currency} exchange rate; valued at 1:1")
        fx = usd_per.get(currency, 1.0)
        mark = float(item.marketPrice) if is_set(item.marketPrice) else float("nan")
        positions.append(VenuePosition(
            venue=VENUE,
            account=acct,
            symbol=contract_symbol(contract),
            asset_class=asset_class(contract),
            quantity=float(item.position),
            avg_price=float(item.averageCost) / mult if is_set(item.averageCost) else float("nan"),
            mark_price=mark,
            multiplier=mult,
            currency=currency,
            fx_to_usd=fx,
            unrealized_usd=float(item.unrealizedPNL) * fx if is_set(item.unrealizedPNL) else 0.0,
            ref=str(contract.conId),
        ))
    init_margin = _summary_value(summary, "InitMarginReq")
    available = _summary_value(summary, "AvailableFunds")
    return VenueAccount(
        venue=VENUE,
        account=acct,
        label=account.name if account.id else f"IB {acct}",
        equity_usd=equity * to_usd,
        cash_usd=_summary_value(summary, "TotalCashValue") * to_usd,
        margin_used_usd=(init_margin if math.isfinite(init_margin) else 0.0) * to_usd,
        available_usd=(available if math.isfinite(available) else 0.0) * to_usd,
        unrealized_usd=sum(position.unrealized_usd for position in positions),
        positions=tuple(positions),
        base_currency=base,
        notes=tuple(notes),
    )


def snapshot(account: Account) -> VenueAccount:
    """Connect read-only, read one IB account, and disconnect."""
    ib = connect(account, "overview", readonly=True)
    try:
        return snapshot_from_ib(ib, account)
    finally:
        ib.disconnect()


# ---------------------------------------------------------------------------
# Contracts and prices
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContractSpec:
    """What a signal asks for, before IB resolves it to one contract."""

    symbol: str
    sec_type: str = "STK"
    exchange: str = ""
    currency: str = "USD"
    expiry: str = ""


def build_contract(spec: ContractSpec):
    """Turn a spec into an unqualified ib_async contract."""
    from ib_async import Contract, Crypto, Forex, Future, Stock

    sec_type = spec.sec_type.upper()
    symbol = spec.symbol.upper()
    if sec_type in {"STK", "ETF"}:
        return Stock(symbol, spec.exchange or "SMART", spec.currency)
    if sec_type == "FUT":
        if not spec.exchange:
            raise ValueError(f"{symbol}: futures need an exchange, e.g. CME or NYMEX.")
        return Future(symbol, spec.expiry, spec.exchange, currency=spec.currency)
    if sec_type == "CASH":
        pair = symbol.replace(".", "").replace("/", "")
        if len(pair) != 6:
            raise ValueError(f"{symbol}: FX pairs look like EURUSD.")
        return Forex(pair, exchange=spec.exchange or "IDEALPRO")
    if sec_type == "CRYPTO":
        return Crypto(symbol, spec.exchange or "PAXOS", spec.currency)
    if sec_type in {"OPT", "FOP"}:
        raise ValueError(f"{symbol}: options are not supported by the executor yet.")
    return Contract(symbol=symbol, secType=sec_type, exchange=spec.exchange or "SMART",
                    currency=spec.currency)


def front_month(details, today: pd.Timestamp | None = None, min_days: int = 5):
    """Pick the nearest futures contract with at least ``min_days`` to expiry."""
    today = pd.Timestamp.now(tz=ACCOUNTING_TIMEZONE) if today is None else pd.Timestamp(today)
    if today.tzinfo is not None:
        today = today.tz_localize(None)
    today = today.normalize()
    dated = []
    for item in details:
        expiry = item.realExpirationDate or item.contract.lastTradeDateOrContractMonth
        try:
            expires = pd.Timestamp(str(expiry)[:8])
        except ValueError:
            continue
        if (expires - today).days >= min_days:
            dated.append((expires, item))
    if not dated:
        raise ValueError("No futures contract has enough time left before expiry.")
    return min(dated, key=lambda pair: pair[0])[1]


def resolve_contract(ib, spec: ContractSpec):
    """Return ``(contract, details)`` for one tradeable contract matching the spec."""
    details = ib.reqContractDetails(build_contract(spec))
    if not details:
        raise ValueError(f"IB found no contract for {spec.symbol} ({spec.sec_type}).")
    if spec.sec_type.upper() == "FUT" and not spec.expiry:
        chosen = front_month(details)
    elif len(details) > 1:
        primary = [item for item in details if item.contract.primaryExchange in
                   {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS"}]
        if len(primary) != 1:
            names = ", ".join(f"{item.contract.symbol}@{item.contract.primaryExchange}" for item in details[:5])
            raise ValueError(f"{spec.symbol} is ambiguous ({names}); give an exchange or currency.")
        chosen = primary[0]
    else:
        chosen = details[0]
    return chosen.contract, chosen


def daily_bars(ib, contract, duration: str = "1 Y") -> pd.DataFrame:
    """Completed daily bars (today's partial bar removed) with high/low/close columns."""
    what = "MIDPOINT" if contract.secType == "CASH" else "AGGTRADES" if contract.secType == "CRYPTO" else "TRADES"
    bars = ib.reqHistoricalData(contract, endDateTime="", durationStr=duration,
                                barSizeSetting="1 day", whatToShow=what, useRTH=True, formatDate=1)
    if not bars:
        errors = "; ".join(getattr(ib, "desk_errors", [])[-2:])
        raise ValueError(f"IB returned no daily history for {contract_symbol(contract)}"
                         + (f" ({errors})" if errors else "") + ".")
    frame = pd.DataFrame([{"date": pd.Timestamp(bar.date), "open": bar.open, "high": bar.high,
                           "low": bar.low, "close": bar.close} for bar in bars])
    today = pd.Timestamp.now(tz=ACCOUNTING_TIMEZONE).tz_localize(None).normalize()
    return frame[frame["date"].dt.tz_localize(None) < today].reset_index(drop=True)


def adjusted_daily_closes(ib, contract, duration: str = "1 Y") -> pd.Series:
    """Completed daily closes adjusted for splits and dividends, dated by session (UTC midnight).

    Uses IB's ADJUSTED_LAST bars, which suit return analysis; falls back to unadjusted
    trade prices where IB offers no adjusted series (non-stock contracts).
    """
    series = None
    for what in ("ADJUSTED_LAST", "TRADES"):
        bars = ib.reqHistoricalData(contract, endDateTime="", durationStr=duration, barSizeSetting="1 day",
                                    whatToShow=what, useRTH=True, formatDate=1)
        if bars:
            series = pd.Series({pd.Timestamp(bar.date).tz_localize(None).normalize(): float(bar.close)
                                for bar in bars if bar.close and bar.close > 0})
            break
    if series is None or series.empty:
        errors = "; ".join(getattr(ib, "desk_errors", [])[-2:])
        raise ValueError(f"IB returned no daily history for {contract_symbol(contract)}"
                         + (f" ({errors})" if errors else "") + ".")
    today = pd.Timestamp.now(tz=ACCOUNTING_TIMEZONE).tz_localize(None).normalize()
    series = series[series.index < today].sort_index()  # today's session is not complete
    series.index = series.index.tz_localize("UTC")
    return series


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    last: float
    close: float
    live: bool

    @property
    def reference(self) -> float:
        """Midpoint when both sides exist, else last trade, else prior close."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        if self.last > 0:
            return self.last
        return self.close


def quote(ib, contract) -> Quote:
    """Snapshot bid/ask/last; ``live`` is False for delayed or frozen data."""
    ticker = ib.reqTickers(contract)[0]

    def clean(value):
        return float(value) if is_set(value) and float(value) > 0 else float("nan")

    bid, ask, last, close = (clean(ticker.bid), clean(ticker.ask), clean(ticker.last), clean(ticker.close))
    return Quote(
        bid=bid if np.isfinite(bid) else 0.0,
        ask=ask if np.isfinite(ask) else 0.0,
        last=last if np.isfinite(last) else 0.0,
        close=close if np.isfinite(close) else 0.0,
        live=getattr(ticker, "marketDataType", 1) == LIVE_MARKET_DATA_TYPE,
    )


def round_to_tick(price: float, tick: float, direction: str = "nearest") -> float:
    """Round a price to the contract's tick; ``down``/``up`` floor or ceil."""
    if not tick or tick <= 0:
        return price
    steps = price / tick
    steps = math.floor(steps + 1e-9) if direction == "down" else math.ceil(steps - 1e-9) \
        if direction == "up" else round(steps)
    decimals = max(0, -int(math.floor(math.log10(tick))) + 2)
    return round(steps * tick, decimals)


def size_step(details) -> float:
    """Smallest order size increment (1 share for most stocks)."""
    for name in ("sizeIncrement", "minSize"):
        value = getattr(details, name, None)
        if is_set(value) and float(value) > 0:
            return float(value)
    return 1.0


def _session_intervals(hours: str, timezone: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Parse IB's ``tradingHours``/``liquidHours`` text into timezone-aware intervals.

    Handles ``20261106:0400-20261106:2000;20261107:CLOSED`` and the older
    ``20261106:0400-2000,2030-2300`` form.
    """
    intervals = []
    for day in filter(None, (part.strip() for part in (hours or "").split(";"))):
        if day.endswith("CLOSED"):
            continue
        date = day.split(":", 1)[0]
        for span in day.split(":", 1)[1].split(","):
            if "-" not in span:
                continue
            start, end = span.split("-", 1)
            end = end if ":" in end else f"{date}:{end}"
            start = start if ":" in start else f"{date}:{start}"
            try:
                intervals.append((pd.Timestamp(f"{start[:8]} {start[9:11]}:{start[11:13]}", tz=timezone),
                                  pd.Timestamp(f"{end[:8]} {end[9:11]}:{end[11:13]}", tz=timezone)))
            except ValueError:
                continue
    return intervals


def session_state(details, now: pd.Timestamp | None = None) -> str:
    """'regular' in liquid hours, 'extended' in other trading hours, else 'closed'."""
    timezone = getattr(details, "timeZoneId", "") or ACCOUNTING_TIMEZONE
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    if any(start <= now < end for start, end in _session_intervals(details.liquidHours, timezone)):
        return "regular"
    if any(start <= now < end for start, end in _session_intervals(details.tradingHours, timezone)):
        return "extended"
    return "closed"


def recent_fills(ib, account: str) -> pd.DataFrame:
    """Executions TWS still holds (about the last 7 days), with commissions and realized PnL."""
    from ib_async import ExecutionFilter

    rows = []
    for fill in ib.reqExecutions(ExecutionFilter(acctCode=account)):
        execution, report = fill.execution, fill.commissionReport
        rows.append({
            "exec_id": execution.execId,
            "time": pd.Timestamp(execution.time).tz_convert("UTC") if pd.Timestamp(execution.time).tzinfo
            else pd.Timestamp(execution.time, tz="UTC"),
            "account": execution.acctNumber,
            "symbol": contract_symbol(fill.contract),
            "sec_type": fill.contract.secType,
            "currency": fill.contract.currency,
            "side": "BUY" if execution.side.upper() in {"BOT", "BUY"} else "SELL",
            "quantity": float(execution.shares),
            "price": float(execution.price),
            "multiplier": multiplier(fill.contract),
            "commission": float(report.commission) if report and is_set(report.commission) else float("nan"),
            "realized_pnl": float(report.realizedPNL) if report and is_set(report.realizedPNL) else float("nan"),
        })
    return pd.DataFrame(rows)
