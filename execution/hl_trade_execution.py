"""Guarded Hyperliquid executor: dry-run by default, live requires local confirmation.

Signals use a ``coin,side,dex`` CSV; position actions use
``action,coin,side,dex,margin_usd``. Private keys are read from the local OS
keychain only for live runs.
"""

from __future__ import annotations

import argparse
import getpass
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from hyperliquid_trade_analytics import (
    DEFAULT_HORIZON_DAYS,
    ExpectedPnl,
    adverse_funding_cost_per_unit,
    expected_position_pnl,
    fetch_collateral_equity,
    fetch_daily_candles,
    fetch_funding_history,
    post_info,
    side_to_direction,
    wilder_atr,
)

KEYCHAIN_SERVICE = "Hyperliquid API wallet"
BASE_URL = "https://api.hyperliquid.xyz"
RESERVE_PCT = 0.20
RISK_PCT_OF_USABLE_MARGIN = 0.05
MAX_TOTAL_RISK_PCT = 0.20
STOP_ATR_MULTIPLE = 1.25
REWARD_RISK_RATIO = 1.0
FUNDING_STRESS_PERCENTILE = 0.95
MIN_ORDER_VALUE_USD = 10.0
ENTRY_TIMEOUT_SECONDS = 5 * 60
CHASE_INTERVAL_SECONDS = 10
CANCEL_CONFIRM_ATTEMPTS = 5
MONITOR_INTERVAL_SECONDS = 15


@dataclass(frozen=True)
class Market:
    """Market metadata and current reference prices used for sizing orders."""

    coin: str
    dex: str
    mark_price: float
    oracle_price: float
    size_decimals: int
    max_leverage: int
    only_isolated: bool = False


@dataclass(frozen=True)
class Position:
    """A wallet's open position together with market and margin details."""

    wallet: str
    coin: str
    dex: str
    quantity: float
    entry_price: float
    mark_price: float
    oracle_price: float
    margin_used: float
    market: Market
    leverage: int = 1


@dataclass(frozen=True)
class AccountSnapshot:
    """Combined account balances, positions, fees, and sizing-wallet totals."""

    equity: float
    margin_used: float
    target_equity: float
    target_margin_used: float
    positions: tuple[Position, ...]
    maker_fee_rate: float
    taker_fee_rate: float
    risk_wallets: tuple[str, ...]


@dataclass(frozen=True)
class PlannedTrade:
    """Risk-checked entry size, reference levels, and expected-PnL estimate."""

    coin: str
    dex: str
    side: str
    quantity: float
    entry_reference: float
    stop_price: float
    target_price: float
    stop_distance: float
    funding_reserve_per_unit: float
    risk_budget: float
    expected_pnl: ExpectedPnl
    maker_fee_rate: float
    taker_fee_rate: float
    leverage: int
    block_reason: str = ""
    price_stop_distance: float = 0.0


@dataclass
class ActiveTrade:
    """Runtime state for a submitted trade monitored by the live guard."""

    plan: PlannedTrade
    quantity: float
    entry_price: float
    baseline_quantity: float
    stop_oid: int
    target_oid: int
    started_at: pd.Timestamp


@dataclass(frozen=True)
class ProtectionPlan:
    """Missing stop/target quantities and levels for an open position."""

    position: Position
    side: str
    stop_price: float
    target_price: float
    stop_quantity_to_add: float
    target_quantity_to_add: float
    stop_order_ids: tuple[int, ...]
    target_order_ids: tuple[int, ...]
    non_reduce_only_stop_order_ids: tuple[int, ...] = ()
    stop_block_reason: str = ""
    target_block_reason: str = ""


@dataclass(frozen=True)
class LadderOrder:
    """One position-protection ladder leg with its size and limit/trigger price."""

    quantity: float
    price: float


@dataclass(frozen=True)
class ProtectionLadderPlan:
    """Complete stop and take-profit ladders plus IDs of orders to replace."""

    position: Position
    side: str
    atr: float
    stop_steps: tuple[LadderOrder, ...]
    target_steps: tuple[LadderOrder, ...]
    existing_stop_order_ids: tuple[int, ...]
    existing_target_order_ids: tuple[int, ...]


@dataclass(frozen=True)
class PositionAction:
    """One requested open, resize, target, or close action for a market."""

    action: str
    coin: str
    dex: str
    side: str
    margin_usd: float | None = None


@dataclass(frozen=True)
class ReconcilePlan:
    """Preflight result describing a position action and its order consequences."""

    action: PositionAction
    position: Position | None
    side: str
    current_quantity: float
    target_quantity: float
    quantity_delta: float
    leverage: int
    mark_price: float
    entry_plan: PlannedTrade | None
    stop_price: float
    target_price: float
    stop_order_ids: tuple[int, ...]
    target_order_ids: tuple[int, ...]
    block_reason: str = ""


def normalize_market(coin: str, dex: str = "") -> tuple[str, str]:
    """Normalize a market symbol and resolve its optional DEX prefix."""
    coin = coin.strip()
    dex = dex.strip()
    if ":" in coin:
        prefix, _ = coin.split(":", 1)
        if dex and dex != prefix:
            raise ValueError(f"Signal coin {coin!r} conflicts with DEX {dex!r}.")
        dex = prefix
    elif dex:
        coin = f"{dex}:{coin}"
    if not coin:
        raise ValueError("Signal coin cannot be empty.")
    return coin, dex


def get_dexes(signals: list[dict[str, str]]) -> list[str]:
    """Return the default DEX, DEXes named by signals, and all listed perp DEXes."""
    dexes = {""}
    for signal in signals:
        _, dex = normalize_market(signal["coin"], signal.get("dex", ""))
        if dex:
            dexes.add(dex)
    response = post_info({"type": "perpDexs"})
    if not isinstance(response, list):
        raise ValueError("Hyperliquid returned an invalid perp DEX list.")
    dexes.update(
        str(item["name"])
        for item in response
        if isinstance(item, dict) and item.get("name")
    )
    return sorted(dexes)


def fetch_markets(dexes: list[str]) -> dict[str, Market]:
    """Fetch metadata and current mark/oracle prices for the requested DEXes."""
    markets: dict[str, Market] = {}
    for dex in dexes:
        request: dict[str, Any] = {"type": "metaAndAssetCtxs"}
        if dex:
            request["dex"] = dex
        response = post_info(request)
        if not isinstance(response, list) or len(response) != 2:
            raise ValueError(f"Invalid market metadata response for DEX {dex or 'default'}.")
        meta, contexts = response
        if not isinstance(meta, dict) or not isinstance(contexts, list):
            raise ValueError(f"Invalid market metadata structure for DEX {dex or 'default'}.")
        universe = meta.get("universe")
        if not isinstance(universe, list) or len(universe) != len(contexts):
            raise ValueError(f"Market metadata and contexts do not align for DEX {dex or 'default'}.")
        for instrument, context in zip(universe, contexts):
            name = str(instrument.get("name", ""))
            if not name or not isinstance(context, dict):
                continue
            coin, normalized_dex = normalize_market(name, dex)
            mark = float(context.get("markPx") or 0)
            oracle = float(context.get("oraclePx") or mark or 0)
            size_decimals = int(instrument.get("szDecimals", 0))
            max_leverage = int(instrument.get("maxLeverage", 1))
            if mark <= 0 or oracle <= 0:
                continue
            markets[coin] = Market(
                coin=coin,
                dex=normalized_dex,
                mark_price=mark,
                oracle_price=oracle,
                size_decimals=size_decimals,
                max_leverage=max_leverage,
                only_isolated=bool(instrument.get("onlyIsolated")),
            )
    return markets


def fetch_user_state(wallet: str, dex: str) -> dict[str, Any]:
    """Fetch and validate one wallet's clearinghouse state for a DEX."""
    request: dict[str, Any] = {"type": "clearinghouseState", "user": wallet}
    if dex:
        request["dex"] = dex
    response = post_info(request)
    if not isinstance(response, dict) or not isinstance(response.get("assetPositions"), list):
        raise ValueError(f"Invalid account state returned for wallet {wallet} on DEX {dex or 'default'}.")
    return response


def fetch_open_orders(wallet: str, dexes: list[str] | None = None) -> list[dict[str, Any]]:
    """Fetch and combine a wallet's open orders across the default and named DEXs."""
    selected_dexes = get_dexes([]) if dexes is None else list(dict.fromkeys(dexes))
    orders_by_id: dict[int, dict[str, Any]] = {}
    for dex in selected_dexes:
        request: dict[str, Any] = {"type": "frontendOpenOrders", "user": wallet}
        if dex:
            request["dex"] = dex
        response = post_info(request)
        if not isinstance(response, list) or any(not isinstance(order, dict) for order in response):
            raise ValueError(
                f"Invalid open-order response returned for wallet {wallet} on DEX {dex or 'default'}."
            )
        for order in response:
            if order.get("oid") is None:
                raise ValueError(
                    f"Open order for wallet {wallet} on DEX {dex or 'default'} has no order ID."
                )
            order_id = int(order["oid"])
            existing = orders_by_id.get(order_id)
            if existing is not None and existing != order:
                raise ValueError(f"Conflicting open-order data returned for order {order_id}.")
            orders_by_id[order_id] = order
    return list(orders_by_id.values())


def parse_fee_rates(wallet: str) -> tuple[float, float]:
    """Return the wallet's maker and taker fee rates from Hyperliquid."""
    response = post_info({"type": "userFees", "user": wallet})
    if not isinstance(response, dict) or "userAddRate" not in response or "userCrossRate" not in response:
        raise ValueError(f"Could not read the maker/taker fee rates for wallet {wallet}.")
    maker, taker = float(response["userAddRate"]), float(response["userCrossRate"])
    if maker < 0 or taker < 0 or not np.isfinite(maker + taker):
        raise ValueError(f"Invalid maker/taker fee rates returned for wallet {wallet}.")
    return maker, taker


def collect_snapshot(wallets: list[str], dexes: list[str], markets: dict[str, Market]) -> AccountSnapshot:
    """Collect validated positions, margin totals, and collateral equity across wallets.

    The first wallet is the execution wallet; its equity and margin are kept
    separately for sizing, while all supplied wallets contribute to aggregate
    risk and total equity.
    """
    wallets = list(dict.fromkeys(wallet.lower() for wallet in wallets))
    if not wallets:
        raise ValueError("At least one risk wallet address is required.")
    all_positions: list[Position] = []
    equity = 0.0
    margin_used = 0.0
    target_equity = 0.0
    target_margin_used = 0.0
    for wallet_index, wallet in enumerate(wallets):
        perp_equity = 0.0
        for dex in dexes:
            state = fetch_user_state(wallet, dex)
            margin_summary = state.get("marginSummary", {})
            try:
                account_value = float(margin_summary["accountValue"])
                used_margin = float(margin_summary["totalMarginUsed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Wallet {wallet} has no usable margin summary for DEX {dex or 'default'}."
                ) from exc
            if account_value < 0 or used_margin < 0:
                raise ValueError(f"Wallet {wallet} returned negative account or margin values.")
            perp_equity += account_value
            margin_used += used_margin
            if wallet_index == 0:
                target_margin_used += used_margin
            for item in state["assetPositions"]:
                position = item.get("position", {})
                quantity = float(position.get("szi") or 0)
                if quantity == 0:
                    continue
                coin = str(position.get("coin", ""))
                market = markets.get(coin)
                if market is None:
                    raise ValueError(
                        f"Cannot price existing position {coin!r}; refusing to size new risk."
                    )
                entry_price = float(position.get("entryPx") or 0)
                if entry_price <= 0:
                    raise ValueError(f"Existing position {coin} has an invalid entry price.")
                leverage_data = position.get("leverage")
                leverage = (
                    int(leverage_data.get("value"))
                    if isinstance(leverage_data, dict) and leverage_data.get("value") is not None
                    else min(3 if dex else 4, market.max_leverage)
                )
                if leverage < 1:
                    raise ValueError(f"Existing position {coin} has invalid leverage metadata.")
                all_positions.append(
                    Position(
                        wallet=wallet,
                        coin=coin,
                        dex=dex,
                        quantity=quantity,
                        entry_price=entry_price,
                        mark_price=market.mark_price,
                        oracle_price=market.oracle_price,
                        margin_used=float(position.get("marginUsed") or 0),
                        market=market,
                        leverage=leverage,
                    )
                )
        wallet_equity = fetch_collateral_equity(wallet, perp_equity)
        equity += wallet_equity
        if wallet_index == 0:
            target_equity = wallet_equity
    if equity <= 0:
        raise ValueError("Combined account equity must be positive.")
    maker_rate, taker_rate = parse_fee_rates(wallets[0])
    return AccountSnapshot(
        equity=equity,
        margin_used=margin_used,
        target_equity=target_equity,
        target_margin_used=target_margin_used,
        positions=tuple(all_positions),
        maker_fee_rate=maker_rate,
        taker_fee_rate=taker_rate,
        risk_wallets=tuple(wallets),
    )


def position_risk(
    position: Position,
    candles: pd.DataFrame,
    funding: pd.DataFrame,
    taker_fee_rate: float,
    maker_fee_rate: float = 0.0,
) -> float:
    """Estimate a position's stop, adverse-funding, and round-trip fee risk."""
    side = "LONG" if position.quantity > 0 else "SHORT"
    atr = wilder_atr(candles, 14)
    funding_reserve = adverse_funding_cost_per_unit(
        funding,
        side,
        position.oracle_price,
        horizon_hours=DEFAULT_HORIZON_DAYS * 24,
        percentile=FUNDING_STRESS_PERCENTILE,
    )
    fee_reserve_per_unit = position.mark_price * (maker_fee_rate + taker_fee_rate)
    return abs(position.quantity) * (
        atr * STOP_ATR_MULTIPLE + funding_reserve + fee_reserve_per_unit
    )


def _order_side_direction(order: dict[str, Any]) -> int:
    """Map an exchange order's side field to buy/sell direction."""
    side = str(order.get("side", "")).strip().upper()
    if side in {"B", "BUY", "BID"}:
        return 1
    if side in {"A", "S", "SELL", "ASK"}:
        return -1
    return 0


def _is_reduce_only(order: dict[str, Any]) -> bool:
    """Return whether an order is explicitly marked reduce-only."""
    value = order.get("reduceOnly", False)
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _matching_protection_orders(
    position: Position,
    open_orders: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate close-side orders matching a position into stops and targets."""
    direction = 1 if position.quantity > 0 else -1
    close_direction = -direction
    stops: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for order in open_orders:
        if order.get("coin") != position.coin or _order_side_direction(order) != close_direction:
            continue
        order_type = str(order.get("orderType", "")).lower()
        try:
            trigger_price = float(order.get("triggerPx") or 0)
        except (TypeError, ValueError):
            trigger_price = 0.0
        if "stop" in order_type or trigger_price > 0:
            stops.append(order)
            continue
        if not _is_reduce_only(order):
            continue
        if "take profit" in order_type:
            targets.append(order)
            continue
        if "limit" not in order_type and order.get("limitPx") is None:
            continue
        try:
            limit_price = float(order.get("limitPx") or 0)
        except (TypeError, ValueError):
            continue
        # A ladder on a losing position rests beyond the mark but short of entry.
        is_favorable_exit = (
            limit_price > min(position.entry_price, position.mark_price)
            if direction > 0
            else 0 < limit_price < max(position.entry_price, position.mark_price)
        )
        if is_favorable_exit:
            targets.append(order)
    return stops, targets


def _open_order_quantity(order: dict[str, Any]) -> float:
    """Read a non-negative finite remaining order size, or zero if invalid."""
    try:
        quantity = float(order.get("sz") or order.get("origSz") or 0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, quantity) if np.isfinite(quantity) else 0.0


def _protection_coverage(
    position: Position,
    open_orders: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float]:
    """Calculate existing stop/target orders and uncovered position quantities."""
    stops, targets = _matching_protection_orders(position, open_orders)
    open_quantity = abs(position.quantity)
    step = 10 ** -position.market.size_decimals
    stop_quantity_to_add = round(
        max(0.0, open_quantity - sum(_open_order_quantity(order) for order in stops)),
        position.market.size_decimals,
    )
    target_quantity_to_add = round(
        max(0.0, open_quantity - sum(_open_order_quantity(order) for order in targets)),
        position.market.size_decimals,
    )
    if stop_quantity_to_add < step:
        stop_quantity_to_add = 0.0
    if target_quantity_to_add < step:
        target_quantity_to_add = 0.0
    return stops, targets, stop_quantity_to_add, target_quantity_to_add


def plan_position_protection(
    position: Position,
    open_orders: list[dict[str, Any]],
    candles: pd.DataFrame,
    funding: pd.DataFrame | None = None,
) -> ProtectionPlan:
    """Plan missing single-level ATR stop and take-profit protection."""
    direction = 1 if position.quantity > 0 else -1
    side = "LONG" if direction > 0 else "SHORT"
    stops, targets, stop_quantity_to_add, target_quantity_to_add = _protection_coverage(
        position,
        open_orders,
    )
    risk_stop_distance = wilder_atr(candles, 14) * STOP_ATR_MULTIPLE
    if funding is None:
        funding = fetch_funding_history(position.coin.split(":")[-1], position.dex)
    funding_reserve = adverse_funding_cost_per_unit(
        funding,
        side,
        position.oracle_price,
        horizon_hours=DEFAULT_HORIZON_DAYS * 24,
        percentile=FUNDING_STRESS_PERCENTILE,
    )
    stop_distance = max(
        _price_tick(position.mark_price, position.market.size_decimals),
        risk_stop_distance - funding_reserve,
    )
    stop_price = _round_price(
        position.entry_price - direction * stop_distance,
        position.market.size_decimals,
    )
    target_price = _round_price(
        position.entry_price + direction * stop_distance * REWARD_RISK_RATIO,
        position.market.size_decimals,
    )
    stop_block_reason = ""
    if stop_quantity_to_add and (
        stop_price <= 0
        or (direction > 0 and stop_price >= position.mark_price)
        or (direction < 0 and stop_price <= position.mark_price)
    ):
        stop_block_reason = "ATR stop is at or beyond the current mark; refusing a potentially immediate trigger"
    target_block_reason = ""
    if target_quantity_to_add and (
        (direction > 0 and target_price <= position.mark_price)
        or (direction < 0 and target_price >= position.mark_price)
    ):
        target_block_reason = "ATR target is at or beyond the current mark; refusing a crossing take-profit"

    return ProtectionPlan(
        position=position,
        side=side,
        stop_price=stop_price,
        target_price=target_price,
        stop_quantity_to_add=stop_quantity_to_add,
        target_quantity_to_add=target_quantity_to_add,
        stop_order_ids=tuple(int(order["oid"]) for order in stops if order.get("oid") is not None),
        target_order_ids=tuple(int(order["oid"]) for order in targets if order.get("oid") is not None),
        non_reduce_only_stop_order_ids=tuple(
            int(order["oid"])
            for order in stops
            if order.get("oid") is not None and not _is_reduce_only(order)
        ),
        stop_block_reason=stop_block_reason,
        target_block_reason=target_block_reason,
    )


def plan_protection_ladder(
    position: Position,
    open_orders: list[dict[str, Any]],
    candles: pd.DataFrame,
    steps: int = 3,
    atr_multiple: float = STOP_ATR_MULTIPLE,
) -> ProtectionLadderPlan:
    """Split an open position into size-valid ATR stop and take-profit steps."""
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("Protection-ladder steps must be a positive integer.")
    if not np.isfinite(atr_multiple) or atr_multiple <= 0:
        raise ValueError("Protection-ladder ATR multiple must be positive and finite.")
    if not np.isfinite(position.quantity) or position.quantity == 0:
        raise ValueError("A protection ladder requires a non-zero position.")
    if not np.isfinite(position.mark_price) or position.mark_price <= 0:
        raise ValueError("A protection ladder requires a positive finite mark price.")

    atr = wilder_atr(candles, 14)
    distance = atr * atr_multiple
    direction = 1 if position.quantity > 0 else -1
    stop_endpoint = position.mark_price - direction * distance
    target_endpoint = position.mark_price + direction * distance
    if stop_endpoint <= 0 or target_endpoint <= 0:
        raise ValueError(f"ATR protection levels for {position.coin} are not positive prices.")

    quantity_step = 10 ** -position.market.size_decimals
    quantity_units = int(round(abs(position.quantity) / quantity_step))
    if quantity_units <= 0 or not np.isclose(
        quantity_units * quantity_step,
        abs(position.quantity),
        atol=quantity_step / 2,
    ):
        raise ValueError(f"Position quantity for {position.coin} does not align to its size precision.")

    level_count = min(steps, quantity_units)
    while level_count > 1:
        fractions = [(index + 1) / level_count for index in range(level_count)]
        stop_prices = [
            _round_price(position.mark_price - direction * distance * fraction,
                         position.market.size_decimals)
            for fraction in fractions
        ]
        target_prices = [
            _round_price(position.mark_price + direction * distance * fraction,
                         position.market.size_decimals)
            for fraction in fractions
        ]
        stop_is_ordered = all(
            direction * (current - previous) < 0
            for previous, current in zip(stop_prices, stop_prices[1:])
        )
        target_is_ordered = all(
            direction * (current - previous) > 0
            for previous, current in zip(target_prices, target_prices[1:])
        )
        stop_is_protective = all(
            direction * (price - position.mark_price) < 0 for price in stop_prices
        )
        target_is_profitable = all(
            direction * (price - position.mark_price) > 0 for price in target_prices
        )
        base_units, remainder = divmod(quantity_units, level_count)
        candidate_quantities = [
            round((base_units + (index < remainder)) * quantity_step, position.market.size_decimals)
            for index in range(level_count)
        ]
        minimum_value_met = all(
            quantity * price >= MIN_ORDER_VALUE_USD
            for quantity, price in zip(candidate_quantities, stop_prices)
        ) and all(
            quantity * price >= MIN_ORDER_VALUE_USD
            for quantity, price in zip(candidate_quantities, target_prices)
        )
        if (
            stop_is_ordered
            and target_is_ordered
            and stop_is_protective
            and target_is_profitable
            and minimum_value_met
        ):
            break
        level_count -= 1

    fractions = [(index + 1) / level_count for index in range(level_count)]
    stop_prices = tuple(
        _round_price(position.mark_price - direction * distance * fraction,
                     position.market.size_decimals)
        for fraction in fractions
    )
    target_prices = tuple(
        _round_price(position.mark_price + direction * distance * fraction,
                     position.market.size_decimals)
        for fraction in fractions
    )
    if any(direction * (price - position.mark_price) >= 0 for price in stop_prices):
        raise ValueError(f"ATR stop levels for {position.coin} are not separated from mark by a price tick.")
    if any(direction * (price - position.mark_price) <= 0 for price in target_prices):
        raise ValueError(f"ATR target levels for {position.coin} are not separated from mark by a price tick.")
    base_units, remainder = divmod(quantity_units, level_count)
    quantities = tuple(
        round((base_units + (index < remainder)) * quantity_step, position.market.size_decimals)
        for index in range(level_count)
    )
    if any(
        quantity * price < MIN_ORDER_VALUE_USD
        for quantity, price in zip(quantities, stop_prices)
    ) or any(
        quantity * price < MIN_ORDER_VALUE_USD
        for quantity, price in zip(quantities, target_prices)
    ):
        raise ValueError(
            f"{position.coin}: position size is too small to place even one "
            f"${MIN_ORDER_VALUE_USD:g}-minimum stop and target order."
        )
    stop_steps = tuple(
        LadderOrder(quantity=quantity, price=price)
        for quantity, price in zip(quantities, stop_prices)
    )
    target_steps = tuple(
        LadderOrder(quantity=quantity, price=price)
        for quantity, price in zip(quantities, target_prices)
    )

    old_stops, old_targets = _matching_protection_orders(position, open_orders)
    def order_ids(orders: list[dict[str, Any]]) -> tuple[int, ...]:
        """Return required exchange IDs for a set of replaceable orders."""
        identifiers = []
        for order in orders:
            oid = order.get("oid")
            if oid is None:
                raise ValueError(f"Existing protection order for {position.coin} has no order ID.")
            identifiers.append(int(oid))
        return tuple(identifiers)

    return ProtectionLadderPlan(
        position=position,
        side="LONG" if direction > 0 else "SHORT",
        atr=atr,
        stop_steps=stop_steps,
        target_steps=target_steps,
        existing_stop_order_ids=order_ids(old_stops),
        existing_target_order_ids=order_ids(old_targets),
    )


def plan_open_position_protection(
    positions: tuple[Position, ...],
    open_orders: list[dict[str, Any]],
) -> list[ProtectionPlan]:
    """Plan missing protections for each position without changing live orders."""
    plans: list[ProtectionPlan] = []
    for position in positions:
        direction = 1 if position.quantity > 0 else -1
        side = "LONG" if direction > 0 else "SHORT"
        stops, targets, stop_add, target_add = _protection_coverage(position, open_orders)
        stop_ids = tuple(int(order["oid"]) for order in stops if order.get("oid") is not None)
        target_ids = tuple(int(order["oid"]) for order in targets if order.get("oid") is not None)
        non_reduce_only_ids = tuple(
            int(order["oid"])
            for order in stops
            if order.get("oid") is not None and not _is_reduce_only(order)
        )
        if stop_add == 0 and target_add == 0:
            plans.append(
                ProtectionPlan(
                    position=position,
                    side=side,
                    stop_price=0,
                    target_price=0,
                    stop_quantity_to_add=0,
                    target_quantity_to_add=0,
                    stop_order_ids=stop_ids,
                    target_order_ids=target_ids,
                    non_reduce_only_stop_order_ids=non_reduce_only_ids,
                )
            )
            continue
        try:
            candles = fetch_daily_candles(position.coin.split(":")[-1], position.dex)
            funding = fetch_funding_history(position.coin.split(":")[-1], position.dex)
            plans.append(plan_position_protection(position, open_orders, candles, funding))
        except (RuntimeError, ValueError) as exc:
            block_reason = f"{type(exc).__name__}: {exc}"
            plans.append(
                ProtectionPlan(
                    position=position,
                    side=side,
                    stop_price=0,
                    target_price=0,
                    stop_quantity_to_add=stop_add,
                    target_quantity_to_add=target_add,
                    stop_order_ids=stop_ids,
                    target_order_ids=target_ids,
                    non_reduce_only_stop_order_ids=non_reduce_only_ids,
                    stop_block_reason=block_reason if stop_add else "",
                    target_block_reason=block_reason if target_add else "",
                )
            )
    return plans


def size_trade(
    coin: str,
    dex: str,
    side: str,
    market: Market,
    candles: pd.DataFrame,
    funding: pd.DataFrame,
    equity: float,
    risk_headroom: float,
    available_margin: float,
    maker_fee_rate: float,
    taker_fee_rate: float,
) -> PlannedTrade:
    """Calculate an entry quantity constrained by risk, margin, and leverage."""
    direction = side_to_direction(side)
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError("Account equity must be positive and finite.")
    atr = wilder_atr(candles, 14)
    stop_distance = atr * STOP_ATR_MULTIPLE
    funding_reserve = adverse_funding_cost_per_unit(
        funding,
        side,
        market.oracle_price,
        horizon_hours=DEFAULT_HORIZON_DAYS * 24,
        percentile=FUNDING_STRESS_PERCENTILE,
    )
    price_risk_per_unit = (
        stop_distance
        + funding_reserve
        + market.mark_price * maker_fee_rate
        + market.mark_price * taker_fee_rate
    )
    per_position_budget = equity * (1 - RESERVE_PCT) * RISK_PCT_OF_USABLE_MARGIN
    leverage_cap = 3 if dex else 4
    leverage = min(leverage_cap, market.max_leverage)
    if leverage < 1:
        raise ValueError(f"Invalid leverage metadata for {coin}.")
    available_risk = max(0.0, min(per_position_budget, risk_headroom))
    raw_quantity = min(
        available_risk / price_risk_per_unit,
        max(0.0, available_margin) * leverage / market.mark_price,
    )
    quantity_step = 10 ** -market.size_decimals
    quantity = round(math.floor(raw_quantity / quantity_step + 1e-12) * quantity_step, market.size_decimals)
    price_stop_distance = max(
        _price_tick(market.mark_price, market.size_decimals),
        stop_distance - funding_reserve,
    )
    stop_price = market.mark_price - direction * price_stop_distance
    target_price = market.mark_price + direction * price_stop_distance * REWARD_RISK_RATIO
    if stop_price <= 0:
        raise ValueError(f"{coin} ATR stop is non-positive at the current mark; refusing to size the trade.")
    expected = expected_position_pnl(
        candles,
        funding,
        side,
        quantity,
        market.mark_price,
        market.oracle_price,
        DEFAULT_HORIZON_DAYS,
    )
    reason = ""
    if risk_headroom <= 0:
        reason = "combined open risk already meets or exceeds the 20% equity cap"
    elif available_margin <= 0:
        reason = "no margin remains above the 20% equity reserve"
    elif quantity <= 0:
        reason = "risk/margin-limited quantity is below the market's minimum size step"
    return PlannedTrade(
        coin=coin,
        dex=dex,
        side=side.upper(),
        quantity=quantity,
        entry_reference=market.mark_price,
        stop_price=stop_price,
        target_price=target_price,
        stop_distance=stop_distance,
        funding_reserve_per_unit=funding_reserve,
        risk_budget=quantity * price_risk_per_unit,
        expected_pnl=expected,
        maker_fee_rate=maker_fee_rate,
        taker_fee_rate=taker_fee_rate,
        leverage=leverage,
        block_reason=reason,
        price_stop_distance=price_stop_distance,
    )


def read_signals(path: Path) -> list[dict[str, str]]:
    """Load and validate unique market-side entries from a signal CSV."""
    import csv

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"coin", "side"}.issubset(reader.fieldnames):
            raise ValueError("Signal CSV must include coin and side columns; dex is optional.")
        signals = []
        seen_markets: set[tuple[str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            coin = (row.get("coin") or "").strip()
            side = (row.get("side") or "").strip().upper()
            if not coin:
                raise ValueError(f"Signal CSV row {row_number} has an empty coin.")
            side_to_direction(side)
            normalized_coin, dex = normalize_market(coin, row.get("dex") or "")
            key = (normalized_coin, dex)
            if key in seen_markets:
                raise ValueError(f"Signal CSV has duplicate entries for market {normalized_coin}.")
            seen_markets.add(key)
            signals.append({"coin": coin, "side": side, "dex": (row.get("dex") or "").strip()})
    if not signals:
        raise ValueError("Signal CSV contains no positions.")
    return signals


def read_position_actions(path: Path) -> list[PositionAction]:
    """Load and validate position-sizing actions from an action CSV."""
    import csv

    allowed = {"target", "open_target", "open_margin", "adjust_margin", "close"}
    actions: list[PositionAction] = []
    seen: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"action", "coin"}.issubset(reader.fieldnames):
            raise ValueError(
                "Action CSV must include action and coin columns; side, dex, and margin_usd are optional."
            )
        for row_number, row in enumerate(reader, start=2):
            action = (row.get("action") or "").strip().lower()
            coin = (row.get("coin") or "").strip()
            if action not in allowed:
                raise ValueError(
                    f"Action CSV row {row_number} has unsupported action {action!r}; "
                    f"choose one of {', '.join(sorted(allowed))}."
                )
            if not coin:
                raise ValueError(f"Action CSV row {row_number} has an empty coin.")
            normalized_coin, dex = normalize_market(coin, row.get("dex") or "")
            if (normalized_coin, dex) in seen:
                raise ValueError(f"Action CSV has duplicate entries for market {normalized_coin}.")
            seen.add((normalized_coin, dex))
            side = (row.get("side") or "").strip().upper()
            if side:
                side_to_direction(side)
            if action in {"open_target", "open_margin"} and not side:
                raise ValueError(f"Action CSV row {row_number} requires LONG or SHORT in side.")
            margin_text = (row.get("margin_usd") or "").strip()
            margin_usd: float | None = None
            if margin_text:
                try:
                    margin_usd = float(margin_text)
                except ValueError as exc:
                    raise ValueError(f"Action CSV row {row_number} has invalid margin_usd.") from exc
                if not np.isfinite(margin_usd):
                    raise ValueError(f"Action CSV row {row_number} has non-finite margin_usd.")
            if action == "open_margin" and (margin_usd is None or margin_usd <= 0):
                raise ValueError(f"Action CSV row {row_number} requires positive margin_usd for open_margin.")
            if action == "adjust_margin" and (margin_usd is None or margin_usd == 0):
                raise ValueError(f"Action CSV row {row_number} requires non-zero signed margin_usd for adjust_margin.")
            if action not in {"open_margin", "adjust_margin"} and margin_usd is not None:
                raise ValueError(f"Action CSV row {row_number} may only set margin_usd for margin actions.")
            actions.append(
                PositionAction(
                    action=action,
                    coin=normalized_coin,
                    dex=dex,
                    side=side,
                    margin_usd=margin_usd,
                )
            )
    if not actions:
        raise ValueError("Action CSV contains no position actions.")
    return actions


def plan_trades(
    signals: list[dict[str, str]],
    markets: dict[str, Market],
    snapshot: AccountSnapshot,
) -> tuple[list[PlannedTrade], float]:
    """Build risk- and margin-limited entry plans and report current portfolio risk."""
    caches: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    current_risk = 0.0
    for position in snapshot.positions:
        key = (position.coin, position.dex)
        if key not in caches:
            caches[key] = (
                fetch_daily_candles(position.coin.split(":")[-1], position.dex),
                fetch_funding_history(position.coin.split(":")[-1], position.dex),
            )
        current_risk += position_risk(
            position,
            caches[key][0],
            caches[key][1],
            snapshot.taker_fee_rate,
            snapshot.maker_fee_rate,
        )

    total_risk_cap = snapshot.equity * MAX_TOTAL_RISK_PCT
    risk_headroom = total_risk_cap - current_risk
    available_margin = max(
        0.0,
        snapshot.target_equity * (1 - RESERVE_PCT) - snapshot.target_margin_used,
    )
    plans: list[PlannedTrade] = []
    for signal in signals:
        coin, dex = normalize_market(signal["coin"], signal.get("dex", ""))
        market = markets.get(coin)
        if market is None:
            raise ValueError(f"Market {coin!r} was not returned by Hyperliquid metadata.")
        own_positions = [
            position
            for position in snapshot.positions
            if position.wallet == snapshot.risk_wallets[0] and position.coin == coin
        ]
        if any(np.sign(position.quantity) != side_to_direction(signal["side"]) for position in own_positions):
            raise ValueError(
                f"{coin} is held in the target wallet on the opposite side; flatten it manually before re-entry."
            )
        key = (coin, dex)
        if key not in caches:
            caches[key] = (
                fetch_daily_candles(coin.split(":")[-1], dex),
                fetch_funding_history(coin.split(":")[-1], dex),
            )
        candles, funding = caches[key]
        plan = size_trade(
            coin,
            dex,
            signal["side"],
            market,
            candles,
            funding,
            snapshot.equity,
            risk_headroom,
            available_margin,
            snapshot.maker_fee_rate,
            snapshot.taker_fee_rate,
        )
        plans.append(plan)
        risk_headroom -= plan.risk_budget
        available_margin -= plan.quantity * market.mark_price / plan.leverage
    return plans, current_risk


def plan_position_actions(
    actions: list[PositionAction],
    markets: dict[str, Market],
    snapshot: AccountSnapshot,
    open_orders: list[dict[str, Any]],
    always_allow: bool = False,
) -> list[ReconcilePlan]:
    """Plan requested position changes within risk and margin limits."""
    candles_by_market: dict[tuple[str, str], pd.DataFrame] = {}
    funding_by_market: dict[tuple[str, str], pd.DataFrame] = {}
    current_risk = 0.0
    risk_by_position: dict[tuple[str, str, str], float] = {}
    for position in snapshot.positions:
        key = (position.coin, position.dex)
        if key not in candles_by_market:
            candles_by_market[key] = fetch_daily_candles(position.coin.split(":")[-1], position.dex)
            funding_by_market[key] = fetch_funding_history(position.coin.split(":")[-1], position.dex)
        risk = position_risk(
            position,
            candles_by_market[key],
            funding_by_market[key],
            snapshot.taker_fee_rate,
            snapshot.maker_fee_rate,
        )
        risk_by_position[(position.wallet, position.coin, position.dex)] = risk
        current_risk += risk

    total_risk_cap = snapshot.equity * MAX_TOTAL_RISK_PCT
    per_position_cap = snapshot.equity * (1 - RESERVE_PCT) * RISK_PCT_OF_USABLE_MARGIN
    margin_available = max(
        0.0,
        snapshot.target_equity * (1 - RESERVE_PCT) - snapshot.target_margin_used,
    )
    target_wallet = snapshot.risk_wallets[0]
    plans: list[ReconcilePlan] = []

    for request in actions:
        market = markets.get(request.coin)
        if market is None:
            raise ValueError(f"Market {request.coin!r} was not returned by Hyperliquid metadata.")
        matches = [
            position
            for position in snapshot.positions
            if position.wallet == target_wallet
            and position.coin == request.coin
            and position.dex == request.dex
        ]
        if len(matches) > 1:
            raise ValueError(f"Multiple positions found for {request.coin}; refusing ambiguous reconciliation.")
        position = matches[0] if matches else None
        current_signed = position.quantity if position else 0.0
        current_quantity = abs(current_signed)
        current_side = "LONG" if current_signed > 0 else "SHORT" if current_signed < 0 else ""
        side = current_side or request.side
        block_reason = ""
        if request.action in {"open_target", "open_margin"} and position is not None:
            block_reason = "open action requires a flat market; use target or adjust_margin for an existing position"
        elif request.action in {"target", "adjust_margin"} and position is None:
            block_reason = f"{request.action} requires an existing position; use an open action instead"
        elif request.action == "close" and position is None:
            block_reason = "there is no open position to close"
        elif position is not None and request.side and request.side != current_side:
            block_reason = "requested side conflicts with the existing position; close it before reversing"
        if not side:
            if request.action != "close":
                block_reason = block_reason or "a side is required when the market is flat"
            side = "LONG"

        key = (request.coin, request.dex)
        if key not in candles_by_market:
            candles_by_market[key] = fetch_daily_candles(request.coin.split(":")[-1], request.dex)
            funding_by_market[key] = fetch_funding_history(request.coin.split(":")[-1], request.dex)
        candles = candles_by_market[key]
        funding = funding_by_market[key]
        direction = side_to_direction(side)
        leverage_cap = 3 if request.dex else 4
        protocol_leverage = min(leverage_cap, market.max_leverage)
        leverage = position.leverage if position else protocol_leverage
        if leverage < 1:
            raise ValueError(f"Invalid leverage for {request.coin}.")
        atr = wilder_atr(candles, 14)
        funding_reserve = adverse_funding_cost_per_unit(
            funding,
            side,
            market.oracle_price,
            horizon_hours=DEFAULT_HORIZON_DAYS * 24,
            percentile=FUNDING_STRESS_PERCENTILE,
        )
        risk_per_unit = (
            atr * STOP_ATR_MULTIPLE
            + funding_reserve
            + market.mark_price * (snapshot.maker_fee_rate + snapshot.taker_fee_rate)
        )
        if not np.isfinite(risk_per_unit) or risk_per_unit <= 0:
            raise ValueError(f"Could not calculate a positive risk per unit for {request.coin}.")

        own_risk = (
            risk_by_position.get((target_wallet, request.coin, request.dex), 0.0)
            if position
            else 0.0
        )
        other_risk = max(0.0, current_risk - own_risk)
        risk_allowed_for_position = (
            per_position_cap
            if always_allow
            else max(
                0.0,
                min(per_position_cap, total_risk_cap - other_risk),
            )
        )
        maximum_risk_quantity = risk_allowed_for_position / risk_per_unit
        available_margin_quantity = margin_available * leverage / market.mark_price
        maximum_total_quantity = (
            maximum_risk_quantity
            if always_allow
            else min(
                maximum_risk_quantity,
                current_quantity + available_margin_quantity,
            )
        )
        quantity_step = 10 ** -market.size_decimals
        if request.action in {"open_target", "target"}:
            target_quantity = maximum_total_quantity
        elif request.action == "open_margin":
            requested_quantity = (request.margin_usd or 0.0) * protocol_leverage / market.mark_price
            target_quantity = min(requested_quantity, maximum_total_quantity)
        elif request.action == "adjust_margin":
            requested_delta = abs(request.margin_usd or 0.0) * leverage / market.mark_price
            if (request.margin_usd or 0.0) > 0:
                if current_quantity > maximum_total_quantity + quantity_step / 2:
                    block_reason = (
                        block_reason
                        or "existing position exceeds risk/margin target; use target to reduce before adding"
                    )
                    target_quantity = current_quantity
                else:
                    target_quantity = min(
                        current_quantity + requested_delta,
                        maximum_total_quantity,
                    )
            else:
                target_quantity = max(0.0, current_quantity - requested_delta)
        else:
            target_quantity = 0.0
        target_quantity = round(math.floor(target_quantity / quantity_step + 1e-12) * quantity_step, market.size_decimals)
        quantity_delta = target_quantity - current_quantity

        if quantity_delta > 0 and not block_reason:
            headroom_for_entry = (
                quantity_delta * risk_per_unit
                if always_allow
                else max(0.0, min(
                    total_risk_cap - current_risk,
                    per_position_cap - own_risk,
                    quantity_delta * risk_per_unit,
                ))
            )
            margin_for_entry = (
                quantity_delta * market.mark_price / leverage
                if always_allow
                else margin_available
            )
            entry_plan = size_trade(
                request.coin,
                request.dex,
                side,
                market,
                candles,
                funding,
                snapshot.equity,
                headroom_for_entry,
                margin_for_entry,
                snapshot.maker_fee_rate,
                snapshot.taker_fee_rate,
            )
            if entry_plan.quantity + quantity_step / 2 < quantity_delta:
                target_quantity = current_quantity + entry_plan.quantity
                quantity_delta = target_quantity - current_quantity
            entry_plan = PlannedTrade(
                **{
                    **entry_plan.__dict__,
                    "quantity": quantity_delta,
                    "risk_budget": quantity_delta * risk_per_unit,
                }
            )
        else:
            entry_plan = None

        reference_entry = position.entry_price if position else market.mark_price
        stop_distance = max(
            _price_tick(market.mark_price, market.size_decimals),
            atr * STOP_ATR_MULTIPLE - funding_reserve,
        )
        stop_price = _round_price(reference_entry - direction * stop_distance, market.size_decimals)
        target_price = _round_price(
            reference_entry + direction * stop_distance * REWARD_RISK_RATIO,
            market.size_decimals,
        )
        existing_stops: list[dict[str, Any]] = []
        existing_targets: list[dict[str, Any]] = []
        if position:
            existing_stops, existing_targets = _matching_protection_orders(position, open_orders)
        if target_quantity > 0 and (
            (direction > 0 and stop_price >= market.mark_price)
            or (direction < 0 and stop_price <= market.mark_price)
        ):
            block_reason = block_reason or "calculated stop is not safely beyond the current mark"
        if quantity_delta > 0 and target_quantity < current_quantity + quantity_step:
            block_reason = block_reason or "risk/margin limits leave less than one market size step to add"

        plans.append(
            ReconcilePlan(
                action=request,
                position=position,
                side=side,
                current_quantity=current_quantity,
                target_quantity=target_quantity,
                quantity_delta=quantity_delta,
                leverage=leverage if request.action != "open_margin" else protocol_leverage,
                mark_price=market.mark_price,
                entry_plan=entry_plan,
                stop_price=stop_price,
                target_price=target_price,
                stop_order_ids=tuple(
                    int(order["oid"]) for order in existing_stops if order.get("oid") is not None
                ),
                target_order_ids=tuple(
                    int(order["oid"]) for order in existing_targets if order.get("oid") is not None
                ),
                block_reason=block_reason,
            )
        )

        if not block_reason:
            next_risk = other_risk + target_quantity * risk_per_unit
            current_risk = next_risk
            if quantity_delta > 0:
                margin_available = max(
                    0.0,
                    margin_available - quantity_delta * market.mark_price / leverage,
                )
            elif quantity_delta < 0:
                margin_available += abs(quantity_delta) * market.mark_price / leverage
            if position:
                risk_by_position[(target_wallet, request.coin, request.dex)] = target_quantity * risk_per_unit

    return plans


def _round_price(price: float, size_decimals: int) -> float:
    """Round a positive price to the exchange precision allowed for a market."""
    if price <= 0:
        raise ValueError("Order price must be positive.")
    max_decimals = 6 - size_decimals
    decimals = min(max_decimals, 4 - math.floor(math.log10(price)))
    return round(price, decimals)


def _price_tick(price: float, size_decimals: int) -> float:
    """Return the smallest valid price increment near the supplied price."""
    if price <= 0 or not np.isfinite(price):
        raise ValueError("Cannot determine a price tick from an invalid price.")
    decimals = min(6 - size_decimals, 4 - math.floor(math.log10(price)))
    return 10 ** -decimals


def post_only_price(coin: str, dex: str, side: str, size_decimals: int) -> float:
    """Choose a non-crossing midpoint-based limit price for a post-only order."""
    request: dict[str, Any] = {"type": "l2Book", "coin": coin}
    if dex:
        request["dex"] = dex
    response = post_info(request)
    if not isinstance(response, dict) or not isinstance(response.get("levels"), list):
        raise ValueError(f"Invalid order book response for {coin}.")
    levels = response["levels"]
    if len(levels) != 2 or not levels[0] or not levels[1]:
        raise ValueError(f"Order book for {coin} has no two-sided market.")
    bid = float(levels[0][0]["px"])
    ask = float(levels[1][0]["px"])
    if bid <= 0 or ask <= bid:
        raise ValueError(f"Order book for {coin} is crossed or has invalid prices.")
    tick = 10 ** -max(0, min(6 - size_decimals, 5 - math.floor(math.log10((bid + ask) / 2))))
    direction = side_to_direction(side)
    desired = (bid + ask) / 2
    price = min(desired, ask - tick) if direction > 0 else max(desired, bid + tick)
    price = _round_price(price, size_decimals)
    if direction > 0 and price >= ask:
        price = _round_price(ask - tick, size_decimals)
    if direction < 0 and price <= bid:
        price = _round_price(bid + tick, size_decimals)
    if price <= 0 or (direction > 0 and price >= ask) or (direction < 0 and price <= bid):
        raise ValueError(f"Could not construct a non-crossing ALO price for {coin}.")
    return price


def _status_rows(response: Any) -> list[dict[str, Any]]:
    """Extract order status entries from a validated exchange response."""
    if not isinstance(response, dict):
        raise ValueError("Exchange returned an invalid order response.")
    statuses = response.get("response", {}).get("data", {}).get("statuses")
    if not isinstance(statuses, list) or not statuses:
        raise ValueError(f"Exchange order response has no status details: {response!r}")
    return statuses


def _oid_from_status(status: dict[str, Any]) -> int | None:
    """Return the exchange order ID for a resting or immediately filled order."""
    item = status.get("resting") or status.get("filled")
    if isinstance(item, dict) and item.get("oid") is not None:
        return int(item["oid"])
    return None


def _order_status(wallet: str, oid: int) -> dict[str, Any]:
    """Fetch the exchange's current status for one wallet order ID."""
    response = post_info({"type": "orderStatus", "user": wallet, "oid": oid})
    if not isinstance(response, dict):
        raise ValueError(f"Could not read order status for order {oid}.")
    return response


def _filled_size(status: dict[str, Any]) -> tuple[float, float]:
    """Return the filled quantity and average execution price from an order status."""
    filled = status.get("filled")
    if isinstance(filled, dict):
        return float(filled.get("totalSz", 0)), float(filled.get("avgPx", 0))
    return 0.0, 0.0


def _order_fill(status: dict[str, Any], size_decimals: int) -> tuple[str, float]:
    """Return an order's state and filled quantity from an orderStatus response."""
    details = status.get("order", {})
    order = details.get("order", {})
    state = str(details.get("status", "")).lower()
    original = float(order.get("origSz") or 0)
    filled = original if state == "filled" else original - float(order.get("sz", original))
    return state, round(filled, size_decimals)


def _place_entry(exchange: Any, wallet: str, plan: PlannedTrade, market: Market) -> tuple[float, float]:
    """Chase the planned entry with post-only limits and return filled size and average price."""
    deadline = time.monotonic() + ENTRY_TIMEOUT_SECONDS
    filled_total = 0.0
    weighted_entry = 0.0
    remaining = plan.quantity
    while remaining > 0 and time.monotonic() < deadline:
        price = post_only_price(plan.coin, plan.dex, plan.side, market.size_decimals)
        is_buy = plan.side == "LONG"
        response = exchange.order(
            plan.coin,
            is_buy,
            remaining,
            price,
            {"limit": {"tif": "Alo"}},
            reduce_only=False,
        )
        status = _status_rows(response)[0]
        oid = _oid_from_status(status)
        if oid is None:
            if filled_total > 0:
                # Earlier fills still need exit protection from the caller.
                print(f"{plan.coin}: entry stopped after a partial fill; order rejected: {status!r}")
                break
            raise RuntimeError(f"ALO entry for {plan.coin} was rejected: {status!r}")
        if "filled" in status:
            size, average = _filled_size(status)
            if size > 0:
                filled_total += size
                weighted_entry += size * average
            break
        time.sleep(CHASE_INTERVAL_SECONDS)
        state, filled = _order_fill(_order_status(wallet, oid), market.size_decimals)
        if state != "filled":
            exchange.cancel(plan.coin, oid)
            # The status endpoint can briefly report a cancelled order as open.
            for _ in range(CANCEL_CONFIRM_ATTEMPTS):
                state, filled = _order_fill(_order_status(wallet, oid), market.size_decimals)
                if state not in {"", "open"}:
                    break
                time.sleep(1)
        # Post-only orders fill as maker at their own limit price.
        filled_total += filled
        weighted_entry += filled * price
        if state in {"", "open"}:
            # Re-placing while this order may still rest could double the position.
            print(f"{plan.coin}: could not confirm order {oid} was cancelled; stopping entry.")
            break
        remaining = round(plan.quantity - filled_total, market.size_decimals)
    if filled_total <= 0:
        raise RuntimeError(
            f"No entry fill for {plan.coin} within the five-minute ALO chase window; no taker order was sent."
        )
    return filled_total, weighted_entry / filled_total


def _place_exit_orders(
    exchange: Any,
    plan: PlannedTrade,
    quantity: float,
    entry_price: float,
    size_decimals: int,
) -> tuple[int, int]:
    """Submit reduce-only stop-market and post-only take-profit orders for a filled entry."""
    direction = side_to_direction(plan.side)
    stop_distance = plan.price_stop_distance or plan.stop_distance
    stop_price = _round_price(entry_price - direction * stop_distance, size_decimals)
    target_price = _round_price(
        entry_price + direction * stop_distance * REWARD_RISK_RATIO,
        size_decimals,
    )
    close_is_buy = direction < 0
    stop_response = exchange.order(
        plan.coin,
        close_is_buy,
        quantity,
        stop_price,
        {"trigger": {"triggerPx": stop_price, "isMarket": True, "tpsl": "sl"}},
        reduce_only=True,
    )
    stop_status = _status_rows(stop_response)[0]
    stop_oid = _oid_from_status(stop_status)
    if stop_oid is None:
        raise RuntimeError(f"Could not confirm the reduce-only stop-market order for {plan.coin}: {stop_status!r}")
    target_response = exchange.order(
        plan.coin,
        close_is_buy,
        quantity,
        target_price,
        {"limit": {"tif": "Alo"}},
        reduce_only=True,
    )
    try:
        target_status = _status_rows(target_response)[0]
    except ValueError:
        exit_side = "SHORT" if direction > 0 else "LONG"
        passive_price = post_only_price(plan.coin, plan.dex, exit_side, size_decimals)
        target_price = (
            max(target_price, passive_price)
            if direction > 0
            else min(target_price, passive_price)
        )
        target_response = exchange.order(
            plan.coin,
            close_is_buy,
            quantity,
            target_price,
            {"limit": {"tif": "Alo"}},
            reduce_only=True,
        )
        try:
            target_status = _status_rows(target_response)[0]
        except ValueError:
            print(
                f"{plan.coin}: take-profit ALO was rejected after passive repricing; "
                "the reduce-only stop remains active."
            )
            return stop_oid, 0
    target_oid = _oid_from_status(target_status)
    if target_oid is None:
        print(
            f"{plan.coin}: take-profit order has no confirmed resting ID; "
            "the reduce-only stop remains active."
        )
        return stop_oid, 0
    return stop_oid, target_oid


def _cancel_if_open(exchange: Any, coin: str, wallet: str, oid: int) -> None:
    """Cancel an order only when the exchange reports it is still active."""
    status = _order_status(wallet, oid).get("order", {})
    if str(status.get("status", "")).lower() in {"open", "triggered"}:
        exchange.cancel(coin, oid)


def _current_position_quantity(wallet: str, dex: str, coin: str) -> float:
    """Return the wallet's signed open quantity for one market, or zero if flat."""
    state = fetch_user_state(wallet, dex)
    for item in state["assetPositions"]:
        position = item.get("position", {})
        if position.get("coin") == coin:
            return float(position.get("szi") or 0)
    return 0.0


def _watch_trades(exchange: Any, wallet: str, active_trades: list[ActiveTrade], maker_rate: float, taker_rate: float) -> None:
    """Monitor managed entries, cancel sibling exits, and enforce the net-loss guard."""
    remaining = list(active_trades)
    while remaining:
        for trade in list(remaining):
            plan = trade.plan
            stop_status = _order_status(wallet, trade.stop_oid).get("order", {})
            target_status = (
                _order_status(wallet, trade.target_oid).get("order", {})
                if trade.target_oid
                else {}
            )
            stop_state = str(stop_status.get("status", "")).lower()
            target_state = str(target_status.get("status", "")).lower()
            if stop_state == "filled" or target_state == "filled":
                sibling = trade.target_oid if stop_state == "filled" else trade.stop_oid
                if sibling:
                    _cancel_if_open(exchange, plan.coin, wallet, sibling)
                remaining.remove(trade)
                print(f"{plan.coin}: {'stop' if stop_state == 'filled' else 'take-profit'} filled.")
                continue

            current_quantity = _current_position_quantity(wallet, plan.dex, plan.coin)
            tranche_quantity = side_to_direction(plan.side) * (current_quantity - trade.baseline_quantity)
            if tranche_quantity <= max(1e-8, trade.quantity * 0.01):
                _cancel_if_open(exchange, plan.coin, wallet, trade.stop_oid)
                if trade.target_oid:
                    _cancel_if_open(exchange, plan.coin, wallet, trade.target_oid)
                remaining.remove(trade)
                print(f"{plan.coin}: managed position tranche is no longer open.")
                continue

            dex_request: dict[str, Any] = {"type": "metaAndAssetCtxs"}
            if plan.dex:
                dex_request["dex"] = plan.dex
            metadata = post_info(dex_request)
            universe = metadata[0]["universe"]
            contexts = metadata[1]
            market_index = next(
                (
                    index
                    for index, item in enumerate(universe)
                    if (f"{plan.dex}:{item.get('name')}" if plan.dex else item.get("name")) == plan.coin
                ),
                None,
            )
            if market_index is None:
                raise RuntimeError(f"Could not refresh mark price for active market {plan.coin}.")
            mark = float(contexts[market_index].get("markPx") or 0)
            if mark <= 0:
                raise RuntimeError(f"Invalid current mark price for active market {plan.coin}.")

            funding = fetch_funding_history(
                plan.coin.split(":")[-1],
                plan.dex,
                start_time=trade.started_at,
                allow_empty=True,
            )
            rates = funding.loc[funding["time"] >= trade.started_at, "fundingRate"]
            direction = side_to_direction(plan.side)
            price_pnl = direction * min(trade.quantity, abs(tranche_quantity)) * (mark - trade.entry_price)
            funding_pnl = -direction * min(trade.quantity, abs(tranche_quantity)) * trade.entry_price * float(rates.sum())
            estimated_close_fee = mark * min(trade.quantity, abs(tranche_quantity)) * taker_rate
            entry_fee = trade.entry_price * trade.quantity * maker_rate
            net_pnl = price_pnl + funding_pnl - estimated_close_fee - entry_fee
            tranche_risk_budget = (
                plan.risk_budget * trade.quantity / plan.quantity if plan.quantity > 0 else 0
            )
            if net_pnl <= -tranche_risk_budget and tranche_risk_budget > 0:
                _cancel_if_open(exchange, plan.coin, wallet, trade.stop_oid)
                if trade.target_oid:
                    _cancel_if_open(exchange, plan.coin, wallet, trade.target_oid)
                exchange.market_close(plan.coin, min(trade.quantity, abs(tranche_quantity)))
                remaining.remove(trade)
                print(
                    f"{plan.coin}: funding-aware net-loss guard exited the tranche "
                    f"(estimated net PnL ${net_pnl:,.2f})."
                )
        if remaining:
            time.sleep(MONITOR_INTERVAL_SECONDS)


def _load_key_from_keychain(wallet: str) -> str:
    """Load the API-wallet private key associated with a main wallet from OS keychain."""
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError(
            "The local keychain package is missing. Install requirements-trading.txt in this environment."
        ) from exc
    key = keyring.get_password(KEYCHAIN_SERVICE, wallet.lower())
    if not key:
        raise RuntimeError(
            f"No API-wallet key is stored for {wallet}. Run this script with --store-key --wallet ADDRESS locally."
        )
    return key


def store_key(wallet: str) -> None:
    """Prompt for and securely store a confirmed API-wallet key in the OS keychain."""
    _validate_address(wallet, "main-wallet address")
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError("Install requirements-trading.txt before storing a local key.") from exc
    first = getpass.getpass("API-wallet private key (input is hidden; never paste it into chat): ").strip()
    if not first:
        raise ValueError("No key was entered.")
    second = getpass.getpass("Confirm API-wallet private key: ").strip()
    if first != second:
        raise ValueError("The two entries did not match; the key was not stored.")
    keyring.set_password(KEYCHAIN_SERVICE, wallet.lower(), first)
    print(f"Stored the API-wallet key in the OS keychain for {wallet}.")


def build_exchange(wallet_address: str, private_key: str, dexes: list[str]) -> Any:
    """Create an authenticated Hyperliquid exchange client for the selected DEXes."""
    try:
        from eth_account import Account
        from hyperliquid.exchange import Exchange
        from hyperliquid.utils.constants import MAINNET_API_URL
    except ImportError as exc:
        raise RuntimeError("Install requirements-trading.txt in the active Python environment.") from exc
    signer = Account.from_key(private_key)
    return Exchange(
        signer,
        MAINNET_API_URL,
        account_address=wallet_address,
        perp_dexs=dexes,
    )


def read_target_wallets(wallet: str, risk_wallets: str | None) -> list[str]:
    """Return the execution wallet first, followed by distinct validated risk wallets."""
    wallets = [item.strip().lower() for item in (risk_wallets or wallet).split(",") if item.strip()]
    target_wallet = wallet.lower()
    result = [target_wallet] + list(dict.fromkeys(item for item in wallets if item != target_wallet))
    for address in result:
        _validate_address(address, "risk-wallet address")
    return result


def _validate_address(address: str, label: str) -> None:
    """Reject wallet strings that are not 42-character hexadecimal addresses."""
    if (
        len(address) != 42
        or not address.startswith("0x")
        or any(character not in "0123456789abcdefABCDEF" for character in address[2:])
    ):
        raise ValueError(f"Provide a valid 42-character 0x-prefixed {label}.")


def print_plan(snapshot: AccountSnapshot, plans: list[PlannedTrade], current_risk: float) -> None:
    """Display account limits, estimated existing risk, and proposed entry details."""
    print(f"Risk wallets: {len(snapshot.risk_wallets)}")
    print(f"Collateral equity: ${snapshot.equity:,.2f}")
    print(f"Perp margin used: ${snapshot.margin_used:,.2f}")
    print(
        f"Execution-wallet margin available above its 20% reserve: "
        f"${max(0.0, snapshot.target_equity * (1 - RESERVE_PCT) - snapshot.target_margin_used):,.2f}"
    )
    print(f"Estimated current stop + 5-day adverse funding risk: ${current_risk:,.2f} "
          f"({current_risk / snapshot.equity:.2%} of collateral equity)")
    print(f"Aggregate stop-risk cap: ${snapshot.equity * MAX_TOTAL_RISK_PCT:,.2f} "
          f"({MAX_TOTAL_RISK_PCT:.0%} of collateral equity)")
    print(
        f"Per-position risk budget: ${snapshot.equity * (1 - RESERVE_PCT) * RISK_PCT_OF_USABLE_MARGIN:,.2f} "
        f"(5% of the 80% usable-equity amount)"
    )
    print("Unified accounts use USDC collateral; other account modes use perp equity.")
    for plan in plans:
        print(
            f"{plan.coin} {plan.side}: quantity={plan.quantity:g}, reference=${plan.entry_reference:,.6g}, "
            f"stop=${plan.stop_price:,.6g}, target=${plan.target_price:,.6g}, "
            f"risk=${plan.risk_budget:,.2f}, funding reserve/unit=${plan.funding_reserve_per_unit:,.6g}, "
            f"estimated entry+stop fees=${plan.quantity * plan.entry_reference * (plan.maker_fee_rate + plan.taker_fee_rate):,.2f}, "
            f"expected 5d PnL=${plan.expected_pnl.net_pnl:,.2f} "
            f"(price ${plan.expected_pnl.price_pnl:,.2f}; funding ${plan.expected_pnl.funding_pnl:,.2f})"
            + (f" — BLOCKED: {plan.block_reason}" if plan.block_reason else "")
        )


def print_protection_plans(plans: list[ProtectionPlan]) -> None:
    """Display current position protection gaps and reasons additions are blocked."""
    if not plans:
        print("No open positions were found in this wallet.")
        return
    for plan in plans:
        position = plan.position
        stop_summary = (
            "covered"
            if plan.stop_quantity_to_add == 0
            else f"add {plan.stop_quantity_to_add:g} at ${plan.stop_price:,.6g}"
        )
        target_summary = (
            "covered"
            if plan.target_quantity_to_add == 0
            else f"add {plan.target_quantity_to_add:g} at ${plan.target_price:,.6g}"
        )
        print(
            f"{position.coin} {plan.side}: position quantity={abs(position.quantity):g} "
            f"(unchanged), entry=${position.entry_price:,.6g}, mark=${position.mark_price:,.6g}, "
            f"SL {stop_summary}"
            + (f" — BLOCKED: {plan.stop_block_reason}" if plan.stop_block_reason else "")
        )
        print(
            f"  TP {target_summary}"
            + (f" — BLOCKED: {plan.target_block_reason}" if plan.target_block_reason else "")
        )
        if plan.stop_order_ids:
            print(f"  existing stop order IDs: {', '.join(map(str, plan.stop_order_ids))}")
        if plan.non_reduce_only_stop_order_ids:
            print(
                "  WARNING: existing stop order(s) are not reduce-only: "
                f"{', '.join(map(str, plan.non_reduce_only_stop_order_ids))}"
            )
        if plan.target_order_ids:
            print(f"  existing reduce-only take-profit order IDs: {', '.join(map(str, plan.target_order_ids))}")


def has_submitable_protection(plans: list[ProtectionPlan]) -> bool:
    """Return whether any protection plan contains an unblocked order to submit."""
    return any(
        (plan.stop_quantity_to_add > 0 and not plan.stop_block_reason)
        or (plan.target_quantity_to_add > 0 and not plan.target_block_reason)
        for plan in plans
    )


def run_open_position_protection(
    wallet: str,
    private_key: str,
    dexes: list[str],
    plans: list[ProtectionPlan],
    markets: dict[str, Market],
) -> None:
    """Submit missing stop/target protections after rechecking live positions and orders."""
    exchange = build_exchange(wallet, private_key, dexes)
    failures: list[str] = []
    positions_by_market: dict[tuple[str, str], tuple[float, float]] = {}
    for dex in dict.fromkeys(plan.position.dex for plan in plans):
        state = fetch_user_state(wallet, dex)
        for item in state["assetPositions"]:
            raw_position = item.get("position", {})
            coin = str(raw_position.get("coin", ""))
            quantity = float(raw_position.get("szi") or 0)
            entry_price = float(raw_position.get("entryPx") or 0)
            if coin and quantity:
                positions_by_market[(coin, dex)] = (quantity, entry_price)
    live_orders = fetch_open_orders(wallet)
    submitted_order_ids: list[int] = []

    for reviewed_plan in plans:
        coin = reviewed_plan.position.coin
        dex = reviewed_plan.position.dex
        market = markets.get(coin)
        if market is None:
            failures.append(f"{coin}: market disappeared after preflight")
            continue
        current_position = positions_by_market.get((coin, dex))
        if current_position is None:
            print(f"{coin}: position is no longer open; skipping protection.")
            continue
        current_signed, entry_price = current_position
        direction = side_to_direction(reviewed_plan.side)
        size_tolerance = 10 ** -market.size_decimals
        if (
            np.sign(current_signed) != direction
            or not np.isclose(
                abs(current_signed),
                abs(reviewed_plan.position.quantity),
                atol=size_tolerance / 2,
            )
            or not np.isclose(
                entry_price,
                reviewed_plan.position.entry_price,
                atol=_price_tick(reviewed_plan.position.entry_price, market.size_decimals),
            )
        ):
            print(f"{coin}: position changed since confirmation; skipping protection.")
            continue
        current_position_record = Position(
            wallet=wallet.lower(),
            coin=coin,
            dex=dex,
            quantity=current_signed,
            entry_price=entry_price,
            mark_price=market.mark_price,
            oracle_price=market.oracle_price,
            margin_used=reviewed_plan.position.margin_used,
            market=market,
            leverage=reviewed_plan.position.leverage,
        )
        close_is_buy = direction < 0

        for order_kind, quantity, price, block_reason in (
            (
                "stop",
                reviewed_plan.stop_quantity_to_add,
                reviewed_plan.stop_price,
                reviewed_plan.stop_block_reason,
            ),
            (
                "take-profit",
                reviewed_plan.target_quantity_to_add,
                reviewed_plan.target_price,
                reviewed_plan.target_block_reason,
            ),
        ):
            if quantity <= 0 or block_reason:
                continue
            _, _, missing_stop, missing_target = _protection_coverage(
                current_position_record,
                live_orders,
            )
            if order_kind == "stop":
                quantity = missing_stop
            else:
                quantity = missing_target
            if quantity <= 0:
                print(f"{coin}: {order_kind} coverage appeared since preflight; no duplicate sent.")
                continue
            try:
                if order_kind == "stop":
                    response = exchange.order(
                        coin,
                        close_is_buy,
                        quantity,
                        price,
                        {"trigger": {"triggerPx": price, "isMarket": True, "tpsl": "sl"}},
                        reduce_only=True,
                    )
                else:
                    response = exchange.order(
                        coin,
                        close_is_buy,
                        quantity,
                        price,
                        {"limit": {"tif": "Alo"}},
                        reduce_only=True,
                    )
                status = _status_rows(response)[0]
                oid = _oid_from_status(status)
                if oid is None or "filled" in status:
                    raise RuntimeError(f"order was not confirmed resting: {status!r}")
                submitted_order_ids.append(oid)
                live_orders.append(
                    {
                        "coin": coin,
                        "side": "B" if close_is_buy else "A",
                        "sz": str(quantity),
                        "limitPx": str(price),
                        "triggerPx": str(price) if order_kind == "stop" else "0.0",
                        "orderType": "Stop Market" if order_kind == "stop" else "Limit",
                        "reduceOnly": True,
                        "oid": oid,
                    }
                )
                print(
                    f"{coin}: added reduce-only {order_kind} "
                    f"for {quantity:g} at ${price:,.6g} (order {oid})."
                )
            except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError) as exc:
                failures.append(
                    f"{coin} {order_kind}: {type(exc).__name__}: {exc}"
                )
                print(f"{coin}: failed to add {order_kind}: {exc}")

    if submitted_order_ids:
        confirmed_orders = fetch_open_orders(wallet)
        confirmed_ids = {
            int(order["oid"])
            for order in confirmed_orders
            if order.get("oid") is not None
        }
        missing_ids = sorted(set(submitted_order_ids) - confirmed_ids)
        if missing_ids:
            failures.append(
                "submitted order IDs not present in the final open-order check: "
                + ", ".join(map(str, missing_ids))
            )
        else:
            print(
                f"Verified {len(submitted_order_ids)} newly submitted protection orders "
                "in the exchange open-order list."
            )
    if failures:
        raise RuntimeError("Some protection orders were not confirmed: " + "; ".join(failures))


def print_reconcile_plans(plans: list[ReconcilePlan]) -> None:
    """Display proposed position sizes, margin estimates, and reconciliation warnings."""
    for plan in plans:
        request = plan.action
        quantity_change = (
            f"add {plan.quantity_delta:g}"
            if plan.quantity_delta > 0
            else f"reduce {abs(plan.quantity_delta):g}"
            if plan.quantity_delta < 0
            else "no size change"
        )
        print(
            f"{request.coin} {plan.side} [{request.action}]: "
            f"current={plan.current_quantity:g}, target={plan.target_quantity:g}, "
            f"{quantity_change}; SL=${plan.stop_price:,.6g}, TP=${plan.target_price:,.6g}"
            + (f" — BLOCKED: {plan.block_reason}" if plan.block_reason else "")
        )
        print(
            f"  target notional ≈ ${plan.target_quantity * plan.mark_price:,.2f}; "
            f"estimated margin ≈ ${plan.target_quantity * plan.mark_price / plan.leverage:,.2f} "
            f"at {plan.leverage}x"
        )
        if plan.action.margin_usd is not None:
            print(f"  requested margin change: ${plan.action.margin_usd:,.2f}")
            actual_margin_change = (
                abs(plan.quantity_delta) * plan.mark_price / plan.leverage
            )
            if actual_margin_change + 0.01 < abs(plan.action.margin_usd):
                print(
                    f"  requested margin change is limited to approximately "
                    f"${actual_margin_change:,.2f} by risk/margin/position limits"
                )
    if any(not plan.block_reason and plan.quantity_delta != 0 for plan in plans):
        print(
            "Warning: after a size change the executor cancels and recreates reduce-only SL/TP orders; "
            "the exchange does not provide an atomic resize, so a brief unprotected interval is possible."
        )


def has_submitable_reconciliation(plans: list[ReconcilePlan]) -> bool:
    """Return whether any unblocked plan changes size or otherwise requires reconciliation."""
    return any(
        not plan.block_reason
        and (
            plan.quantity_delta != 0
            or plan.action.action in {"target", "open_target", "open_margin", "adjust_margin"}
        )
        for plan in plans
    )


def _cancel_protection_orders(exchange: Any, coin: str, order_ids: tuple[int, ...]) -> None:
    """Cancel each specified protection order and raise if the exchange rejects a cancellation."""
    for oid in order_ids:
        response = exchange.cancel(coin, oid)
        if not isinstance(response, dict) or response.get("status") != "ok":
            raise RuntimeError(f"Could not cancel existing protection order {oid}: {response!r}")


def _current_position_details(wallet: str, dex: str, coin: str) -> tuple[float, float]:
    """Return the current signed position quantity and entry price for one market."""
    state = fetch_user_state(wallet, dex)
    for item in state["assetPositions"]:
        position = item.get("position", {})
        if position.get("coin") == coin:
            quantity = float(position.get("szi") or 0)
            entry_price = float(position.get("entryPx") or 0)
            return quantity, entry_price
    return 0.0, 0.0


def _submit_protection_order(
    exchange: Any,
    coin: str,
    side: str,
    quantity: float,
    price: float,
    order_kind: str,
) -> int:
    """Place one reduce-only stop or post-only target and return its resting order ID."""
    direction = side_to_direction(side)
    close_is_buy = direction < 0
    if order_kind == "stop":
        response = exchange.order(
            coin,
            close_is_buy,
            quantity,
            price,
            {"trigger": {"triggerPx": price, "isMarket": True, "tpsl": "sl"}},
            reduce_only=True,
        )
    else:
        response = exchange.order(
            coin,
            close_is_buy,
            quantity,
            price,
            {"limit": {"tif": "Alo"}},
            reduce_only=True,
        )
    status = _status_rows(response)[0]
    oid = _oid_from_status(status)
    if oid is None or "filled" in status:
        raise RuntimeError(f"reduce-only {order_kind} order was not confirmed resting: {status!r}")
    return oid


def run_protection_ladder_replacement(
    wallet: str,
    private_key: str,
    dexes: list[str],
    plans: list[ProtectionLadderPlan],
) -> None:
    """Replace reviewed protections with verified ladders, retaining old orders until new placement succeeds."""
    if not plans:
        print("No open positions were found; no protection orders were changed.")
        return
    wallet = wallet.lower()
    if any(plan.position.wallet.lower() != wallet for plan in plans):
        raise ValueError("Every protection-ladder plan must belong to the signing wallet.")

    exchange = build_exchange(wallet, private_key, dexes)
    for plan in plans:
        position = plan.position
        coin = position.coin
        dex = position.dex
        quantity_step = 10 ** -position.market.size_decimals
        current_quantity, current_entry = _current_position_details(wallet, dex, coin)
        if (
            not np.isclose(
                abs(current_quantity),
                abs(position.quantity),
                atol=quantity_step / 2,
            )
            or np.sign(current_quantity) != np.sign(position.quantity)
            or not np.isclose(
                current_entry,
                position.entry_price,
                atol=_price_tick(position.entry_price, position.market.size_decimals),
            )
        ):
            raise RuntimeError(f"{coin}: position changed since ladder review; no orders were changed.")

        live_orders = fetch_open_orders(wallet)
        current_stops, current_targets = _matching_protection_orders(position, live_orders)
        current_stop_ids = tuple(int(order["oid"]) for order in current_stops)
        current_target_ids = tuple(int(order["oid"]) for order in current_targets)
        if (
            set(current_stop_ids) != set(plan.existing_stop_order_ids)
            or set(current_target_ids) != set(plan.existing_target_order_ids)
        ):
            raise RuntimeError(f"{coin}: existing protection orders changed since review; no orders were changed.")

        new_order_ids: list[int] = []
        try:
            for kind, steps in (
                ("stop", plan.stop_steps),
                ("take-profit", plan.target_steps),
            ):
                for step in steps:
                    new_order_ids.append(
                        _submit_protection_order(
                            exchange,
                            coin,
                            plan.side,
                            step.quantity,
                            step.price,
                            kind,
                        )
                    )
                    time.sleep(0.25)
            confirmed_orders = fetch_open_orders(wallet)
            confirmed_ids = {int(order["oid"]) for order in confirmed_orders if order.get("oid") is not None}
            if not set(new_order_ids).issubset(confirmed_ids):
                raise RuntimeError(f"{coin}: one or more new ladder orders are not visible as open.")
            refreshed_quantity, refreshed_entry = _current_position_details(wallet, dex, coin)
            if (
                not np.isclose(
                    abs(refreshed_quantity),
                    abs(position.quantity),
                    atol=quantity_step / 2,
                )
                or np.sign(refreshed_quantity) != np.sign(position.quantity)
                or not np.isclose(
                    refreshed_entry,
                    position.entry_price,
                    atol=_price_tick(position.entry_price, position.market.size_decimals),
                )
            ):
                raise RuntimeError(f"{coin}: position changed during ladder placement.")
        except Exception as exc:
            cleanup_errors: list[str] = []
            for oid in reversed(new_order_ids):
                try:
                    response = exchange.cancel(coin, oid)
                    if not isinstance(response, dict) or response.get("status") != "ok":
                        cleanup_errors.append(f"{oid}: {response!r}")
                    time.sleep(0.25)
                except Exception as cleanup_exc:
                    cleanup_errors.append(f"{oid}: {type(cleanup_exc).__name__}: {cleanup_exc}")
            detail = (
                f" New-order cleanup was incomplete: {'; '.join(cleanup_errors)}."
                if cleanup_errors
                else " Previously open protections were left in place."
            )
            raise RuntimeError(f"{coin}: ladder placement failed: {exc}.{detail}") from exc

        old_order_ids = (*plan.existing_stop_order_ids, *plan.existing_target_order_ids)
        _cancel_protection_orders(exchange, coin, old_order_ids)
        final_orders = fetch_open_orders(wallet)
        final_ids = {int(order["oid"]) for order in final_orders if order.get("oid") is not None}
        missing_new = set(new_order_ids) - final_ids
        lingering_old = set(old_order_ids) & final_ids
        if missing_new or lingering_old:
            raise RuntimeError(
                f"{coin}: post-replacement verification failed "
                f"(missing new IDs={sorted(missing_new)}, old IDs still open={sorted(lingering_old)})."
            )
        print(
            f"{coin}: replaced {len(old_order_ids)} existing protection orders with "
            f"{len(plan.stop_steps)} trigger-market stops and "
            f"{len(plan.target_steps)} post-only take-profits; "
            f"new order IDs: {', '.join(map(str, new_order_ids))}."
        )


def run_position_reconciliation(
    wallet: str,
    private_key: str,
    dexes: list[str],
    plans: list[ReconcilePlan],
    markets: dict[str, Market],
) -> None:
    """Apply approved position-size changes, then restore reduce-only protections."""
    exchange = build_exchange(wallet, private_key, dexes)
    failures: list[str] = []
    for plan in plans:
        if plan.block_reason:
            continue
        coin = plan.action.coin
        dex = plan.action.dex
        try:
            current_signed, _ = _current_position_details(wallet, dex, coin)
            if not np.isclose(abs(current_signed), plan.current_quantity, atol=1e-8):
                raise RuntimeError(
                    f"position changed after preflight (planned {plan.current_quantity:g}, "
                    f"now {abs(current_signed):g}); refresh and review before retrying"
                )
            if plan.current_quantity and np.sign(current_signed) != side_to_direction(plan.side):
                raise RuntimeError("position side changed after preflight; refusing to act")

            if plan.quantity_delta > 0:
                if plan.entry_plan is None:
                    raise RuntimeError("increase has no risk-checked entry plan")
                if plan.current_quantity == 0:
                    leverage_result = exchange.update_leverage(
                        plan.leverage,
                        coin,
                        is_cross=not markets[coin].only_isolated,
                    )
                    if (
                        not isinstance(leverage_result, dict)
                        or leverage_result.get("status") != "ok"
                    ):
                        raise RuntimeError(
                            f"could not confirm {plan.leverage}x leverage for {coin}: "
                            f"{leverage_result!r}"
                        )
                _place_entry(exchange, wallet, plan.entry_plan, markets[coin])
            elif plan.quantity_delta < 0:
                result = exchange.market_close(coin, abs(plan.quantity_delta))
                if not isinstance(result, dict) or result.get("status") != "ok":
                    raise RuntimeError(f"reduce-only market adjustment was not confirmed: {result!r}")

            actual_signed, actual_entry = _current_position_details(wallet, dex, coin)
            actual_quantity = abs(actual_signed)
            existing_orders = fetch_open_orders(wallet)
            existing_position = plan.position
            if existing_position is not None:
                stops, targets = _matching_protection_orders(existing_position, existing_orders)
                existing_stop_ids = tuple(
                    int(order["oid"]) for order in stops if order.get("oid") is not None
                )
                existing_target_ids = tuple(
                    int(order["oid"]) for order in targets if order.get("oid") is not None
                )
            else:
                existing_stop_ids, existing_target_ids = (), ()

            if actual_quantity == 0:
                _cancel_protection_orders(
                    exchange,
                    coin,
                    existing_stop_ids + existing_target_ids,
                )
                print(f"{coin}: position is flat; cancelled its existing reduce-only SL/TP orders.")
                continue
            if np.sign(actual_signed) != side_to_direction(plan.side):
                raise RuntimeError("resulting position side differs from the requested side")

            position_changed = plan.quantity_delta != 0
            if position_changed:
                _cancel_protection_orders(exchange, coin, existing_stop_ids + existing_target_ids)
                open_orders = [
                    order
                    for order in existing_orders
                    if order.get("coin") == coin
                    and int(order.get("oid", -1)) not in set(existing_stop_ids + existing_target_ids)
                ]
            else:
                open_orders = existing_orders

            market = markets[coin]
            position = Position(
                wallet=wallet.lower(),
                coin=coin,
                dex=dex,
                quantity=actual_signed,
                entry_price=actual_entry,
                mark_price=market.mark_price,
                oracle_price=market.oracle_price,
                margin_used=0.0,
                market=market,
                leverage=plan.position.leverage if plan.position else min(3 if dex else 4, market.max_leverage),
            )
            candles = fetch_daily_candles(coin.split(":")[-1], dex)
            protection = plan_position_protection(position, open_orders, candles)
            if protection.stop_block_reason or protection.target_block_reason:
                raise RuntimeError(
                    protection.stop_block_reason or protection.target_block_reason
                )
            if protection.stop_quantity_to_add > 0:
                stop_oid = _submit_protection_order(
                    exchange,
                    coin,
                    protection.side,
                    protection.stop_quantity_to_add,
                    protection.stop_price,
                    "stop",
                )
                print(f"{coin}: reduce-only stop active at ${protection.stop_price:,.6g} (order {stop_oid}).")
            if protection.target_quantity_to_add > 0:
                target_oid = _submit_protection_order(
                    exchange,
                    coin,
                    protection.side,
                    protection.target_quantity_to_add,
                    protection.target_price,
                    "target",
                )
                print(
                    f"{coin}: reduce-only take-profit active at "
                    f"${protection.target_price:,.6g} (order {target_oid})."
                )
        except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"{coin}: {type(exc).__name__}: {exc}")
            print(f"{coin}: reconciliation failed: {exc}")
    if failures:
        raise RuntimeError(
            "Some position actions/protection orders failed; review exchange state before retrying: "
            + "; ".join(failures)
        )


def run_live(wallet: str, private_key: str, dexes: list[str], plans: list[PlannedTrade], markets: dict[str, Market]) -> None:
    """Execute planned entries, place protective exits, and monitor resulting trades."""
    exchange = build_exchange(wallet, private_key, dexes)
    active: list[ActiveTrade] = []
    failures: list[str] = []
    for plan in plans:
        if plan.quantity <= 0:
            continue
        try:
            baseline_quantity = _current_position_quantity(wallet, plan.dex, plan.coin)
            leverage_result = exchange.update_leverage(
                plan.leverage,
                plan.coin,
                is_cross=not markets[plan.coin].only_isolated,
            )
            if not isinstance(leverage_result, dict) or leverage_result.get("status") != "ok":
                raise RuntimeError(
                    f"Could not confirm leverage {plan.leverage}x for {plan.coin}: {leverage_result!r}"
                )
            quantity, entry_price = _place_entry(exchange, wallet, plan, markets[plan.coin])
            try:
                stop_oid, target_oid = _place_exit_orders(
                    exchange,
                    plan,
                    quantity,
                    entry_price,
                    markets[plan.coin].size_decimals,
                )
            except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError) as exc:
                exchange.market_close(plan.coin, quantity)
                failures.append(
                    f"{plan.coin}: exit protection failed; attempted a reduce-only market close "
                    f"for the new fill ({type(exc).__name__}: {exc})"
                )
                continue
        except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"{plan.coin}: {type(exc).__name__}: {exc}")
            print(f"Skipping {plan.coin} after a live preflight/order error: {exc}")
            continue
        active.append(
            ActiveTrade(
                plan=plan,
                quantity=quantity,
                entry_price=entry_price,
                baseline_quantity=baseline_quantity,
                stop_oid=stop_oid,
                target_oid=target_oid,
                started_at=pd.Timestamp.now(tz="UTC"),
            )
        )
        print(
            f"{plan.coin}: filled {quantity:g} at average ${entry_price:,.6g}; "
            f"reduce-only stop-market is active."
        )
        if target_oid:
            print(f"{plan.coin}: reduce-only post-only take-profit is active.")
    if active:
        print("Monitoring funding-adjusted net PnL. Keep this process running for the guard to remain active.")
        _watch_trades(exchange, wallet, active, plans[0].maker_fee_rate, plans[0].taker_fee_rate)
    if failures:
        raise RuntimeError("Some live orders did not complete successfully: " + "; ".join(failures))


def main(argv: list[str] | None = None) -> int:
    """Parse command-line options and run the selected planning or execution workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wallet", required=True, help="Main wallet address linked to the API wallet key.")
    parser.add_argument("--risk-wallets", help="Comma-separated public addresses included in aggregate risk.")
    parser.add_argument("--signals", type=Path, help="CSV file with coin,side,dex columns.")
    parser.add_argument(
        "--actions",
        type=Path,
        help="CSV with action,coin,side,dex,margin_usd columns for position reconciliation.",
    )
    parser.add_argument(
        "--protect-open",
        action="store_true",
        help="Inspect open positions and add only missing reduce-only stop/target orders.",
    )
    parser.add_argument("--store-key", action="store_true", help="Prompt locally and store the API-wallet key in Keychain.")
    parser.add_argument("--live", action="store_true", help="Allow live orders after typing the confirmation phrase.")
    args = parser.parse_args(argv)
    try:
        if args.store_key:
            store_key(args.wallet)
            return 0
        _validate_address(args.wallet, "--wallet main-wallet address")
        if args.protect_open:
            if args.signals is not None or args.actions is not None:
                parser.error("--protect-open cannot be combined with --signals or --actions.")
            if args.risk_wallets:
                parser.error("--risk-wallets applies only to --signals and cannot be used with --protect-open.")
            dexes = get_dexes([])
            markets = fetch_markets(dexes)
            snapshot = collect_snapshot([args.wallet], dexes, markets)
            plans = plan_open_position_protection(
                snapshot.positions,
                fetch_open_orders(args.wallet),
            )
            print_protection_plans(plans)
            if not has_submitable_protection(plans):
                print("No missing, safely placeable protection orders were found.")
                if not args.live:
                    print("Dry run only. No signing key was loaded and no order was sent.")
                return 0
            if not args.live:
                print("Dry run only. No signing key was loaded and no order was sent.")
                return 0

            confirmation = input(
                "Type LIVE TRADING to review protection orders for existing positions: "
            ).strip()
            if confirmation != "LIVE TRADING":
                print("Confirmation did not match; no order was sent.")
                return 0
            fresh_markets = fetch_markets(dexes)
            fresh_snapshot = collect_snapshot([args.wallet], dexes, fresh_markets)
            fresh_plans = plan_open_position_protection(
                fresh_snapshot.positions,
                fetch_open_orders(args.wallet),
            )
            print("Refreshed protection check immediately before live submission:")
            print_protection_plans(fresh_plans)
            if not has_submitable_protection(fresh_plans):
                print("The refreshed check found no missing, safely placeable orders.")
                return 0
            final_confirmation = input(
                "Type SUBMIT PROTECTION ORDERS to add only the missing orders: "
            ).strip()
            if final_confirmation != "SUBMIT PROTECTION ORDERS":
                print("Confirmation did not match; no order was sent.")
                return 0
            private_key = _load_key_from_keychain(args.wallet)
            run_open_position_protection(
                args.wallet,
                private_key,
                dexes,
                fresh_plans,
                fresh_markets,
            )
            return 0

        if args.actions is not None:
            if args.signals is not None:
                parser.error("--actions cannot be combined with --signals.")
            actions = read_position_actions(args.actions)
            dexes = get_dexes(
                [{"coin": action.coin, "dex": action.dex} for action in actions]
            )
            markets = fetch_markets(dexes)
            risk_wallets = read_target_wallets(args.wallet, args.risk_wallets)
            snapshot = collect_snapshot(risk_wallets, dexes, markets)
            plans = plan_position_actions(
                actions,
                markets,
                snapshot,
                fetch_open_orders(args.wallet),
            )
            print_reconcile_plans(plans)
            if not args.live:
                print("Dry run only. No signing key was loaded and no order was sent.")
                return 0
            if not has_submitable_reconciliation(plans):
                print("No unblocked position actions are available to submit.")
                return 2
            confirmation = input(
                "Type LIVE TRADING to review the position actions: "
            ).strip()
            if confirmation != "LIVE TRADING":
                print("Confirmation did not match; no order was sent.")
                return 0
            fresh_markets = fetch_markets(dexes)
            fresh_snapshot = collect_snapshot(risk_wallets, dexes, fresh_markets)
            fresh_plans = plan_position_actions(
                actions,
                fresh_markets,
                fresh_snapshot,
                fetch_open_orders(args.wallet),
            )
            print("Refreshed position-action preflight immediately before submission:")
            print_reconcile_plans(fresh_plans)
            if not has_submitable_reconciliation(fresh_plans):
                print("The refreshed check blocks every action; no order was sent.")
                return 2
            final_confirmation = input(
                "Type SUBMIT POSITION ACTIONS to execute the refreshed plan: "
            ).strip()
            if final_confirmation != "SUBMIT POSITION ACTIONS":
                print("Confirmation did not match; no order was sent.")
                return 0
            private_key = _load_key_from_keychain(args.wallet)
            run_position_reconciliation(
                args.wallet,
                private_key,
                dexes,
                fresh_plans,
                fresh_markets,
            )
            return 0

        if args.signals is None:
            parser.error("--signals is required unless --store-key, --protect-open, or --actions is used.")
        signals = read_signals(args.signals)
        dexes = get_dexes(signals)
        markets = fetch_markets(dexes)
        risk_wallets = read_target_wallets(args.wallet, args.risk_wallets)
        snapshot = collect_snapshot(risk_wallets, dexes, markets)
        plans, current_risk = plan_trades(signals, markets, snapshot)
        print_plan(snapshot, plans, current_risk)
        if not args.live:
            print("Dry run only. No signing key was loaded and no order was sent.")
            return 0
        if not any(plan.quantity > 0 for plan in plans):
            print("No live order can be sent because every proposed quantity is zero.")
            return 2
        confirmation = input("Type LIVE TRADING to submit these mainnet orders: ").strip()
        if confirmation != "LIVE TRADING":
            print("Confirmation did not match; no order was sent.")
            return 0
        fresh_markets = fetch_markets(dexes)
        fresh_snapshot = collect_snapshot(risk_wallets, dexes, fresh_markets)
        fresh_plans, fresh_current_risk = plan_trades(signals, fresh_markets, fresh_snapshot)
        print("Refreshed preflight immediately before live submission:")
        print_plan(fresh_snapshot, fresh_plans, fresh_current_risk)
        if not any(plan.quantity > 0 for plan in fresh_plans):
            print("The refreshed risk/margin check blocks every order; no order was sent.")
            return 2
        final_confirmation = input("Type SUBMIT ORDERS to send the refreshed plan: ").strip()
        if final_confirmation != "SUBMIT ORDERS":
            print("Confirmation did not match; no order was sent.")
            return 0
        private_key = _load_key_from_keychain(args.wallet)
        run_live(args.wallet, private_key, dexes, fresh_plans, fresh_markets)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Trade execution stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
