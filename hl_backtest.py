"""Historical ATR-stop backtest for Hyperliquid perpetual signals.

Run with an input CSV containing ``coin`` and ``side`` columns. The backtest
does not sign requests or submit orders.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hyperliquid_trade_analytics import (
    DEFAULT_HORIZON_DAYS,
    adverse_funding_cost_per_unit,
    fetch_daily_candles,
    fetch_funding_history,
    side_to_direction,
    wilder_atr,
)

DEFAULT_RESERVE_PCT = 0.20
DEFAULT_POSITION_RISK_PCT_OF_USABLE_MARGIN = 0.05
DEFAULT_STOP_ATR_MULTIPLE = 1.25
DEFAULT_REWARD_RISK_RATIO = 1.0
FUNDING_STRESS_PERCENTILE = 0.95


@dataclass(frozen=True)
class BacktestSettings:
    starting_equity: float = 10_000.0
    horizon_days: int = DEFAULT_HORIZON_DAYS
    atr_period: int = 14
    stop_atr_multiple: float = DEFAULT_STOP_ATR_MULTIPLE
    reward_risk_ratio: float = DEFAULT_REWARD_RISK_RATIO
    reserve_pct: float = DEFAULT_RESERVE_PCT
    position_risk_pct_of_usable_margin: float = DEFAULT_POSITION_RISK_PCT_OF_USABLE_MARGIN
    max_leverage: float = 4.0
    maker_fee_bps: float | None = None
    taker_fee_bps: float | None = None

    def validate(self) -> None:
        positive = {
            "starting_equity": self.starting_equity,
            "horizon_days": self.horizon_days,
            "atr_period": self.atr_period,
            "stop_atr_multiple": self.stop_atr_multiple,
            "reward_risk_ratio": self.reward_risk_ratio,
            "max_leverage": self.max_leverage,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        if not 0 <= self.reserve_pct < 1:
            raise ValueError("reserve_pct must be in [0, 1).")
        if not 0 < self.position_risk_pct_of_usable_margin <= 1:
            raise ValueError("position_risk_pct_of_usable_margin must be in (0, 1].")
        if self.maker_fee_bps is None or self.taker_fee_bps is None:
            raise ValueError("Set both maker_fee_bps and taker_fee_bps explicitly.")
        if (
            not np.isfinite(self.maker_fee_bps)
            or not np.isfinite(self.taker_fee_bps)
            or self.maker_fee_bps < 0
            or self.taker_fee_bps < 0
        ):
            raise ValueError("Fee assumptions must be finite and non-negative.")


def backtest_signal(
    candles: pd.DataFrame,
    funding: pd.DataFrame,
    side: str,
    settings: BacktestSettings,
) -> tuple[list[dict[str, Any]], list[str]]:
    settings.validate()
    maker_fee_bps = float(settings.maker_fee_bps)
    taker_fee_bps = float(settings.taker_fee_bps)
    direction = side_to_direction(side)
    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(candles.columns):
        raise ValueError(f"Candles are missing required columns: {sorted(required - set(candles.columns))}")
    ordered = candles.copy()
    ordered["time"] = pd.to_datetime(ordered["time"], utc=True)
    ordered = ordered.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if not {"time", "fundingRate"}.issubset(funding.columns):
        raise ValueError("Funding data must contain time and fundingRate columns.")
    funding_ordered = funding.copy()
    funding_ordered["time"] = pd.to_datetime(funding_ordered["time"], utc=True)
    funding_ordered = funding_ordered.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

    trades: list[dict[str, Any]] = []
    skipped: list[str] = []
    next_entry_index = settings.atr_period
    while next_entry_index < len(ordered):
        entry_index = next_entry_index
        if entry_index + settings.horizon_days > len(ordered):
            break
        entry_candle = ordered.iloc[entry_index]
        entry_time = pd.Timestamp(entry_candle["time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        previous_candles = ordered.iloc[:entry_index]
        try:
            atr = wilder_atr(previous_candles, settings.atr_period)
        except ValueError as exc:
            skipped.append(f"{entry_time.isoformat()}: {exc}")
            next_entry_index += 1
            continue

        prior_funding = funding_ordered.loc[funding_ordered["time"] < entry_time]
        try:
            stress_funding_per_unit = adverse_funding_cost_per_unit(
                prior_funding,
                side,
                float(entry_candle["open"]),
                horizon_hours=settings.horizon_days * 24,
                percentile=FUNDING_STRESS_PERCENTILE,
            )
        except ValueError as exc:
            skipped.append(f"{entry_time.isoformat()}: {exc}")
            next_entry_index += 1
            continue

        entry_price = float(entry_candle["open"])
        stop_distance = atr * settings.stop_atr_multiple
        stop_price = entry_price - direction * stop_distance
        target_price = entry_price + direction * stop_distance * settings.reward_risk_ratio
        stop_exit_fee_per_unit = max(stop_price, 0) * taker_fee_bps / 10_000
        entry_fee_per_unit = entry_price * maker_fee_bps / 10_000
        risk_per_unit = stop_distance + stress_funding_per_unit + entry_fee_per_unit + stop_exit_fee_per_unit
        risk_budget = (
            settings.starting_equity
            * (1 - settings.reserve_pct)
            * settings.position_risk_pct_of_usable_margin
        )
        max_notional = settings.starting_equity * (1 - settings.reserve_pct) * settings.max_leverage
        quantity = min(
            risk_budget / risk_per_unit,
            max_notional / entry_price,
        )
        if not np.isfinite(quantity) or quantity <= 0:
            skipped.append(f"{entry_time.isoformat()}: position size is not positive.")
            next_entry_index += 1
            continue

        last_index = min(entry_index + settings.horizon_days - 1, len(ordered) - 1)
        exit_index = last_index
        exit_price = float(ordered.iloc[last_index]["close"])
        exit_reason = "time"
        for index in range(entry_index, last_index + 1):
            bar = ordered.iloc[index]
            stop_hit = (
                float(bar["low"]) <= stop_price
                if direction > 0
                else float(bar["high"]) >= stop_price
            )
            target_hit = (
                float(bar["high"]) >= target_price
                if direction > 0
                else float(bar["low"]) <= target_price
            )
            if stop_hit:
                exit_index = index
                open_price = float(bar["open"])
                exit_price = min(stop_price, open_price) if direction > 0 else max(stop_price, open_price)
                exit_reason = "stop"
                break
            if target_hit:
                exit_index = index
                exit_price = target_price
                exit_reason = "target"
                break

        exit_time = pd.Timestamp(ordered.iloc[exit_index]["time"])
        if exit_time.tzinfo is None:
            exit_time = exit_time.tz_localize("UTC")
        exit_time += pd.Timedelta(days=1)
        held_rates = funding_ordered.loc[
            (funding_ordered["time"] >= entry_time)
            & (funding_ordered["time"] < exit_time),
            "fundingRate",
        ].astype(float)
        funding_pnl = -direction * quantity * entry_price * float(held_rates.sum())
        price_pnl = direction * quantity * (exit_price - entry_price)
        exit_fee_bps = maker_fee_bps if exit_reason == "target" else taker_fee_bps
        fees = (
            entry_price * quantity * maker_fee_bps / 10_000
            + exit_price * quantity * exit_fee_bps / 10_000
        )
        trades.append(
            {
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": side.upper(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "notional": entry_price * quantity,
                "stop_price": stop_price,
                "target_price": target_price,
                "exit_reason": exit_reason,
                "price_pnl": price_pnl,
                "funding_pnl": funding_pnl,
                "fees": fees,
                "net_pnl": price_pnl + funding_pnl - fees,
                "risk_budget": risk_budget,
                "funding_stress_reserve": stress_funding_per_unit * quantity,
                "funding_observations": len(held_rates),
            }
        )
        next_entry_index = max(entry_index + 1, exit_index + 1)
    return trades, skipped


def read_signals(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"coin", "side"}.issubset(reader.fieldnames):
            raise ValueError("Signal CSV must have coin and side columns; dex is optional.")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            coin = (row.get("coin") or "").strip()
            side = (row.get("side") or "").strip().upper()
            if not coin:
                raise ValueError(f"Signal CSV row {row_number} has an empty coin.")
            side_to_direction(side)
            rows.append({"coin": coin, "side": side, "dex": (row.get("dex") or "").strip()})
    if not rows:
        raise ValueError("Signal CSV contains no trades.")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True, help="CSV file with coin,side,dex columns.")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--maker-fee-bps", type=float, required=True, help="Your current maker fee in basis points.")
    parser.add_argument("--taker-fee-bps", type=float, required=True, help="Your current taker fee in basis points.")
    parser.add_argument("--output", type=Path, help="Optional path for the trade-level CSV results.")
    args = parser.parse_args(argv)
    try:
        signals = read_signals(args.signals)
        settings = BacktestSettings(
            starting_equity=args.equity,
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
        )
        all_trades: list[dict[str, Any]] = []
        all_skipped: list[str] = []
        for signal in signals:
            candles = fetch_daily_candles(signal["coin"], signal["dex"], args.lookback_days)
            funding = fetch_funding_history(signal["coin"], signal["dex"], args.lookback_days)
            market_settings = replace(settings, max_leverage=3.0 if signal["dex"] else 4.0)
            trades, skipped = backtest_signal(
                candles,
                funding,
                signal["side"],
                market_settings,
            )
            for trade in trades:
                trade["coin"] = signal["coin"]
                trade["dex"] = signal["dex"]
            all_trades.extend(trades)
            all_skipped.extend(f"{signal['coin']}: {message}" for message in skipped)
            print(f"{signal['coin']} {signal['side']}: {len(trades)} completed historical trades.")

        if not all_trades:
            raise ValueError("No backtest trades could be built; see skipped-entry diagnostics.")
        results = pd.DataFrame(all_trades)
        print(
            f"Trades: {len(results)} | Win rate: {(results['net_pnl'] > 0).mean():.1%} | "
            f"Mean net PnL/trade: ${results['net_pnl'].mean():,.2f} | "
            f"Total net PnL (independent trades): ${results['net_pnl'].sum():,.2f}"
        )
        print("Stops are assumed to fill first when stop and target touch in the same daily candle.")
        print("Funding PnL uses the entry price as the hourly notional-price proxy.")
        if all_skipped:
            print(f"Skipped entries: {len(all_skipped)} (first 10 shown)")
            for message in all_skipped[:10]:
                print(f"  {message}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(args.output, index=False)
            print(f"Saved trade-level results to {args.output}")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Backtest failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
