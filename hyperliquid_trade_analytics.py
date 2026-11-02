"""Shared price, funding, risk, and expected-value calculations for Hyperliquid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

INFO_URL = "https://api.hyperliquid.xyz/info"
MAX_RATE_LIMIT_RETRIES = 4
DEFAULT_HORIZON_DAYS = 5
FUNDING_INTERVAL_HOURS = 1
FUNDING_TIMESTAMP_TOLERANCE = pd.Timedelta(seconds=2)


def _rate_limit_delay(response: requests.Response, retry_number: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            delay = float(retry_after)
            if math.isfinite(delay):
                return max(0.0, delay)
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_time - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return float(2 ** (retry_number - 1))


def post_info(payload: dict[str, Any], timeout: float = 30) -> Any:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            response = requests.post(INFO_URL, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"Hyperliquid info request failed ({type(exc).__name__}).") from exc
        if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
            time.sleep(_rate_limit_delay(response, attempt + 1))
            continue
        if not response.ok:
            raise RuntimeError(f"Hyperliquid info request failed (HTTP {response.status_code}).")
        break
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Hyperliquid returned invalid JSON for an info request.") from exc


def normalize_market_symbol(coin: str, dex: str = "") -> tuple[str, str]:
    coin = coin.strip()
    dex = dex.strip()
    if ":" in coin:
        prefix, base_coin = coin.split(":", 1)
        if dex and dex != prefix:
            raise ValueError(f"Coin {coin!r} conflicts with DEX {dex!r}.")
        dex, coin = prefix, base_coin
    if not coin:
        raise ValueError("coin must be non-empty.")
    return (f"{dex}:{coin}" if dex else coin), dex


def fetch_daily_candles(
    coin: str,
    dex: str = "",
    lookback_days: int = 365,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if not coin or lookback_days <= 0:
        raise ValueError("coin must be non-empty and lookback_days must be positive.")
    end = pd.Timestamp.now(tz="UTC") if end_time is None else pd.Timestamp(end_time)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    end = end.tz_convert("UTC")
    start = end - pd.Timedelta(days=lookback_days)
    market_coin, dex = normalize_market_symbol(coin, dex)
    payload: dict[str, Any] = {
        "type": "candleSnapshot",
        "req": {
            "coin": market_coin,
            "interval": "1d",
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
        },
    }
    if dex:
        payload["dex"] = dex
    response = post_info(payload)
    if not isinstance(response, list):
        raise ValueError(f"Hyperliquid returned invalid daily candles for {coin}.")
    frame = pd.DataFrame(response)
    required = {"t", "o", "h", "l", "c"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"No usable daily OHLC candles were returned for {coin}.")
    frame["time"] = pd.to_datetime(pd.to_numeric(frame["t"], errors="coerce"), unit="ms", utc=True)
    for column in ("o", "h", "l", "c"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time", "o", "h", "l", "c"])
    frame = frame.loc[frame["time"] < end.normalize()].copy()
    frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    frame = frame.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No completed daily OHLC candles are available for {coin}.")
    return frame[["time", "open", "high", "low", "close"]]


def fetch_mark_and_oracle(coin: str, dex: str = "") -> tuple[float, float]:
    market_coin, dex = normalize_market_symbol(coin, dex)
    coin = market_coin.split(":", 1)[-1]
    payload: dict[str, Any] = {"type": "metaAndAssetCtxs"}
    if dex:
        payload["dex"] = dex
    response = post_info(payload)
    if not isinstance(response, list) or len(response) != 2:
        raise ValueError(f"Invalid market context response for {coin}.")
    meta, contexts = response
    if not isinstance(meta, dict) or not isinstance(contexts, list):
        raise ValueError(f"Invalid market context structure for {coin}.")
    universe = meta.get("universe")
    if not isinstance(universe, list) or len(universe) != len(contexts):
        raise ValueError(f"Market metadata and contexts do not align for {coin}.")
    for instrument, context in zip(universe, contexts):
        if instrument.get("name") == coin:
            mark = float(context.get("markPx") or 0)
            oracle = float(context.get("oraclePx") or mark or 0)
            if mark <= 0 or oracle <= 0:
                raise ValueError(f"Market {coin} returned an invalid mark or oracle price.")
            return mark, oracle
    raise ValueError(f"Market {coin!r} was not found on DEX {dex or 'default'}.")


def fetch_funding_history(
    coin: str,
    dex: str = "",
    lookback_days: int = 20,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not coin or lookback_days <= 0:
        raise ValueError("coin must be non-empty and lookback_days must be positive.")
    end = pd.Timestamp.now(tz="UTC") if end_time is None else pd.Timestamp(end_time)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    end = end.tz_convert("UTC")
    start = (
        end - pd.Timedelta(days=lookback_days)
        if start_time is None
        else pd.Timestamp(start_time)
    )
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    start = start.tz_convert("UTC")
    if start >= end:
        raise ValueError("Funding-history start_time must precede end_time.")

    market_coin, dex = normalize_market_symbol(coin, dex)
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[dict[str, Any]] = []
    while cursor_ms < end_ms:
        payload: dict[str, Any] = {
            "type": "fundingHistory",
            "coin": market_coin,
            "startTime": cursor_ms,
            "endTime": end_ms,
        }
        if dex:
            payload["dex"] = dex
        batch = post_info(payload)
        if not isinstance(batch, list):
            raise ValueError(f"Hyperliquid returned invalid funding history for {coin}.")
        if not batch:
            break
        rows.extend(batch)
        times = [int(item["time"]) for item in batch if item.get("time") is not None]
        if not times:
            raise ValueError(f"Funding history for {coin} contained no timestamps.")
        next_cursor = max(times) + 1
        if next_cursor <= cursor_ms:
            raise RuntimeError(f"Funding-history pagination did not advance for {coin}.")
        cursor_ms = next_cursor
        if len(batch) < 500:
            break

    frame = pd.DataFrame(rows)
    if frame.empty and allow_empty:
        return pd.DataFrame(columns=["time", "fundingRate"])
    if frame.empty or not {"time", "fundingRate"}.issubset(frame.columns):
        raise ValueError(f"No usable hourly funding history was returned for {coin}.")
    frame["time"] = pd.to_datetime(pd.to_numeric(frame["time"], errors="coerce"), unit="ms", utc=True)
    frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    frame = frame.dropna(subset=["time", "fundingRate"])
    return frame.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def true_range(candles: pd.DataFrame) -> pd.Series:
    required = {"high", "low", "close"}
    if not required.issubset(candles.columns):
        raise ValueError(f"Daily candle data is missing columns: {sorted(required - set(candles.columns))}")
    previous_close = candles["close"].shift(1)
    ranges = pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - previous_close).abs(),
            (candles["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return ranges.dropna()


def wilder_atr(candles: pd.DataFrame, period: int = 14) -> float:
    if not isinstance(period, int) or period <= 0:
        raise ValueError("ATR period must be a positive integer.")
    ranges = true_range(candles)
    if len(ranges) < period:
        raise ValueError(f"ATR({period}) requires at least {period} completed true-range values; got {len(ranges)}.")
    atr = float(ranges.iloc[:period].mean())
    for value in ranges.iloc[period:]:
        atr = (atr * (period - 1) + float(value)) / period
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError(f"ATR({period}) is not a positive finite value.")
    return atr


def rolling_funding_sums(
    funding: pd.DataFrame,
    horizon_hours: int = DEFAULT_HORIZON_DAYS * 24,
) -> pd.Series:
    if not isinstance(horizon_hours, int) or horizon_hours <= 0:
        raise ValueError("Funding horizon must be a positive number of hours.")
    if "fundingRate" not in funding:
        raise ValueError("Funding data is missing the fundingRate column.")
    ordered = funding.sort_values("time") if "time" in funding.columns else funding
    valid_rates = pd.to_numeric(ordered["fundingRate"], errors="coerce").notna()
    rates = pd.to_numeric(ordered.loc[valid_rates, "fundingRate"], errors="coerce").reset_index(drop=True)
    if len(rates) < horizon_hours:
        raise ValueError(
            f"A {horizon_hours}-hour funding estimate requires at least "
            f"{horizon_hours} hourly rates; got {len(rates)}."
        )
    sums = rates.rolling(horizon_hours, min_periods=horizon_hours).sum()
    if "time" in ordered.columns:
        times = pd.to_datetime(ordered.loc[valid_rates, "time"], utc=True).reset_index(drop=True)
        if horizon_hours > 1:
            expected_interval = pd.Timedelta(hours=FUNDING_INTERVAL_HOURS)
            contiguous = (times.diff() - expected_interval).abs().le(
                FUNDING_TIMESTAMP_TOLERANCE
            )
            complete_window = contiguous.rolling(
                horizon_hours - 1,
                min_periods=horizon_hours - 1,
            ).sum().eq(horizon_hours - 1)
            sums = sums.where(complete_window)
    sums = sums.dropna()
    if sums.empty:
        raise ValueError(
            f"No contiguous {horizon_hours}-hour funding windows are available."
        )
    return sums


def adverse_funding_cost_per_unit(
    funding: pd.DataFrame,
    side: str,
    oracle_price: float,
    horizon_hours: int = DEFAULT_HORIZON_DAYS * 24,
    percentile: float = 0.95,
) -> float:
    direction = side_to_direction(side)
    if not np.isfinite(oracle_price) or oracle_price <= 0:
        raise ValueError("Oracle price must be positive and finite.")
    if not 0.5 < percentile < 1:
        raise ValueError("Funding stress percentile must be greater than 0.5 and less than 1.")
    sums = rolling_funding_sums(funding, horizon_hours)
    return max(0.0, float((direction * sums).quantile(percentile))) * oracle_price


def side_to_direction(side: str) -> int:
    normalized = str(side).strip().upper()
    if normalized in {"LONG", "BUY", "B"}:
        return 1
    if normalized in {"SHORT", "SELL", "S"}:
        return -1
    raise ValueError(f"Unsupported position side: {side!r}. Use LONG or SHORT.")


def consolidate_current_funding(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, float | int]] = {}
    for row in rows:
        coin = str(row["Coin"])
        side = str(row["Side"]).strip().upper()
        direction = side_to_direction(side)
        notional = abs(float(row["Pos"]))
        size = abs(float(row["Sz"]))
        if not np.isfinite(notional) or not np.isfinite(size):
            raise ValueError(f"Invalid position size or notional for {coin} {side}.")

        key = (coin, side)
        group = grouped.setdefault(
            key,
            {
                "Positions": 0,
                "Size": 0.0,
                "Position Value ($)": 0.0,
                "Funding-Covered Value ($)": 0.0,
                "Weighted Funding Rate": 0.0,
                "Funding PnL / hour ($)": 0.0,
            },
        )
        group["Positions"] += 1
        group["Size"] += size
        group["Position Value ($)"] += notional

        try:
            rate_pct = float(row["Funding (1h)"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(rate_pct):
            continue
        group["Funding-Covered Value ($)"] += notional
        group["Weighted Funding Rate"] += notional * rate_pct
        group["Funding PnL / hour ($)"] += -direction * notional * rate_pct / 100.0

    consolidated = []
    for (coin, side), group in sorted(grouped.items()):
        total_notional = float(group["Position Value ($)"])
        covered_notional = float(group["Funding-Covered Value ($)"])
        funding_rate_pct = (
            float(group["Weighted Funding Rate"]) / covered_notional
            if covered_notional > 0
            else float("nan")
        )
        hourly_funding_pnl = float(group["Funding PnL / hour ($)"])
        consolidated.append(
            {
                "Coin": coin,
                "Side": side,
                "Positions": int(group["Positions"]),
                "Size": float(group["Size"]),
                "Position Value ($)": total_notional,
                "Current Funding / 1h (%)": funding_rate_pct,
                "Funding APR (%)": funding_rate_pct * 24 * 365,
                "Estimated Funding PnL / hour ($)": hourly_funding_pnl,
                "Estimated Funding PnL / day ($)": hourly_funding_pnl * 24,
                "Funding Coverage (%)": (
                    covered_notional / total_notional * 100.0
                    if total_notional > 0
                    else float("nan")
                ),
            }
        )
    return consolidated


def summarize_current_funding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    long_notional = 0.0
    short_notional = 0.0
    long_positions = 0
    short_positions = 0
    funded_notional = 0.0
    funding_pnl_per_hour = 0.0

    for row in rows:
        side = str(row["Side"]).strip().upper()
        direction = side_to_direction(side)
        notional = abs(float(row["Pos"]))
        if not np.isfinite(notional):
            raise ValueError(f"Invalid position notional for {row.get('Coin', 'unknown')} {side}.")

        if direction > 0:
            long_notional += notional
            long_positions += 1
        else:
            short_notional += notional
            short_positions += 1

        try:
            rate_pct = float(row["Funding (1h)"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(rate_pct):
            continue
        funded_notional += notional
        funding_pnl_per_hour += -direction * notional * rate_pct / 100.0

    gross_notional = long_notional + short_notional
    net_notional = long_notional - short_notional
    net_funding_rate_pct = (
        funding_pnl_per_hour / funded_notional * 100.0
        if funded_notional > 0
        else float("nan")
    )
    return {
        "Net Direction": (
            "NET LONG" if net_notional > 0.01 else "NET SHORT" if net_notional < -0.01 else "FLAT"
        ),
        "Long Positions": long_positions,
        "Short Positions": short_positions,
        "Long Notional ($)": long_notional,
        "Short Notional ($)": short_notional,
        "Gross Notional ($)": gross_notional,
        "Net Notional ($)": net_notional,
        "Net Exposure / Gross (%)": (
            net_notional / gross_notional * 100.0 if gross_notional > 0 else float("nan")
        ),
        "Net Funding Rate / 1h (%) (covered)": net_funding_rate_pct,
        "Net Funding APR (%) (simple, covered)": net_funding_rate_pct * 24 * 365,
        "Estimated Net Funding PnL / hour ($)": funding_pnl_per_hour,
        "Estimated Net Funding PnL / day ($)": funding_pnl_per_hour * 24,
        "Funding-Covered Notional ($)": funded_notional,
        "Funding Coverage (%)": (
            funded_notional / gross_notional * 100.0 if gross_notional > 0 else float("nan")
        ),
    }


@dataclass(frozen=True)
class ExpectedPnl:
    price_pnl: float
    funding_pnl: float
    net_pnl: float
    mean_forward_return: float
    forward_observations: int
    funding_observations: int


def expected_position_pnl(
    candles: pd.DataFrame,
    funding: pd.DataFrame,
    side: str,
    quantity: float,
    mark_price: float,
    oracle_price: float,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> ExpectedPnl:
    if not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("Holding horizon must be a positive integer number of daily bars.")
    if not np.isfinite(quantity) or quantity < 0:
        raise ValueError("Position quantity must be non-negative and finite.")
    if not np.isfinite(mark_price) or mark_price <= 0:
        raise ValueError("Mark price must be positive and finite.")
    if not np.isfinite(oracle_price) or oracle_price <= 0:
        raise ValueError("Oracle price must be positive and finite.")
    direction = side_to_direction(side)
    ordered = candles.sort_values("time").drop_duplicates("time", keep="last")
    closes = pd.to_numeric(ordered["close"], errors="coerce").dropna()
    forward_returns = closes.shift(-horizon_days).div(closes).sub(1).dropna()
    if forward_returns.empty:
        raise ValueError(f"Need more than {horizon_days} completed candles to estimate expected PnL.")
    funding_sums = rolling_funding_sums(funding, horizon_days * 24)
    mean_funding_rate_sum = float(funding_sums.mean())
    price_pnl = direction * quantity * mark_price * float(forward_returns.mean())
    funding_pnl = -direction * quantity * oracle_price * mean_funding_rate_sum
    return ExpectedPnl(
        price_pnl=price_pnl,
        funding_pnl=funding_pnl,
        net_pnl=price_pnl + funding_pnl,
        mean_forward_return=float(forward_returns.mean()),
        forward_observations=len(forward_returns),
        funding_observations=len(funding_sums),
    )


def expected_forward_return(closes: pd.Series, horizon_days: int = DEFAULT_HORIZON_DAYS) -> tuple[float, int]:
    if not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("Holding horizon must be a positive integer number of daily bars.")
    values = pd.to_numeric(closes, errors="coerce").dropna()
    returns = values.shift(-horizon_days).div(values).sub(1).dropna()
    if returns.empty:
        raise ValueError(f"Need more than {horizon_days} closes to estimate expected return.")
    return float(returns.mean()), int(len(returns))


def add_expected_pnl_columns(
    rows: list[dict[str, Any]],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[list[dict[str, Any]], list[str]]:
    enriched: list[dict[str, Any]] = []
    warnings: list[str] = []
    cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for source in rows:
        row = dict(source)
        coin = str(row.get("Coin", "")).strip()
        dex = str(row.get("Dex", "") or "").strip()
        if ":" in coin:
            prefix, base_coin = coin.split(":", 1)
            dex = dex or prefix
            coin = base_coin
        key = (coin, dex)
        try:
            if key not in cache:
                cache[key] = (
                    fetch_daily_candles(coin, dex, lookback_days=365),
                    fetch_funding_history(coin, dex, lookback_days=20),
                )
            quantity = abs(float(row["Sz"]))
            mark_price = float(row["Mark"])
            oracle_price = float(row.get("Oracle", mark_price))
            expected = expected_position_pnl(
                cache[key][0],
                cache[key][1],
                str(row["Side"]),
                quantity,
                mark_price,
                oracle_price,
                horizon_days,
            )
        except ValueError as exc:
            message = f"{row.get('Coin', 'unknown')}: {exc}"
            warnings.append(message)
            expected = None
        row[f"Expected Price PnL ({horizon_days}d)"] = (
            expected.price_pnl if expected is not None else np.nan
        )
        row[f"Expected Funding PnL ({horizon_days}d)"] = (
            expected.funding_pnl if expected is not None else np.nan
        )
        row[f"Expected PnL incl. funding ({horizon_days}d)"] = (
            expected.net_pnl if expected is not None else np.nan
        )
        enriched.append(row)
    return enriched, warnings
