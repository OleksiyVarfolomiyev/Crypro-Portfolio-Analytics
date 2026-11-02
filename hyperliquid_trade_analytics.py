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
    """Return how long to wait before retrying a rate-limited request.

    Uses the server's ``Retry-After`` header when it contains a valid delay or
    date, and otherwise falls back to exponential backoff.
    """
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
    """Send a request to Hyperliquid's public info endpoint.

    Rate-limit responses are retried with backoff. Transport errors, other
    unsuccessful HTTP responses, and invalid JSON are reported as exceptions.
    """
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
    """Return a canonical market symbol and its DEX name.

    A DEX prefix embedded in ``coin`` is extracted. If both arguments specify
    different DEXes, the function raises ``ValueError``.
    """
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
    """Fetch completed daily OHLC candles for a market.

    The returned frame has UTC ``time`` values and ``open``, ``high``, ``low``,
    and ``close`` columns. The still-forming current daily candle is excluded.
    """
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
    """Fetch a market's current mark and oracle prices.

    Returns ``(mark_price, oracle_price)`` and raises ``ValueError`` if the
    market context is missing, malformed, or contains invalid prices.
    """
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
    """Fetch and normalize hourly funding observations for a market.

    Results are paginated as needed and returned in ascending time order with
    UTC ``time`` and numeric ``fundingRate`` columns. An empty result is
    returned only when ``allow_empty`` is true.
    """
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
    """Calculate true range for each candle that has a previous close.

    ``candles`` must contain ``high``, ``low``, and ``close`` columns. The
    result is a series containing the maximum of the intraday range and both
    previous-close gaps; the first candle is omitted.
    """
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
    """Calculate Wilder's average true range for completed candles.

    The first ATR value is the mean of the first ``period`` true ranges;
    subsequent values use Wilder's recursive smoothing. Insufficient history
    or a non-positive result raises ``ValueError``.
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("ATR period must be a positive integer.")
    ranges = true_range(candles)
    if len(ranges) == 0:
        raise ValueError("No completed daily true-range values are available.")
    effective_period = min(period, len(ranges))
    atr = float(ranges.iloc[:effective_period].mean())
    for value in ranges.iloc[effective_period:]:
        atr = (atr * (effective_period - 1) + float(value)) / effective_period
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError(f"ATR({effective_period}) is not a positive finite value.")
    return atr


def rolling_funding_sums(
    funding: pd.DataFrame,
    horizon_hours: int = DEFAULT_HORIZON_DAYS * 24,
) -> pd.Series:
    """Sum funding rates over complete, contiguous hourly windows.

    Returns one sum per ``horizon_hours`` window. When timestamps are present,
    gaps larger than the configured hourly tolerance invalidate a window.
    """
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
    """Estimate adverse funding cost per coin over a holding horizon.

    The requested percentile is applied to rolling funding sums after
    adjusting for position direction. Favorable funding is floored at zero,
    and the resulting rate is converted to quote-currency value using the
    oracle price.
    """
    direction = side_to_direction(side)
    if not np.isfinite(oracle_price) or oracle_price <= 0:
        raise ValueError("Oracle price must be positive and finite.")
    if not 0.5 < percentile < 1:
        raise ValueError("Funding stress percentile must be greater than 0.5 and less than 1.")
    sums = rolling_funding_sums(funding, horizon_hours)
    return max(0.0, float((direction * sums).quantile(percentile))) * oracle_price


def side_to_direction(side: str) -> int:
    """Convert a long/buy or short/sell label to ``1`` or ``-1``."""
    normalized = str(side).strip().upper()
    if normalized in {"LONG", "BUY", "B"}:
        return 1
    if normalized in {"SHORT", "SELL", "S"}:
        return -1
    raise ValueError(f"Unsupported position side: {side!r}. Use LONG or SHORT.")


def consolidate_current_funding(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate current funding metrics by market and position side.

    Each input row is expected to contain ``Coin``, ``Side``, ``Pos``, ``Sz``,
    and optionally ``Funding (1h)`` (expressed as a percentage). Missing or
    invalid funding rates are excluded from funding-weighted values, while
    their position notionals remain included in coverage calculations.
    """
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


def fetch_collateral_equity(wallet: str, perp_equity: float) -> float:
    """Return a wallet's margin collateral, using USDC balance on unified accounts.

    A unified account's perp account value is already included in its USDC
    balance, so the two are never added. Other account modes keep perp equity.
    """
    if post_info({"type": "userAbstraction", "user": wallet}) != "unifiedAccount":
        return perp_equity
    response = post_info({"type": "spotClearinghouseState", "user": wallet})
    balances = response.get("balances") if isinstance(response, dict) else None
    if not isinstance(balances, list):
        raise ValueError(f"Invalid spot balance response returned for wallet {wallet}.")
    usdc = next((item for item in balances if item.get("coin") == "USDC"), None)
    try:
        collateral = float(usdc["total"]) if usdc is not None else 0.0
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Wallet {wallet} returned an invalid USDC balance.") from exc
    if not np.isfinite(collateral) or collateral < 0:
        raise ValueError(f"Wallet {wallet} returned an invalid USDC balance.")
    return collateral


def apply_fill_fees(fills: pd.DataFrame) -> pd.DataFrame:
    """Make ``closedPnl`` net of USDC fees and keep the exchange value as ``grossPnl``.

    Hyperliquid reports ``closedPnl`` before fees, on opening and closing fills
    alike. Fees paid in other tokens cannot be valued here and are not subtracted.
    """
    fills = fills.copy()
    fills["grossPnl"] = fills["closedPnl"]
    usdc_fee = fills["fee"].where(fills["feeToken"] == "USDC", 0.0).fillna(0.0)
    fills["closedPnl"] = fills["closedPnl"] - usdc_fee
    return fills


def drop_refreshed_daily_rows(saved: pd.DataFrame, sources: list[str], days: pd.Series) -> pd.DataFrame:
    """Remove saved daily rows of the given sources that fresh API data will rebuild.

    A row dated differently from its rebuilt replacement would otherwise stay and
    double-count. Rows from before the fetched data begin are kept, in case the API
    stops returning its oldest events.
    """
    if saved.empty or days.empty:
        return saved
    covered = pd.to_datetime(saved["ts"], utc=True) >= pd.to_datetime(days, utc=True).min()
    return saved.loc[~(saved["source"].isin(sources) & covered)].copy()


def inactive_periods(
    active_days: Any, min_gap_days: int = 30
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return ``(start, end)`` day ranges with no activity for more than ``min_gap_days``.

    ``start`` is the day after one active day and ``end`` is the next active day,
    which is the half-open form a Plotly range break expects.
    """
    days = pd.DatetimeIndex(active_days)
    if days.tz is not None:
        days = days.tz_localize(None)
    days = days.normalize().unique().sort_values()
    return [
        (previous + pd.Timedelta(days=1), current)
        for previous, current in zip(days[:-1], days[1:])
        if current - previous > pd.Timedelta(days=min_gap_days)
    ]


def visible_month_starts(
    first: pd.Timestamp, last: pd.Timestamp, periods: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> list[pd.Timestamp]:
    """Return month starts from ``first`` to ``last`` that are not in a hidden period."""
    months = pd.date_range(first.normalize().replace(day=1), last, freq="MS")
    return [
        month for month in months
        if month >= first and not any(start <= month < end for start, end in periods)
    ]


def drop_inactive_weeks(
    weekly_returns: pd.Series, periods: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.Series:
    """Drop weekly returns, indexed by week start, that fall wholly inside a period."""
    if not periods or weekly_returns.empty:
        return weekly_returns
    weeks = pd.DatetimeIndex(weekly_returns.index)
    if weeks.tz is not None:
        weeks = weeks.tz_localize(None)
    inactive = np.zeros(len(weeks), dtype=bool)
    for start, end in periods:
        inactive |= (weeks >= start) & (weeks + pd.Timedelta(days=7) <= end)
    return weekly_returns[~inactive]


def time_weighted_index(histories: list[pd.DataFrame]) -> pd.Series:
    """Build a growth-of-1 index that excludes deposits, withdrawals, and transfers.

    Each history has ``ts``, ``accountValue``, and cumulative ``pnl`` columns. The
    cash flow in an interval is the value change not explained by PnL. Inflows are
    treated as arriving at the start of an interval and outflows as leaving at the
    end, so PnL is measured against the larger balance. Wallets are summed first,
    which cancels transfers between them.
    """
    frames = [
        history.set_index("ts")[["accountValue", "pnl"]].groupby(level=0).last()
        for history in histories
        if not history.empty
    ]
    if not frames:
        return pd.Series(dtype=float, name="index")
    times = frames[0].index
    for frame in frames[1:]:
        times = times.union(frame.index)
    combined = sum(frame.reindex(times).ffill().fillna(0.0) for frame in frames)
    pnl_change = combined["pnl"].diff().fillna(0.0)
    flow = combined["accountValue"].diff().fillna(0.0) - pnl_change
    base = combined["accountValue"].shift(1).fillna(0.0) + flow.clip(lower=0.0)
    if not (base > 0).any():
        return pd.Series(dtype=float, name="index")
    returns = (pnl_change / base.where(base > 0)).fillna(0.0).clip(lower=-1.0)
    index = (1.0 + returns).cumprod().rename("index")
    first_funded = int(np.argmax((base > 0).to_numpy()))
    return index.iloc[max(first_funded - 1, 0):]


def net_funding_apr_history(
    events: list[dict[str, Any]],
    timezone: str = "America/New_York",
    window_days: int = 7,
) -> pd.DataFrame:
    """Build daily and rolling net funding APR from ``userFunding`` events.

    Uses the formula behind ``Net Funding APR (%) (simple, covered)``: net funding
    cash flow divided by gross notional, annualized. Cash divided by the hourly
    rate gives notional-hours, which holds for hourly events and for the older
    daily events that total 24 hours of funding. Zero-rate events carry no
    recoverable notional and are excluded.
    """
    rows = []
    for event in events:
        delta = event.get("delta", {})
        if delta.get("type", "funding") != "funding":
            continue
        rate = float(delta["fundingRate"])
        cash = float(delta["usdc"])
        if not np.isfinite(rate) or not np.isfinite(cash) or rate == 0:
            continue
        rows.append({"time": event["time"], "funding_pnl": cash, "notional_hours": abs(cash / rate)})
    columns = ["funding_pnl", "notional_hours", "apr_pct", "rolling_apr_pct"]
    if not rows:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    frame = pd.DataFrame(rows)
    frame["date"] = (
        pd.to_datetime(frame["time"], unit="ms", utc=True)
        .dt.tz_convert(timezone)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    daily = frame.groupby("date")[["funding_pnl", "notional_hours"]].sum().sort_index()
    daily["apr_pct"] = daily["funding_pnl"] / daily["notional_hours"] * 24 * 365 * 100.0
    rolling = daily[["funding_pnl", "notional_hours"]].rolling(f"{window_days}D").sum()
    daily["rolling_apr_pct"] = rolling["funding_pnl"] / rolling["notional_hours"] * 24 * 365 * 100.0
    return daily[columns]


def summarize_current_funding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize gross/net exposure and current funding cash flow.

    Input rows use ``Pos`` for notional and optionally ``Funding (1h)`` as a
    percentage. The returned dictionary includes long/short exposure, net
    direction, funding coverage, and estimated hourly/daily funding PnL.
    """
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
    """Hold-period PnL estimate and the observations used to calculate it.

    Price and funding PnL are expressed in quote currency. The mean forward
    return is a decimal return, and the observation fields count usable
    historical samples.
    """

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
    """Estimate price, funding, and net PnL for a position over a horizon.

    Historical mean forward close-to-close returns estimate price PnL, and
    the mean contiguous funding sum estimates funding PnL. ``quantity`` is an
    absolute coin amount; ``side`` determines the direction of both effects.
    """
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
    """Return the mean historical forward return and its sample count.

    Each observation compares a close with the close ``horizon_days`` later.
    Raises ``ValueError`` when there are not enough usable closes.
    """
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
    """Add historical price/funding PnL estimates to position rows.

    Candle and funding data are cached per market during the call. Rows with
    insufficient or invalid history are retained with ``NaN`` estimates, and
    a corresponding explanation is returned in the warnings list.
    """
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
