"""Summarize what changed in the portfolio, split into went-well and needs-attention rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

STATE_FILE_NAME = "hl_highlights_state.json"
MIN_MOVE_USD = 1.0
MAX_TICKERS_LISTED = 5
LOW_MARGIN_PCT = 10.0
LOW_LIQUIDATION_PCT = 25.0
RESIZE_TOLERANCE_PCT = 0.5


def _usd(value: float) -> str:
    return f"-${abs(value):,.2f}" if value < 0 else f"+${value:,.2f}"


def _position_key(row: dict[str, Any]) -> str:
    return f"{row['Wallet']}|{row['Coin']}|{row['Side']}"


def _ticker_list(amounts: pd.Series) -> str:
    """List the largest amounts by ticker, noting how many were left out."""
    ordered = amounts.reindex(amounts.abs().sort_values(ascending=False).index)
    listed = ", ".join(f"{ticker} {_usd(value)}" for ticker, value in ordered.head(MAX_TICKERS_LISTED).items())
    hidden = len(ordered) - MAX_TICKERS_LISTED
    return f"{listed}, +{hidden} more" if hidden > 0 else listed


def make_state(now: pd.Timestamp, cumulative_pnl: float, positions: list[dict[str, Any]]) -> dict[str, Any]:
    """Capture what the next run compares against."""
    return {
        "ts": now.isoformat(),
        "cumulative_pnl": cumulative_pnl,
        "positions": {
            _position_key(row): {"size": abs(float(row["Sz"])), "upnl": float(row["uPnL"])}
            for row in positions
        },
    }


def load_state(data_dir: Path | str, file_name: str = STATE_FILE_NAME) -> dict[str, Any] | None:
    """Return the last run's saved state, or ``None`` when there is no usable file."""
    try:
        return json.loads((Path(data_dir) / file_name).read_text())
    except (FileNotFoundError, ValueError):
        return None


def save_state(data_dir: Path | str, state: dict[str, Any], file_name: str = STATE_FILE_NAME) -> None:
    (Path(data_dir) / file_name).write_text(json.dumps(state, indent=1))


def position_changes(baseline: dict[str, Any], positions: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Name positions opened, closed, or resized since the baseline."""
    previous = baseline["positions"]
    current = {_position_key(row): abs(float(row["Sz"])) for row in positions}
    def label(key: str) -> str:
        wallet, coin, side = key.split("|")
        return f"{coin} {side.lower()} ({wallet})"
    resized = [
        label(key) for key, size in current.items()
        if key in previous and previous[key]["size"] > 0
        and abs(size / previous[key]["size"] - 1) * 100 > RESIZE_TOLERANCE_PCT
    ]
    return {
        "Opened": [label(key) for key in current if key not in previous],
        "Closed": [label(key) for key in previous if key not in current],
        "Resized": resized,
    }


def _window_rows(
    window: str,
    start: pd.Timestamp,
    pnl: float,
    equity: float,
    fills: pd.DataFrame,
    funding: pd.DataFrame,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Classify PnL, closed trades, and funding for one time window."""
    well: list[dict[str, str]] = []
    attention: list[dict[str, str]] = []
    def add(update: str, amount: float, detail: str) -> None:
        (well if amount > 0 else attention).append({"Window": window, "Update": update, "Detail": detail})
    if pd.notna(pnl) and pnl != 0:
        base = equity - pnl
        share = f" ({pnl / base * 100:+.2f}%)" if base > 0 else ""
        add("PnL including open positions", pnl, f"{_usd(pnl)}{share}")
    if not fills.empty:
        # Opening fills only carry a fee, so gross PnL identifies actual closes.
        closes = fills.loc[(pd.to_datetime(fills["ts"], utc=True) >= start) & (fills["grossPnl"] != 0)]
        by_ticker = closes.groupby("coin")["closedPnl"].sum()
        winners, losers = by_ticker[by_ticker > 0], by_ticker[by_ticker < 0]
        if not winners.empty:
            add("Closed at a profit", 1.0, f"{_usd(winners.sum())}: {_ticker_list(winners)}")
        if not losers.empty:
            add("Closed at a loss", -1.0, f"{_usd(losers.sum())}: {_ticker_list(losers)}")
    if not funding.empty:
        net = float(funding.loc[pd.to_datetime(funding["ts"], utc=True) >= start, "fundingPnl"].sum())
        if net:
            add("Funding received" if net > 0 else "Funding paid", net, _usd(net))
    return well, attention


def build_highlights(
    now: pd.Timestamp,
    baseline: dict[str, Any] | None,
    positions: list[dict[str, Any]],
    fills: pd.DataFrame,
    funding: pd.DataFrame,
    cumulative_pnl: float,
    pnl_24h: float,
    equity: float,
    risk_used: float,
    risk_cap: float,
    margin_remaining: float,
    margin_capacity: float,
    missing_stops: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return went-well and needs-attention rows, using ``Now`` for current-state checks."""
    well, attention = _window_rows("Last 24h", now - pd.Timedelta(hours=24), pnl_24h, equity, fills, funding)
    if baseline is not None:
        last_run = pd.Timestamp(baseline["ts"])
        run_well, run_attention = _window_rows(
            "Since last run", last_run, cumulative_pnl - baseline["cumulative_pnl"], equity, fills, funding
        )
        current = {_position_key(row): row for row in positions}
        moves = pd.Series({
            f"{row['Coin']} ({row['Wallet']})": float(row["uPnL"]) - baseline["positions"][key]["upnl"]
            for key, row in current.items() if key in baseline["positions"]
        }, dtype=float)
        moves = moves[moves.abs() >= MIN_MOVE_USD]
        gains, losses = moves[moves > 0].nlargest(3), moves[moves < 0].nsmallest(3)
        if not gains.empty:
            run_well.append({"Window": "Since last run", "Update": "Biggest open-position gains",
                             "Detail": _ticker_list(gains)})
        if not losses.empty:
            run_attention.append({"Window": "Since last run", "Update": "Biggest open-position losses",
                                  "Detail": _ticker_list(losses)})
        well, attention = run_well + well, run_attention + attention

    def now_row(update: str, detail: str) -> dict[str, str]:
        return {"Window": "Now", "Update": update, "Detail": detail}
    risk_detail = f"${risk_used:,.2f} used vs ${risk_cap:,.2f} cap"
    if risk_used > risk_cap:
        attention.insert(0, now_row("Aggregate risk over cap", f"{risk_detail} ({_usd(risk_cap - risk_used)} room)"))
    else:
        well.append(now_row("Aggregate risk within cap", risk_detail))
    if missing_stops:
        attention.insert(0, now_row("Positions missing full stop coverage", ", ".join(missing_stops)))
    elif positions:
        well.append(now_row("Stop coverage", f"All {len(positions)} positions fully covered"))
    if margin_capacity > 0 and margin_remaining / margin_capacity * 100 < LOW_MARGIN_PCT:
        attention.append(now_row(
            "Available margin low",
            f"${margin_remaining:,.2f} left of ${margin_capacity:,.2f} above the reserve",
        ))
    near_liquidation = sorted(
        (float(row["Liq %"]), row["Coin"]) for row in positions
        if pd.notna(row.get("Liq %")) and float(row["Liq %"]) <= LOW_LIQUIDATION_PCT
    )
    if near_liquidation:
        attention.append(now_row(
            f"Within {LOW_LIQUIDATION_PCT:g}% of liquidation",
            ", ".join(f"{coin} {distance:.1f}%" for distance, coin in near_liquidation),
        ))
    return well, attention
