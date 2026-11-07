"""Live desk dashboard: the desk_accounts notebook as a local web page that keeps itself current.

Run from the repository root with the py314 environment (it needs plotly):

    python -m desk.dashboard            # http://127.0.0.1:8050
    python -m desk.dashboard --open --interval 120

A background thread re-reads every account in accounts.json (HL wallets, their tokens on other
chains, and IB through a read-only TWS connection) every ``--interval`` seconds. HL value history,
IB Flex history, benchmark prices and the risk statistics are refreshed hourly; IB account
snapshots are appended to the saved IB history at most every 30 minutes. The page polls for the
latest result and redraws in place.

The server listens on 127.0.0.1 only and has no login, because the page shows account balances.
It never places orders.
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import math
import signal
import threading
import time
import traceback
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from desk.desk_accounts import ROOT_DATA, WINDOWS, DeskAccounts
from desk.risk import MAX_TOTAL_RISK_PCT, per_position_budget

STATIC_DIR = Path(__file__).resolve().parent / "dashboard_static"
DEFAULT_PORT = 8050
DEFAULT_INTERVAL_SECONDS = 180
MIN_INTERVAL_SECONDS = 30
HISTORY_EVERY_SECONDS = 3600
IB_RECORD_EVERY_SECONDS = 1800
MOVERS = 5
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
}


# ----------------------------------------------------------------------------------------------
# Payload: everything the page shows, as plain JSON.

def json_safe(value):
    """Convert numpy/pandas values to JSON types; NaN and infinity become null."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()
    if value is None or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, str) else str(value)


def table(frame: pd.DataFrame, formats: dict[str, str], total: dict | None = None,
          total_last: bool = False) -> dict:
    """A table for the page: column formats, body rows, and an optional total row."""
    if frame is None or frame.empty:
        return {"columns": [], "rows": [], "total": None}
    rows = frame.to_dict(orient="records")
    if total_last:
        rows, total = rows[:-1], rows[-1]
    return {"columns": [{"key": column, "fmt": formats.get(column, "text")} for column in frame.columns],
            "rows": rows, "total": total}


def _daily(series: pd.Series) -> pd.Series:
    series = series.sort_index()
    return series.resample("1D").last().ffill().dropna()


def build_payload(desk: DeskAccounts, risk_table: pd.DataFrame | None) -> dict:
    """The page's data from a fetched DeskAccounts: tiles, highlights, tables and chart series."""
    overview = desk.overview
    trading, wallets = overview._trading(), overview._wallets()
    trading_equity = sum(snapshot.equity_usd for snapshot in trading)
    gross = sum(snapshot.gross_usd for snapshot in trading)
    net = sum(snapshot.net_usd for snapshot in trading)
    unrealized = sum(snapshot.unrealized_usd for snapshot in trading)
    positions = overview.positions_frame()
    returns = desk.desk_returns()
    peak = desk.index.cummax()
    kpis = {
        "desk_value": overview.desk_equity,
        "trading_equity": trading_equity,
        "wallet_value": sum(snapshot.equity_usd for snapshot in wallets),
        "unrealized": unrealized,
        "unrealized_share": unrealized / gross if gross else None,
        "gross": gross, "net": net,
        "gross_leverage": gross / trading_equity if trading_equity else None,
        "net_leverage": net / trading_equity if trading_equity else None,
        "positions": int(sum(len(snapshot.positions) for snapshot in trading)),
        "returns": {"since_start": returns.get("since start"), **{label: returns.get(label) for label in WINDOWS}},
        "since": desk.common_start,
        "drawdown": float(desk.index.iloc[-1] / peak.iloc[-1] - 1),
        "peak_date": desk.index.idxmax(),
    }
    if overview.book:
        equity, used, cap = overview.risk_totals()
        kpis.update(risk_used=used, risk_cap=cap, risk_cap_pct=MAX_TOTAL_RISK_PCT,
                    position_budget=per_position_budget(equity))
    if len(positions):
        top = positions.loc[positions["Value (USD)"].abs().idxmax()]
        kpis["largest"] = {"symbol": top["Symbol"], "side": top["Side"], "account": top["Account"],
                           "value": abs(top["Value (USD)"]), "share": top["Exposure / desk value"]}

    well, attention, _, _ = desk.highlight_rows()

    accounts = overview.accounts_frame()
    account_total = None
    if len(accounts):
        sums = accounts.drop(columns=["Venue", "Account", "% of desk"]).sum(numeric_only=True).to_dict()
        account_total = {"Venue": "Total", "Account": "", **sums, "% of desk": 1.0}
    money = {column: "usd" for column in ("Equity", "Cash", "Margin used", "Available", "Gross", "Value", "Risk equity",
                                           "Open risk", "Long", "Short", "VaR 95% ($)", "VaR 99% ($)")}
    signed = {column: "susd" for column in ("Unrealized PnL", "Net", "Value (USD)")}
    tables = {
        "accounts": table(accounts, {**money, **signed, "Positions": "int", "% of desk": "pct"}, account_total),
        "performance": table(desk.performance_frame(),
                             {"Value": "usd", **{f"PnL {label}": "susd" for label in WINDOWS},
                              **{f"Return {label}": "spct" for label in WINDOWS}}, total_last=True),
        "exposure": table(overview.exposure_frame(), {**money, "Net": "susd", "Gross / equity": "pct",
                                                      "Net / equity": "spct"}, total_last=True),
        "movers": table(overview.movers_frame(MOVERS), {"Value (USD)": "susd", "Unrealized PnL": "susd",
                                                        "% of position": "spct"}),
    }
    if len(positions):
        ordered = positions.assign(_abs=positions["Value (USD)"].abs()).sort_values(
            ["Asset class", "_abs"], ascending=[True, False]).drop(columns="_abs")
        total = {column: None for column in ordered.columns}
        total.update({"Asset class": "Total", "Value (USD)": ordered["Value (USD)"].sum(),
                      "Unrealized PnL": ordered["Unrealized PnL"].sum(), "Weight": ordered["Weight"].sum(),
                      "Exposure / desk value": ordered["Exposure / desk value"].sum()})
        tables["positions"] = table(ordered, {"Quantity": "num", "Avg price": "num", "Mark": "num", **signed,
                                              "Weight": "pct", "Exposure / desk value": "pct"}, total)
    else:
        tables["positions"] = table(positions, {})
    if overview.book:
        tables["open_risk"] = table(overview.risk_frame(), {**money, "Positions": "int", "% of desk equity": "pct"},
                                    total_last=True)
    holdings = overview.wallet_holdings_frame()
    if len(holdings):
        tables["wallets"] = table(holdings, {"Amount": "num", "Price": "num", "Value": "usd", "% of desk": "pct"},
                                  {"Wallet": "Total", "Value": holdings["Value"].sum(),
                                   "% of desk": holdings["% of desk"].sum()})
    if risk_table is not None and len(risk_table):
        tables["risk"] = table(risk_table, {
            **money, "Periods": "int", "Return (total)": "spct", "Volatility (/yr)": "pct", "Sharpe": "num2",
            "Max drawdown": "spct", "Beta BTC": "num2", "Beta SPY": "num2",
            "Alpha vs BTC (/yr)": "spct", "Alpha vs SPY (/yr)": "spct"})

    start = desk.common_start
    value_series = []
    for name, history in desk.histories.items():
        series = _daily(history.set_index(pd.to_datetime(history["ts"], utc=True))["accountValue"])
        series = series[series.index >= start.normalize()]
        value_series.append({"name": name, "x": [ts.strftime("%Y-%m-%d") for ts in series.index],
                             "y": series.round(2).tolist()})
    index = _daily(desk.index)
    index.index = index.index.normalize()
    twr = [{"name": "Desk (net of deposits)",
            "x": [ts.strftime("%Y-%m-%d") for ts in index.index],
            "y": (100 * index / index.iloc[0]).round(3).tolist()}]
    for name in ("BTC", "SPY"):
        if name in desk.benchmarks:
            prices = desk.benchmarks[name].dropna()
            prices = prices[prices.index >= index.index[0]]
            if len(prices):
                twr.append({"name": name, "x": [ts.strftime("%Y-%m-%d") for ts in prices.index],
                            "y": (100 * prices / prices.iloc[0]).round(3).tolist()})

    notes = list(desk.notes)
    notes += [f"{snapshot.label}: {note}" for snapshot in overview.snapshots for note in snapshot.notes]
    return {
        "kpis": kpis,
        "highlights": {"well": [list(row) for row in well], "attention": [list(row) for row in attention]},
        "tables": tables,
        "charts": {"value": value_series, "twr": twr},
        "problems": list(overview.problems),
        "notes": notes,
    }


# ----------------------------------------------------------------------------------------------
# Refresh loop

class Dashboard:
    """Holds the latest payload and refreshes it in a background thread."""

    def __init__(self, interval: int = DEFAULT_INTERVAL_SECONDS, data_dir: Path | str = ROOT_DATA,
                 wallet_data_dir: Path | str | None = None):
        self.interval = max(MIN_INTERVAL_SECONDS, int(interval))
        self.desk = DeskAccounts(data_dir=data_dir, wallet_data_dir=wallet_data_dir)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.payload: dict | None = None
        self.payload_json = b""
        self.risk_table: pd.DataFrame | None = None
        self.last_history = 0.0
        self.last_ib_record = 0.0
        self.last_seen: dict[str, str] = {}
        self.status = {"refreshing": False, "last_ok": None, "last_attempt": None, "error": None,
                       "duration_s": None, "history_at": None, "interval_s": self.interval}

    def account_status(self, problems: list[str]) -> list[dict]:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        rows = []
        for account in self.desk.accounts:
            problem = next((item for item in problems if item.split(":")[0].strip() == account.name), None)
            if problem is None:
                self.last_seen[account.name] = now
            rows.append({"name": account.name, "venue": account.venue, "ok": problem is None,
                         "problem": problem, "last_seen": self.last_seen.get(account.name)})
        return rows

    def refresh(self) -> None:
        started = time.time()
        history = started - self.last_history >= HISTORY_EVERY_SECONDS
        record = started - self.last_ib_record >= IB_RECORD_EVERY_SECONDS
        with self.lock:
            self.status.update(refreshing=True, last_attempt=pd.Timestamp.now(tz="UTC").isoformat())
            self._publish()
        try:
            self.desk.fetch_data(refresh_history=history, record_ib_history=record)
            if history or self.risk_table is None:
                self.risk_table = self.desk.risk_metrics()
            payload = build_payload(self.desk, self.risk_table)
            payload["accounts_status"] = self.account_status(payload["problems"])
            with self.lock:
                self.payload = json_safe(payload)
                self.status.update(last_ok=pd.Timestamp.now(tz="UTC").isoformat(), error=None,
                                   duration_s=round(time.time() - started, 1))
                if history:
                    self.last_history = started
                    self.status["history_at"] = self.status["last_ok"]
                if record and any(snapshot.venue == "ib" for snapshot in self.desk.overview.snapshots):
                    self.last_ib_record = started
        except Exception as exc:  # keep serving the last good data and say what failed
            traceback.print_exc()
            with self.lock:
                self.status["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.status["refreshing"] = False
                self._publish()

    def _publish(self) -> None:
        """Serialize payload and status once per change (the page polls often)."""
        document = {"status": self.status, "data": self.payload,
                    "server_time": pd.Timestamp.now(tz="UTC").isoformat()}
        self.payload_json = json.dumps(json_safe(document), allow_nan=False).encode()

    def state_json(self) -> bytes:
        with self.lock:
            if not self.payload_json:
                self._publish()
            return self.payload_json

    def run_forever(self) -> None:
        # ib_async needs an event loop in this thread; the desk code calls it directly here.
        asyncio.set_event_loop(asyncio.new_event_loop())
        while True:
            self.refresh()
            self.wake.wait(self.interval)
            self.wake.clear()


# ----------------------------------------------------------------------------------------------
# HTTP

def make_handler(dashboard: Dashboard, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    plotly_js = None

    class Handler(BaseHTTPRequestHandler):
        server_version = "DeskDashboard/1.0"

        def _host_ok(self) -> bool:
            # Rejects other hostnames pointed at this port (DNS rebinding) so only this machine's
            # browser, addressing it as localhost, can read the balances.
            if self.headers.get("Host", "") not in allowed_hosts:
                self.send_error(403, "Use http://127.0.0.1 or http://localhost")
                return False
            return True

        def _send(self, body: bytes, content_type: str, cache: bool = False) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            nonlocal plotly_js
            if not self._host_ok():
                return
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                self._send(dashboard.state_json(), "application/json")
            elif path in STATIC_FILES:
                name, content_type = STATIC_FILES[path]
                self._send((STATIC_DIR / name).read_bytes(), content_type)
            elif path == "/plotly.min.js":
                if plotly_js is None:
                    import plotly

                    plotly_js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_bytes()
                self._send(plotly_js, "text/javascript; charset=utf-8", cache=True)
            else:
                self.send_error(404)

        def do_POST(self):
            if not self._host_ok():
                return
            origin = self.headers.get("Origin")
            if origin and origin.split("//", 1)[-1] not in allowed_hosts:
                self.send_error(403)
                return
            if self.path == "/api/refresh":
                dashboard.wake.set()
                self._send(b'{"ok": true}', "application/json")
            else:
                self.send_error(404)

        def log_message(self, format, *args):  # quiet: the page polls every few seconds
            if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
                super().log_message(format, *args)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live desk dashboard on http://127.0.0.1 (read-only).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS,
                        help=f"seconds between account refreshes (minimum {MIN_INTERVAL_SECONDS})")
    parser.add_argument("--data-dir", default=str(ROOT_DATA), help="desk data folder (default: the repo's data/)")
    parser.add_argument("--wallet-data-dir", default=None, help="HL history folder (default: the repo's data/)")
    parser.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    args = parser.parse_args(argv)

    # `kill -USR1 <pid>` writes every thread's stack to the log: shows where a slow refresh is waiting.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    dashboard = Dashboard(args.interval, args.data_dir, args.wallet_data_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(dashboard, args.port))
    threading.Thread(target=dashboard.run_forever, name="desk-refresh", daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Desk dashboard on {url} (refresh every {dashboard.interval}s; Ctrl+C stops). Read-only.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
