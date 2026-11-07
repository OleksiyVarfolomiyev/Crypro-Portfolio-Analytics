"""Desk overview: balances, positions, exposure, and open risk across every venue account.

Reads each account in accounts.json (HL wallets through the public API, IB
through TWS with a read-only connection) and never places orders. IB account values and
executions are saved under data/ so IB history builds up from run to run.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from IPython.display import display

from desk.accounts import Account, accounts_for, ib_data_dir, load_accounts
from desk.hl.account_analytics import display_interactive_table
from desk.notebook_highlights import HighlightsState, money, percent, signed_money
from desk.notebook_highlights import show as show_highlights
from desk.risk import (
    MAX_TOTAL_RISK_PCT,
    BookEntry,
    hl_book_entry,
    ib_book_entry,
    per_position_budget,
)
from desk.venues import hl as hl_venue
from desk.venues import ib as ib_venue
from desk.venues import onchain
from desk.venues.base import VenueAccount

OVERVIEW_HIGHLIGHTS_STATE = "desk_overview_highlights_state.json"
HIGHLIGHT_MOVERS = 3
CONCENTRATION_PCT = 0.25   # one position above this share of desk value is flagged
HIGH_LEVERAGE = 2.0        # gross exposure / trading equity
LOW_AVAILABLE_PCT = 0.10   # available margin / trading equity
NEAR_CAP_PCT = 0.80        # open risk above this share of the cap is flagged
IB_HISTORY_COLUMNS = ["ts", "equity_usd", "cash_usd", "margin_used_usd", "unrealized_usd", "gross_usd", "net_usd"]


def _money(value):
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"${value:,.2f}"


def _percent(value):
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.2%}"


def _signed_style(row_index, column, value, row):
    if isinstance(value, (int, float)) and math.isfinite(value) and column in {"Unrealized PnL", "Net", "Value (USD)"}:
        return "color:#188038" if value > 0 else "color:#d93025" if value < 0 else "color:inherit"
    # Balances, quantities and prices are not gains or losses, so keep them uncoloured.
    return "color:inherit"


def _pnl_share(position) -> float:
    """Unrealized PnL as a share of the position's value at entry."""
    entry_value = abs(position.quantity * position.avg_price * position.multiplier * position.fx_to_usd)
    return position.unrealized_usd / entry_value if entry_value and math.isfinite(entry_value) else float("nan")


def save_ib_history(snapshot: VenueAccount, data_dir: Path) -> Path:
    """Append this run's IB account values to ``ib_account_history_<account>.csv``."""
    path = data_dir / f"ib_account_history_{snapshot.account}.csv"
    row = pd.DataFrame([{
        "ts": snapshot.as_of.isoformat(), "equity_usd": snapshot.equity_usd, "cash_usd": snapshot.cash_usd,
        "margin_used_usd": snapshot.margin_used_usd, "unrealized_usd": snapshot.unrealized_usd,
        "gross_usd": snapshot.gross_usd, "net_usd": snapshot.net_usd,
    }], columns=IB_HISTORY_COLUMNS)
    frame = pd.concat([pd.read_csv(path), row], ignore_index=True) if path.exists() else row
    frame.to_csv(path, index=False)
    return path


def merge_ib_fills(fills: pd.DataFrame, data_dir: Path, account: str) -> pd.DataFrame:
    """Merge fills into ``ib_fills_<account>.csv`` by execution ID and return all saved fills.

    TWS only returns about a week of executions, so saving them each run keeps the history.
    """
    path = data_dir / f"ib_fills_{account}.csv"
    saved = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if fills.empty:
        return saved
    fills = fills.assign(time=fills["time"].astype(str))
    merged = pd.concat([saved, fills], ignore_index=True).drop_duplicates("exec_id", keep="last")
    merged = merged.sort_values("time").reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


class DeskOverview:
    """Read every configured account and show desk-wide tables."""

    def __init__(self, accounts: list[Account] | None = None, data_dir: str | Path = "data"):
        self.accounts = load_accounts() if accounts is None else accounts
        if not self.accounts:
            raise ValueError("No accounts configured: copy accounts.example.json to accounts.json and edit it.")
        self.data_dir = Path(data_dir)
        self.snapshots: list[VenueAccount] = []
        self.book: list[BookEntry] = []
        self.ib_fills: dict[str, pd.DataFrame] = {}
        self.wallet_activity: dict[str, pd.DataFrame] = {}
        self.problems: list[str] = []
        self._highlights_state = None

    def fetch_data(self, include_risk: bool = True, include_wallet_activity: bool = False):
        """Read balances and positions (and, by default, ATR risk) for every account.

        Wallet transfer history is the slowest read, so it is fetched only on request.
        """
        snapshots = ib_venue.run_off_event_loop(self._fetch_data, include_risk, include_wallet_activity)
        for problem in self.problems:
            print(f"WARNING: {problem}")
        return snapshots

    def _fetch_data(self, include_risk: bool = True, include_wallet_activity: bool = False):
        self.snapshots, self.book, self.problems, self.ib_fills = [], [], [], {}
        self._highlights_state = None  # a fresh fetch compares with the last saved run
        self.wallet_activity = {}
        self.data_dir.mkdir(parents=True, exist_ok=True)
        hl_accounts = accounts_for("hl", self.accounts)
        for account in hl_accounts:
            try:
                self.snapshots.append(hl_venue.snapshot(account))
            except Exception as exc:  # network and API shape errors alike
                self.problems.append(f"{account.name}: {type(exc).__name__}: {exc}")
        for account in hl_accounts if include_risk else []:
            # Per wallet, so one unreadable wallet does not hide the others' risk.
            try:
                self.book.append(hl_book_entry([account.id], account.name))
            except Exception as exc:
                self.problems.append(f"{account.name} risk: {type(exc).__name__}: {exc}")
        for account in accounts_for("ib", self.accounts):
            try:
                ib = ib_venue.connect(account, "overview", readonly=True)
            except Exception as exc:
                self.problems.append(f"{account.name}: {str(exc) or type(exc).__name__}")
                continue
            try:
                if include_risk:
                    entry, snapshot = ib_book_entry(account, ib)
                    self.book.append(entry)
                else:
                    snapshot = ib_venue.snapshot_from_ib(ib, account)
                self.snapshots.append(snapshot)
                save_ib_history(snapshot, ib_data_dir())
                self.ib_fills[snapshot.account] = merge_ib_fills(
                    ib_venue.recent_fills(ib, snapshot.account), ib_data_dir(), snapshot.account)
            except Exception as exc:
                self.problems.append(f"{account.name}: {str(exc) or type(exc).__name__}")
            finally:
                ib.disconnect()
        for account in hl_accounts:
            if not account.onchain:
                continue
            # The same address on other EVM chains: holdings and transfers outside HL.
            try:
                snapshot, _ = onchain.wallet_snapshot(account)
                self.snapshots.append(snapshot)
            except Exception as exc:
                self.problems.append(f"{account.name} wallet: {type(exc).__name__}: {exc}")
            if not include_wallet_activity:
                continue
            try:
                activity, notes = onchain.update_activity_file(
                    account, self.data_dir,
                    desk_wallets={item.id.lower(): item.name for item in hl_accounts})
                self.wallet_activity[account.name] = activity
                self.problems.extend(f"{account.name} wallet: {note}" for note in notes
                                     if "unavailable" in note)
            except Exception as exc:
                self.problems.append(f"{account.name} wallet activity: {type(exc).__name__}: {exc}")
        return self.snapshots

    @property
    def desk_equity(self) -> float:
        return sum(snapshot.equity_usd for snapshot in self.snapshots)

    def _require(self) -> bool:
        """Return whether any account was read; otherwise say why and show nothing."""
        if self.snapshots:
            return True
        if self.problems:
            print("No account could be read: " + "; ".join(self.problems))
        else:
            print("No account data yet: run fetch_data() first.")
        return False

    def show_accounts(self):
        """One row per account plus a desk total."""
        if not self._require():
            return
        rows = [{
            "Venue": snapshot.venue, "Account": snapshot.label, "Equity": snapshot.equity_usd,
            "Cash": snapshot.cash_usd, "Margin used": snapshot.margin_used_usd,
            "Available": snapshot.available_usd, "Unrealized PnL": snapshot.unrealized_usd,
            "Gross": snapshot.gross_usd, "Net": snapshot.net_usd, "Positions": len(snapshot.positions),
            "% of desk": snapshot.equity_usd / self.desk_equity if self.desk_equity else float("nan"),
        } for snapshot in self.snapshots]
        frame = pd.DataFrame(rows)
        total = frame.drop(columns=["Venue", "Account", "% of desk"]).sum(numeric_only=True)
        frame.loc[len(frame)] = {"Venue": "Total", "Account": "", **total.to_dict(), "% of desk": 1.0}
        money = {column: _money for column in ["Equity", "Cash", "Margin used", "Available",
                                                "Unrealized PnL", "Gross", "Net"]}
        display_interactive_table(frame, formatters={**money, "% of desk": _percent, "Positions": "{:.0f}".format},
                                  cell_style=_signed_style, bold_last=True, title="DESK ACCOUNTS")
        for snapshot in self.snapshots:
            for note in snapshot.notes:
                print(f"{snapshot.label}: {note}")
        print("All values are in USD; IB values in other currencies use IB's own exchange rates. "
              "Onchain rows are each HL wallet's own token holdings on other chains, "
              "valued at DefiLlama prices; they count toward desk value but not toward trading "
              "risk equity, since the executors cannot trade them.")

    def positions_frame(self) -> pd.DataFrame:
        equity = self.desk_equity
        rows = []
        for snapshot in self.snapshots:
            for position in snapshot.positions:
                rows.append({
                    "Asset class": position.asset_class, "Venue": position.venue, "Account": snapshot.label,
                    "Symbol": position.symbol, "Side": position.side, "Quantity": position.quantity,
                    "Avg price": position.avg_price, "Mark": position.mark_price, "Currency": position.currency,
                    "Value (USD)": position.notional_usd, "Unrealized PnL": position.unrealized_usd,
                    "Exposure / desk value": abs(position.notional_usd) / equity if equity else float("nan"),
                })
        frame = pd.DataFrame(rows)
        if len(frame):
            # Weight is the position's share of all gross exposure, so it sums to 100% across every
            # table; exposure / desk value sums to gross leverage, which leveraged perps push past 100%.
            gross = frame["Value (USD)"].abs().sum()
            frame.insert(len(frame.columns) - 1, "Weight", frame["Value (USD)"].abs() / gross if gross else float("nan"))
        return frame

    def show_positions(self):
        """Open positions on every venue, one table per asset class."""
        if not self._require():
            return
        frame = self.positions_frame()
        if frame.empty:
            print("No open positions.")
            return
        number = "{:,.6g}".format
        for klass, group in frame.groupby("Asset class", sort=True):
            group = group.sort_values("Value (USD)", key=abs, ascending=False).drop(columns="Asset class")
            total = {"Venue": "Total", "Account": "", "Symbol": "", "Side": "", "Quantity": float("nan"),
                     "Avg price": float("nan"), "Mark": float("nan"), "Currency": "",
                     "Value (USD)": group["Value (USD)"].sum(), "Unrealized PnL": group["Unrealized PnL"].sum(),
                     "Weight": group["Weight"].sum(), "Exposure / desk value": group["Exposure / desk value"].sum()}
            group = pd.concat([group, pd.DataFrame([total])], ignore_index=True)
            display_interactive_table(
                group, formatters={"Quantity": number, "Avg price": number, "Mark": number,
                                   "Value (USD)": _money, "Unrealized PnL": _money, "Weight": _percent,
                                   "Exposure / desk value": _percent},
                cell_style=_signed_style, bold_last=True, title=f"OPEN POSITIONS: {klass.upper()}",
                blank_missing_cells={(len(group) - 1, column) for column in ("Quantity", "Avg price", "Mark")},
            )
        frame = frame.assign(gross=frame["Value (USD)"].abs())
        leverage = frame["gross"].sum() / self.desk_equity if self.desk_equity else float("nan")
        print("Value is signed (shorts negative). Weight is each position's share of gross exposure (absolute "
              "values), so weights sum to 100% across all tables. Exposure / desk value is gross value over desk "
              f"value; summed over every table it is gross leverage ({leverage:.0%} now), above 100% because HL "
              "perps are traded on margin.")

    def show_exposure(self):
        """Long, short, net, and gross exposure by asset class."""
        if not self._require():
            return
        frame = self.positions_frame()
        if frame.empty:
            print("No open positions.")
            return
        value = frame["Value (USD)"]
        summary = pd.DataFrame({
            "Long": value.clip(lower=0).groupby(frame["Asset class"]).sum(),
            "Short": value.clip(upper=0).groupby(frame["Asset class"]).sum(),
        })
        summary["Net"] = summary["Long"] + summary["Short"]
        summary["Gross"] = summary["Long"] - summary["Short"]
        summary.loc["Total"] = summary.sum()
        summary["Gross / equity"] = summary["Gross"] / self.desk_equity
        summary["Net / equity"] = summary["Net"] / self.desk_equity
        summary = summary.reset_index(names="Asset class")
        display_interactive_table(
            summary, formatters={**{column: _money for column in ["Long", "Short", "Net", "Gross"]},
                                 "Gross / equity": _percent, "Net / equity": _percent},
            cell_style=_signed_style, bold_last=True, title="EXPOSURE BY ASSET CLASS")

    def show_risk(self):
        """Estimated open risk per venue against the desk-wide aggregate cap."""
        if not self._require():
            return
        if not self.book:
            print("Risk was not computed; call fetch_data(include_risk=True).")
            return
        # Executor equity: HL margin collateral (USDC on unified accounts) plus IB
        # net liquidation, so the cap shown here is the one the executors enforce.
        equity = sum(entry.equity_usd for entry in self.book)
        cap = equity * MAX_TOTAL_RISK_PCT
        rows = [{"Venue": entry.venue, "Account": entry.label, "Positions": entry.positions,
                 "Risk equity": entry.equity_usd, "Open risk": entry.risk_usd,
                 "% of desk equity": entry.risk_usd / equity if equity else float("nan")}
                for entry in self.book]
        used = sum(entry.risk_usd for entry in self.book)
        rows.append({"Venue": "Total", "Account": "", "Positions": sum(entry.positions for entry in self.book),
                     "Risk equity": equity, "Open risk": used,
                     "% of desk equity": used / equity if equity else float("nan")})
        display_interactive_table(
            pd.DataFrame(rows), formatters={"Risk equity": _money, "Open risk": _money,
                                            "% of desk equity": _percent, "Positions": "{:.0f}".format},
            cell_style=lambda *args: "color:inherit", bold_last=True, title="DESK OPEN RISK")
        if self.problems:
            print("WARNING: unreadable accounts are missing from this risk total.")
        print(f"Aggregate cap: {_money(cap)} ({MAX_TOTAL_RISK_PCT:.0%} of desk equity {_money(equity)}); "
              f"used {_money(used)}; room {_money(cap - used)}.")
        print(f"Per-position risk budget for a new trade: {_money(per_position_budget(equity))}.")
        for entry in self.book:
            for symbol, risk, basis in entry.position_risks:
                if "no price history" in basis:
                    print(f"{entry.label} {symbol}: {_money(risk)} ({basis})")
        print("Risk is each position's 1.25 x ATR(14) stop distance plus round-trip costs; HL "
              "also adds a 5-day adverse funding reserve. Risk equity counts HL margin "
              "collateral (USDC on unified accounts, so other spot tokens are excluded) and IB net "
              "liquidation. Both executors enforce this same cap.")

    def _wallets(self):
        return [snapshot for snapshot in self.snapshots if snapshot.venue == "onchain"]

    def _trading(self):
        return [snapshot for snapshot in self.snapshots if snapshot.venue != "onchain"]

    def show_wallet_holdings(self, min_usd: float = 1.0):
        """Each HL wallet's own tokens on other chains: chain, amount, price and value."""
        if not self._require():
            return
        rows = []
        for snapshot in self._wallets():
            for position in snapshot.positions:
                token, _, chain = position.symbol.partition(" (")
                detail = chain.rstrip(")")
                chain_name = detail.rsplit(", ", 1)[-1]
                note = detail.rsplit(", ", 1)[0] if ", " in detail else ""
                if abs(position.notional_usd) < min_usd and "rewards" not in note:
                    continue
                rows.append({"Wallet": snapshot.label, "Chain": chain_name, "Token": token, "Note": note,
                             "Amount": position.quantity, "Price": position.mark_price,
                             "Value": position.notional_usd,
                             "% of desk": position.notional_usd / self.desk_equity if self.desk_equity else float("nan")})
        if not rows:
            print("No wallet holdings outside HL.")
            return
        frame = pd.DataFrame(rows).sort_values("Value", ascending=False)
        total = {"Wallet": "Total", "Chain": "", "Token": "", "Note": "", "Amount": float("nan"), "Price": float("nan"),
                 "Value": frame["Value"].sum(), "% of desk": frame["% of desk"].sum()}
        frame = pd.concat([frame, pd.DataFrame([total])], ignore_index=True)
        last = len(frame) - 1
        number = "{:,.6g}".format
        display_interactive_table(frame, formatters={"Amount": number, "Price": number, "Value": _money,
                                                     "% of desk": _percent},
                                  cell_style=lambda *args: "color:inherit", bold_last=True,
                                  blank_missing_cells={(last, "Amount"), (last, "Price")},
                                  title="WALLET HOLDINGS OUTSIDE HL")
        for snapshot in self._wallets():
            for note in snapshot.notes:
                print(f"{snapshot.label}: {note}")
        print("Tokens are valued at DefiLlama prices; staked tokens at the price of the token they represent. "
              "Unpriced (spam) tokens are left out. Wallet holdings count toward desk value but not toward "
              "trading risk equity.")

    def show_unrealized_pnl(self, top: int = 5):
        """Unrealized PnL by account, and the largest winning and losing positions."""
        if not self._require():
            return
        accounts = []
        for snapshot in self._trading():
            gross = snapshot.gross_usd
            accounts.append({"Venue": snapshot.venue, "Account": snapshot.label, "Unrealized PnL": snapshot.unrealized_usd,
                             "% of account value": snapshot.unrealized_usd / snapshot.equity_usd if snapshot.equity_usd else float("nan"),
                             "% of position value": snapshot.unrealized_usd / gross if gross else float("nan"),
                             "Positions": len(snapshot.positions)})
        frame = pd.DataFrame(accounts)
        total = {"Venue": "Total", "Account": "", "Unrealized PnL": frame["Unrealized PnL"].sum(),
                 "% of account value": frame["Unrealized PnL"].sum() / sum(s.equity_usd for s in self._trading()),
                 "% of position value": frame["Unrealized PnL"].sum() / sum(s.gross_usd for s in self._trading()),
                 "Positions": frame["Positions"].sum()}
        frame = pd.concat([frame, pd.DataFrame([total])], ignore_index=True)
        display_interactive_table(frame, formatters={"Unrealized PnL": _money, "% of account value": _percent,
                                                     "% of position value": _percent, "Positions": "{:.0f}".format},
                                  cell_style=_signed_style, bold_last=True, title="UNREALIZED PNL BY ACCOUNT")
        positions = pd.DataFrame([{
            "Account": snapshot.label, "Symbol": position.symbol, "Side": position.side,
            "Value (USD)": position.notional_usd, "Unrealized PnL": position.unrealized_usd,
            "% of position": _pnl_share(position),
        } for snapshot in self._trading() for position in snapshot.positions])
        if positions.empty:
            return
        ranked = positions.sort_values("Unrealized PnL")
        movers = pd.concat([ranked.tail(top).iloc[::-1], ranked.head(top)]).drop_duplicates()
        display_interactive_table(movers, formatters={"Value (USD)": _money, "Unrealized PnL": _money, "% of position": _percent},
                                  cell_style=_signed_style, title=f"LARGEST WINNERS AND LOSERS (TOP {top} EACH)")
        print("Unrealized PnL is on open trading positions (HL and IB); wallet tokens have no cost basis here. "
              "% of position compares the PnL with the position's value at entry.")

    def show_desk_metrics(self):
        """One summary of the whole desk: value, exposure, leverage, PnL and risk."""
        if not self._require():
            return
        trading = self._trading()
        wallets = self._wallets()
        trading_equity = sum(snapshot.equity_usd for snapshot in trading)
        gross = sum(snapshot.gross_usd for snapshot in trading)
        net = sum(snapshot.net_usd for snapshot in trading)
        positions = [(snapshot.label, position) for snapshot in trading for position in snapshot.positions]
        largest = max(positions, key=lambda item: abs(item[1].notional_usd), default=None)
        rows = [
            ("Desk value (all accounts)", _money(self.desk_equity)),
            ("Trading accounts (HL, IB)", _money(trading_equity)),
            ("Wallet holdings outside HL", _money(sum(snapshot.equity_usd for snapshot in wallets))),
            ("Cash", _money(sum(snapshot.cash_usd for snapshot in trading if snapshot.venue == "ib"))
             + " IB cash; HL collateral is in the account values"),
            ("Margin used", _money(sum(snapshot.margin_used_usd for snapshot in trading))),
            ("Gross exposure", f"{_money(gross)} ({gross / trading_equity:.2f}x trading equity)" if trading_equity else _money(gross)),
            ("Net exposure", f"{_money(net)} ({net / trading_equity:+.2f}x trading equity)" if trading_equity else _money(net)),
            ("Long / short", f"{_money(sum(p.notional_usd for _, p in positions if p.quantity > 0))} / "
                             f"{_money(sum(p.notional_usd for _, p in positions if p.quantity < 0))}"),
            ("Unrealized PnL", f"{_money(sum(snapshot.unrealized_usd for snapshot in trading))}"),
            ("Open positions", f"{len(positions)} across {len(trading)} trading accounts"),
        ]
        if largest:
            label, position = largest
            rows.append(("Largest position", f"{position.symbol} {position.side.lower()} {_money(abs(position.notional_usd))} "
                                             f"({abs(position.notional_usd) / self.desk_equity:.1%} of desk, {label})"))
        if self.book:
            equity = sum(entry.equity_usd for entry in self.book)
            used = sum(entry.risk_usd for entry in self.book)
            cap = equity * MAX_TOTAL_RISK_PCT
            rows.append(("Open risk vs cap", f"{_money(used)} of {_money(cap)} ({used / cap:.0%} of the "
                                             f"{MAX_TOTAL_RISK_PCT:.0%} cap); room {_money(cap - used)}"))
            rows.append(("Next position budget", _money(per_position_budget(equity))))
        if self.problems:
            rows.append(("Unreadable accounts", "; ".join(problem.split(":")[0] for problem in self.problems)))
        display_interactive_table(pd.DataFrame(rows, columns=["Metric", "Value"]), cell_style=lambda *args: "color:inherit",
                                  title="DESK METRICS")

    def highlight_rows(self) -> tuple[list, list, dict, dict]:
        """What stands out and what needs attention now, plus the key numbers compared run to run."""
        well, attention = [], []
        trading, wallets = self._trading(), self._wallets()
        trading_equity = sum(snapshot.equity_usd for snapshot in trading)
        wallet_value = sum(snapshot.equity_usd for snapshot in wallets)
        gross = sum(snapshot.gross_usd for snapshot in trading)
        net = sum(snapshot.net_usd for snapshot in trading)
        unrealized = sum(snapshot.unrealized_usd for snapshot in trading)
        positions = [(snapshot.label, position) for snapshot in trading for position in snapshot.positions]
        well.append(("Desk value", f"{money(self.desk_equity)}: trading accounts {money(trading_equity)}, "
                                   f"wallet holdings {money(wallet_value)}; {len(positions)} open positions."))
        if positions:
            ranked = sorted(positions, key=lambda item: item[1].unrealized_usd)
            winners = [item for item in ranked[::-1] if item[1].unrealized_usd > 0][:HIGHLIGHT_MOVERS]
            losers = [item for item in ranked if item[1].unrealized_usd < 0][:HIGHLIGHT_MOVERS]
            (well if unrealized >= 0 else attention).append(
                ("Unrealized PnL", f"{signed_money(unrealized)} on {money(gross)} of positions "
                                   f"({percent(unrealized / gross, signed=True) if gross else 'n/a'})."))
            if winners:
                well.append(("Largest winners", ", ".join(f"{p.symbol} {signed_money(p.unrealized_usd)}"
                                                          for _, p in winners) + "."))
            if losers:
                attention.append(("Largest losers", ", ".join(f"{p.symbol} {signed_money(p.unrealized_usd)}"
                                                              for _, p in losers) + "."))
            label, largest = max(positions, key=lambda item: abs(item[1].notional_usd))
            share = abs(largest.notional_usd) / self.desk_equity if self.desk_equity else float("nan")
            if share > CONCENTRATION_PCT:
                attention.append(("Concentration", f"{largest.symbol} {largest.side.lower()} is {share:.0%} of desk "
                                                   f"value ({money(abs(largest.notional_usd))}, {label})."))
        if trading_equity:
            leverage = gross / trading_equity
            text = f"Gross {leverage:.2f}x and net {net / trading_equity:+.2f}x trading equity."
            (attention if leverage > HIGH_LEVERAGE else well).append(("Leverage", text))
            available = sum(snapshot.available_usd for snapshot in trading)
            if available / trading_equity < LOW_AVAILABLE_PCT:
                attention.append(("Available margin low", f"{money(available)} available, "
                                                          f"{available / trading_equity:.0%} of trading equity."))
        risk_used = risk_cap = float("nan")
        if self.book:
            equity = sum(entry.equity_usd for entry in self.book)
            risk_used = sum(entry.risk_usd for entry in self.book)
            risk_cap = equity * MAX_TOTAL_RISK_PCT
            text = (f"{money(risk_used)} of {money(risk_cap)} ({risk_used / risk_cap:.0%} of the cap); "
                    f"next position budget {money(per_position_budget(equity))}.")
            if risk_used > risk_cap:
                attention.insert(0, ("Open risk over cap", text))
            elif risk_used > risk_cap * NEAR_CAP_PCT:
                attention.append(("Open risk near cap", text))
            else:
                well.append(("Open risk within cap", text))
        if self.problems:
            attention.insert(0, ("Unreadable accounts", "; ".join(self.problems) + " Totals leave them out."))
        metrics = {"Desk value": (self.desk_equity, "usd"), "Trading accounts": (trading_equity, "usd"),
                   "Wallet holdings": (wallet_value, "usd"), "Unrealized PnL": (unrealized, "usd"),
                   "Gross exposure": (gross, "usd"), "Net exposure": (net, "usd"),
                   "Open risk": (risk_used, "usd"), "Open positions": (len(positions), "int")}
        items = {"Positions": [f"{p.symbol} {p.side.lower()} ({label})" for label, p in positions]}
        return well, attention, metrics, items

    def show_highlights(self):
        """What stands out, what needs attention, and what changed since the last run."""
        if not self._require():
            return
        well, attention, metrics, items = self.highlight_rows()
        if self._highlights_state is None:
            self._highlights_state = HighlightsState(self.data_dir / OVERVIEW_HIGHLIGHTS_STATE)
        since = self._highlights_state.compare(metrics, items)
        show_highlights(well, attention, since,
                        footnote="Changes in desk value include deposits, withdrawals and price moves in wallet tokens.")

    def show_wallet_activity(self, days: int = 30, min_usd: float = 1.0):
        """Transfers in and out of each wallet on other chains, newest first.

        Values use current prices. Transfers with the HL bridge are labelled; they
        are deposits to and withdrawals from HL itself.
        """
        frames = [frame.assign(Wallet=name) for name, frame in self.wallet_activity.items() if not frame.empty]
        if not frames:
            print("No priced on-chain wallet activity found.")
            return
        activity = pd.concat(frames, ignore_index=True)
        activity["time"] = pd.to_datetime(activity["time"], utc=True, format="mixed")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        recent = activity[(activity["time"] >= cutoff) & (activity["value_usd"].abs() >= min_usd)]
        if recent.empty:
            print(f"No priced on-chain transfers of ${min_usd:g} or more in the last {days} days.")
            return
        sign = recent["direction"].map({"in": 1, "out": -1, "self": 0})
        table = pd.DataFrame({
            "Time": recent["time"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M"),
            "Wallet": recent["Wallet"], "Chain": recent["chain"], "Direction": recent["direction"],
            "Token": recent["symbol"], "Amount": recent["amount"], "Value (USD)": recent["value_usd"] * sign,
            "Counterparty": [label or f"{str(party)[:6]}…{str(party)[-4:]}"
                             for label, party in zip(recent["label"].fillna(""), recent["counterparty"])],
            "Tx": [f"{str(tx)[:10]}…" for tx in recent["hash"]],
        })
        display_interactive_table(table, formatters={"Amount": "{:,.6g}".format, "Value (USD)": _money},
                                  cell_style=_signed_style, title=f"WALLET ACTIVITY OUTSIDE HYPERLIQUID: LAST {days} DAYS")
        labels = recent["label"].fillna("")
        kind = labels.map(lambda label: "the HL bridge" if label == "HL bridge"
                          else "other desk wallets" if label.startswith("Desk wallet") else "everyone else")
        for (wallet, group), total in (recent["value_usd"] * sign).groupby([recent["Wallet"], kind]).sum().items():
            print(f"{wallet}: net {_money(total)} with {group}.")
        print("Values use current prices, not prices at transfer time. Unpriced (spam) tokens are omitted.")

    def show_ib_fills(self, days: int = 30):
        """Saved IB executions from the last ``days`` days, newest first."""
        frames = [frame for frame in self.ib_fills.values() if not frame.empty]
        if not frames:
            print("No saved IB fills yet.")
            return
        fills = pd.concat(frames, ignore_index=True)
        fills["time"] = pd.to_datetime(fills["time"], utc=True, format="mixed")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        fills = fills[fills["time"] >= cutoff].sort_values("time", ascending=False)
        display(fills.drop(columns=["exec_id"]).reset_index(drop=True))
