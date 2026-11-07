"""Combined account analytics for every desk account: the equivalent of hl_accounts and ib_account.

Combines the HL wallets (saved value and PnL history) and the IB account (IB Flex daily NAV
with deposits and withdrawals) into one desk time-weighted return, so deposits, withdrawals
and transfers between accounts are never performance. Wallet tokens on other chains have no
value history, so they appear in current totals only.

HL keeps detailed history for about a month (older saved points are weekly), so risk
statistics use weekly returns over the full shared history, plus daily returns over the days
every account has daily data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display

from desk.accounts import accounts_for, ib_data_dir, load_accounts
from desk.hl.account_analytics import display_interactive_table, fetch_btc_candles, update_portfolio_history
from desk.hl.trade_analytics import bootstrap_btc_regression, bootstrap_var, time_weighted_index
from desk.notebook_highlights import HighlightsState, percent, signed_money
from desk.notebook_highlights import show as show_highlights
from desk.overview import DeskOverview

ROOT_DATA = Path(__file__).resolve().parents[1] / "data"
RESEARCH_PRICES = Path(__file__).resolve().parents[1] / "research" / "data" / "daily_prices_usd.csv"
# IB NAV is end of the US trading day; stamp it at 21:00 UTC (16:00-17:00 New York).
IB_CLOSE_HOUR_UTC = 21
WINDOWS = {"7D": 7, "30D": 30, "90D": 90}
ACCOUNTS_HIGHLIGHTS_STATE = "desk_accounts_highlights_state.json"
DEEP_DRAWDOWN = -0.10   # the desk this far below its peak is flagged


def ib_history(nav: pd.DataFrame) -> pd.DataFrame:
    """IB Flex NAV as ``ts, accountValue, pnl`` (cumulative PnL = NAV change less deposits)."""
    frame = nav.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    change = frame["nav"].diff() - frame["flows"]
    change.iloc[0] = 0.0  # the first day only sets the starting value
    return pd.DataFrame({
        "ts": frame["date"] + pd.Timedelta(hours=IB_CLOSE_HOUR_UTC),
        "accountValue": frame["nav"].to_numpy(),
        "pnl": change.cumsum().to_numpy(),
    })


def daily_points(history: pd.DataFrame) -> pd.DatetimeIndex:
    """UTC dates on which an account has at least one saved value."""
    return pd.DatetimeIndex(pd.to_datetime(history["ts"], utc=True).dt.normalize().unique())


def period_returns(index: pd.Series, rule: str, valid_days: pd.DatetimeIndex | None = None) -> pd.Series:
    """Returns between consecutive periods of a growth index; gaps are never one period.

    With ``valid_days``, a daily return needs both its day and the day before in that set,
    so a weekly-sampled account's move is never pinned on a single day.
    """
    closes = index.resample(rule).last().dropna()
    closes.index = closes.index.normalize()
    step = pd.Timedelta(days=1) if rule == "1D" else pd.Timedelta(days=7)
    returns = closes.pct_change()
    keep = closes.index.to_series().diff().eq(step)
    if valid_days is not None:
        days = set(valid_days)
        keep &= closes.index.to_series().map(lambda day: day in days and day - step in days)
    return returns[keep].dropna()


class DeskAccounts:
    """Every account in accounts.json as one desk: current state, performance and risk."""

    def __init__(self, data_dir: str | Path = "data", wallet_data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir)
        self.wallet_data_dir = Path(wallet_data_dir) if wallet_data_dir else ROOT_DATA
        self.accounts = load_accounts()
        self.overview = DeskOverview(self.accounts, data_dir=self.data_dir)
        self.histories: dict[str, pd.DataFrame] = {}
        # Accounts whose history starts where a data window starts (IB Flex keeps a year), as
        # opposed to HL wallets, whose history starts when they were funded.
        self.windowed: set[str] = set()
        self.notes: list[str] = []
        self.index = pd.Series(dtype=float)
        self.benchmarks = pd.DataFrame()
        self._highlights_state = None

    # ------------------------------------------------------------------
    def fetch_data(self, refresh_history: bool = True):
        """Current state of every account, plus each account's value and PnL history."""
        self.notes = []
        self._highlights_state = None  # a fresh fetch compares with the last saved run
        self.overview.fetch_data()
        self.histories, self.windowed = {}, set()
        for account in accounts_for("hl", self.accounts):
            try:
                history = (update_portfolio_history(account.id, self.wallet_data_dir) if refresh_history else
                           pd.read_csv(self.wallet_data_dir / f"hl_portfolio_history_{account.id.lower()}.csv"))
                history["ts"] = pd.to_datetime(history["ts"], utc=True, format="mixed")
                self.histories[account.name] = history[["ts", "accountValue", "pnl"]]
            except Exception as exc:
                self.notes.append(f"{account.name}: no value history ({type(exc).__name__}: {exc}).")
        for account in accounts_for("ib", self.accounts):
            if refresh_history:
                try:
                    from desk.venues import ib_flex

                    ib_flex.update_nav_history(account, ib_data_dir())
                except Exception as exc:
                    self.notes.append(f"{account.name}: IB Flex not refreshed ({str(exc) or type(exc).__name__}).")
            for path in sorted(ib_data_dir().glob("ib_nav_history_*.csv")):
                label = f"IB {path.stem.rsplit('_', 1)[-1]}"
                self.histories[label] = ib_history(pd.read_csv(path))
                self.windowed.add(label)
        if not self.histories:
            raise RuntimeError("No account has value history yet.")
        # The combined desk starts where every account's value is known. An HL wallet's history
        # starts when it was funded (it was worth nothing before), so it can join later as an
        # inflow; an IB history starts where the Flex window starts, which limits the desk start.
        firsts = {name: history["ts"].min() for name, history in self.histories.items()}
        start = max([firsts[name] for name in self.windowed] or [min(firsts.values())])
        clipped = []
        for history in self.histories.values():
            part = history[history["ts"] >= start].copy()
            # HL PnL is cumulative since the wallet opened; measure it from the first point in the
            # window, or the lifetime PnL so far would land as one day's loss or gain.
            if len(part):
                part["pnl"] = part["pnl"] - part["pnl"].iloc[0]
            clipped.append(part)
        self.index = time_weighted_index(clipped)
        self.index.index = pd.to_datetime(self.index.index, utc=True)
        self.common_start = start
        joined = [f"{name} ({first:%Y-%m-%d})" for name, first in firsts.items() if first > start + pd.Timedelta(days=1)]
        if joined:
            self.notes.append("Joined the desk after it starts, so their first funding counts as an inflow: "
                              + ", ".join(joined) + ".")
        # A day counts for daily returns when every account has a value that day, or did not exist yet.
        self.daily_days = None
        for name, history in self.histories.items():
            days = set(daily_points(history))
            if name not in self.windowed:
                first = pd.Timestamp(firsts[name]).normalize()
                days |= set(pd.date_range(start.normalize(), first - pd.Timedelta(days=1), freq="D", tz="UTC"))
            self.daily_days = days if self.daily_days is None else self.daily_days & days
        self.daily_days = pd.DatetimeIndex(sorted(self.daily_days or []))
        self.benchmarks = self._benchmarks(start - pd.Timedelta(days=10))
        if any(account.venue == "hl" and account.onchain for account in self.accounts):
            self.notes.append("Wallet tokens on other chains have no value history; they count in current totals only.")
        for note in self.notes:
            print(f"note: {note}")
        return self.index

    def _benchmarks(self, start: pd.Timestamp) -> pd.DataFrame:
        """Daily BTC and SPY closes for the comparison period."""
        frame = pd.DataFrame()
        try:
            candles = fetch_btc_candles(int(start.timestamp() * 1000),
                                        int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), interval="1d")
            dates = pd.to_datetime(candles["t"], unit="ms", utc=True).dt.normalize()
            frame["BTC"] = pd.Series(candles["c"].astype(float).to_numpy(), index=dates)
        except Exception as exc:
            self.notes.append(f"BTC prices unavailable ({type(exc).__name__}).")
        spy, ib_problem = None, "no IB account configured"
        ib = accounts_for("ib", self.accounts)
        if ib:
            try:
                from desk.venues import ib as ib_venue

                def read():
                    client = ib_venue.connect(ib[0], "analytics", readonly=True, timeout=20)
                    try:
                        contract, _ = ib_venue.resolve_contract(client, ib_venue.ContractSpec("SPY"))
                        return ib_venue.adjusted_daily_closes(client, contract, "2 Y")
                    finally:
                        client.disconnect()
                spy = ib_venue.run_off_event_loop(read)
            except Exception as exc:
                spy, ib_problem = None, str(exc).split(";")[0].rstrip(".") or type(exc).__name__
        if spy is None and RESEARCH_PRICES.exists():
            prices = pd.read_csv(RESEARCH_PRICES, parse_dates=["date"]).set_index("date")
            if "SPY" in prices:
                spy = pd.to_numeric(prices["SPY"], errors="coerce").dropna()
                spy.index = pd.to_datetime(spy.index, utc=True).normalize()
                self.notes.append(f"SPY from the saved research prices, through {spy.index.max():%Y-%m-%d} "
                                  f"(IB: {ib_problem}).")
        if spy is not None:
            frame = frame.join(spy.rename("SPY"), how="outer") if len(frame) else spy.rename("SPY").to_frame()
        return frame.sort_index()

    def _require(self) -> bool:
        if self.index.empty:
            print("Call fetch_data() first.")
            return False
        return True

    # ------------------------------------------------------------------
    def show_accounts(self):
        self.overview.show_accounts()

    def show_exposure(self):
        self.overview.show_exposure()

    def show_positions(self):
        self.overview.show_positions()

    def show_risk(self):
        self.overview.show_risk()

    def account_pnl(self) -> pd.DataFrame:
        """Each account's PnL (net of deposits) and return over standard windows."""
        now = pd.Timestamp.now(tz="UTC")
        rows = []
        for name, history in self.histories.items():
            frame = history.set_index(pd.to_datetime(history["ts"], utc=True)).sort_index()
            row = {"Account": name, "Value": float(frame["accountValue"].iloc[-1])}
            index = time_weighted_index([history])
            index.index = pd.to_datetime(index.index, utc=True)
            for label, days in WINDOWS.items():
                cutoff = now - pd.Timedelta(days=days)
                before = frame[frame.index <= cutoff]
                if before.empty:
                    row[f"PnL {label}"] = row[f"Return {label}"] = np.nan
                    continue
                row[f"PnL {label}"] = float(frame["pnl"].iloc[-1] - before["pnl"].iloc[-1])
                prior = index[index.index <= cutoff]
                row[f"Return {label}"] = float(index.iloc[-1] / prior.iloc[-1] - 1) if len(prior) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def show_performance(self):
        """Desk and per-account returns and PnL over 7, 30 and 90 days and since the shared start."""
        if not self._require():
            return
        table = self.account_pnl()
        now = pd.Timestamp.now(tz="UTC")
        desk = {"Account": "Desk (combined)", "Value": table["Value"].sum()}
        for label, days in WINDOWS.items():
            prior = self.index[self.index.index <= now - pd.Timedelta(days=days)]
            desk[f"Return {label}"] = float(self.index.iloc[-1] / prior.iloc[-1] - 1) if len(prior) else np.nan
            desk[f"PnL {label}"] = table[f"PnL {label}"].sum(min_count=1)
        table = pd.concat([table, pd.DataFrame([desk])], ignore_index=True)
        columns = ["Account", "Value"] + [f"{kind} {label}" for label in WINDOWS for kind in ("PnL", "Return")]
        money = lambda value: "n/a" if not np.isfinite(value) else f"${value:,.2f}"
        percent = lambda value: "n/a" if not np.isfinite(value) else f"{value:+.2%}"
        display_interactive_table(table[columns], bold_last=True, title="DESK PERFORMANCE (NET OF DEPOSITS)",
                                  formatters={column: (percent if column.startswith("Return") else money)
                                              for column in columns[1:]})
        since = float(self.index.iloc[-1] / self.index.iloc[0] - 1)
        print(f"Desk time-weighted return since {self.common_start:%Y-%m-%d} (first date every account has history): "
              f"{since:+.2%}. PnL is each account's value change less deposits and withdrawals; transfers between "
              "desk accounts cancel in the combined return. Wallet tokens on other chains are not included.")

    def desk_returns(self) -> dict[str, float]:
        """Desk time-weighted return over each window and since the shared start."""
        now = pd.Timestamp.now(tz="UTC")
        returns = {"since start": float(self.index.iloc[-1] / self.index.iloc[0] - 1)}
        for label, days in WINDOWS.items():
            prior = self.index[self.index.index <= now - pd.Timedelta(days=days)]
            returns[label] = float(self.index.iloc[-1] / prior.iloc[-1] - 1) if len(prior) else np.nan
        return returns

    def show_highlights(self):
        """Desk performance and risk highlights plus the overview's current-state ones, with changes since the last run."""
        if not self._require():
            return
        well, attention, metrics, items = self.overview.highlight_rows() if self.overview.snapshots else ([], [], {}, {})
        returns = self.desk_returns()
        text = (f"{percent(returns['since start'], signed=True)} since {self.common_start:%Y-%m-%d}; "
                + ", ".join(f"{label} {percent(returns[label], signed=True)}" for label in WINDOWS) + ".")
        (well if returns.get("30D", 0) >= 0 else attention).insert(0, ("Desk return (net of deposits)", text))
        table = self.account_pnl()
        recent = table.dropna(subset=["Return 30D"]) if "Return 30D" in table else table.iloc[0:0]
        if len(recent) > 1:
            best, worst = recent.loc[recent["Return 30D"].idxmax()], recent.loc[recent["Return 30D"].idxmin()]
            well.insert(1, ("Best account, 30D", f"{best['Account']} {percent(best['Return 30D'], signed=True)} "
                                                 f"({signed_money(best['PnL 30D'])})."))
            if worst["Return 30D"] < 0:
                attention.insert(1, ("Worst account, 30D", f"{worst['Account']} {percent(worst['Return 30D'], signed=True)} "
                                                           f"({signed_money(worst['PnL 30D'])})."))
        drawdown = float(self.index.iloc[-1] / self.index.cummax().iloc[-1] - 1)
        peak_day = self.index.idxmax()
        text = f"{percent(drawdown)} below the peak of {peak_day:%Y-%m-%d}." if drawdown < 0 else "At its peak."
        (attention if drawdown <= DEEP_DRAWDOWN else well).append(("Drawdown now", text))
        weekly = period_returns(self.index, "W-SUN")
        sharpe = beta_spy = beta_btc = np.nan
        if len(weekly) >= 8 and weekly.std() > 0:
            sharpe = float(weekly.mean() / weekly.std() * math.sqrt(52))
            bench = self.benchmarks.resample("W-SUN").last().pct_change() if len(self.benchmarks) else pd.DataFrame()
            betas = {}
            for name in ("BTC", "SPY"):
                if name in bench:
                    pair = pd.concat([weekly, bench[name]], axis=1).dropna()
                    if len(pair) >= 8 and pair.iloc[:, 1].var() > 0:
                        betas[name] = float(pair.cov().iloc[0, 1] / pair.iloc[:, 1].var())
            beta_btc, beta_spy = betas.get("BTC", np.nan), betas.get("SPY", np.nan)
            text = (f"Weekly Sharpe {sharpe:.2f}, volatility {weekly.std() * math.sqrt(52):.0%}/yr over "
                    f"{len(weekly)} weeks" + "".join(f"; beta {name} {value:.2f}" for name, value in betas.items()) + ".")
            (well if sharpe > 0 else attention).append(("Risk-adjusted", text))
        for note in self.notes:
            if "not refreshed" in note or ": no value history" in note:
                attention.append(("History", note))
        metrics.update({"Desk return since start": (returns["since start"], "pct"),
                        "Desk return 30D": (returns.get("30D", np.nan), "pct"),
                        "Drawdown now": (drawdown, "pct"), "Weekly Sharpe": (sharpe, "num"),
                        "Beta SPY (weekly)": (beta_spy, "num"), "Beta BTC (weekly)": (beta_btc, "num")})
        if self._highlights_state is None:
            self._highlights_state = HighlightsState(self.data_dir / ACCOUNTS_HIGHLIGHTS_STATE)
        since = self._highlights_state.compare(metrics, items)
        show_highlights(well, attention, since,
                        footnote="Returns are time-weighted and net of deposits and withdrawals. Betas here are "
                                 "simple weekly estimates; the risk metrics section has bootstrap ranges.")

    def risk_metrics(self) -> pd.DataFrame:
        """Volatility, Sharpe, drawdown, VaR and benchmark betas, weekly and (where possible) daily."""
        equity = sum(float(history["accountValue"].iloc[-1]) for history in self.histories.values())
        rows = []
        bench_weekly = self.benchmarks.resample("W-SUN").last().pct_change() if len(self.benchmarks) else pd.DataFrame()
        bench_daily = self.benchmarks.pct_change() if len(self.benchmarks) else pd.DataFrame()
        for label, returns, periods, bench in (
            ("Weekly", period_returns(self.index, "W-SUN"), 52, bench_weekly),
            ("Daily", period_returns(self.index, "1D", self.daily_days), 365, bench_daily),
        ):
            if len(returns) < 8:
                continue
            path = (1 + returns).cumprod()
            row = {"Frequency": label, "Periods": len(returns),
                   "From": f"{returns.index.min():%Y-%m-%d}", "To": f"{returns.index.max():%Y-%m-%d}",
                   "Return (total)": float(path.iloc[-1] - 1),
                   "Volatility (/yr)": float(returns.std() * math.sqrt(periods)),
                   "Sharpe": float(returns.mean() / returns.std() * math.sqrt(periods)) if returns.std() > 0 else np.nan,
                   "Max drawdown": float((path / path.cummax() - 1).min())}
            if len(returns) >= 20:
                for confidence in (0.95, 0.99):
                    var = bootstrap_var(returns, confidence)
                    row[f"VaR {confidence:.0%} ($)"] = var["var"] * equity
            for name in ("BTC", "SPY"):
                if name in bench:
                    pair = pd.concat([returns.rename("desk"), bench[name].rename(name)], axis=1).dropna()
                    stats = bootstrap_btc_regression(pair["desk"], pair[name], periods_per_year=periods, samples=2000)
                    row[f"Beta {name}"] = stats["beta"]
                    row[f"Beta {name} 90% range"] = (f"{stats['beta_low']:.2f} to {stats['beta_high']:.2f}"
                                                     if np.isfinite(stats.get("beta_low", np.nan)) else "n/a")
                    row[f"Alpha vs {name} (/yr)"] = stats["alpha"] / 100
            rows.append(row)
        return pd.DataFrame(rows)

    def show_risk_metrics(self):
        if not self._require():
            return
        table = self.risk_metrics()
        if table.empty:
            print("Not enough shared history for risk statistics yet.")
            return
        formats = {"Return (total)": "{:+.1%}", "Volatility (/yr)": "{:.1%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}",
                   "VaR 95% ($)": "${:,.2f}", "VaR 99% ($)": "${:,.2f}", "Beta BTC": "{:.2f}", "Beta SPY": "{:.2f}",
                   "Alpha vs BTC (/yr)": "{:+.1%}", "Alpha vs SPY (/yr)": "{:+.1%}", "Periods": "{:.0f}"}
        display(table.set_index("Frequency").style.format({k: v for k, v in formats.items() if k in table}, na_rep="n/a"))
        print("Weekly statistics cover the whole shared history; daily ones only the days every account has a daily "
              "value (HL keeps detailed history for about a month). VaR is one period's bootstrap loss on today's desk "
              "value. Betas use 2,000 bootstrap resamples; alpha is versus each benchmark alone.")
        self.risk_table = table

    def plot_value(self):
        """Desk value stacked by account, and the desk return against BTC and SPY."""
        if not self._require():
            return
        figure = go.Figure()
        for name, history in self.histories.items():
            series = history.set_index(pd.to_datetime(history["ts"], utc=True))["accountValue"].sort_index()
            series = series.resample("1D").last().ffill()
            figure.add_trace(go.Scatter(x=series.index, y=series, name=name, stackgroup="value", mode="lines"))
        figure.update_layout(title="Desk value by account (USD; wallet tokens excluded)", template="plotly_white",
                             hovermode="x unified", yaxis_tickprefix="$")
        figure.show()
        index = self.index.resample("1D").last().dropna()
        index.index = index.index.normalize()
        comparison = go.Figure(go.Scatter(x=index.index, y=100 * index / index.iloc[0], name="Desk (net of deposits)"))
        for name, color in (("BTC", "#F7931A"), ("SPY", "#16a34a")):
            if name in self.benchmarks:
                prices = self.benchmarks[name].dropna()
                prices = prices[prices.index >= index.index[0]]
                if len(prices):
                    comparison.add_trace(go.Scatter(x=prices.index, y=100 * prices / prices.iloc[0], name=name,
                                                    line={"color": color}))
        comparison.update_layout(title=f"Desk time-weighted return vs BTC and SPY (start = 100, from {index.index[0]:%Y-%m-%d})",
                                 template="plotly_white", hovermode="x unified")
        comparison.show()
