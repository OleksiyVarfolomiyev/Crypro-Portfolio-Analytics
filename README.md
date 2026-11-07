# Trading Desk

A multi-venue trading desk: analytics, research, and guarded execution across **HL** wallets and **IB** accounts, plus the same wallets' holdings and transfers on other blockchains. Every account in `accounts.json` shares one equity base and one aggregate risk cap, so a trade on either venue is sized and limited by the risk open on both.

| Part | Venues | Where |
| --- | --- | --- |
| Desk overview: accounts, desk metrics, exposure, positions, unrealized PnL, wallet holdings outside HL, desk-wide open risk | All | [`desk_overview.ipynb`](desk_overview.ipynb), `desk/overview.py` |
| Desk accounts: every account combined into one desk return net of deposits; performance by account, risk metrics vs BTC and SPY, charts | All | [`desk_accounts.ipynb`](desk_accounts.ipynb), `desk/desk_accounts.py` |
| Live dashboard: the desk accounts view as a self-refreshing local web page | All | `python -m desk.dashboard`, `desk/dashboard.py` |
| Guarded execution: dry run by default, typed confirmations, desk-wide risk cap, shared kill switch | HL, IB | `execution/hl_trade.py`, `execution/ib_trade.py` |
| Account analytics: PnL, funding, risk metrics, attribution, charts | HL | `hl_account*.ipynb`, `desk/hl/` |
| Account analytics: balances, today's PnL, positions with ATR levels, current-book risk vs SPY, fills, attribution | IB | `ib_account.ipynb`, `desk/ib/` |
| Trade analytics and backtest | HL | `hl_trade_analytics.ipynb`, `hl_backtest.ipynb` |
| Portfolio research: Kelly weights, correlations, benchmark exposures (crypto vs BTC, TradFi vs SPY), factors, market scans | HL, wallets, IB | `research/` |

All notebooks are read-only. Orders are sent only by the execution scripts, which default to a dry run.

## Layout

```
accounts.json            your accounts (not committed; copy accounts.example.json)
desk/                    venue-neutral core
  accounts.py            reads accounts.json
  risk.py                shared sizing rules and the desk-wide open-risk book
  overview.py            cross-venue report used by desk_overview.ipynb
  venues/                one adapter per venue, all returning USD snapshots
    hl.py                public HL API
    ib.py                TWS / IB Gateway through ib_async
    onchain.py           the HL wallets on other EVM chains (Blockscout, Etherscan, DefiLlama)
  hl/                    HL analytics, research, highlights and backtest modules
  ib/                    IB account analytics
execution/               one executor per venue, plus the shared KILL_SWITCH file
research/                portfolio research notebooks and modules
tests/                   unit tests (python -m unittest discover -s tests)
data/                    saved history and caches
```

## Accounts

Copy `accounts.example.json` to `accounts.json` and list every account the desk should report on and risk together:

```json
[
  {"venue": "hl", "id": "0xYOUR_WALLET", "label": "HL main"},
  {"venue": "ib", "id": "U1234567", "label": "IB", "port": 7496}
]
```

HL ids are public wallet addresses. An IB `id` may be left empty to use the first account TWS manages. The file holds no secrets and is ignored by git because it names brokerage accounts.

## IB setup

IB account records (net-liquidation snapshots, fills and Flex NAV history) are saved in `trading-desk-data/ib/`, a folder next to this repository, so account numbers and balances never enter git; set `DESK_IB_DATA_DIR` to use another folder. `data/ib_*` is ignored by git as a safeguard.

The IB adapter talks to a running Trader Workstation (or IB Gateway); TWS's own login and two-factor authentication are the credentials, so no API key is stored.

1. In TWS open **File → Global Configuration → API → Settings** (on a Mac, **Trader Workstation → Global Configuration**).
2. Tick **Enable ActiveX and Socket Clients**.
3. Set **Socket port** to the `port` in `accounts.json`: 7496 for a live login, 7497 for paper.
4. Keep **Allow connections from localhost only** ticked.
5. Untick **Read-Only API** if `execution/ib_trade.py` should place orders; leave it ticked for reporting only.
6. Click **Apply** and **OK**. If the port still refuses connections, restart TWS.
7. On the first connection TWS asks whether to accept it; click **Accept** (or add 127.0.0.1 under **Trusted IPs**).

Install `ib_async` into each environment that talks to IB: `python -m pip install ib_async`. Real-time quotes need IB market-data subscriptions; without them the desk falls back to delayed quotes for reports and dry runs, and live IB entries are refused.

## On-chain wallets

Every HL wallet is an ordinary EVM address, so the desk also reads that address on other chains: token holdings appear as `onchain` accounts and positions, and transfers in and out appear in the overview's wallet-activity table. Transfers with the HL bridge on Arbitrum are labelled as deposits to and withdrawals from HL. Set `"onchain": false` on a wallet's entry in `accounts.json` to skip it.

- Ethereum, Optimism, Unichain, Scroll, zkSync and Ink are read from Blockscout's public API without a key.
- Arbitrum and Polygon need a free Etherscan API key. Create one at etherscan.io, then run `python -m desk.venues.onchain --store-key` to save it in the macOS Keychain, or set `ETHERSCAN_API_KEY`. BNB Chain and Avalanche need a paid Etherscan plan and are listed as skipped.
- Base is read through Alchemy when a free Alchemy key is stored (`python -m desk.venues.onchain --store-alchemy-key`, or `ALCHEMY_API_KEY`): every token balance and the full transfer history, with the key sent in a header. Base counterparties are not named, because Base's explorers need paid plans. Without an Alchemy key, Base is read without any key through Base's public RPC: native ETH and the main Base tokens (USDC, USDbC, WETH, cbBTC, cbETH, DAI, AERO, EURC, wstETH, USDT), with no transfer history, because free public RPCs block history queries. A Blockscout PRO API key on a paid plan (Builder or above; Blockscout's free plan excludes Base as a "featured chain") adds every Base token, transfer history and contract names: run `python -m desk.venues.onchain --store-blockscout-key`, or set `BLOCKSCOUT_API_KEY`. With a free key, Base keeps the keyless balances. Keys are sent in headers where the API allows it, and stored keys are scrubbed from every printed message.
- Counterparties are labelled: the HL bridge, the desk's other wallets (shown separately so transfers between your own wallets are not counted as outside flows), mints and burns, and verified contract names from the chain's explorer (cached in `data/api_cache/contract_names.json`). Unlabelled counterparties are ordinary wallets or unverified contracts.
- Tokens are valued with DefiLlama's public prices. Tokens it cannot price confidently, which includes the scam airdrops most wallets collect, are counted in a note but never valued or listed; balances under $1 are hidden. Transfer values use current prices, not prices at the time of transfer.
- Staked positions are included. VVV staked with Venice on Base appears as sVVV, valued one-for-one at the VVV price, and its unclaimed rewards (the staking contract's `pendingRewards`) appear as a separate line even when under $1. More staking contracts can be added in `PRICED_AS` and `STAKING_REWARDS` in `desk/venues/onchain.py`.
- On-chain balances count toward desk value in the overview but not toward the executors' risk equity, because the executors cannot trade them. Transfers are saved to `data/onchain_activity_<wallet>.csv`; the first run reads a year of history.

## When an account cannot be read

Every account is read on its own, so one failure never hides another. The overview reports each unreadable account and shows everything else; with nothing readable, each section says why. In both executors a dry run warns and leaves the unreadable account out of desk equity and risk. A live entry stops, because that account's risk is unknown, unless `--ignore-unreachable` is given. A broken `accounts.json` is treated the same way. Protective commands (`--protect-open`, `--reattach-guards`, `--kill-switch`) never depend on other accounts, and the HL analytics notebooks do not read `accounts.json` at all.

## IB account notebook

Open [`ib_account.ipynb`](ib_account.ipynb) and run all cells; it is the IB counterpart of `hl_account.ipynb` and uses a read-only TWS connection. It shows net liquidation, cash, buying power, margin, unrealized PnL and today's PnL; open positions with average cost, unrealized PnL, the 1.25 × ATR(14) stop and 1:1 target the executor would place now, risk to that stop, and 60-day beta to SPY; and PnL attribution by symbol.

IB's API has no account-value history, so beta and alpha to SPY, Sharpe, volatility, bootstrap VaR and maximum drawdown price **today's holdings** over the past year of IB daily closes, with 90% bootstrap ranges. They describe the current book, not past account performance. Real history builds up from run to run: each run saves a net-liquidation snapshot to `trading-desk-data/ib/ib_account_history_<account>.csv` and merges executions into `trading-desk-data/ib/ib_fills_<account>.csv` (TWS keeps about a week of them), and the history and realized-PnL sections read those files.

## Live dashboard

`desk_accounts.ipynb` as a web page that keeps itself current. With the py314 environment (it needs plotly), from any folder:

```bash
/Users/admin/anaconda3/envs/py314/bin/python /Users/admin/Dropbox/Investing/trading-desk/run_dashboard.py --open
```

From the repository root, `python -m desk.dashboard --open` does the same.

It serves http://127.0.0.1:8050 (`--port` to change) and re-reads every account every 3 minutes (`--interval` in seconds, minimum 30); the **Refresh now** button refreshes immediately. The page shows key-number tiles (desk value, unrealized PnL, returns, drawdown, leverage, open risk against the cap, largest position), which accounts were read, highlights, desk value by account and the desk return against BTC and SPY, and the performance, accounts, risk-metrics, positions, exposure, open-risk, winners-and-losers and wallet tables. HL value history, IB Flex history, benchmark prices and risk statistics refresh hourly; IB account snapshots are added to the saved IB history at most every 30 minutes. An unreadable account is named in a red banner and left out of every total; a failed refresh keeps the last good data on screen and says how old it is. During IBKR maintenance, IB reads give up after 30 seconds instead of hanging.

The server listens on 127.0.0.1 only and has no login, because the page shows balances. It is read-only and never places orders. `kill -USR1 <pid>` writes every thread's stack to its output if a refresh seems stuck. Code: `desk/dashboard.py`, page in `desk/dashboard_static/`.

## Notebook highlights

Every notebook opens with a Highlights block: what stands out, what needs attention, and, for the account and portfolio notebooks, which key numbers and positions changed since the last run (baselines are saved as `*highlights_state*.json`; the IB one is saved with the IB records). The account, desk and market research notebooks fill it in right after fetching data. The portfolio, backtest and trade analytics notebooks reserve it at the top, and a later cell fills it in once the sections it summarizes have run, so use Run All. Shared rendering lives in `desk/notebook_highlights.py`.

## Desk overview notebook

Open [`desk_overview.ipynb`](desk_overview.ipynb) and run all cells. It reads every account in `accounts.json` and shows account balances in USD, desk metrics (value split between trading accounts and wallet holdings, margin, gross and net exposure and leverage, unrealized PnL, the largest position, open risk against the cap and the next position budget), exposure by asset class, open positions by asset class, desk open risk, each HL wallet's holdings on other chains, unrealized PnL by account with the largest winners and losers, and recent IB fills. An account that cannot be read is reported and left out. Wallet transfer history is no longer fetched by default (`fetch_data(include_wallet_activity=True)` and `show_wallet_activity()` still provide it). Each run appends IB account values to `trading-desk-data/ib/ib_account_history_<account>.csv` and merges IB executions into `trading-desk-data/ib/ib_fills_<account>.csv`.

## Desk accounts notebook

Open [`desk_accounts.ipynb`](desk_accounts.ipynb), the combined equivalent of `hl_accounts.ipynb` and `ib_account.ipynb`. HL wallets contribute their saved value and PnL history and IB its Flex daily NAV with deposits and withdrawals; together they form one desk time-weighted return, so deposits, withdrawals and transfers between accounts are never performance. The desk starts where IB's Flex window starts; an HL wallet funded later joins with its first funding counted as an inflow, and each wallet's PnL is measured from its first point in the window. Sections: accounts; performance (PnL and return net of deposits over 7, 30 and 90 days, per account and combined); risk metrics (weekly over the whole shared history and daily over the days every account has daily data: volatility, Sharpe, max drawdown, bootstrap VaR in dollars, beta and alpha against BTC and SPY with 90% ranges); value stacked by account and the desk return against BTC and SPY; and exposure, positions and open risk. Wallet tokens on other chains have no value history, so they count in current totals only.

## IB execution

Create a signals CSV with `symbol,side` and optional `sec_type,exchange,currency,expiry` columns:

```csv
symbol,side,sec_type,exchange,currency,expiry
AAPL,LONG,,,,
SPY,SHORT,,,,
ES,LONG,FUT,CME,USD,
EURUSD,SHORT,CASH,,,
```

Defaults are a stock (`STK`) routed through SMART in USD. Futures need an exchange and use the nearest contract with at least five days to expiry unless `expiry` is given. FX pairs use IDEALPRO; `CRYPTO` uses PAXOS. Options are not supported.

```bash
.venv/bin/python execution/ib_trade.py --signals ib_signals.csv          # dry run
.venv/bin/python execution/ib_trade.py --signals ib_signals.csv --live   # live, after confirmations
```

Sizing follows the same rules as HL: ATR(14) from IB daily bars, a stop 1.25 × ATR from the reference price, a 1:1 target, a risk budget of 5% of the 80% usable part of desk equity per position, and a 20% desk-wide aggregate cap (`--max-risk-pct` changes it for one run, at most 50%). Risk per unit is measured from the entry limit to the stop plus round-trip commission, scaled by the contract multiplier and converted to USD. The dry run asks IB for a what-if estimate of initial margin and commission and limits size to IB margin above a 20% reserve.

A live run needs `--live`, typing `LIVE TRADING`, reviewing a refreshed plan, and typing `SUBMIT ORDERS`. Each entry is a marketable DAY limit at most two ticks through the spread (and never past the planned limit), bracketed by a GTC stop and GTC take-profit that cancel each other. IB holds all three orders, so no local process needs to keep running. An entry is skipped without a real-time quote, when the price is already through the stop, or after a move of more than half an ATR since the plan. An entry not filled within 60 seconds is cancelled; a partial fill keeps its exits resized to the filled quantity. A contract held on the opposite side must be closed first.

`--protect-open` adds GTC stops and take-profits to open IB positions that have none, such as positions bought in TWS. Stops sit 1.25 × ATR(14) from the current mark and targets the same distance the other way. `--ladder N` splits each position into up to N parts at 1/N, 2/N … of that distance, like the HL ladder; parts shrink to whole shares worth at least $10, so small positions get fewer steps. `--stops-only` leaves out the targets. A dry run lists every level; `--live` asks for `LIVE TRADING`, re-reads positions and orders, then asks for `SUBMIT PROTECTION ORDERS`.

Stops and targets work in extended hours unless `--rth-only` (see below). IB stock orders have no reduce-only flag, so each ladder step's stop and target are linked one-cancels-other: at most one of each pair fills, and the orders can never sell more than the position. For the same reason a position that already has some exit orders is left alone and reported, rather than given unlinked orders on top. IB's API rejects fractional orders, so only whole shares are protected; a fractional remainder (or a position under one share) is reported to set in TWS. Protection works while the kill switch is engaged.

```bash
.venv/bin/python execution/ib_trade.py --protect-open --ladder 3          # dry run
.venv/bin/python execution/ib_trade.py --protect-open --ladder 3 --live   # place, after confirmations
```

**Extended hours.** By default IB orders also work in pre-market and after-hours (US stocks 04:00-09:30 and 16:00-20:00 ET). Entries and targets are limit orders allowed outside regular hours. Stops are GTC stop-limits, the IB order type that can trigger outside regular hours, with the limit half an ATR past the stop (at most 1.5%, because IB rejects a limit too far through the stop) so a fast move can still fill; a gap past that limit leaves the order resting unfilled. If IB still rejects a stop-limit, the executor replaces it at once with a plain regular-hours stop linked to the same target, and if that is rejected too it cancels the target, so no position is left with a target but no stop. `--rth-only` keeps a run to regular hours with plain stop orders. `--protect-open --replace` cancels the desk's own exit orders for each position (only orders it placed, marked `desk-protect`) and places fresh ones, for example to move older regular-hours orders to extended hours or to a new ladder. A cash account treats overlapping sell orders as a short sale, so the new orders are sent only after IB confirms the old ones cancelled.

`--kill-switch` shows (and with `--live` and `SHUT IT ALL DOWN`, executes) closing every IB position. It halts new entries on every venue, cancels non-exit orders, then for each position cancels its exit orders (a cash account will not accept a sell while resting sells claim the shares) and places a limit close that works in extended hours, 2% through the price and widened to 5% and 10%. A close that still does not fill is cancelled and the position gets its protective stop back. When no session is open the closes are left queued at the 10% limit for the next session, with no stop meanwhile. Fractional shares cannot be sold through the API and are listed to close in TWS.

## Desk-wide kill switch

`execution/kill_switch.py` shows (and with `--live` and `SHUT IT ALL DOWN`, executes) the kill switch on every venue at once: it engages the shared halt, flattens every HL wallet in `accounts.json` with the HL kill switch, then closes the IB account as above, and prints a summary per venue. A venue that cannot be reached is reported and the others still run.

```bash
.venv/bin/python execution/kill_switch.py          # dry run: what every venue would close
.venv/bin/python execution/kill_switch.py --live   # after typing SHUT IT ALL DOWN
```

Both executors share `execution/KILL_SWITCH`: while it exists neither opens new positions (protection still works); `--clear-kill-switch` on either removes it.

## Desk-wide risk

Both executors build the same risk book before sizing. Desk equity is HL margin collateral plus IB net liquidation, in USD. Open risk is each position's 1.25 × ATR(14) stop distance plus round-trip costs, plus a 5-day adverse funding reserve on HL; an IB position without usable price history counts 10% of its value. The HL executor reads IB through TWS, and the IB executor reads every configured HL wallet. If a configured account cannot be read, a dry run warns and leaves it out, and a live run stops; `--ignore-unreachable` trades without that account's risk.

## Notebooks

The two notebooks below contain the complete workflows, including price-history refresh.

| Notebook | What it does | Main data files |
| --- | --- | --- |
| [`research/portfolio_analytics.ipynb`](research/portfolio_analytics.ipynb) | Discovers live perp positions and spot balances from configured HL wallets; fetches daily perp prices from HL (including HIP-3 markets) and supported spot prices plus market data from CoinGecko; calculates returns and correlations; optimizes signed long/short constrained full-Kelly weights; fits BTC and available CoinGecko cross-sectional factors; compares portfolio results with BTC; writes metrics and factor outputs to CSV. | `data/daily_prices_usd.csv`, `data/daily_returns.csv`, `data/return_correlation.csv`, `data/kelly_full_weights.csv`, `data/kelly_weights.csv`, `data/portfolio_daily_returns.csv`, `data/portfolio_metrics.csv`, `data/market_data.csv`, `data/portfolio_factor_model.csv`, `data/portfolio_factor_fit.csv`, `data/factor_returns.csv` |
| [`hl_account.ipynb`](hl_account.ipynb) | Analyzes one HL account by reading current account and position data; saves account history locally; reports spot/perp balances, fee-net realized PnL with signed perp funding, unrealized PnL, risk metrics, PnL attribution, and charts against BTC. | `data/hl_account_history_<wallet>.csv` |
| [`hl_accounts.ipynb`](hl_accounts.ipynb) | Consolidates account history, balances, positions, fills, funding, and PnL across configured HL wallets. Displays combined account/risk summaries and charts, pooled PnL attribution, and per-wallet open positions. Position sizing uses a configurable fixed-fractional risk budget and ATR-based stop/take-profit levels. | `data/hl_account_history_<wallet>.csv` for each wallet |
| [`hl_trade_analytics.ipynb`](hl_trade_analytics.ipynb) | Interactive, read-only view of ATR stop distance, historical adverse funding reserve, and expected five-day price/funding PnL for one perp position. | Public HL daily candles and hourly funding history |
| [`hl_backtest.ipynb`](hl_backtest.ipynb) | Interactive, read-only backtest for a list of coin/direction signals. Shows trade-level PnL, funding, fees, and exit reasons. | Public HL daily candles and hourly funding history |

Modules in `desk/hl/` (and `research/*_implementation.py`) provide code used by the portfolio and account reports; run the user-facing notebooks in the table above. The trade notebooks import shared helpers from `desk/hl/trade_analytics.py` and the backtest engine from `desk/hl/backtest.py`. All CSV and spreadsheet data files are stored in the `data/` subfolder. The account notebooks create this folder when needed and read/write their files there. Keep the existing files in `data/` between runs to preserve price and account history.

## Setup

Use Python 3.10 or later and install Jupyter plus the libraries used by the notebooks:

```bash
python -m pip install jupyter pandas numpy matplotlib plotly scipy requests seaborn ib_async
```

Start Jupyter from this repository directory:

```bash
jupyter lab
```

Open either notebook and run its cells from top to bottom. Each notebook is independent; run only the one whose analysis you need.

## 1. Portfolio analytics notebook

Open **research/portfolio_analytics.ipynb** and run all cells in order.

- Set `WALLET_ADDRESSES` in the notebook's setup cell. The refresh discovers nonzero perp positions and spot balances across those wallets using HL's public API, the same wallets' tokens on other chains (as in the desk overview; staked VVV counts as VVV, stablecoins are treated as cash, tokens under $10 are skipped), and IB stock positions through TWS (`include_wallet_tokens=False` or `include_ib=False` turns either off). An explicit `assets` list, as in `portfolio_low_correlation.ipynb`, replaces all discovery. It does not place trades.
- Assets split into two groups, each measured against its own benchmark on its own calendar. **Crypto** (HL crypto perps and spot, wallet tokens) is measured against BTC over every day with 365-day annualization. **TradFi** (IB stocks and HL HIP-3 stock, index, commodity and FX perps) is measured against SPY over US trading days with 252-day annualization; 24/7 HL prices are sampled at each trading day's close, so a weekend move counts in Monday's return. Wallet tokens are priced from DefiLlama by contract address and IB stocks and SPY from IB's split- and dividend-adjusted daily closes. Without TWS, IB positions are skipped and TradFi falls back to BTC if no SPY prices are saved.
- `show_benchmark_exposures()` lists each asset's correlation, beta (with a 90% bootstrap range), alpha, volatility and Sharpe against its benchmark over its full pairwise history, plus its correlation to the other benchmark. Correlation heatmaps are drawn per group, plus a crypto-vs-TradFi heatmap on trading days.
- A book holding both groups runs the Kelly optimizer, performance and factor model on US trading days with 252-day annualization: performance reports beta to BTC and to SPY with alpha net of both, a sleeve table splits the book into Crypto and TradFi, and the factor model adds SPY market excess while compounding the daily crypto factors onto trading days. These sections use only dates every asset shares, so only assets with at least `min_history_days` of prices (default 180; `MIN_HISTORY_DAYS` in the setup cell) enter them; shorter-history assets stay in the benchmark table and the heatmaps, which compute each pair over its own shared dates.
- `estimate_factor_premia()` tests whether characteristics earn a premium across crypto assets with t-stats that respect cross-sectional correlation. Fama-MacBeth runs one cross-sectional regression per day of every asset's return on its standardized, one-day-lagged characteristic and takes the t-stat from the time series of daily slopes (Newey-West), so a day when every asset moves together moves only that day's slope; a portfolio sort (long top half, short bottom half) gives the same test as a spread. Characteristics are 30-day momentum, 30-day volatility and 60-day BTC beta (every crypto asset, from prices), size and turnover (assets with CoinGecko market data), and 7-day funding (perps), plus a joint regression of the broadly covered ones. The factor model's table also shows each factor premium's own t-stat. With about 20 assets the cross-section is small, so |t| near 2 is suggestive rather than settled.
- CoinGecko's public API is used without an API key. It is rate-limited, so refreshes may need to be retried if the public service returns HTTP 429.
- Perp prices are refreshed from HL's daily candle API, preserving DEX-qualified names for HIP-3 markets. CoinGecko supplies prices for supported spot-only assets and historical market capitalization/volume. Perps without a unique CoinGecko symbol still enter the modeled universe when HL provides candles; spot-only assets without a unique CoinGecko mapping are reported and skipped. Assets with fewer than 30 common daily returns with BTC are reported and skipped.
- Prices reach back `history_years` (default 5; `HISTORY_YEARS` in the setup cell). CoinGecko's public API serves only the latest 365 days, so it supplies that year of prices and all market-cap and volume history; older prices are backfilled for free from HL daily candles (each perp back to its HL listing) and DefiLlama (CoinGecko-listed coins by CoinGecko ID and wallet tokens by contract address, back to their listing). Backfills fill only missing dates, so a newer source is never overwritten, and DefiLlama pages are requested from midnight UTC so they give the same daily closes as CoinGecko and HL. When a source has nothing older, the date is kept in `data/price_history_floor.json` so later runs do not ask again. IB stocks and SPY come from IB for the same number of years. Etherscan's free API has no price history and Alchemy's covers 365 days per request, so neither is used for prices.
- `fit_style_model()` is the industry-standard multi-factor regression of the book's daily excess return: alpha + Σ beta × factor + residual, with HAC t-stats. Equity style factors come from the Ken French data library (US market, size SMB, value HML, quality RMW, investment CMA, 12-1 month momentum, 1-month short-term reversal, developed-ex-US minus US for domicile, and business equipment minus market for the tech sector), plus a dividend factor from IB prices (SCHD minus SPY, since French has no daily dividend series); the crypto block adds BTC, alt market minus BTC, BTC momentum and funding carry. Alpha is the intercept, the average daily return no factor explains; specific return is alpha plus each day's residual, the return attributable to selection, reported with its volatility and information ratio. An attribution table and chart split the book's return into factor contributions and specific return. Factor files are cached in `data/french/` for a day; French updates monthly, so the latest weeks enter after its next release. Equity factors apply to a book on US trading days; a crypto-only book uses the crypto block. Betas are estimated from returns, not from holdings' characteristics as commercial risk models do.
- `fit_style_model_hl_book()` and `fit_style_model_ib_book()` run the same regression on the books actually traded, so their alpha is the desk's real specific return, unlike the Kelly model portfolio's in-sample alpha. The HL book combines every configured wallet into one time-weighted index from the saved HL history (`data/hl_portfolio_history_<wallet>.csv`), so deposits, withdrawals and transfers between wallets are not counted as performance. HL keeps detailed history for about a month, so the model uses weekly returns, compounding the daily factors into the same weeks, until 120 consecutive daily returns are saved, then daily. With few periods only the core factors are kept. The IB book uses IB Flex (below) or, without it, the desk's saved net-liquidation snapshots, which are flagged because they cannot separate deposits from performance.
- **IB Flex history.** TWS has no account-value history, but IB's Flex Web Service returns up to a year of daily NAV with deposits and withdrawals. One-time setup in IB's Client Portal: under Performance & Reports > Flex Queries, create an Activity Flex Query with the "Net Asset Value (NAV) in Base" and "Cash Transactions" (Deposits & Withdrawals) sections, XML format, period "Last 365 Calendar Days" and Breakout by Day; note its Query ID. Then enable Flex Web Service Configuration and generate a token (read-only reports). Store the token with `python -m desk.venues.ib_flex --store-token` and add `"flex_query_id": "<id>"` to the IB entry in `accounts.json`. Each run merges the statement into `trading-desk-data/ib/ib_nav_history_<account>.csv`, so history keeps growing past IB's one-year window.
- `show_highlights()` summarizes every section that ran: what stands out (universe, strongest benchmark exposures and least-correlated assets, significant factor exposures and premia, actual-book alpha) and what needs attention (in-sample-only results, a failed out-of-sample alpha test, assets left out for short history, very volatile assets, IB history without Flex, the Ken French data lag). Key metrics (Kelly Sharpe and betas, actual-book alpha and t-stat, asset counts) and the asset list are compared with the previous run, saved in `data/portfolio_highlights_state.json`.
- `estimate_factor_premia()` reports size twice: from CoinGecko's actual market caps (latest year), and as "Size (estimated, full history)", which keeps those actual caps and, before CoinGecko's year, uses price × the earliest circulating supply CoinGecko shows (its market cap divided by price). Coins whose implied supply moved more than 25% across the observed year, typically tokens still unlocking, are left at actual caps only, because a constant supply would misstate their past size several-fold. If the two size rows agree, the estimate is not distorting the result. Turnover cannot be extended this way: it needs historical market-wide volume, which only CoinGecko has.
- Market-cap and volume history stays at one year, so the CoinGecko-based factors (alt market, size, momentum spread, turnover) are left out of the time-series factor model when they would halve its sample; `estimate_factor_premia()` still tests them over the dates they cover.
- The remaining cells calculate daily returns and correlations, then optimize signed long/short full-Kelly weights and report portfolio performance and risk versus BTC. The optimizer table includes a historical expected five-day price-PnL contribution per $100 of portfolio equity; this estimate reflects signed weights and excludes funding, fees, and slippage. To run just the optimizer after loading price history and calculating returns, call `analysis.optimize_portfolio()`. The optimizer may select a direction independently of current wallet position sides; its weights are theoretical historical-return estimates, not the wallets' actual positions. Later cells estimate factor exposures to BTC market returns, altcoin-market performance, size, momentum, and trading turnover.
- The final export cell writes the weights, return series, correlations, performance metrics, factor betas, model fit statistics, estimated factor return contributions, and factor series to CSV files.

The discovered, priceable wallet assets form the modeled basket. CoinGecko cross-sectional factors use only assets with CoinGecko market-cap or volume history; all priceable assets remain in the portfolio-return series. The benchmark is BTC; signed weight bounds and other model assumptions are documented in the implementation notebook. The historical model does not include short borrow costs, perp funding, margin requirements, or liquidation risk.

## 2. HL account analytics notebook

Open **hl_account.ipynb** and run all cells in order.

- The notebook queries the configured HL account through the public API and fetches current spot balances, perp positions, fills, and BTC candles. No API secret is required for these public account-data requests.
- It creates or updates `data/hl_account_history_<wallet>.csv` with new account snapshots, realized trade PnL, and perp funding history. Keep that wallet's CSV between runs to preserve the local history used by the charts and statistics. Daily funding and fill-PnL rows are rebuilt from HL's data on each refresh, so a wrong saved row cannot persist.
- Each run fetches HL's all-time account history and saves any points newer than the CSV, so the account-value history and risk statistics stay current.
- Daily, weekly, and monthly return statistics, Sharpe, alpha, and beta use a time-weighted index built from HL's value and PnL history, so deposits, withdrawals, and transfers do not move them. A daily return needs history points on consecutive days, so older periods with weekly history count toward weekly and monthly returns but not daily statistics. Each run saves new value and PnL points to `data/hl_portfolio_history_<wallet>.csv`, which HL only keeps at this detail for about a month, so keep these files to let the daily sample grow. Account maximum drawdown uses the same index.
- Current account value counts a unified account's perp equity once, because it is already part of the USDC balance; other account modes add spot balances and perp equity.
- The notebook's report sections match the consolidated notebook and share its code: highlights, position tables, net exposure and funding, margin and aggregate risk, and the net funding APR chart. Highlights save a per-wallet baseline in `data/hl_highlights_state_<wallet>.json`, separate from the consolidated one. Adjust `MARGIN_RESERVE_PCT`, `AGGREGATE_RISK_PCT`, `ATR_PERIOD`, and `ATR_STOP_MULTIPLE` in the setup cell.
- The notebook reports account value, open-position metrics, fee-net spot/perp realized PnL with signed funding (HL's `closedPnl` is before fees, so USDC fees on every fill are subtracted, and win/loss statistics are after fees), fees paid, unrealized PnL, historical and bootstrap VaR, historical volatility, and asset-level PnL attribution. It also plots account and BTC comparisons.
- The open-positions table includes historical expected five-day price PnL, expected funding PnL, and their sum. These estimates use completed daily candles and historical hourly funding; fees, slippage, and future-rate uncertainty are not included in the expected values.

To analyze another account, update `WALLET_ADDRESS` in the setup cell with that account’s public HL address. The notebook may show less historical detail for an account if it has fewer saved snapshots or fills.

## 3. HL consolidated account analytics notebook

Open **hl_accounts.ipynb** and run its cells from top to bottom.

- Set `WALLET_ADDRESSES` in the setup cell to the public HL addresses to combine. The notebook reads each account through HL's public API; no API secret is needed.
- The notebooks use a separate `data/hl_account_history_<wallet>.csv` history file for each wallet. Preserve these files between runs to retain saved account history.
- Combined account-value history and account/risk metrics use dates with coverage from every configured wallet. In the account charts, the oldest wallet's saved history is shown before shared coverage begins, then the combined account value is used. The daily account-vs-BTC comparison is a time-weighted return built from HL's account value and PnL history, so deposits, withdrawals, and transfers do not move it. Weekly returns follow the same oldest-wallet-then-combined coverage; traded-volume/PnL charts already pool fills across wallets. Daily rollups and calendar-period cutoffs use US Eastern time. Time-axis charts collapse periods with no realized PnL or funding for more than 30 days and label what is hidden; the weekly-return histogram omits weeks wholly inside them, while statistics still include them.
- The open-positions section groups same-side positions across wallets and shows them separately from an aggregate account summary. The aggregate summary includes cumulative open-position funding as a percentage of combined account equity and total position value. Reserve-adjusted available margin is combined margin collateral (USDC on unified accounts, matching the executor) less the reserve; the summary shows this capacity, margin used, remaining available margin in dollars and as a percentage of capacity, and position value as a percentage of equity. `MARGIN_RESERVE_PCT` defaults to 20%. Position rows include funding and fees since open as dollar amounts, plus fees since open as a percentage of position value and position value as a percentage of wallet account value. They also show historical expected five-day price PnL, expected funding PnL, and expected PnL including funding. These expected values exclude fees and slippage. Fee ratios are shown for complete USDC-denominated fee history. Fixed-fractional sizing defaults to a risk budget of 5% of that available margin per asset, ATR(14) from completed daily candles, and a stop distance of 1.25 × ATR. The sizing table adds a directionally adverse 95th-percentile five-day funding reserve to the target-size and open-risk calculations; its displayed stop remains a price trigger and does not itself enforce a funding-adjusted net-loss threshold. Take-profit is placed the same distance in the favorable direction (1:1 risk/reward). If both long and short positions exist for an asset, each side shares the asset's risk budget equally. `Open - Target` compares a row with its own budget only; `Addable Risk ($)` and `Addable Value ($)` also apply the aggregate risk cap, and the summary shows that cap, risk used, and room. The room is one shared pool, so rows cannot all add their amounts. A separate chart shows `Net Funding APR (%) (simple, covered)` over time, rebuilt from HL funding history across wallets, as daily and notional-weighted 7-day values.
- Adjust `MARGIN_RESERVE_PCT`, `AGGREGATE_RISK_PCT`, `ATR_PERIOD`, and `ATR_STOP_MULTIPLE` in the setup cell to change the reserve and sizing assumptions. This is sizing analysis only; the notebook does not create, change, or submit orders.

## 4. HL trade analytics notebook

Open [`hl_trade_analytics.ipynb`](hl_trade_analytics.ipynb) and run its titled cells in order. Change the coin, DEX, side, and quantity in the configuration cell. The notebook reads completed daily candles and hourly funding history from public endpoints. It estimates the directionally adverse 95th-percentile five-day funding reserve from contiguous hourly windows and adds expected historical funding PnL separately from price PnL. Historical estimates are not guarantees and exclude fees, slippage, gaps, and funding-rate changes.

## 5. HL backtest notebook

Open [`hl_backtest.ipynb`](hl_backtest.ipynb) and run its titled cells in order. Edit the signal rows and set your current maker/taker fee rates in basis points before running. With no dated signal rules, it simulates persistent long/short entries at daily opens whenever that market is flat, one position at a time per market. The backtest uses ATR(14), a 1.25×ATR stop, a 1:1 target, and a five-daily-bar time exit by default; leverage is capped at 4× on default-Dex markets and 3× on HIP-3 DEX markets. When stop and target both touch the same daily candle, it assumes the stop fills first. Funding PnL uses entry price as the hourly notional-price proxy. Each market is tested independently; overlapping positions across assets are not combined into a portfolio equity curve or checked against the live 20% aggregate-risk cap.

## 6. HL guarded trade execution

Execution is deliberately separate from the notebooks. Install its optional dependencies into the Python environment that will run it:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-trading.txt
```

Create a local signals CSV with `coin,side,dex` columns; `execution/trade_targets.example.csv` shows the format. Store the API-wallet private key directly in the macOS Keychain by running:

```bash
.venv/bin/python execution/hl_trade.py --wallet 0xYOUR_MAIN_WALLET_ADDRESS --store-key
```

The script prompts for the API-wallet signing key with hidden input. The key is never written to the repository or included in command-line arguments. Do not send private keys, seed phrases, or API secrets in chat. Use a dedicated API wallet, enable only the access it needs, and avoid using a wallet holding funds beyond the intended trading allocation.

To check every open position and add only missing reduce-only stop-loss/take-profit coverage, run:

```bash
.venv/bin/python execution/hl_trade.py \
  --wallet 0xYOUR_MAIN_WALLET_ADDRESS \
  --protect-open
```

This defaults to a dry run. The proposed stop uses ATR(14) × 1.25, reduced by the directionally adverse 95th-percentile five-day funding reserve; the target uses the same adjusted distance in the favorable direction. Existing reduce-only stops and profitable reduce-only limits count toward covered size and are not changed by this protection-only mode.

For managed size changes, create an actions CSV with columns `action,coin,side,dex,margin_usd` and run it using `--actions`. Supported actions:

- `open_margin`: open a flat market using the requested positive USD margin. Leverage is the market's existing protocol leverage cap; actual size can be scaled down by the 5%-of-usable-equity per-position risk budget, the 20% aggregate risk cap, and available margin.
- `open_target`: open a flat market at the maximum size allowed by that same risk protocol. Include `LONG` or `SHORT` in `side`.
- `adjust_margin`: add or subtract a signed USD margin amount from an existing position. Positive values add in the existing direction; negative values reduce it. The current position leverage is used, reductions are capped at flat (never reverse), and increases are scaled by risk and margin limits. Use `close` for an intentional full close.
- `target`: move an existing position (or a flat market with an explicit side) to the current maximum risk-protocol size. This can add or reduce size. If other positions already use the whole aggregate risk cap, the allowed size is zero and `target` closes the position; check the dry run first.
- `close`: reduce the existing position to flat without opening the opposite side.

Open actions submit a maker-only ALO entry immediately and reprice without crossing for up to five minutes; a fill is not guaranteed. This version does not use a market entry for opening, to avoid unbounded slippage.

Example action rows (replace the values before use):

```csv
action,coin,side,dex,margin_usd
open_margin,BTC,LONG,,100
open_target,ETH,SHORT,,
adjust_margin,SOL,,,25
adjust_margin,ADA,,,-10
target,xyz:AAPL,LONG,,
close,ARB,,,
```

Run a dry check first:

```bash
.venv/bin/python execution/hl_trade.py \
  --wallet 0xYOUR_MAIN_WALLET_ADDRESS \
  --risk-wallets 0xWALLET_ONE,0xWALLET_TWO \
  --actions position_actions.csv
```

Additions use the existing maker-only ALO entry approach; a resting entry keeps its queue position and is only replaced when its target price changes. Within its five-minute window an entry retries after price moves, failed reads, and failed sends. A failed send is retried only after the exchange shows nothing was placed, and rejections retrying cannot fix, such as insufficient margin, stop at once. Reductions and closes use reduce-only market orders, and never reverse a position. After a size change, the executor cancels and recreates the recognized reduce-only SL/TP orders to match the actual remaining position. The exchange cannot atomically resize the position and its exit orders, so a brief unprotected interval is possible; the CLI warns before live confirmation. Use one action per market in a CSV and review every refreshed quantity and order level.

First run without `--live` to inspect the proposed sizing and current aggregate risk:

```bash
.venv/bin/python execution/hl_trade.py \
  --wallet 0xYOUR_MAIN_WALLET_ADDRESS \
  --risk-wallets 0xWALLET_ONE,0xWALLET_TWO \
  --signals trade_targets.csv
```

The script defaults to dry-run, does not load a key in dry-run mode, and does not send orders. A live run requires `--live`, typing `LIVE TRADING` to refresh the preflight, then reviewing the refreshed plan and typing the mode-specific final phrase (`SUBMIT ORDERS`, `SUBMIT POSITION ACTIONS`, or `SUBMIT PROTECTION ORDERS`). It uses maker-only ALO entries that reprice around mid without crossing for up to five minutes, stop-market exits, and reduce-only post-only take-profits. The funding-aware net-loss guard for newly opened signal trades runs in the foreground; keep the process running after entry. Exchange-side stop and take-profit orders remain active if the local process stops.

Sizing defaults are a 5% risk budget on the 80% usable-equity amount per position, a 20% aggregate stop-plus-five-day-adverse-funding risk cap, a 20% equity margin reserve, 4× leverage for default-Dex crypto perps and a conservative 3× cap for HIP-3 DEX markets. The script scales new orders down to remaining risk/margin capacity and blocks risk-increasing orders when the cap is already full. Existing open positions are included in risk checks and opposite-side positions in the target wallet must be flattened manually before re-entry.

Equity for these limits is margin collateral, matching how HL margins the account. On a unified account that is the wallet's USDC balance: perp account value is already part of it, so the two are not added, and non-USDC spot tokens such as HYPE are not counted. Other account modes use perp account value. Available margin is 80% of collateral less margin used. With `--risk-wallets`, the per-position and aggregate risk caps use combined collateral across those wallets, while available margin uses the execution wallet alone.

The caps limit estimated loss (stop distance, adverse funding, and fees), not margin, so the same margin can carry very different risk across markets and leverage levels. The sizing protocol for in-the-money positions is:

- A position over the per-position risk cap is trimmed down to the cap.
- A position under the cap can be increased up to the cap, but only while the aggregate risk cap and available margin have room. The aggregate cap often binds first.
- Remaining room goes to in-the-money positions with the lowest unrealized PnL percentage first.

The executor applies the limits to each action but does not choose positions; that ordering is applied when writing the actions CSV.

## HL kill switch

To see what would be shut down (read-only; no key is loaded):

```bash
.venv/bin/python execution/hl_trade.py --kill-switch \
  --wallet 0xWALLET_ONE --risk-wallets 0xWALLET_ONE,0xWALLET_TWO
```

Add `--live` and type `SHUT IT ALL DOWN` to execute. It first halts new entries, including entry loops already running on this machine, then for each wallet cancels entry orders and market-closes every perp position with reduce-only IOC orders, retrying with up to 10% slippage. A position's stop and take-profit are cancelled only once it is confirmed flat, so a failed close keeps its protection. It sweeps a second time, re-reads the exchange, and exits with status 2 and a list of anything still open.

It needs each wallet's API key in the keychain and runs from `.venv` or the `py314` environment; both have the trading packages. It does not touch spot balances, vaults, or staking, and it cannot stop orders placed elsewhere, such as the HL app. Live entries and position actions stay blocked until `--clear-kill-switch`; `--protect-open` still works.

## Notes

- Completed daily candles, past market funding, and wallet funding events are saved in `data/api_cache/` (not committed) and downloaded only once. Delete that folder to force a full download. Live balances, positions, orders, and fills are always read fresh.

- The Kelly weights are historical estimates based on the available price sample; they are not a guarantee of future performance.
- Expected PnL values use historical mean forward returns/funding and can be unstable for short or changing market samples.
- The factor model is a descriptive in-sample regression over the selected assets. Factor betas and contributions are estimates, not causal explanations or forecasts.
- HL account history is stored locally in per-wallet CSV files. Back up the relevant `data/hl_account_history_<wallet>.csv` files if you want to preserve those records.
- Multi-wallet daily account comparisons require overlapping saved history for all configured wallets; a short overlap can make drawdown and other historical statistics unrepresentative of the full account history.
- Files in `data/` can contain account-level financial history and portfolio data. Avoid committing them to a public repository unless you intend to publish that data.
