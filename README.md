# Cryptocurrency Portfolio Analytics

This repository contains Jupyter notebooks for portfolio analytics, Hyperliquid account analysis, trade analytics, and backtesting, plus a separate guarded trade-execution script:

1. **Wallet-based portfolio analytics** — discovers the asset universe from nonzero perp positions and spot balances in configured Hyperliquid wallets, then models that basket from historical market data, estimates constrained Kelly weights, and compares modeled performance with BTC. The Kelly weights are model outputs, not the wallets' actual allocations, and the notebook does not place trades.
2. **Single-wallet Hyperliquid account analytics** — reads the current account, balances, and open perp positions from Hyperliquid, and tracks fee-net realized PnL, signed perp funding, unrealized PnL, and account performance.
3. **Consolidated Hyperliquid account analytics** — combines account history, balances, positions, fills, and PnL across multiple configured wallets, with ATR-based stop and target sizing for open positions.
4. **Hyperliquid trade analytics** — estimates historical five-day expected price/funding PnL and ATR-based funding-aware risk for a perp position.
5. **Hyperliquid backtest** — backtests selected directions with ATR stops/targets, historical funding, and configured maker/taker fees.

All notebooks are read-only: they analyze data and do not place trades. Live orders are possible only through the separate execution script, which defaults to dry-run and requires explicit confirmation.

## Notebooks

The two notebooks below contain the complete workflows, including price-history refresh.

| Notebook | What it does | Main data files |
| --- | --- | --- |
| [`crypto_portfolio_analytics.ipynb`](crypto_portfolio_analytics.ipynb) | Discovers live perp positions and spot balances from configured Hyperliquid wallets; fetches daily perp prices from Hyperliquid (including HIP-3 markets) and supported spot prices plus market data from CoinGecko; calculates returns and correlations; optimizes signed long/short constrained full-Kelly weights; fits BTC and available CoinGecko cross-sectional factors; compares portfolio results with BTC; writes metrics and factor outputs to CSV. | `data/crypto_daily_prices_usd.csv`, `data/crypto_daily_returns.csv`, `data/crypto_return_correlation.csv`, `data/crypto_kelly_full_weights.csv`, `data/crypto_kelly_weights.csv`, `data/crypto_portfolio_daily_returns.csv`, `data/crypto_portfolio_metrics.csv`, `data/crypto_market_data.csv`, `data/crypto_portfolio_factor_model.csv`, `data/crypto_portfolio_factor_fit.csv`, `data/crypto_factor_returns.csv` |
| [`hyperliquid_account_analytics.ipynb`](hyperliquid_account_analytics.ipynb) | Analyzes one Hyperliquid account by reading current account and position data; saves account history locally; reports spot/perp balances, fee-net realized PnL with signed perp funding, unrealized PnL, risk metrics, PnL attribution, and charts against BTC. | `data/hyperliquid_account_history_<wallet>.csv` |
| [`hyperliquid_multi_account_analytics.ipynb`](hyperliquid_multi_account_analytics.ipynb) | Consolidates account history, balances, positions, fills, funding, and PnL across configured Hyperliquid wallets. Displays combined account/risk summaries and charts, pooled PnL attribution, and per-wallet open positions. Position sizing uses a configurable fixed-fractional risk budget and ATR-based stop/take-profit levels. | `data/hyperliquid_account_history_<wallet>.csv` for each wallet |
| [`hyperliquid_trade_analytics.ipynb`](hyperliquid_trade_analytics.ipynb) | Interactive, read-only view of ATR stop distance, historical adverse funding reserve, and expected five-day price/funding PnL for one perp position. | Public Hyperliquid daily candles and hourly funding history |
| [`hyperliquid_backtest.ipynb`](hyperliquid_backtest.ipynb) | Interactive, read-only backtest for a list of coin/direction signals. Shows trade-level PnL, funding, fees, and exit reasons. | Public Hyperliquid daily candles and hourly funding history |

Implementation modules (`*_implementation.py`) provide code used by the portfolio and account reports; run the user-facing notebooks in the table above. The trade notebooks import shared helpers from `hyperliquid_trade_analytics.py` and the backtest engine from `hyperliquid_backtest.py`. All CSV and spreadsheet data files are stored in the `data/` subfolder. The account notebooks create this folder when needed and read/write their files there. Keep the existing files in `data/` between runs to preserve price and account history.

## Setup

Use Python 3.10 or later and install Jupyter plus the libraries used by the notebooks:

```bash
python -m pip install jupyter pandas numpy matplotlib requests seaborn
```

Start Jupyter from this repository directory:

```bash
jupyter lab
```

Open either notebook and run its cells from top to bottom. Each notebook is independent; run only the one whose analysis you need.

## 1. Portfolio analytics notebook

Open **crypto_portfolio_analytics.ipynb** and run all cells in order.

- Set `WALLET_ADDRESSES` in the notebook's setup cell. The refresh discovers nonzero perp positions and spot balances across those wallets using Hyperliquid's public API; it does not place trades.
- CoinGecko's public API is used without an API key. It is rate-limited, so refreshes may need to be retried if the public service returns HTTP 429.
- Perp prices are refreshed from Hyperliquid's daily candle API, preserving DEX-qualified names for HIP-3 markets. CoinGecko supplies prices for supported spot-only assets and historical market capitalization/volume. Perps without a unique CoinGecko symbol still enter the modeled universe when Hyperliquid provides candles; spot-only assets without a unique CoinGecko mapping are reported and skipped. Assets with fewer than 30 common daily returns with BTC are reported and skipped.
- Public historical requests are limited to the latest 365 days. The refresh caps requests to that window and retains older prices already stored in the CSV; older market-cap and volume factor history is retained locally but cannot be newly fetched from the public endpoint.
- The remaining cells calculate daily returns and correlations, then optimize signed long/short full-Kelly weights and report portfolio performance and risk versus BTC. The optimizer table includes a historical expected five-day price-PnL contribution per $100 of portfolio equity; this estimate reflects signed weights and excludes funding, fees, and slippage. To run just the optimizer after loading price history and calculating returns, call `analysis.optimize_portfolio()`. The optimizer may select a direction independently of current wallet position sides; its weights are theoretical historical-return estimates, not the wallets' actual positions. Later cells estimate factor exposures to BTC market returns, altcoin-market performance, size, momentum, and trading turnover.
- The final export cell writes the weights, return series, correlations, performance metrics, factor betas, model fit statistics, estimated factor return contributions, and factor series to CSV files.

The discovered, priceable wallet assets form the modeled basket. CoinGecko cross-sectional factors use only assets with CoinGecko market-cap or volume history; all priceable assets remain in the portfolio-return series. The benchmark is BTC; signed weight bounds and other model assumptions are documented in the implementation notebook. The historical model does not include short borrow costs, perp funding, margin requirements, or liquidation risk.

## 2. Hyperliquid account analytics notebook

Open **hyperliquid_account_analytics.ipynb** and run all cells in order.

- The notebook queries the configured Hyperliquid account through the public API and fetches current spot balances, perp positions, fills, and BTC candles. No API secret is required for these public account-data requests.
- It creates or updates `data/hyperliquid_account_history_<wallet>.csv` with new account snapshots, realized trade PnL, and perp funding history. Keep that wallet's CSV between runs to preserve the local history used by the charts and statistics. Daily funding and fill-PnL rows are rebuilt from Hyperliquid's data on each refresh, so a wrong saved row cannot persist.
- By default, previously saved history is reused and only new observations are collected. Set `FETCH_FULL_HISTORY = True` in the account data cell to request the all-time history again.
- The notebook reports account value, open-position metrics, fee-net spot/perp realized PnL with signed funding (Hyperliquid's `closedPnl` is before fees, so USDC fees on every fill are subtracted, and win/loss statistics are after fees), fees paid, unrealized PnL, VaR and historical volatility, and asset-level PnL attribution. It also plots account and BTC comparisons.
- The open-positions table includes historical expected five-day price PnL, expected funding PnL, and their sum. These estimates use completed daily candles and historical hourly funding; fees, slippage, and future-rate uncertainty are not included in the expected values.

To analyze another account, update `ADDR` in the setup cell with that account’s public Hyperliquid address. The notebook may show less historical detail for an account if it has fewer saved snapshots or fills.

## 3. Hyperliquid consolidated account analytics notebook

Open **hyperliquid_multi_account_analytics.ipynb** and run its cells from top to bottom.

- Set `WALLET_ADDRESSES` in the setup cell to the public Hyperliquid addresses to combine. The notebook reads each account through Hyperliquid's public API; no API secret is needed.
- The notebooks use a separate `data/hyperliquid_account_history_<wallet>.csv` history file for each wallet. Preserve these files between runs to retain saved account history.
- Combined account-value history and account/risk metrics use dates with coverage from every configured wallet. In the account charts, the oldest wallet's saved history is shown before shared coverage begins, then the combined account value is used. The daily account-vs-BTC comparison is a time-weighted return built from Hyperliquid's account value and PnL history, so deposits, withdrawals, and transfers do not move it. Weekly returns follow the same oldest-wallet-then-combined coverage; traded-volume/PnL charts already pool fills across wallets. Daily rollups and calendar-period cutoffs use US Eastern time. Time-axis charts collapse periods with no realized PnL or funding for more than 30 days and label what is hidden; the weekly-return histogram omits weeks wholly inside them, while statistics still include them.
- The open-positions section groups same-side positions across wallets and shows them separately from an aggregate account summary. The aggregate summary includes cumulative open-position funding as a percentage of combined account equity and total position value. Reserve-adjusted available margin is combined margin collateral (USDC on unified accounts, matching the executor) less the reserve; the summary shows this capacity, margin used, remaining available margin in dollars and as a percentage of capacity, and position value as a percentage of equity. `MARGIN_RESERVE_PCT` defaults to 20%. Position rows include funding and fees since open as dollar amounts, plus fees since open as percentages of wallet equity and position value and position value as a percentage of wallet account value. They also show historical expected five-day price PnL, expected funding PnL, and expected PnL including funding. These expected values exclude fees and slippage. Fee ratios are shown for complete USDC-denominated fee history. Fixed-fractional sizing defaults to a risk budget of 5% of that available margin per asset, ATR(14) from completed daily candles, and a stop distance of 1.25 × ATR. The sizing table adds a directionally adverse 95th-percentile five-day funding reserve to the target-size and open-risk calculations; its displayed stop remains a price trigger and does not itself enforce a funding-adjusted net-loss threshold. Take-profit is placed the same distance in the favorable direction (1:1 risk/reward). If both long and short positions exist for an asset, each side shares the asset's risk budget equally. `Open - Target` compares a row with its own budget only; `Addable Risk ($)` and `Addable Value ($)` also apply the aggregate risk cap, and the summary shows that cap, risk used, and room. The room is one shared pool, so rows cannot all add their amounts. A separate chart shows `Net Funding APR (%) (simple, covered)` over time, rebuilt from Hyperliquid funding history across wallets, as daily and notional-weighted 7-day values.
- Adjust `MARGIN_RESERVE_PCT`, `AGGREGATE_RISK_PCT`, `RISK_BUDGET_PCT`, `ATR_PERIOD`, and `ATR_STOP_MULTIPLE` in the setup cell to change the reserve and sizing assumptions. This is sizing analysis only; the notebook does not create, change, or submit orders.

## 4. Hyperliquid trade analytics notebook

Open [`hyperliquid_trade_analytics.ipynb`](hyperliquid_trade_analytics.ipynb) and run its titled cells in order. Change the coin, DEX, side, and quantity in the configuration cell. The notebook reads completed daily candles and hourly funding history from public endpoints. It estimates the directionally adverse 95th-percentile five-day funding reserve from contiguous hourly windows and adds expected historical funding PnL separately from price PnL. Historical estimates are not guarantees and exclude fees, slippage, gaps, and funding-rate changes.

## 5. Hyperliquid backtest notebook

Open [`hyperliquid_backtest.ipynb`](hyperliquid_backtest.ipynb) and run its titled cells in order. Edit the signal rows and set your current maker/taker fee rates in basis points before running. With no dated signal rules, it simulates persistent long/short entries at daily opens whenever that market is flat, one position at a time per market. The backtest uses ATR(14), a 1.25×ATR stop, a 1:1 target, and a five-daily-bar time exit by default; leverage is capped at 4× on default-Dex markets and 3× on HIP-3 DEX markets. When stop and target both touch the same daily candle, it assumes the stop fills first. Funding PnL uses entry price as the hourly notional-price proxy. Each market is tested independently; overlapping positions across assets are not combined into a portfolio equity curve or checked against the live 20% aggregate-risk cap.

## 6. Guarded trade execution

Execution is deliberately separate from the notebooks. Install its optional dependencies into the Python environment that will run it:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-trading.txt
```

Create a local signals CSV with `coin,side,dex` columns; `trade_targets.example.csv` shows the format. Store the API-wallet private key directly in the macOS Keychain by running:

```bash
.venv/bin/python hyperliquid_trade_execution.py --wallet 0xYOUR_MAIN_WALLET_ADDRESS --store-key
```

The script prompts for the API-wallet signing key with hidden input. The key is never written to the repository or included in command-line arguments. Do not send private keys, seed phrases, or API secrets in chat. Use a dedicated API wallet, enable only the access it needs, and avoid using a wallet holding funds beyond the intended trading allocation.

To check every open position and add only missing reduce-only stop-loss/take-profit coverage, run:

```bash
.venv/bin/python hyperliquid_trade_execution.py \
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
.venv/bin/python hyperliquid_trade_execution.py \
  --wallet 0xYOUR_MAIN_WALLET_ADDRESS \
  --risk-wallets 0xWALLET_ONE,0xWALLET_TWO \
  --actions position_actions.csv
```

Additions use the existing maker-only ALO entry approach. Reductions and closes use reduce-only market orders, and never reverse a position. After a size change, the executor cancels and recreates the recognized reduce-only SL/TP orders to match the actual remaining position. The exchange cannot atomically resize the position and its exit orders, so a brief unprotected interval is possible; the CLI warns before live confirmation. Use one action per market in a CSV and review every refreshed quantity and order level.

First run without `--live` to inspect the proposed sizing and current aggregate risk:

```bash
.venv/bin/python hyperliquid_trade_execution.py \
  --wallet 0xYOUR_MAIN_WALLET_ADDRESS \
  --risk-wallets 0xWALLET_ONE,0xWALLET_TWO \
  --signals trade_targets.csv
```

The script defaults to dry-run, does not load a key in dry-run mode, and does not send orders. A live run requires `--live`, typing `LIVE TRADING` to refresh the preflight, then reviewing the refreshed plan and typing the mode-specific final phrase (`SUBMIT ORDERS`, `SUBMIT POSITION ACTIONS`, or `SUBMIT PROTECTION ORDERS`). It uses maker-only ALO entries that reprice around mid without crossing for up to five minutes, stop-market exits, and reduce-only post-only take-profits. The funding-aware net-loss guard for newly opened signal trades runs in the foreground; keep the process running after entry. Exchange-side stop and take-profit orders remain active if the local process stops.

Sizing defaults are a 5% risk budget on the 80% usable-equity amount per position, a 20% aggregate stop-plus-five-day-adverse-funding risk cap, a 20% equity margin reserve, 4× leverage for default-Dex crypto perps and a conservative 3× cap for HIP-3 DEX markets. The script scales new orders down to remaining risk/margin capacity and blocks risk-increasing orders when the cap is already full. Existing open positions are included in risk checks and opposite-side positions in the target wallet must be flattened manually before re-entry.

Equity for these limits is margin collateral, matching how Hyperliquid margins the account. On a unified account that is the wallet's USDC balance: perp account value is already part of it, so the two are not added, and non-USDC spot tokens such as HYPE are not counted. Other account modes use perp account value. Available margin is 80% of collateral less margin used. With `--risk-wallets`, the per-position and aggregate risk caps use combined collateral across those wallets, while available margin uses the execution wallet alone.

The caps limit estimated loss (stop distance, adverse funding, and fees), not margin, so the same margin can carry very different risk across markets and leverage levels. The sizing protocol for in-the-money positions is:

- A position over the per-position risk cap is trimmed down to the cap.
- A position under the cap can be increased up to the cap, but only while the aggregate risk cap and available margin have room. The aggregate cap often binds first.
- Remaining room goes to in-the-money positions with the lowest unrealized PnL percentage first.

The executor applies the limits to each action but does not choose positions; that ordering is applied when writing the actions CSV.

## Notes

- The Kelly weights are historical estimates based on the available price sample; they are not a guarantee of future performance.
- Expected PnL values use historical mean forward returns/funding and can be unstable for short or changing market samples.
- The factor model is a descriptive in-sample regression over the selected assets. Factor betas and contributions are estimates, not causal explanations or forecasts.
- Hyperliquid account history is stored locally in per-wallet CSV files. Back up the relevant `data/hyperliquid_account_history_<wallet>.csv` files if you want to preserve those records.
- Multi-wallet daily account comparisons require overlapping saved history for all configured wallets; a short overlap can make drawdown and other historical statistics unrepresentative of the full account history.
- Files in `data/` can contain account-level financial history and portfolio data. Avoid committing them to a public repository unless you intend to publish that data.
