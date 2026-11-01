# Cryptocurrency Portfolio Analytics

This repository contains three Jupyter notebooks for complementary views of a crypto portfolio:

1. **Wallet-based portfolio analytics** — discovers the asset universe from nonzero perp positions and spot balances in configured Hyperliquid wallets, then models that basket from historical market data, estimates constrained Kelly weights, and compares modeled performance with BTC. The Kelly weights are model outputs, not the wallets' actual allocations, and the notebook does not place trades.
2. **Single-wallet Hyperliquid account analytics** — reads the current account, balances, and open perp positions from Hyperliquid, and tracks fee-net realized PnL, signed perp funding, unrealized PnL, and account performance.
3. **Consolidated Hyperliquid account analytics** — combines account history, balances, positions, fills, and PnL across multiple configured wallets, with ATR-based stop and target sizing for open positions.

All three notebooks are read-only: they analyze data and do not place trades.

## Notebooks

The two notebooks below contain the complete workflows, including price-history refresh.

| Notebook | What it does | Main data files |
| --- | --- | --- |
| [`crypto_portfolio_analytics.ipynb`](crypto_portfolio_analytics.ipynb) | Discovers a modeled asset universe from configured Hyperliquid wallets; fetches daily USD prices, market capitalization, and trading volume from CoinGecko's public API; calculates returns and correlations; optimizes signed long/short constrained full-Kelly weights; fits a BTC/altcoin/size/momentum/turnover factor model; compares portfolio results with BTC; writes metrics and factor outputs to CSV. | `data/crypto_daily_prices_usd.csv`, `data/crypto_daily_returns.csv`, `data/crypto_return_correlation.csv`, `data/crypto_kelly_full_weights.csv`, `data/crypto_kelly_weights.csv`, `data/crypto_portfolio_daily_returns.csv`, `data/crypto_portfolio_metrics.csv`, `data/crypto_market_data.csv`, `data/crypto_portfolio_factor_model.csv`, `data/crypto_portfolio_factor_fit.csv`, `data/crypto_factor_returns.csv` |
| [`hyperliquid_account_analytics.ipynb`](hyperliquid_account_analytics.ipynb) | Analyzes one Hyperliquid account by reading current account and position data; saves account history locally; reports spot/perp balances, fee-net realized PnL with signed perp funding, unrealized PnL, risk metrics, PnL attribution, and charts against BTC. | `data/hyperliquid_account_history_<wallet>.csv` |
| [`hyperliquid_multi_account_analytics.ipynb`](hyperliquid_multi_account_analytics.ipynb) | Consolidates account history, balances, positions, fills, funding, and PnL across configured Hyperliquid wallets. Displays combined account/risk summaries and charts, pooled PnL attribution, and per-wallet open positions. Position sizing uses a configurable fixed-fractional risk budget and ATR-based stop/take-profit levels. | `data/hyperliquid_account_history_<wallet>.csv` for each wallet |

Implementation notebooks provide code used by the user-facing notebooks; run the user-facing notebooks in the table above. All CSV and spreadsheet data files are stored in the `data/` subfolder. The notebooks create this folder when needed and read/write their files there. Keep the existing files in `data/` between runs to preserve price and account history.

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
- The notebook fetches daily USD prices plus historical market capitalization and total volume for the discovered, CoinGecko-matched assets, updating `data/crypto_daily_prices_usd.csv` and `data/crypto_market_data.csv`. Assets with ambiguous or missing CoinGecko symbols, or fewer than 30 daily returns overlapping BTC, are reported and skipped.
- Public historical requests are limited to the latest 365 days. The refresh caps requests to that window and retains older prices already stored in the CSV; older market-cap and volume factor history is retained locally but cannot be newly fetched from the public endpoint.
- The remaining cells calculate daily returns and correlations, then optimize signed long/short full-Kelly weights and report portfolio performance and risk versus BTC. To run just the optimizer after loading price history and calculating returns, call `analysis.optimize_portfolio()`. The optimizer may select a direction independently of current wallet position sides; its weights are theoretical historical-return estimates, not the wallets' actual positions. Later cells estimate factor exposures to BTC market returns, altcoin-market performance, size, momentum, and trading turnover.
- The final export cell writes the weights, return series, correlations, performance metrics, factor betas, model fit statistics, estimated factor return contributions, and factor series to CSV files.

The discovered wallet assets form the modeled basket. The benchmark is BTC; signed weight bounds and other model assumptions are documented in the implementation notebook. The historical model does not include short borrow costs, perp funding, margin requirements, or liquidation risk.

## 2. Hyperliquid account analytics notebook

Open **hyperliquid_account_analytics.ipynb** and run all cells in order.

- The notebook queries the configured Hyperliquid account through the public API and fetches current spot balances, perp positions, fills, and BTC candles. No API secret is required for these public account-data requests.
- It creates or updates `data/hyperliquid_account_history_<wallet>.csv` with new account snapshots, realized trade PnL, and perp funding history. Keep that wallet's CSV between runs to preserve the local history used by the charts and statistics.
- By default, previously saved history is reused and only new observations are collected. Set `FETCH_FULL_HISTORY = True` in the account data cell to request the all-time history again.
- The notebook reports account value, open-position metrics, fee-net spot/perp realized PnL with signed funding, fees paid, unrealized PnL, VaR and historical volatility, and asset-level PnL attribution. It also plots account and BTC comparisons.

To analyze another account, update `ADDR` in the setup cell with that account’s public Hyperliquid address. The notebook may show less historical detail for an account if it has fewer saved snapshots or fills.

## 3. Hyperliquid consolidated account analytics notebook

Open **hyperliquid_multi_account_analytics.ipynb** and run its cells from top to bottom.

- Set `WALLET_ADDRESSES` in the setup cell to the public Hyperliquid addresses to combine. The notebook reads each account through Hyperliquid's public API; no API secret is needed.
- The notebooks use a separate `data/hyperliquid_account_history_<wallet>.csv` history file for each wallet. Preserve these files between runs to retain saved account history.
- Combined account-value history is limited to dates with coverage from every configured wallet. Daily rollups and calendar-period cutoffs use US Eastern time.
- The open-positions section groups same-side positions across wallets and shows them separately from an aggregate account summary. Fixed-fractional sizing defaults to a 5% account-value risk budget per asset, ATR(14) from completed daily candles, and a stop distance of 1.25 × ATR. Take-profit is placed the same distance in the favorable direction (1:1 risk/reward). If both long and short positions exist for an asset, each side shares the asset's risk budget equally.
- Adjust `RISK_BUDGET_PCT`, `ATR_PERIOD`, and `ATR_STOP_MULTIPLE` in the setup cell to change sizing assumptions. This is sizing analysis only; the notebook does not create, change, or submit orders.

## Notes

- The Kelly weights are historical estimates based on the available price sample; they are not a guarantee of future performance.
- The factor model is a descriptive in-sample regression over the selected assets. Factor betas and contributions are estimates, not causal explanations or forecasts.
- Hyperliquid account history is stored locally in per-wallet CSV files. Back up the relevant `data/hyperliquid_account_history_<wallet>.csv` files if you want to preserve those records.
- Multi-wallet daily account comparisons require overlapping saved history for all configured wallets; a short overlap can make drawdown and other historical statistics unrepresentative of the full account history.
- Files in `data/` can contain account-level financial history and portfolio data. Avoid committing them to a public repository unless you intend to publish that data.
