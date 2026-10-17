# Cryptocurrency Portfolio Analytics

This repository contains two Jupyter notebooks for complementary views of a crypto portfolio:

1. **Portfolio analytics** — analyzes a selected basket of crypto assets, estimates constrained Kelly weights, and compares portfolio performance with BTC.
2. **Hyperliquid account analytics** — tracks a Hyperliquid account’s balances, perp positions, PnL, and performance over time.

## Notebooks

The two main notebooks provide portfolio and Hyperliquid account analytics. `crypto_read_daily_returns.ipynb` is a supporting notebook for refreshing the shared crypto price history.

| Notebook | What it does | Main data files |
| --- | --- | --- |
| [`Crypto correlations, constrained Kelly, and portfolio metrics.ipynb`](Crypto%20correlations%2C%20constrained%20Kelly%2C%20and%20portfolio%20metrics.ipynb) | Fetches daily USD prices from CoinGecko; calculates returns and correlations; optimizes long-only constrained full-Kelly weights; compares portfolio results with BTC; writes metrics and return series to CSV. | `data/crypto_daily_prices_usd.csv`, `data/crypto_daily_returns.csv`, `data/crypto_return_correlation.csv`, `data/crypto_kelly_full_weights.csv`, `data/crypto_kelly_weights.csv`, `data/crypto_portfolio_daily_returns.csv`, `data/crypto_portfolio_metrics.csv` |
| [`hyperliquid.ipynb`](hyperliquid.ipynb) | Reads current account and position data from Hyperliquid; saves account history locally; reports spot/perp balances, realized and unrealized PnL, risk metrics, PnL attribution, and charts against BTC. | `data/hyperliquid_account_history.csv` |

All CSV and spreadsheet data files are stored in the `data/` subfolder. The notebooks create this folder when needed and read/write their files there. Keep the existing files in `data/` between runs to preserve price and account history.

## Setup

Use Python 3.10 or later and install Jupyter plus the libraries used by the notebooks:

```bash
python -m pip install jupyter pandas numpy matplotlib requests scipy seaborn
```

Start Jupyter from this repository directory:

```bash
jupyter lab
```

Open either notebook and run its cells from top to bottom. Each notebook is independent; run only the one whose analysis you need.

## 1. Portfolio analytics notebook

Open **Crypto correlations, constrained Kelly, and portfolio metrics.ipynb** and run all cells in order.

- The first data-fetch cell asks for a CoinGecko Demo API key using hidden input. Create a key in your CoinGecko account and paste it when prompted. The key is not saved in the notebook.
- The notebook fetches daily USD price history for its configured coin basket and updates `data/crypto_daily_prices_usd.csv`.
- The remaining cells calculate daily returns, a correlation matrix, constrained full-Kelly weights, and portfolio performance metrics versus BTC.
- The final export cell writes the weights, daily return series, correlations, and performance metrics to CSV files.

The default basket, weight bounds, and benchmark assumptions are documented in the notebook’s opening markdown cell. Edit those settings there if you want a different analysis.

## 2. Hyperliquid account analytics notebook

Open **hyperliquid.ipynb** and run all cells in order.

- The notebook queries the configured Hyperliquid account through the public API and fetches current spot balances, perp positions, fills, and BTC candles. No API secret is required for these public account-data requests.
- It creates or updates `data/hyperliquid_account_history.csv` with new account snapshots and realized PnL. Keep `data/hyperliquid_account_history.csv` between runs to preserve the local history used by the charts and statistics.
- By default, previously saved history is reused and only new observations are collected. Set `FETCH_FULL_HISTORY = True` in the account data cell to request the all-time history again.
- The notebook reports account value, open-position metrics, spot and perp realized PnL, unrealized PnL, VaR and historical volatility, and asset-level PnL attribution. It also plots account and BTC comparisons.

To analyze another account, update `ADDR` in the setup cell with that account’s public Hyperliquid address. The notebook may show less historical detail for an account if it has fewer saved snapshots or fills.

## Notes

- The Kelly weights are historical estimates based on the available price sample; they are not a guarantee of future performance.
- Hyperliquid history is stored locally in CSV. Back up `data/hyperliquid_account_history.csv` if you want to preserve that record.
- Files in `data/` can contain account-level financial history and portfolio data. Avoid committing them to a public repository unless you intend to publish that data.
