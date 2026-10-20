# Cryptocurrency Portfolio Analytics

This repository contains two Jupyter notebooks for complementary views of a crypto portfolio:

1. **Static portfolio analytics** — models a selected basket of crypto assets from historical market data, estimates constrained Kelly weights, and compares modeled performance with BTC. It does not read exchange positions or place trades.
2. **Tradable Hyperliquid account analytics** — reads the current account, balances, and open perp positions from Hyperliquid, and tracks fee-net realized PnL, signed perp funding, unrealized PnL, and account performance. It analyzes an account that can be traded; the notebook itself only reads account data and does not place trades.

## Notebooks

The two notebooks below contain the complete workflows, including price-history refresh.

| Notebook | What it does | Main data files |
| --- | --- | --- |
| [`crypto_portfolio_analytics.ipynb`](crypto_portfolio_analytics.ipynb) | Analyzes a static, modeled portfolio; fetches daily USD prices, market capitalization, and trading volume from CoinGecko; calculates returns and correlations; optimizes long-only constrained full-Kelly weights; fits a BTC/altcoin/size/momentum/turnover factor model; compares portfolio results with BTC; writes metrics and factor outputs to CSV. | `data/crypto_daily_prices_usd.csv`, `data/crypto_daily_returns.csv`, `data/crypto_return_correlation.csv`, `data/crypto_kelly_full_weights.csv`, `data/crypto_kelly_weights.csv`, `data/crypto_portfolio_daily_returns.csv`, `data/crypto_portfolio_metrics.csv`, `data/crypto_market_data.csv`, `data/crypto_portfolio_factor_model.csv`, `data/crypto_portfolio_factor_fit.csv`, `data/crypto_factor_returns.csv` |
| [`hyperliquid_account_analytics.ipynb`](hyperliquid_account_analytics.ipynb) | Analyzes a tradable Hyperliquid account by reading current account and position data from Hyperliquid; saves account history locally; reports spot/perp balances, fee-net realized PnL with signed perp funding, unrealized PnL, risk metrics, PnL attribution, and charts against BTC. | `data/hyperliquid_account_history.csv` |

All CSV and spreadsheet data files are stored in the `data/` subfolder. The notebooks create this folder when needed and read/write their files there. Keep the existing files in `data/` between runs to preserve price and account history.

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

- The first data-fetch cell asks you to choose a CoinGecko Demo or Pro plan, then enter its API key using hidden input. Create a key in your CoinGecko account and paste it when prompted. The key is not saved in the notebook. Select the same plan as the key you generated (Demo key with Demo, Pro key with Pro).
- The notebook fetches daily USD prices plus historical market capitalization and total volume for its configured coin basket, updating `data/crypto_daily_prices_usd.csv` and `data/crypto_market_data.csv`.
- CoinGecko Demo/Public access is limited to the latest 365 days of historical API data. The refresh cell caps requests to that window and keeps older prices already stored in the CSV; older market-cap and volume factor history can only be filled if it is already saved locally or the API plan allows it.
- The remaining cells calculate daily returns, a correlation matrix, descending constrained full-Kelly weights, and portfolio performance and risk metrics versus BTC. They also estimate factor exposures to BTC market returns, altcoin-market performance, size, momentum, and trading turnover.
- The final export cell writes the weights, return series, correlations, performance metrics, factor betas, model fit statistics, estimated factor return contributions, and factor series to CSV files.

The default basket, weight bounds, and benchmark assumptions are documented in the notebook’s opening markdown cell. Edit those settings there if you want a different analysis.

## 2. Hyperliquid account analytics notebook

Open **hyperliquid_account_analytics.ipynb** and run all cells in order.

- The notebook queries the configured Hyperliquid account through the public API and fetches current spot balances, perp positions, fills, and BTC candles. No API secret is required for these public account-data requests.
- It creates or updates `data/hyperliquid_account_history.csv` with new account snapshots, realized trade PnL, and perp funding history. Keep `data/hyperliquid_account_history.csv` between runs to preserve the local history used by the charts and statistics.
- By default, previously saved history is reused and only new observations are collected. Set `FETCH_FULL_HISTORY = True` in the account data cell to request the all-time history again.
- The notebook reports account value, open-position metrics, fee-net spot/perp realized PnL with signed funding, fees paid, unrealized PnL, VaR and historical volatility, and asset-level PnL attribution. It also plots account and BTC comparisons.

To analyze another account, update `ADDR` in the setup cell with that account’s public Hyperliquid address. The notebook may show less historical detail for an account if it has fewer saved snapshots or fills.

## Notes

- The Kelly weights are historical estimates based on the available price sample; they are not a guarantee of future performance.
- The factor model is a descriptive in-sample regression over the selected assets. Factor betas and contributions are estimates, not causal explanations or forecasts.
- Hyperliquid history is stored locally in CSV. Back up `data/hyperliquid_account_history.csv` if you want to preserve that record.
- Files in `data/` can contain account-level financial history and portfolio data. Avoid committing them to a public repository unless you intend to publish that data.
