"""Crypto Portfolio Analytics implementation.

Implementation for the matching report calls in `portfolio_analytics.ipynb`. Run
the dashboard notebook to execute these methods in order.
"""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from hl_trade_analytics import expected_forward_return, fetch_funding_history, post_info
from factor_model import (
    alpha_verdict,
    btc_momentum_factor,
    btc_realized_vol,
    fit_factor_regression,
    independent_factors,
    split_in_half,
)
from portfolio_risk import calculate_portfolio_risk_metrics
from hl_market_research import asset_class
from IPython.display import display


def _signed_table(frame, formatters=None):
    """Format a table and color positive numeric values green and negatives red."""
    def sign_color(value):
        if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
            if pd.notna(value):
                if value > 0:
                    return 'color: #188038;'
                if value < 0:
                    return 'color: #c62828;'
        return ''
    return frame.style.format(formatters or {}, na_rep='N/A').map(sign_color)


class _TableStyleMixin:
    """Shared table formatting methods."""
    def show_table(self, frame, formatters=None):
        display(_signed_table(frame, formatters))


# ----------------------------------------------------------------------------
# Market data refresh
# ----------------------------------------------------------------------------
#
# Dashboard call: `analysis.refresh_market_data()`. Discovers live perp positions and spot
# balances. Perp USD prices come from Hyperliquid daily candles, including namespaced HIP-3
# markets; CoinGecko provides prices and market-cap/volume history for supported assets.
# Spot-only assets without a unique CoinGecko mapping and assets without 30 common daily
# returns against BTC are reported and skipped.


class _MarketDataMixin:
    def _post_hyperliquid(self, payload):
        try:
            response = requests.post('https://api.hyperliquid.xyz/info', json=payload, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError(
                f'Hyperliquid market-data request failed ({type(exc).__name__}).'
            ) from exc
        if not response.ok:
            raise RuntimeError(
                f'Hyperliquid market-data request failed (HTTP {response.status_code}).'
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError('Hyperliquid returned invalid JSON for a market-data request.') from exc

    def _wallet_assets(self):
        def post_info(payload):
            return self._post_hyperliquid(payload)

        self.perp_asset_dex = {}
        self.spot_wallet_assets = []

        dexes = post_info({'type': 'perpDexs'})
        if not isinstance(dexes, list):
            raise ValueError('Hyperliquid returned an invalid perp DEX list.')
        dex_names = [''] + [dex['name'] for dex in dexes if dex and dex.get('name')]
        assets = []
        for wallet in self.wallet_addresses:
            for dex in dex_names:
                state = post_info({'type': 'clearinghouseState', 'user': wallet, 'dex': dex})
                if not isinstance(state, dict):
                    raise ValueError(f'Hyperliquid returned invalid perp state for wallet {wallet}.')
                for item in state.get('assetPositions', []):
                    position = item.get('position', {})
                    try:
                        size = float(position.get('szi') or 0)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f'Invalid open position size returned for wallet {wallet}.'
                        ) from exc
                    coin = str(position.get('coin', '')).strip()
                    ticker = coin if ':' in coin else coin.upper()
                    if size != 0 and ticker:
                        previous_dex = self.perp_asset_dex.get(ticker)
                        if previous_dex is not None and previous_dex != dex:
                            raise ValueError(f'Hyperliquid returned ambiguous DEX mappings for {ticker}.')
                        self.perp_asset_dex[ticker] = dex
                        if ticker not in assets:
                            assets.append(ticker)
            spot_state = post_info({'type': 'spotClearinghouseState', 'user': wallet})
            if not isinstance(spot_state, dict):
                raise ValueError(f'Hyperliquid returned invalid spot state for wallet {wallet}.')
            for balance in spot_state.get('balances', []):
                try:
                    amount = float(balance.get('total') or 0)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'Invalid spot balance returned for wallet {wallet}.') from exc
                ticker = str(balance.get('coin', '')).strip().upper()
                if amount > 0 and ticker:
                    if ticker not in self.spot_wallet_assets:
                        self.spot_wallet_assets.append(ticker)
                    if ticker not in assets:
                        assets.append(ticker)
        return assets

    def _coingecko_get(self, url, params=None, timeout=30):
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            elapsed = time.monotonic() - self._coingecko_last_request_at
            if elapsed < 2.6:
                time.sleep(2.6 - elapsed)
            self._coingecko_last_request_at = time.monotonic()
            response = requests.get(url, params=params, timeout=timeout)
            rate_limited = response.status_code == 429
            if not rate_limited:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                status = payload.get('status', {}) if isinstance(payload, dict) else {}
                rate_limited = isinstance(status, dict) and status.get('error_code') == 429
            if not rate_limited:
                return response
            if attempt == max_attempts:
                raise RuntimeError(
                    f'CoinGecko rate limit persisted after {max_attempts} attempts; '
                    'market data refresh stopped for this request.'
                )
            retry_after = getattr(response, 'headers', {}).get('Retry-After')
            try:
                delay = float(retry_after)
                if delay < 0:
                    raise ValueError('Retry-After cannot be negative.')
            except (TypeError, ValueError):
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    delay = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    delay = min(2.6 * (2 ** (attempt - 1)), 60)
            print(
                f'CoinGecko rate limit (HTTP 429); retrying attempt '
                f'{attempt + 1}/{max_attempts} in {delay:.1f} seconds.'
            )
            time.sleep(delay)

    def _coingecko_ids(self, assets):
        api_base = 'https://api.coingecko.com/api/v3'
        try:
            response = self._coingecko_get(
                f'{api_base}/coins/list', params={'include_platform': 'false'}, timeout=60,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f'CoinGecko public coin-list request failed ({type(exc).__name__}).'
            ) from exc
        if not response.ok:
            raise RuntimeError(
                f'CoinGecko public coin-list request failed (HTTP {response.status_code}); '
                'the public API may be rate-limited.'
            )
        listing = response.json()
        if not isinstance(listing, list):
            raise ValueError('CoinGecko returned an invalid public coin list.')
        known_ids = {
            'BTC': 'bitcoin', 'VVV': 'venice-token', 'HYPE': 'hyperliquid',
            'USDC': 'usd-coin', 'USDT': 'tether', 'DAI': 'dai',
            'USDE': 'ethena-usde', 'USDH': 'hyperliquid-usd',
        }
        by_symbol = {}
        for coin in listing:
            if isinstance(coin, dict) and coin.get('symbol') and coin.get('id'):
                by_symbol.setdefault(str(coin['symbol']).upper(), []).append(coin['id'])
        coin_ids, unresolved = {}, []
        for ticker in assets:
            if ':' in ticker:
                unresolved.append(f'{ticker} (Hyperliquid DEX symbol)')
                continue
            if ticker in known_ids:
                coin_ids[ticker] = known_ids[ticker]
                continue
            candidates = list(dict.fromkeys(by_symbol.get(ticker, [])))
            if len(candidates) == 1:
                coin_ids[ticker] = candidates[0]
            elif candidates:
                unresolved.append(f'{ticker} (ambiguous CoinGecko symbol)')
            else:
                unresolved.append(f'{ticker} (not listed by CoinGecko)')
        return coin_ids, unresolved

    def refresh_market_data(self):
        self._coingecko_last_request_at = time.monotonic() - 2.6
        self.wallet_assets = self._wallet_assets()
        if not self.wallet_assets:
            raise ValueError('No nonzero perp positions or spot balances were found in the configured wallets.')
        coin_ids, unresolved = self._coingecko_ids(self.wallet_assets)
        self.assets = [
            ticker for ticker in self.wallet_assets
            if ticker in coin_ids or ticker in self.perp_asset_dex
        ]
        if not self.assets:
            raise ValueError('No wallet assets have either a CoinGecko mapping or a Hyperliquid perp market.')
        warnings = []
        unsupported = [ticker for ticker in unresolved if ticker.split(' (',1)[0] not in self.perp_asset_dex]
        if unsupported:
            warnings.append(
                f'Skipping spot-only wallet asset(s) without a unique CoinGecko ID: {", ".join(unsupported)}.'
            )
        perp_without_coingecko = [
            ticker for ticker in self.perp_asset_dex if ticker not in coin_ids
        ]
        if perp_without_coingecko:
            warnings.append(
                'Using Hyperliquid candles for perp asset(s) without a CoinGecko mapping: '
                f'{", ".join(perp_without_coingecko)}.'
            )
        api_base = 'https://api.coingecko.com/api/v3'

        def error_details(response):
            try:
                payload = response.json()
            except ValueError:
                return ''
            if isinstance(payload, dict) and payload.get('error'):
                return f' CoinGecko error: {payload["error"]}'
            status = payload.get('status', {}) if isinstance(payload, dict) else {}
            if isinstance(status, dict) and status.get('error_message'):
                return f' CoinGecko error: {status["error_message"]}'
            return ''

        data_dir = self.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        price_file = data_dir / 'daily_prices_usd.csv'
        market_file = data_dir / 'market_data.csv'
        coin_ids.setdefault(self.benchmark, 'bitcoin')
        requested_assets = list(dict.fromkeys(
            [ticker for ticker in self.assets if ticker in coin_ids] + [self.benchmark]
        ))
        coins = {ticker: coin_ids[ticker] for ticker in requested_assets}
        end_date = datetime.now(timezone.utc)
        public_earliest = end_date - timedelta(days=365)
        fallback_start = public_earliest
        prices = (pd.read_csv(price_file, parse_dates=['date']) if price_file.exists()
                  else pd.DataFrame(columns=['date']))
        if not prices.empty:
            prices['date'] = pd.to_datetime(prices['date'], utc=True).dt.normalize()
            prices = prices.set_index('date').sort_index()
        else:
            prices = pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC', name='date'))
        if market_file.exists():
            market = pd.read_csv(market_file, parse_dates=['date'])
            market['date'] = pd.to_datetime(market['date'], utc=True).dt.normalize()
            market = market.set_index('date').sort_index()
        else:
            market = pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC', name='date'))

        def points_to_series(points, column):
            if not points:
                return pd.Series(dtype=float, name=column)
            frame = pd.DataFrame(points, columns=['timestamp_ms',column])
            frame['date'] = pd.to_datetime(frame.pop('timestamp_ms'),unit='ms',utc=True).dt.normalize()
            return pd.to_numeric(
                frame.drop_duplicates('date',keep='last').set_index('date')[column], errors='coerce'
            )

        updated = {}
        for ticker, coin_id in coins.items():
            if ticker not in prices:
                prices[ticker] = pd.NA
            saved = pd.to_numeric(prices[ticker],errors='coerce').dropna()
            price_start = (saved.index.max()-pd.Timedelta(days=1)).to_pydatetime() if not saved.empty else fallback_start
            cap_col, volume_col = f'{ticker}_market_cap', f'{ticker}_volume'
            latest = [market[c].dropna().index.max() for c in (cap_col,volume_col)
                      if c in market and market[c].notna().any()]
            if len(latest)==2:
                market_start=(min(latest)-pd.Timedelta(days=1)).to_pydatetime()
            elif not saved.empty:
                market_start=saved.index.min().to_pydatetime()
            else:
                market_start=fallback_start
            start_date=min(price_start,market_start).replace(hour=0,minute=0,second=0,microsecond=0)
            start_date=max(start_date,public_earliest)
            url=f'{api_base}/coins/{coin_id}/market_chart/range'
            params={'vs_currency':'usd','from':int(start_date.timestamp()),
                    'to':int(end_date.timestamp()),'interval':'daily'}
            try:
                response=self._coingecko_get(url,params=params,timeout=30)
            except requests.RequestException as exc:
                raise RuntimeError(
                    f'{ticker}: CoinGecko public API request failed ({type(exc).__name__}).'
                ) from None
            if not response.ok:
                raise RuntimeError(
                    f'{ticker}: CoinGecko request failed (HTTP {response.status_code}).'
                    f'{error_details(response)} Check the public API rate limit and requested date range.'
                )
            data=response.json()
            if not isinstance(data, dict) or 'prices' not in data:
                raise ValueError(f'{ticker}: CoinGecko response did not contain prices.')
            fetched=points_to_series(data['prices'],ticker)
            if not fetched.empty:
                prices=prices.reindex(prices.index.union(fetched.index)).sort_index()
                prices.loc[fetched.index,ticker]=fetched
            else:
                warnings.append(f'{ticker}: CoinGecko returned no new price observations.')
            updated[ticker]=len(fetched)
            for response_key,column in [('market_caps',cap_col),('total_volumes',volume_col)]:
                series=points_to_series(data.get(response_key,[]),column)
                if series.empty:
                    warnings.append(f'{ticker}: no {response_key} history was returned.')
                    continue
                market=market.reindex(market.index.union(series.index)).sort_index()
                market.loc[series.index,column]=series
        for ticker, dex in self.perp_asset_dex.items():
            if ticker not in self.assets:
                continue
            # Saved closes stay valid, so only request days from the last saved one onward.
            saved=pd.to_numeric(prices[ticker],errors='coerce').dropna() if ticker in prices else pd.Series(dtype=float)
            candle_start=saved.index.max()-pd.Timedelta(days=1) if not saved.empty else public_earliest
            payload = {
                'type':'candleSnapshot',
                'req':{
                    'coin':ticker,'interval':'1d',
                    'startTime':int(candle_start.timestamp()*1000),
                    'endTime':int(end_date.timestamp()*1000),
                },
            }
            if dex:
                payload['dex']=dex
            candles = self._post_hyperliquid(payload)
            if not isinstance(candles, list):
                raise ValueError(f'Hyperliquid returned invalid daily candles for {ticker}.')
            candle_frame = pd.DataFrame(candles)
            if not candle_frame.empty and {'t','c'}.issubset(candle_frame.columns):
                candle_frame['date'] = pd.to_datetime(
                    pd.to_numeric(candle_frame['t'],errors='coerce'),unit='ms',utc=True
                ).dt.normalize()
                candle_frame['close'] = pd.to_numeric(candle_frame['c'],errors='coerce')
                candle_series = candle_frame.dropna(subset=['date','close']).groupby(
                    'date',sort=True
                )['close'].last()
                candle_series = candle_series.where(candle_series>0).dropna()
            else:
                candle_series = pd.Series(dtype=float)
            if candle_series.empty:
                if ticker not in coin_ids or ticker not in prices or prices[ticker].notna().sum()==0:
                    warnings.append(f'{ticker}: no usable Hyperliquid candles or CoinGecko price history.')
                else:
                    warnings.append(f'{ticker}: no usable Hyperliquid candles; using CoinGecko prices.')
                continue
            prices = prices.reindex(prices.index.union(candle_series.index)).sort_index()
            prices.loc[candle_series.index,ticker] = candle_series
            updated[ticker] = len(candle_series)
        prices=prices.sort_index()
        daily_returns=prices.pct_change(fill_method=None).apply(pd.to_numeric,errors='coerce')
        daily_returns=daily_returns.where(np.isfinite(daily_returns))
        if self.benchmark not in daily_returns:
            raise ValueError(f'No saved or fetched {self.benchmark} prices are available for the benchmark.')
        unavailable_assets = [
            ticker for ticker in self.assets
            if ticker not in prices or not pd.to_numeric(prices[ticker],errors='coerce').notna().any()
        ]
        if unavailable_assets:
            warnings.append(
                f'Skipping wallet asset(s) without usable price history: {", ".join(unavailable_assets)}.'
            )
            self.assets = [ticker for ticker in self.assets if ticker not in unavailable_assets]
        retained_assets=[]
        skipped_assets=[]
        for ticker in self.assets:
            required=[self.benchmark]+retained_assets+[ticker]
            common=daily_returns[required].dropna()
            if len(common)>=30:
                retained_assets.append(ticker)
            else:
                skipped_assets.append(f'{ticker} ({len(common)} common daily returns)')
        self.assets=retained_assets
        if skipped_assets:
            warnings.append(
                'Skipping wallet asset(s) without at least 30 common daily returns with '
                f'{self.benchmark}: {", ".join(skipped_assets)}.'
            )
        prices=prices.sort_index()
        prices.to_csv(price_file,date_format='%Y-%m-%dT%H:%M:%SZ')
        market=market.sort_index()
        market.index.name='date'
        market.reset_index().to_csv(market_file,index=False,date_format='%Y-%m-%dT%H:%M:%SZ')
        self.updated_observations=updated
        self.refresh_warnings=warnings
        self.market_data=market
        message='Refreshed saved prices and market data: '+', '.join(
            f'{ticker} {count:,} days' for ticker,count in updated.items()
        )+'.'
        message+=f' Assets discovered from {len(self.wallet_addresses)} wallet(s): {", ".join(self.assets)}.'
        message+=' Public historical requests are limited to the latest 365 days; older saved rows were retained.'
        print(message)
        for warning in warnings:
            print(warning)
        if not self.assets:
            raise ValueError(
                f'No wallet assets have at least 30 common daily returns with {self.benchmark}; '
                'the refreshed price data was saved, but portfolio analysis cannot continue.'
            )


# ----------------------------------------------------------------------------
# Price history
# ----------------------------------------------------------------------------
#
# Dashboard call: `analysis.load_price_history()`. Load and validate the saved daily USD
# price series.


class _PriceHistoryMixin(_TableStyleMixin):
    def load_price_history(self):
        price_file=self.data_dir/'daily_prices_usd.csv'
        if not price_file.exists():
            raise FileNotFoundError(f'Missing saved price file: {price_file.resolve()}')
        if not self.assets:
            raise RuntimeError('Run refresh_market_data() first to discover wallet assets.')
        prices=pd.read_csv(price_file,parse_dates=['date']).set_index('date').sort_index()
        prices.index=pd.to_datetime(prices.index,utc=True).normalize()
        required=list(dict.fromkeys(self.assets+[self.benchmark]))
        missing=[ticker for ticker in required if ticker not in prices.columns]
        if missing:
            raise ValueError(f'Missing saved price columns: {missing}')
        self.prices=prices[required].apply(pd.to_numeric,errors='coerce').sort_index()


# ----------------------------------------------------------------------------
# Daily returns and correlations
# ----------------------------------------------------------------------------
#
# Dashboard call: `analysis.calculate_returns()`. Build the complete daily-return sample and
# its Pearson correlation matrix.


class _ReturnAnalysisMixin(_TableStyleMixin):
    def calculate_returns(self):
        returns=self.prices.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan)
        required=list(dict.fromkeys(self.assets+[self.benchmark]))
        common=returns.dropna(subset=required)
        if len(common)<30:
            raise ValueError(f'Only {len(common)} common daily returns; need at least 30.')
        self.returns=returns
        self.common_returns=common
        self.correlation=common[self.assets].corr(method='pearson')


# ----------------------------------------------------------------------------
# Correlation heatmap
# ----------------------------------------------------------------------------
#
# Dashboard call: `analysis.plot_correlations()`. Display the correlation matrix as an
# interactive Plotly heatmap.


class _CorrelationPlotMixin(_TableStyleMixin):
    def plot_correlations(self):
        categories=dict(post_info({'type':'perpCategories'}))
        groups={'Crypto':[],'Tradfi':[],'HIP-3':[]}
        for ticker in self.assets:
            groups[asset_class(self.perp_asset_dex.get(ticker),categories.get(ticker))].append(ticker)
        for name,tickers in groups.items():
            if len(tickers)<2:
                print(f'{name} correlations need at least two assets; held: {", ".join(tickers) or "none"}.')
                continue
            # Each group uses its own shared dates, so short histories elsewhere don't shrink it.
            sample=self.returns[tickers].dropna()
            corr=sample.corr(method='pearson')
            size=min(800,max(350,70*len(tickers)+150))
            fig=go.Figure(go.Heatmap(
                z=corr.to_numpy(),x=corr.columns.tolist(),y=corr.index.tolist(),
                zmin=-1,zmax=1,colorscale='RdBu',reversescale=True,
                text=np.round(corr.to_numpy(),2),texttemplate='%{text:.2f}',
                hovertemplate='%{y} vs %{x}: %{z:.3f}<extra></extra>',
                showscale=False,
            ))
            fig.update_layout(title=f'{name} daily return correlations ({len(sample)} days)',
                              template='plotly_white',width=size+100,height=size)
            fig.update_yaxes(autorange='reversed')
            fig.show()


# ----------------------------------------------------------------------------
# Constrained long/short full-Kelly portfolio
# ----------------------------------------------------------------------------
#
# Dashboard call: `analysis.optimize_portfolio()`. Optimize signed risky-asset weights from
# historical returns, keep net unallocated capital as cash, and display long and short
# weights with signed formatting. Position signs are selected by the optimizer and are not
# forced to match current wallet sides.


def _project_weights(weights, lower, upper, budget=1.0):
    projected=np.clip(weights,lower,upper)
    if projected.sum()<=budget:
        return projected
    left=np.min(weights-upper)-1.0
    right=np.max(weights-lower)+1.0
    for _ in range(100):
        shift=(left+right)/2.0
        projected=np.clip(weights-shift,lower,upper)
        if projected.sum()>budget:
            left=shift
        else:
            right=shift
    return np.clip(weights-right,lower,upper)


def _optimize_full_kelly(returns, lower, upper, cash_daily_return, max_iterations=20000):
    def objective(weights):
        portfolio_returns=returns@weights+(1.0-weights.sum())*cash_daily_return
        if np.any(portfolio_returns<=-1.0):
            return float('inf')
        return -np.log1p(portfolio_returns).mean()
    def gradient(weights):
        portfolio_returns=returns@weights+(1.0-weights.sum())*cash_daily_return
        marginal_returns=returns-cash_daily_return
        return -(marginal_returns/(1.0+portfolio_returns[:,None])).mean(axis=0)
    weights=_project_weights(np.full(returns.shape[1],1.0/returns.shape[1]),lower,upper)
    step_size=1.0
    for _ in range(max_iterations):
        current=objective(weights)
        grad=gradient(weights)
        trial=step_size
        for _ in range(100):
            candidate=_project_weights(weights-trial*grad,lower,upper)
            direction=candidate-weights
            candidate_value=objective(candidate)
            bound=current+grad@direction+(direction@direction)/(2.0*trial)
            if candidate_value<=bound+1e-14:
                break
            trial*=0.5
        else:
            raise RuntimeError('Kelly optimizer could not find a stable step.')
        if np.max(np.abs(direction))<1e-10:
            return candidate
        weights=candidate
        step_size=min(trial*1.2,1e6)
    raise RuntimeError('Kelly optimizer did not converge within the iteration limit.')


class _KellyMixin(_TableStyleMixin):
    def optimize_portfolio(self):
        self.min_weight=-0.15
        self.max_weight=0.15
        self.risk_free_annual=0.05
        self.trading_days_per_year=365
        self.risk_free_daily=(1+self.risk_free_annual)**(1/self.trading_days_per_year)-1
        returns=self.common_returns[self.assets].to_numpy(dtype=float)
        weights=_optimize_full_kelly(
            returns,self.min_weight,self.max_weight,self.risk_free_daily
        )
        self.full_kelly=pd.Series(weights,index=self.assets,name='weight').sort_values(ascending=False)
        self.full_portfolio=pd.concat([
            self.full_kelly,pd.Series({'Cash':1.0-self.full_kelly.sum()},name='weight')
        ])
        completed_prices=self.prices[self.assets].copy()
        completed_prices.index=pd.to_datetime(completed_prices.index,utc=True)
        completed_prices=completed_prices.loc[
            completed_prices.index<pd.Timestamp.now(tz='UTC').normalize()
        ]
        expected_returns={}
        expected_warnings=[]
        for asset in self.assets:
            try:
                expected_returns[asset],_=expected_forward_return(
                    completed_prices[asset],horizon_days=5
                )
            except ValueError as exc:
                expected_returns[asset]=float('nan')
                expected_warnings.append(f'{asset}: {exc}')
        expected_pnl_contribution=pd.Series(index=self.full_portfolio.index,dtype=float)
        for asset in self.assets:
            expected_pnl_contribution[asset]=(
                100*self.full_portfolio.loc[asset]*expected_returns[asset]
            )
        expected_pnl_contribution['Cash']=(
            100*self.full_portfolio.loc['Cash']*((1+self.risk_free_daily)**5-1)
        )
        self.expected_pnl_contribution=expected_pnl_contribution
        weight_table=pd.DataFrame({
            'Weight (%)':100*self.full_portfolio,
            'Expected Price PnL / $100 equity (5d)':expected_pnl_contribution,
        })
        long_exposure=weights[weights>0].sum()
        short_exposure=-weights[weights<0].sum()
        print('Constrained long/short full-Kelly weights, sorted by coin weight (%)')
        print(f'Gross long exposure: {long_exposure:.1%}; gross short exposure: {short_exposure:.1%}; net exposure: {weights.sum():.1%}.')
        print('Expected price PnL is a historical mean-forward-return contribution per $100 of portfolio equity; it excludes funding, fees, and slippage.')
        self.show_table(weight_table,{
            'Weight (%)':'{:.2f}',
            'Expected Price PnL / $100 equity (5d)':'${:.3f}',
        })
        if expected_warnings:
            print('Expected-PnL history warnings: ' + '; '.join(expected_warnings))


# ----------------------------------------------------------------------------
# Portfolio performance and risk metrics
# ----------------------------------------------------------------------------
#
# Dashboard call: `analysis.analyze_performance()`. Show Plotly portfolio-vs-BTC growth and
# styled trailing-return, drawdown, and risk tables.


class _PerformanceMixin(_TableStyleMixin):
    def analyze_performance(self):
        benchmark='BTC'
        if benchmark not in self.common_returns.columns:
            raise ValueError(f'Benchmark {benchmark} is not in the return data.')
        weights=self.full_kelly.reindex(self.assets).to_numpy()
        portfolio_returns=pd.Series(
            self.common_returns[self.assets].to_numpy()@weights
            +self.full_portfolio.loc['Cash']*self.risk_free_daily,
            index=self.common_returns.index,name='Portfolio',
        )
        benchmark_returns=self.common_returns[benchmark]
        excess=portfolio_returns-self.risk_free_daily
        benchmark_excess=benchmark_returns-self.risk_free_daily
        beta=portfolio_returns.cov(benchmark_returns)/benchmark_returns.var(ddof=1)
        risk_metrics=calculate_portfolio_risk_metrics(
            portfolio_returns,self.risk_free_daily,self.trading_days_per_year
        )
        alpha=(excess.mean()-beta*benchmark_excess.mean())*self.trading_days_per_year
        initial_value=100.0
        initial_date=self.common_returns.index[0]-pd.Timedelta(days=1)
        portfolio_value=pd.concat([
            pd.Series([initial_value],index=pd.DatetimeIndex([initial_date])),
            initial_value*(1+portfolio_returns).cumprod(),
        ])
        btc_value=pd.concat([
            pd.Series([initial_value],index=pd.DatetimeIndex([initial_date])),
            initial_value*(1+benchmark_returns).cumprod(),
        ])
        portfolio_value.name='Portfolio value ($)'
        btc_value.name='BTC value ($)'
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_value.index,y=portfolio_value,name='Constrained long/short full-Kelly',
                                 line={'color':'#2563eb','width':2}))
        fig.add_trace(go.Scatter(x=btc_value.index,y=btc_value,name='BTC',
                                 line={'color':'#F7931A','width':2}))
        fig.update_layout(title='Since-inception value: constrained long/short full-Kelly vs. BTC',
                          xaxis_title='Date',yaxis_title='Value of $100 invested',
                          template='plotly_white',hovermode='x unified')
        fig.show()
        as_of=portfolio_value.index[-1]
        def trailing_return(days=None,months=None):
            target=as_of-(pd.DateOffset(months=months) if months is not None else pd.Timedelta(days=days))
            prior=portfolio_value.loc[portfolio_value.index<=target]
            if prior.empty:
                return np.nan
            return portfolio_value.iloc[-1]/prior.iloc[-1]-1
        period_returns=pd.Series({
            '1D':trailing_return(days=1),'7D':trailing_return(days=7),
            'M (1 month)':trailing_return(months=1),'Q (3 months)':trailing_return(months=3),
            'Since Inception':portfolio_value.iloc[-1]/100-1,
        },name='Return')
        self.show_table(period_returns.to_frame(),{'Return':'{:.2%}'})
        daily_summary=pd.Series({
            'Average daily return':portfolio_returns.mean(),
            'Largest daily gain':portfolio_returns.max(),
            'Largest daily loss':portfolio_returns.min(),
        },name='Daily return')
        self.show_table(daily_summary.to_frame(),{'Daily return':'{:.2%}'})
        peak=portfolio_value.cummax()
        drawdown=portfolio_value/peak-1
        drawdown_dollars=portfolio_value-peak
        max_drawdown=drawdown.min()
        investment_summary=pd.DataFrame([{
            'All-Time High ($)':portfolio_value.max(),
            'Minimum Portfolio Value ($)':portfolio_value.min(),
            'Max Drawdown ($)':drawdown_dollars.min(),
            'Max Drawdown (%)':max_drawdown,
        }],index=['Since inception'])
        self.show_table(investment_summary,{
            'All-Time High ($)':'${:,.2f}','Minimum Portfolio Value ($)':'${:,.2f}',
            'Max Drawdown ($)':lambda v:f'-${abs(v):,.2f}' if v<0 else f'${v:,.2f}',
            'Max Drawdown (%)':'{:.2%}',
        })
        risk_free_annual=(1+self.risk_free_daily)**self.trading_days_per_year-1
        metrics=pd.DataFrame([{
            'Sharpe (excess, sqrt 365)':risk_metrics['Sharpe (annualized)'],
            'Sharpe (excess, sqrt 252)':risk_metrics['Sharpe (excess, 252d)'],
            'Sharpe (raw, sqrt 252)':risk_metrics['Sharpe (raw, 252d)'],
            'Sortino (365d)':risk_metrics['Sortino (annualized)'],
            'Beta':beta,'Alpha':alpha,
            'Arithmetic Return (365d, linear)':risk_metrics['Arithmetic Return (annualized)'],
            'Hist Vol (daily)':risk_metrics['Hist Vol (daily)'],
            'Hist Vol (annualized, 365d)':risk_metrics['Hist Vol (annualized)'],
            'Historical 1D VaR 95%':risk_metrics['Historical VaR 95%'],
            'Historical VaR 95% ($ / $100)':risk_metrics['Historical VaR 95%']*100,
            'Historical 1D VaR 99%':risk_metrics['Historical VaR 99%'],
            'Historical VaR 99% ($ / $100)':risk_metrics['Historical VaR 99%']*100,
            'Normal 1D VaR 95%':risk_metrics['Normal VaR 95%'],
            'Normal VaR 95% ($ / $100)':risk_metrics['Normal VaR 95%']*100,
            'Normal 1D VaR 99%':risk_metrics['Normal VaR 99%'],
            'Normal VaR 99% ($ / $100)':risk_metrics['Normal VaR 99%']*100,
            'Normal 1D VaR 99% (zero mean)':risk_metrics['Normal VaR 99% (zero mean)'],
            'Normal VaR 99% ($ / $100, zero mean)':risk_metrics['Normal VaR 99% (zero mean)']*100,
            'Daily Return Observations':risk_metrics['Daily Return Observations'],
            'Historical VaR 99% Tail Observations':risk_metrics['Historical VaR 99% Tail Observations'],
            'Risk-Free Rate':risk_free_annual,'Max Drawdown':max_drawdown,
        }],index=['In-sample: weights fit and evaluated on same window'])
        self.show_table(metrics,{
            'Sharpe (excess, sqrt 365)':'{:.3f}',
            'Sharpe (excess, sqrt 252)':'{:.3f}',
            'Sharpe (raw, sqrt 252)':'{:.3f}',
            'Sortino (365d)':'{:.3f}',
            'Beta':'{:.3f}','Alpha':'{:.2%}',
            'Arithmetic Return (365d, linear)':'{:.2%}',
            'Hist Vol (daily)':'{:.2%}','Hist Vol (annualized, 365d)':'{:.2%}',
            'Historical 1D VaR 95%':'{:.2%}',
            'Historical VaR 95% ($ / $100)':'${:,.2f}',
            'Historical 1D VaR 99%':'{:.2%}',
            'Historical VaR 99% ($ / $100)':'${:,.2f}',
            'Normal 1D VaR 95%':'{:.2%}',
            'Normal VaR 95% ($ / $100)':'${:,.2f}',
            'Normal 1D VaR 99%':'{:.2%}',
            'Normal VaR 99% ($ / $100)':'${:,.2f}',
            'Normal 1D VaR 99% (zero mean)':'{:.2%}',
            'Normal VaR 99% ($ / $100, zero mean)':'${:,.2f}',
            'Risk-Free Rate':'{:.2%}',
            'Max Drawdown':'{:.2%}',
        })
        print(f"Risk estimates use {risk_metrics['Daily Return Observations']} daily observations "
              f"and 365-day crypto annualization. Historical 99% VaR has only "
              f"{risk_metrics['Historical VaR 99% Tail Observations']} observed tail "
              "observation(s); treat it as a weak tail estimate. Normal VaR is a "
              "Gaussian-model comparison, not a tail guarantee. Arithmetic return is "
              "the daily mean multiplied by 365, not a forecast or compounded return. "
              "These are in-sample results: Kelly weights were optimized and evaluated "
              "on the same return window; use a walk-forward backtest for out-of-sample evidence.")
        self.portfolio_daily_returns=portfolio_returns
        self.portfolio_value=portfolio_value
        self.btc_value=btc_value
        self.performance_metrics=metrics
        self.beta=beta
        self.max_drawdown=max_drawdown
        self.portfolio_period=(self.common_returns.index.min(),self.common_returns.index.max())


# ----------------------------------------------------------------------------
# Factor model
# ----------------------------------------------------------------------------
#
# Dashboard calls: `analysis.fit_factor_model()` and `analysis.test_alpha_out_of_sample()`.
# Fit excess returns against BTC, BTC momentum, BTC realized volatility, a Hyperliquid funding
# spread, and available CoinGecko cross-sectional factors; Hyperliquid-only assets remain in the
# modeled portfolio but are excluded from factors that require CoinGecko market-cap or volume
# history.


def _long_short_factor(signal, asset_returns, long_low=False):
    factor=pd.Series(np.nan,index=asset_returns.index,dtype=float)
    for date in asset_returns.index:
        available=signal.loc[date].dropna().index.intersection(asset_returns.columns)
        if len(available)<4:
            continue
        ranked=signal.loc[date,available].sort_values()
        group_size=len(ranked)//2
        if long_low:
            long_assets,short_assets=ranked.index[:group_size],ranked.index[-group_size:]
        else:
            long_assets,short_assets=ranked.index[-group_size:],ranked.index[:group_size]
        daily_returns=asset_returns.loc[date]
        factor.loc[date]=daily_returns[long_assets].mean()-daily_returns[short_assets].mean()
    return factor


def _normalize_daily_index(data):
    normalized=data.copy()
    normalized.index=pd.to_datetime(normalized.index,utc=True).normalize()
    return normalized.groupby(level=0,sort=True).last()


CORE_FACTORS=['BTC market excess','BTC momentum','BTC realized vol','Funding (low - high)']
STATE_FACTORS=('BTC realized vol',)


class _FactorModelMixin(_TableStyleMixin):
    def _funding_signal(self):
        """Return each held perp's lagged trailing 7-day average daily funding."""
        start=self.common_returns.index.min()-pd.Timedelta(days=10)
        daily={}
        for ticker in self.assets:
            if ticker not in self.perp_asset_dex:
                continue
            history=fetch_funding_history(
                ticker.split(':')[-1],self.perp_asset_dex[ticker],start_time=start,allow_empty=True
            )
            if not history.empty:
                daily[ticker]=history.set_index('time')['fundingRate'].resample('1D').sum()
        daily=pd.DataFrame(daily)
        signal=daily.rolling(7,min_periods=3).mean().shift(1)
        signal.index=pd.to_datetime(signal.index,utc=True).normalize()
        return signal

    def fit_factor_model(self):
        market_file=self.data_dir/'market_data.csv'
        if not market_file.exists():
            raise FileNotFoundError(f'Missing {market_file}. Run the CoinGecko refresh cell first.')
        market=pd.read_csv(market_file,parse_dates=['date'])
        market['date']=pd.to_datetime(market['date'],utc=True).dt.normalize()
        market=market.set_index('date').sort_index()
        market_caps=pd.DataFrame(index=self.common_returns.index)
        volumes=pd.DataFrame(index=self.common_returns.index)
        market_assets=[]
        for asset in self.assets:
            cap_col,volume_col=f'{asset}_market_cap',f'{asset}_volume'
            if cap_col not in market or not market[cap_col].notna().any():
                continue
            market_caps[asset]=pd.to_numeric(market[cap_col],errors='coerce').reindex(self.common_returns.index).ffill(limit=2)
            if volume_col in market:
                volumes[asset]=pd.to_numeric(market[volume_col],errors='coerce').reindex(self.common_returns.index).ffill(limit=2)
            else:
                volumes[asset]=np.nan
            market_assets.append(asset)
        alt_assets=[asset for asset in market_assets if asset!='BTC']
        factors=pd.DataFrame(index=self.common_returns.index)
        factors['BTC market excess']=self.common_returns['BTC']-self.risk_free_daily
        btc_history=self.returns['BTC'].dropna()
        factors['BTC momentum']=btc_momentum_factor(btc_history,self.risk_free_daily).reindex(factors.index)
        factors['BTC realized vol']=btc_realized_vol(
            btc_history,periods_per_year=self.trading_days_per_year
        ).reindex(factors.index)
        funding_signal=self._funding_signal()
        funding_assets=list(funding_signal.columns)
        # Price-only spread, matching the portfolio returns, which also exclude funding payments.
        funding_spread=_long_short_factor(
            funding_signal.reindex(_normalize_daily_index(self.common_returns).index),
            _normalize_daily_index(self.common_returns)[funding_assets],long_low=True,
        ) if len(funding_assets)>=4 else None
        if alt_assets:
            lagged_caps=market_caps[alt_assets].shift(1)
            alt_weights=lagged_caps.div(lagged_caps.sum(axis=1),axis=0)
            alt_return=(self.common_returns[alt_assets]*alt_weights).sum(axis=1,min_count=len(alt_assets))
            factors['Alt market - BTC']=alt_return-self.common_returns['BTC']
        factor_notes=[]
        if funding_spread is None:
            factor_notes.append(f'Funding factor omitted: {len(funding_assets)} perps have funding history; at least 4 are needed.')
        missing_market_assets=[asset for asset in self.assets if asset not in market_assets]
        if missing_market_assets:
            factor_notes.append(
                'CoinGecko cross-sectional factors exclude assets without market-cap history: '
                f'{", ".join(missing_market_assets)}.'
            )
        if not alt_assets:
            factor_notes.append('Alt market factor omitted: no selected assets have CoinGecko market-cap history.')
        if len(alt_assets)>=4:
            momentum=self.prices[alt_assets].pct_change(30).shift(1).reindex(self.common_returns.index)
            turnover=volumes[alt_assets].div(market_caps[alt_assets].where(market_caps[alt_assets]>0))
            lagged_turnover=turnover.shift(1).rolling(30,min_periods=20).mean()
            factors['Small - large']=_long_short_factor(lagged_caps[alt_assets],self.common_returns[alt_assets],long_low=True)
            factors['Momentum']=_long_short_factor(momentum,self.common_returns[alt_assets])
            if volumes[alt_assets].notna().any().any():
                factors['High - low turnover']=_long_short_factor(lagged_turnover,self.common_returns[alt_assets])
            else:
                factor_notes.append('Turnover factor omitted: CoinGecko volume history is unavailable.')
        else:
            factor_notes.append(
                f'Size, momentum, and turnover spreads omitted: {len(alt_assets)} CoinGecko-priced '
                'altcoins; at least 4 are needed.'
            )
        factor_note=' '.join(factor_notes) or None
        factors=_normalize_daily_index(factors)
        if funding_spread is not None:
            factors['Funding (low - high)']=funding_spread
        portfolio_excess=_normalize_daily_index(
            (self.portfolio_daily_returns-self.risk_free_daily).rename('Portfolio excess')
        )
        core_factors=[name for name in CORE_FACTORS if name in factors]
        model_data=pd.concat([portfolio_excess,factors],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(model_data)<60 and len(factors.columns)>len(core_factors):
            full_factor_observations=len(model_data)
            factors=factors[core_factors]
            model_data=pd.concat([portfolio_excess,factors],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
            factor_notes.append(
                f'CoinGecko cross-sectional factors left only {full_factor_observations} complete observations; '
                'omitting them to retain the available portfolio history.'
            )
            factor_note=' '.join(factor_notes)
        if len(model_data)<30:
            raise ValueError(
                f'Factor model has only {len(model_data)} complete aligned daily observations; at least 30 are required.'
            )
        names=list(factors.columns)
        state_factors=tuple(name for name in names if name in STATE_FACTORS)
        fit_result=fit_factor_regression(model_data['Portfolio excess'],model_data[names],state_factors)
        annual_alpha=fit_result.alpha*self.trading_days_per_year
        annual_premiums=model_data[names].mean()*self.trading_days_per_year
        # A state factor is a condition, not a return, so it has no premium or contribution.
        annual_premiums.loc[list(state_factors)]=np.nan
        annual_contributions=fit_result.betas*annual_premiums
        summary=pd.DataFrame({
            'Beta':fit_result.betas,'HAC t-stat':fit_result.t_stats,
            'Factor premium (%/yr)':annual_premiums,
            'Portfolio contribution (%/yr)':annual_contributions,
        },index=names)
        summary.loc['Alpha']=[np.nan,fit_result.alpha_t_stat,np.nan,annual_alpha]
        fit=pd.Series({
            'Observations':fit_result.observations,'R-squared':fit_result.r_squared,
            'Adjusted R-squared':fit_result.adjusted_r_squared,
            'Annual alpha':annual_alpha,
            'BTC beta (factor model)':fit_result.betas['BTC market excess'],'BTC beta (single factor)':self.beta,
        },name='Factor model fit')
        print(f'Factor model: R²={fit_result.r_squared:.1%} | adjusted R²={fit_result.adjusted_r_squared:.1%} | '
              f'annual alpha={annual_alpha:.2%} | n={fit_result.observations} days')
        print(f'Single-factor BTC beta={self.beta:.3f} | factor-model BTC beta={summary.loc["BTC market excess","Beta"]:.3f}')
        print('BTC realized vol beta is the daily return per one standard deviation of lagged 30-day volatility; '
              'alpha is at average volatility. Funding (low - high) is long low-funding and short high-funding '
              'held perps, by lagged 7-day average funding. Returns exclude funding payments and trading costs.')
        if factor_note:
            print(factor_note)
        self.show_table(summary,{
            'Beta':'{:.3f}','HAC t-stat':'{:.2f}',
            'Factor premium (%/yr)':'{:.2%}',
            'Portfolio contribution (%/yr)':'{:.2%}',
        })
        self.show_table(fit.to_frame())
        self.factor_returns=factors
        self.factor_model_summary=summary
        self.factor_model_fit=fit
        self.factor_model_dates=model_data.index

    def _weighted_excess_returns(self,fit_dates):
        """Fit Kelly weights on ``fit_dates`` only and return excess returns for every date."""
        returns=_normalize_daily_index(self.common_returns)[self.assets]
        weights=_optimize_full_kelly(
            returns.loc[fit_dates].to_numpy(dtype=float),self.min_weight,self.max_weight,self.risk_free_daily
        )
        portfolio=returns.to_numpy(dtype=float)@weights+(1.0-weights.sum())*self.risk_free_daily
        return pd.Series(portfolio-self.risk_free_daily,index=returns.index)

    def test_alpha_out_of_sample(self):
        factors=self.factor_returns
        names=list(factors.columns)
        state_factors=tuple(name for name in names if name in STATE_FACTORS)
        dates=self.factor_model_dates
        first,second=split_in_half(dates)
        runs=[
            ('In-sample','Full sample','Full sample',dates,dates),
            ('In-sample','Full sample','First half',dates,first),
            ('In-sample','Full sample','Second half',dates,second),
            ('In-sample','First half','First half',first,first),
            ('In-sample','Second half','Second half',second,second),
            ('Out-of-sample','First half','Second half',first,second),
            ('Out-of-sample','Second half','First half',second,first),
        ]
        rows=[]
        for kind,fit_label,test_label,fit_dates,test_dates in runs:
            usable=independent_factors(factors.loc[test_dates])
            result=fit_factor_regression(
                self._weighted_excess_returns(fit_dates).loc[test_dates],factors.loc[test_dates,usable],
                tuple(name for name in state_factors if name in usable),
            )
            rows.append({
                'Test':kind,'Weights fit on':fit_label,'Evaluated on':test_label,
                'Days':result.observations,'Annual alpha':result.alpha*self.trading_days_per_year,
                'Alpha t-stat':result.alpha_t_stat,'R-squared':result.r_squared,
                **{f'{name} beta':result.betas.get(name,np.nan) for name in names},
                'Factors dropped':', '.join(name for name in names if name not in usable),
            })
        table=pd.DataFrame(rows)
        print(f'First half: {first.min():%Y-%m-%d} to {first.max():%Y-%m-%d} ({len(first)} days). '
              f'Second half: {second.min():%Y-%m-%d} to {second.max():%Y-%m-%d} ({len(second)} days).')
        print('Only the out-of-sample rows test the alpha: their weights never saw the dates they are '
              'evaluated on. Every in-sample row reuses dates the weights were fit on.')
        self.show_table(table.set_index(['Test','Weights fit on','Evaluated on']),{
            'Annual alpha':'{:.2%}','Alpha t-stat':'{:.2f}','R-squared':'{:.1%}',
            **{f'{name} beta':'{:.3f}' for name in names},
        })
        forward=table.iloc[-2]
        print(f"Forward test (fit on first half, evaluated on second): alpha "
              f"{alpha_verdict(forward['Annual alpha'],forward['Alpha t-stat'])}.")
        print('A factor is dropped from a window when it duplicates another there, such as BTC momentum '
              'when BTC stays on one side of its 30-day trend throughout.')
        print(f'Each half has about {len(first)} days and {len(names)} factors, and the weights cover '
              f'{len(self.assets)} assets, so treat these estimates as very noisy.')
        self.alpha_split_results=table


# ----------------------------------------------------------------------------
# Save analysis outputs
# ----------------------------------------------------------------------------
#
# Dashboard call: `analysis.save_outputs()`. Write the refreshed source data, portfolio
# results, daily returns, and factor estimates to `data/`.


class _OutputMixin:
    def save_outputs(self):
        data=self.data_dir
        data.mkdir(parents=True,exist_ok=True)
        self.correlation.to_csv(data/'return_correlation.csv')
        self.full_portfolio.to_csv(data/'kelly_full_weights.csv',header=True)
        self.full_portfolio.to_csv(data/'kelly_weights.csv',header=True)
        self.performance_metrics.to_csv(data/'portfolio_metrics.csv')
        self.portfolio_daily_returns.to_csv(data/'portfolio_daily_returns.csv',header=True)
        self.prices.to_csv(data/'daily_prices_usd.csv')
        self.market_data.sort_index().rename_axis('date').reset_index().to_csv(
            data/'market_data.csv',index=False,date_format='%Y-%m-%dT%H:%M:%SZ'
        )
        self.common_returns.to_csv(data/'daily_returns.csv')
        self.factor_model_summary.to_csv(data/'portfolio_factor_model.csv')
        self.factor_model_fit.to_csv(data/'portfolio_factor_fit.csv')
        self.factor_returns.to_csv(data/'factor_returns.csv')
        self.alpha_split_results.to_csv(data/'alpha_split_results.csv',index=False)
        print('Saved portfolio, factor, market, and return outputs to data/.')


# ----------------------------------------------------------------------------
# Analytics object
# ----------------------------------------------------------------------------
#
# The report notebook creates one `PortfolioAnalytics` instance and calls the named
# methods above.


class PortfolioAnalytics(
    _MarketDataMixin, _PriceHistoryMixin, _ReturnAnalysisMixin, _CorrelationPlotMixin,
    _KellyMixin, _PerformanceMixin,
    _FactorModelMixin, _OutputMixin, _TableStyleMixin,
):
    def __init__(self, data_dir, wallet_addresses):
        self.data_dir=Path(data_dir)
        if not isinstance(wallet_addresses, (list, tuple)) or not wallet_addresses:
            raise ValueError('wallet_addresses must be a non-empty list of wallet addresses.')
        if any(not isinstance(wallet, str) for wallet in wallet_addresses):
            raise ValueError('Every configured wallet address must be a string.')
        self.wallet_addresses=list(dict.fromkeys(wallet_addresses))
        for wallet in self.wallet_addresses:
            if not isinstance(wallet, str) or not re.fullmatch(r'0x[0-9a-fA-F]{40}', wallet):
                raise ValueError(f'Invalid wallet address: {wallet!r}')
        self.assets=[]
        self.wallet_assets=[]
        self.benchmark='BTC'
        self.data_dir.mkdir(parents=True,exist_ok=True)
