"""Portfolio Analytics implementation: HL wallets, the same wallets on other chains, and IB.

Implementation for the matching report calls in `portfolio_analytics.ipynb`. Run
the dashboard notebook to execute these methods in order.

Assets fall into two groups, each measured against its own benchmark on its own calendar:

* Crypto (HL crypto perps and spot, wallet tokens on other chains): against BTC, on every
  calendar day, annualized over 365 days.
* TradFi (IB stocks and HL HIP-3 stock, index, commodity and FX perps): against SPY, on
  US trading days, annualized over 252 days. 24/7 HL prices are sampled at each trading
  day's close, so a weekend move counts in Monday's return, as it would for a stock.

The combined Kelly portfolio uses the trading-day calendar when it holds both groups, and
reports beta to BTC and to SPY.
"""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import io
import json
import re
import time
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from desk.hl.trade_analytics import expected_forward_return, fetch_funding_history, post_info
from factor_model import (
    alpha_verdict,
    fama_macbeth,
    newey_west_mean,
    sort_spread,
    btc_momentum_factor,
    btc_realized_vol,
    fit_factor_regression,
    independent_factors,
    split_in_half,
)
from portfolio_risk import calculate_portfolio_risk_metrics
from desk.hl.market_research import asset_class
from desk.hl.trade_analytics import bootstrap_btc_regression
from desk.accounts import Account, accounts_for, ib_data_dir, load_accounts
from IPython.display import display
from desk.notebook_highlights import HighlightsPanel, SinceLastRun
from desk.notebook_highlights import show as show_highlight_block

CRYPTO, TRADFI = 'Crypto', 'TradFi'
BENCHMARKS = {CRYPTO: 'BTC', TRADFI: 'SPY'}
PERIODS_PER_YEAR = {CRYPTO: 365, TRADFI: 252}
# Stablecoins track a currency, so they are cash here, not risk assets.
STABLECOINS = {'USDC', 'USDT', 'DAI', 'USDE', 'USDH', 'USDBC', 'PYUSD', 'FDUSD', 'TUSD', 'EURC', 'USDS', 'GHO',
               'FRAX', 'LUSD', 'CRVUSD', 'USD0', 'RLUSD'}
# Wallet tokens worth less than this are left out as dust.
MIN_WALLET_TOKEN_USD = 10.0
# A trading day takes the last price at most this old (covers long weekends).
MAX_CALENDAR_GAP_DAYS = 4
MIN_COMMON_RETURNS = 30
# DefiLlama serves at most 500 points per request, so longer histories are fetched in pages.
DEFILLAMA_PAGE_DAYS = 500
# Saved history older than this is kept, but new history is only requested back to here.
DEFAULT_HISTORY_YEARS = 5
HISTORY_FLOOR_FILE = 'price_history_floor.json'
# Estimated market caps before CoinGecko's year: price x the earliest observed supply, for coins
# whose implied circulating supply moved less than this over the observed year (no big unlocks).
# Ken French data library: the standard free source of daily equity style-factor returns.
FRENCH_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
FRENCH_FILES = {
    'us5': 'F-F_Research_Data_5_Factors_2x3_daily_CSV.zip',
    'momentum': 'F-F_Momentum_Factor_daily_CSV.zip',
    'short_reversal': 'F-F_ST_Reversal_Factor_daily_CSV.zip',
    'industries': '12_Industry_Portfolios_Daily_CSV.zip',
    'developed_ex_us': 'Developed_ex_US_5_Factors_Daily_CSV.zip',
}
# Desk-level history files (HL wallets, IB) live in the repository's data folder.
DESK_DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
# Actual-book attribution switches from weekly to daily returns once this many consecutive days exist.
MIN_DAILY_BOOK_RETURNS = 120
# Dividend style factor: no daily French series, so the standard ETF proxy (high-dividend minus market).
DIVIDEND_ETF = 'SCHD'
SUPPLY_DRIFT_LIMIT = 0.25
SUPPLY_ANCHOR_DAYS = 30


def parse_french_csv(text):
    """First daily table of a Ken French CSV as decimal returns indexed by UTC date.

    The files hold notes, then a header line starting with a comma, then ``YYYYMMDD,...`` rows
    in percent; some files hold several tables (value- then equal-weighted), and only the first
    is read.
    """
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(','))
    columns = [column.strip() for column in lines[start].split(',')[1:]]
    rows = []
    for line in lines[start + 1:]:
        fields = [field.strip() for field in line.split(',')]
        if len(fields[0]) != 8 or not fields[0].isdigit():
            break
        rows.append([fields[0]] + fields[1:len(columns) + 1])
    frame = pd.DataFrame(rows, columns=['date'] + columns)
    frame['date'] = pd.to_datetime(frame['date'], format='%Y%m%d', utc=True)
    frame = frame.set_index('date').apply(pd.to_numeric, errors='coerce')
    return frame.where(frame > -99) / 100  # -99.99 marks missing data


def load_french(name, cache_dir, max_age_hours=24):
    """Download (or reuse a recent cached copy of) one Ken French daily file."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / FRENCH_FILES[name]
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_hours * 3600:
        response = requests.get(FRENCH_URL + FRENCH_FILES[name], timeout=120,
                                headers={'User-Agent': 'trading-desk/1.0 (research)'})
        response.raise_for_status()
        path.write_bytes(response.content)
    archive = zipfile.ZipFile(io.BytesIO(path.read_bytes()))
    return parse_french_csv(archive.read(archive.namelist()[0]).decode('latin-1'))


def estimate_market_caps(caps, prices, drift_limit=SUPPLY_DRIFT_LIMIT, anchor_days=SUPPLY_ANCHOR_DAYS):
    """Extend market caps back before CoinGecko's window as price x the earliest observed supply.

    Implied circulating supply is CoinGecko market cap / price. The anchor is its median over the
    first ``anchor_days`` CoinGecko observed, not today's supply, which halves the error for
    tokens that keep unlocking. Coins whose supply moved more than ``drift_limit`` across the
    observed year are not estimated. Actual market caps are never changed.

    Returns (caps with estimates added, estimated assets, {excluded asset: supply change}).
    """
    combined = caps.copy()
    estimated, excluded = [], {}
    for asset in caps.columns:
        if asset not in prices:
            continue
        supply = (caps[asset] / prices[asset]).replace([np.inf, -np.inf], np.nan).dropna()
        supply = supply[supply > 0]
        if len(supply) < 2 * anchor_days:
            continue
        anchor = supply.iloc[:anchor_days].median()
        drift = supply.iloc[-anchor_days:].median() / anchor - 1
        if abs(drift) > drift_limit:
            excluded[asset] = float(drift)
            continue
        before = (prices.index < supply.index.min()) & combined[asset].isna()
        filled = prices.loc[before, asset] * anchor
        combined.loc[before, asset] = filled
        if filled.notna().any():
            estimated.append(asset)
    return combined, estimated, excluded


def calendar_prices(prices, calendar, max_gap_days=MAX_CALENDAR_GAP_DAYS):
    """Prices on each calendar date: the last close on or before it, at most ``max_gap_days`` old."""
    prices = prices.sort_index()
    return prices.reindex(calendar, method='ffill', tolerance=pd.Timedelta(days=max_gap_days))


def compound_to_calendar(returns, calendar):
    """Compound daily returns into the periods that end on each calendar date.

    A daily return on date d counts toward the first calendar date on or after d. The first
    calendar date's period has no known start, so it is left empty.
    """
    returns = returns.sort_index()
    positions = calendar.searchsorted(returns.index, side='left')
    inside = positions < len(calendar)
    labels = calendar[positions[inside]]
    compounded = (1 + returns[inside]).groupby(labels).prod(min_count=1) - 1
    compounded = compounded.reindex(calendar)
    compounded.iloc[0] = np.nan
    return compounded


def wallet_token_ticker(symbol, chain, token):
    """Analysis ticker for a wallet token: staking receipts become the token they represent."""
    from desk.venues import onchain

    base = symbol.split(' (')[0].strip()
    if (chain, token.lower()) in onchain.PRICED_AS and base.lower().startswith('s'):
        base = base[1:]  # sVVV is VVV staked, valued one-for-one
    return base.upper()


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
# balances. Perp USD prices come from HL daily candles, including namespaced HIP-3
# markets; CoinGecko provides prices and market-cap/volume history for supported assets.
# Spot-only assets without a unique CoinGecko mapping and assets without 30 common daily
# returns against BTC are reported and skipped.


def _coingecko_daily_series(points, column):
    """Convert CoinGecko points to daily values dated by the day they close.

    A point at exactly 00:00 UTC is the previous day's close, so it takes that date;
    this matches HL candles, which are dated by the day they cover.
    """
    if not points:
        return pd.Series(dtype=float, name=column)
    frame = pd.DataFrame(points, columns=['timestamp_ms', column])
    times = pd.to_datetime(frame.pop('timestamp_ms'), unit='ms', utc=True)
    days = times.dt.normalize()
    frame['date'] = days.where(times != days, days - pd.Timedelta(days=1))
    return pd.to_numeric(
        frame.drop_duplicates('date', keep='last').set_index('date')[column], errors='coerce'
    )


class _CalendarMixin:
    """Which benchmark and calendar each asset group uses, and the return tables built on them."""

    def _benchmark_for(self, group, prices=None):
        prices = self.prices if prices is None else prices
        benchmark = BENCHMARKS[group]
        if benchmark in prices and pd.to_numeric(prices[benchmark], errors='coerce').notna().any():
            return benchmark
        return 'BTC'  # no SPY prices: TradFi falls back to BTC on the daily calendar

    def _trading_calendar(self, prices):
        """US trading days, taken from the dates SPY has a close."""
        if 'SPY' not in prices:
            return None
        dates = pd.to_numeric(prices['SPY'], errors='coerce').dropna().index
        return dates if len(dates) else None

    def _has_tradfi(self, assets):
        return any(self.asset_group.get(ticker) == TRADFI for ticker in assets)

    def _uses_trading_calendar(self, assets, prices):
        return self._has_tradfi(assets) and self._trading_calendar(prices) is not None

    def _group_frame(self, prices, tickers, group):
        """Returns of ``tickers`` and their benchmark on the group's own calendar."""
        benchmark = self._benchmark_for(group, prices)
        columns = list(dict.fromkeys([ticker for ticker in tickers if ticker in prices] + [benchmark]))
        frame = prices[columns].apply(pd.to_numeric, errors='coerce').sort_index()
        calendar = self._trading_calendar(prices) if group == TRADFI else None
        if calendar is not None:
            frame = calendar_prices(frame, calendar)
        returns = frame.pct_change(fill_method=None)
        return returns.where(np.isfinite(returns))

    def _required_columns(self, assets, prices):
        columns = list(assets) + ['BTC']
        if self._uses_trading_calendar(assets, prices):
            columns.append('SPY')
        return list(dict.fromkeys(columns))

    def _return_frames(self, prices, assets):
        """Per-group returns on each group's calendar, and combined returns for the whole book.

        With TradFi assets the combined book uses US trading days; otherwise every day.
        """
        prices = prices.sort_index()
        prices = prices[~prices.index.duplicated(keep='last')]
        groups = {}
        for group in (CRYPTO, TRADFI):
            members = [ticker for ticker in assets if self.asset_group.get(ticker, CRYPTO) == group]
            if members:
                groups[group] = self._group_frame(prices, members, group)
        columns = [column for column in self._required_columns(assets, prices) if column in prices]
        frame = prices[columns].apply(pd.to_numeric, errors='coerce')
        if self._uses_trading_calendar(assets, prices):
            frame = calendar_prices(frame, self._trading_calendar(prices))
        combined = frame.pct_change(fill_method=None)
        return groups, combined.where(np.isfinite(combined))


class _MarketDataMixin:
    def _post_hl(self, payload):
        try:
            response = requests.post('https://api.hyperliquid.xyz/info', json=payload, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError(
                f'HL market-data request failed ({type(exc).__name__}).'
            ) from exc
        if not response.ok:
            raise RuntimeError(
                f'HL market-data request failed (HTTP {response.status_code}).'
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError('HL returned invalid JSON for a market-data request.') from exc

    def _wallet_assets(self):
        def post_info(payload):
            return self._post_hl(payload)

        self.perp_asset_dex = {}
        self.spot_wallet_assets = []
        if self.configured_assets:
            # An explicit list replaces wallet discovery; "dex:COIN" names a HIP-3 market.
            self.perp_asset_dex = {
                ticker: ticker.split(':', 1)[0] if ':' in ticker else ''
                for ticker in self.configured_assets
            }
            return list(self.configured_assets)

        dexes = post_info({'type': 'perpDexs'})
        if not isinstance(dexes, list):
            raise ValueError('HL returned an invalid perp DEX list.')
        dex_names = [''] + [dex['name'] for dex in dexes if dex and dex.get('name')]
        assets = []
        for wallet in self.wallet_addresses:
            for dex in dex_names:
                state = post_info({'type': 'clearinghouseState', 'user': wallet, 'dex': dex})
                if not isinstance(state, dict):
                    raise ValueError(f'HL returned invalid perp state for wallet {wallet}.')
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
                            raise ValueError(f'HL returned ambiguous DEX mappings for {ticker}.')
                        self.perp_asset_dex[ticker] = dex
                        if ticker not in assets:
                            assets.append(ticker)
            spot_state = post_info({'type': 'spotClearinghouseState', 'user': wallet})
            if not isinstance(spot_state, dict):
                raise ValueError(f'HL returned invalid spot state for wallet {wallet}.')
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

    def _classify_hl_assets(self, tickers):
        """Put HL perps and spot into Crypto or TradFi (HIP-3 stocks, indices, commodities, FX)."""
        try:
            categories = dict(self._post_hl({'type': 'perpCategories'}))
        except (RuntimeError, TypeError, ValueError):
            categories = {}
        for ticker in tickers:
            if ticker in self.perp_asset_dex:
                kind = asset_class(self.perp_asset_dex[ticker], categories.get(ticker))
                self.asset_group[ticker] = TRADFI if kind == 'Tradfi' else CRYPTO
                self.asset_source[ticker] = 'HL perp'
            else:
                self.asset_group[ticker] = CRYPTO
                self.asset_source[ticker] = 'HL spot'

    def _wallet_token_assets(self, warnings):
        """Tokens the HL wallets hold on other chains, keyed by ticker, with DefiLlama price keys."""
        from desk.venues import onchain

        found = {}
        for wallet in self.wallet_addresses:
            try:
                snapshot, _ = onchain.wallet_snapshot(Account('hl', wallet))
            except Exception as exc:  # one wallet's chains failing must not stop the analysis
                warnings.append(f'Wallet {wallet[-4:]} tokens unavailable ({type(exc).__name__}: {exc}).')
                continue
            for position in snapshot.positions:
                chain, token = position.ref.split(':', 1)
                if token.startswith('rewards:') or not abs(position.notional_usd) >= MIN_WALLET_TOKEN_USD:
                    continue
                ticker = wallet_token_ticker(position.symbol, chain, token)
                if ticker in STABLECOINS:
                    continue
                entry = found.setdefault(ticker, {'key': onchain.price_key(onchain.CHAINS[chain], token), 'value': 0.0})
                entry['value'] += abs(position.notional_usd)
        return found

    def _ib_assets(self, warnings):
        """Adjusted daily closes for every IB stock position and for SPY, read through TWS."""
        accounts = accounts_for('ib', load_accounts())
        if not accounts:
            return {}, None
        from desk.venues import ib as ib_venue

        def read():
            from ib_async import Contract

            closes, spy = {}, None
            for account in accounts:
                ib = ib_venue.connect(account, 'analytics', readonly=True, timeout=20)
                try:
                    snapshot = ib_venue.snapshot_from_ib(ib, account)
                    for position in snapshot.positions:
                        if position.asset_class != 'equity':
                            warnings.append(f'IB {position.symbol}: {position.asset_class} positions are not analysed.')
                            continue
                        contract = Contract(conId=int(position.ref))
                        try:
                            ib.qualifyContracts(contract)
                            closes[position.symbol.upper()] = ib_venue.adjusted_daily_closes(
                                ib, contract, f'{self.history_years} Y')
                        except Exception as exc:
                            warnings.append(f'IB {position.symbol}: no daily history ({exc}).')
                    if spy is None:
                        contract, _ = ib_venue.resolve_contract(ib, ib_venue.ContractSpec('SPY'))
                        spy = ib_venue.adjusted_daily_closes(ib, contract, f'{self.history_years} Y')
                        try:  # dividend style factor (high-dividend ETF minus SPY)
                            contract, _ = ib_venue.resolve_contract(ib, ib_venue.ContractSpec(DIVIDEND_ETF))
                            closes['__' + DIVIDEND_ETF] = ib_venue.adjusted_daily_closes(
                                ib, contract, f'{self.history_years} Y')
                        except Exception as exc:
                            warnings.append(f'{DIVIDEND_ETF} unavailable for the dividend factor ({exc}).')
                finally:
                    ib.disconnect()
            return closes, spy

        try:
            return ib_venue.run_off_event_loop(read)
        except Exception as exc:
            warnings.append(f'IB positions unavailable ({exc}); saved IB prices are used where present.')
            return {}, None

    def _defillama_history(self, key, start, end=None):
        """Daily USD closes for one DefiLlama coin key (``coingecko:<id>`` or ``chain:address``).

        Pages start at midnight UTC: DefiLlama steps from the requested start time, so a
        midnight start yields closes that match CoinGecko's and HL's daily convention.
        """
        start = pd.Timestamp(start).tz_convert('UTC').normalize() if pd.Timestamp(start).tzinfo \
            else pd.Timestamp(start, tz='UTC').normalize()
        end = pd.Timestamp.now(tz='UTC') if end is None else pd.Timestamp(end)
        pages = []
        while start < end:
            days = max(1, min(DEFILLAMA_PAGE_DAYS, int((end - start).days) + 1))
            response = requests.get(f'https://coins.llama.fi/chart/{key}',
                                    params={'start': int(start.timestamp()), 'span': days, 'period': '1d'}, timeout=60)
            if not response.ok:
                raise RuntimeError(f'DefiLlama price history request failed (HTTP {response.status_code}).')
            points = (response.json().get('coins', {}).get(key) or {}).get('prices', [])
            if points:
                pages.append(_coingecko_daily_series(
                    [[int(point['timestamp']) * 1000, point['price']] for point in points], key))
            start += pd.Timedelta(days=days)
        if not pages:
            return pd.Series(dtype=float, name=key)
        series = pd.concat(pages)
        series = series[~series.index.duplicated(keep='last')].sort_index()
        return series[series.index < pd.Timestamp(end).tz_convert('UTC').normalize()]

    def _hl_daily_closes(self, ticker, dex, start, end):
        """HL daily candle closes for one perp (HL returns up to 5,000 candles per request)."""
        payload = {'type': 'candleSnapshot', 'req': {
            'coin': ticker, 'interval': '1d',
            'startTime': int(pd.Timestamp(start).timestamp() * 1000),
            'endTime': int(pd.Timestamp(end).timestamp() * 1000),
        }}
        if dex:
            payload['dex'] = dex
        candles = self._post_hl(payload)
        if not isinstance(candles, list):
            raise ValueError(f'HL returned invalid daily candles for {ticker}.')
        frame = pd.DataFrame(candles)
        if frame.empty or not {'t', 'c'}.issubset(frame.columns):
            return pd.Series(dtype=float)
        frame['date'] = pd.to_datetime(pd.to_numeric(frame['t'], errors='coerce'), unit='ms', utc=True).dt.normalize()
        frame['close'] = pd.to_numeric(frame['c'], errors='coerce')
        series = frame.dropna(subset=['date', 'close']).groupby('date', sort=True)['close'].last()
        return series.where(series > 0).dropna()

    def _backfill(self, prices, coin_ids, warnings):
        """Extend each asset's saved prices back toward ``history_start`` from free deep sources.

        HL candles go back to each perp's listing; DefiLlama serves CoinGecko-listed coins and
        wallet tokens for years. Only missing dates are filled, so newer data from the primary
        source is never overwritten. Where a source has nothing older, that date is remembered
        in the data folder so later runs do not ask again.
        """
        floor_file = self.data_dir / HISTORY_FLOOR_FILE
        try:
            floor = json.loads(floor_file.read_text()) if floor_file.exists() else {}
        except ValueError:
            floor = {}
        start = self.history_start
        added = {}

        def earliest(ticker):
            saved = pd.to_numeric(prices[ticker], errors='coerce').dropna() if ticker in prices else pd.Series(dtype=float)
            return saved.index.min() if len(saved) else None

        def needs(ticker):
            first = earliest(ticker)
            if first is None:
                return pd.Timestamp.now(tz='UTC').normalize()
            if first <= start + pd.Timedelta(days=3):
                return None
            if ticker in floor and pd.Timestamp(floor[ticker], tz='UTC') >= first - pd.Timedelta(days=1):
                return None  # the sources had nothing older last time
            return first

        def fill(ticker, series):
            nonlocal prices
            if series.empty:
                return 0
            existing = prices[ticker].reindex(series.index) if ticker in prices else pd.Series(index=series.index, dtype=float)
            missing = series.index[pd.to_numeric(existing, errors='coerce').isna()]
            if len(missing):
                prices = prices.reindex(prices.index.union(missing)).sort_index()
                prices.loc[missing, ticker] = series.loc[missing].to_numpy()
            return len(missing)

        sources = []
        for ticker in self.assets + [self.benchmark]:
            if ticker in self.perp_asset_dex:
                sources.append((ticker, 'HL', self.perp_asset_dex[ticker]))
            if ticker in coin_ids:
                sources.append((ticker, 'DefiLlama', f'coingecko:{coin_ids[ticker]}'))
            if ticker in self.wallet_token_keys:
                sources.append((ticker, 'DefiLlama', self.wallet_token_keys[ticker]))
        for ticker, source, reference in sources:
            until = needs(ticker)
            if until is None:
                continue
            try:
                series = (self._hl_daily_closes(ticker, reference, start, until) if source == 'HL'
                          else self._defillama_history(reference, start, until))
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                warnings.append(f'{ticker}: {source} history backfill failed ({exc}).')
                continue
            added[ticker] = added.get(ticker, 0) + fill(ticker, series)
        for ticker in {ticker for ticker, _, _ in sources}:
            first = earliest(ticker)
            if first is not None and first > start + pd.Timedelta(days=3):
                floor[ticker] = first.strftime('%Y-%m-%d')
        floor_file.write_text(json.dumps(floor, indent=1, sort_keys=True))
        return prices, {ticker: count for ticker, count in added.items() if count}

    def _coingecko_get(self, url, params=None, timeout=30):
        max_attempts = 8
        network_failures = 0
        for attempt in range(1, max_attempts + 1):
            elapsed = time.monotonic() - self._coingecko_last_request_at
            if elapsed < 2.6:
                time.sleep(2.6 - elapsed)
            self._coingecko_last_request_at = time.monotonic()
            try:
                response = requests.get(url, params=params, timeout=timeout)
            except (requests.Timeout, requests.ConnectionError):
                network_failures += 1
                if network_failures >= 3 or attempt == max_attempts:
                    raise
                time.sleep(5 * network_failures)
                continue
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
                unresolved.append(f'{ticker} (HL DEX symbol)')
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
        warnings = []
        self.wallet_assets = self._wallet_assets()
        if not self.configured_assets:
            stable = [ticker for ticker in self.wallet_assets if ticker in STABLECOINS]
            if stable:
                warnings.append(f'Treating stablecoin balance(s) as cash, not risk assets: {", ".join(stable)}.')
            self.wallet_assets = [ticker for ticker in self.wallet_assets if ticker not in STABLECOINS]
        self.asset_group, self.asset_source = {}, {}
        self._classify_hl_assets(self.wallet_assets)
        wallet_tokens, ib_closes, spy_closes = {}, {}, None
        if not self.configured_assets:
            if self.include_wallet_tokens:
                wallet_tokens = self._wallet_token_assets(warnings)
            if self.include_ib:
                ib_closes, spy_closes = self._ib_assets(warnings)
        if not self.wallet_assets and not wallet_tokens and not ib_closes:
            raise ValueError('No positions or balances were found in the configured wallets or IB.')
        coin_ids, unresolved = self._coingecko_ids(self.wallet_assets) if self.wallet_assets else ({}, [])
        self.assets = [
            ticker for ticker in self.wallet_assets
            if ticker in coin_ids or ticker in self.perp_asset_dex
        ]
        self.wallet_token_keys = {}
        for ticker, info in wallet_tokens.items():
            if ticker in self.asset_group:
                continue  # also held on HL, whose prices already cover it
            self.asset_group[ticker], self.asset_source[ticker] = CRYPTO, 'Wallet'
            self.wallet_token_keys[ticker] = info['key']
            self.assets.append(ticker)
        renamed_ib = {}
        reference_closes = {symbol[2:]: closes for symbol, closes in ib_closes.items() if symbol.startswith('__')}
        ib_closes = {symbol: closes for symbol, closes in ib_closes.items() if not symbol.startswith('__')}
        for symbol, closes in ib_closes.items():
            ticker = symbol if symbol not in self.asset_group else f'{symbol}.US'  # never merge with a crypto ticker
            self.asset_group[ticker], self.asset_source[ticker] = TRADFI, 'IB'
            renamed_ib[ticker] = closes
            self.assets.append(ticker)
        if not self.assets:
            raise ValueError('No assets have a price source (CoinGecko, HL, DefiLlama or IB).')
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
                'Using HL candles for perp asset(s) without a CoinGecko mapping: '
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
            fetched=_coingecko_daily_series(data['prices'],ticker)
            # Perps use HL candles only, so one venue sets their closes and dates.
            if ticker in self.perp_asset_dex:
                updated[ticker]=0
            elif not fetched.empty:
                prices=prices.reindex(prices.index.union(fetched.index)).sort_index()
                prices.loc[fetched.index,ticker]=fetched
            else:
                warnings.append(f'{ticker}: CoinGecko returned no new price observations.')
            if ticker not in self.perp_asset_dex:
                updated[ticker]=len(fetched)
            for response_key,column in [('market_caps',cap_col),('total_volumes',volume_col)]:
                series=_coingecko_daily_series(data.get(response_key,[]),column)
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
            candle_start=saved.index.max()-pd.Timedelta(days=1) if not saved.empty else self.history_start
            candle_series=self._hl_daily_closes(ticker,dex,candle_start,end_date)
            if candle_series.empty:
                if ticker not in coin_ids or ticker not in prices or prices[ticker].notna().sum()==0:
                    warnings.append(f'{ticker}: no usable HL candles or CoinGecko price history.')
                else:
                    warnings.append(f'{ticker}: no usable HL candles; using CoinGecko prices.')
                continue
            prices = prices.reindex(prices.index.union(candle_series.index)).sort_index()
            prices.loc[candle_series.index,ticker] = candle_series
            updated[ticker] = len(candle_series)
        for ticker, key in self.wallet_token_keys.items():
            # Wallet tokens: DefiLlama daily prices by contract address, so tickers cannot be confused.
            saved=pd.to_numeric(prices[ticker],errors='coerce').dropna() if ticker in prices else pd.Series(dtype=float)
            start=saved.index.max()-pd.Timedelta(days=1) if not saved.empty else self.history_start
            try:
                series=self._defillama_history(key,start)
            except (requests.RequestException,RuntimeError,ValueError) as exc:
                warnings.append(f'{ticker}: no DefiLlama price history ({exc}).')
                continue
            if series.empty:
                warnings.append(f'{ticker}: DefiLlama returned no prices.')
                continue
            prices=prices.reindex(prices.index.union(series.index)).sort_index()
            prices.loc[series.index,ticker]=series.to_numpy()
            updated[ticker]=len(series)
        for ticker, series in (list(renamed_ib.items())+([('SPY',spy_closes)] if spy_closes is not None else [])
                               +list(reference_closes.items())):
            # Adjusted closes restate the past after a dividend or split, so every fetched day is replaced.
            prices=prices.reindex(prices.index.union(series.index)).sort_index()
            prices.loc[series.index,ticker]=series.to_numpy()
            updated[ticker]=len(series)
        prices,backfilled=self._backfill(prices,coin_ids,warnings)
        if backfilled:
            warnings.append('Backfilled older daily prices (HL candles, DefiLlama): '+', '.join(
                f'{ticker} +{count:,} days' for ticker,count in backfilled.items())+'.')
        if any(group==TRADFI for ticker,group in self.asset_group.items() if ticker in self.assets) \
                and ('SPY' not in prices or not pd.to_numeric(prices['SPY'],errors='coerce').notna().any()):
            warnings.append('No SPY prices (IB unreachable and none saved): TradFi assets are analysed without '
                            'their SPY benchmark, on the crypto calendar.')
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
        # Each asset needs enough returns alongside its own benchmark on its own calendar. Whether
        # it has enough history for the Kelly and factor sections is decided in calculate_returns().
        retained_assets=[]
        skipped_assets=[]
        for ticker in self.assets:
            group=self.asset_group.get(ticker,CRYPTO)
            benchmark=self._benchmark_for(group,prices)
            own=self._group_frame(prices,[ticker],group)
            paired=own[[ticker,benchmark]].dropna() if benchmark in own else own[[ticker]].dropna()
            if len(paired)<MIN_COMMON_RETURNS:
                skipped_assets.append(f'{ticker} ({len(paired)} returns with {benchmark})')
            else:
                retained_assets.append(ticker)
        self.assets=retained_assets
        self.all_assets=list(retained_assets)
        if skipped_assets:
            warnings.append(
                f'Skipping asset(s) without at least {MIN_COMMON_RETURNS} common returns: '
                f'{", ".join(skipped_assets)}.'
            )
        prices=prices.sort_index()
        prices.index.name='date'  # merges above can drop the name, which the saved file needs
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
        for group in (CRYPTO, TRADFI):
            members=[f'{ticker} ({self.asset_source.get(ticker,"")})' for ticker in self.assets
                     if self.asset_group.get(ticker,CRYPTO)==group]
            if members:
                message+=f' {group} vs {BENCHMARKS[group]}: {", ".join(members)}.'
        message+=(f' Prices reach back up to {self.history_years} years (CoinGecko covers the latest 365 days and all '
                  'market-cap and volume data; HL candles and DefiLlama extend prices further).')
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
        if not self.all_assets:
            self.all_assets=list(self.assets)
        if not self.all_assets:
            raise RuntimeError('Run refresh_market_data() first to discover wallet assets.')
        prices=pd.read_csv(price_file,parse_dates=['date']).set_index('date').sort_index()
        prices.index=pd.to_datetime(prices.index,utc=True).normalize()
        required=list(dict.fromkeys(self.all_assets+[self.benchmark]))
        if self._has_tradfi(self.all_assets) and 'SPY' in prices.columns:
            required.append('SPY')
        if DIVIDEND_ETF in prices.columns:
            required.append(DIVIDEND_ETF)  # reference series for the dividend style factor
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
    def history_days(self):
        """Calendar days from each asset's first saved price to its last."""
        spans={}
        for ticker in self.all_assets:
            closes=pd.to_numeric(self.prices[ticker],errors='coerce').dropna() if ticker in self.prices else pd.Series(dtype=float)
            spans[ticker]=(closes.index.max()-closes.index.min()).days if len(closes) else 0
        return pd.Series(spans,name='History (days)')

    def calculate_returns(self):
        # Daily returns stay available for the crypto factors (BTC momentum, volatility, spreads).
        self.returns=self.prices.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan)
        # Every asset stays in the benchmark table and heatmaps; only assets with enough history
        # enter the Kelly, performance and factor sections, so a new listing cannot shrink them.
        history=self.history_days()
        self.short_history_assets=history[history<self.min_history_days]
        self.assets=[ticker for ticker in self.all_assets if history.get(ticker,0)>=self.min_history_days]
        if not self.assets:
            raise ValueError(f'No asset has {self.min_history_days} days of price history; lower min_history_days.')
        if len(self.short_history_assets):
            print(f'Left out of the Kelly, performance and factor sections (under {self.min_history_days} days of '
                  'history; still in the benchmark table and heatmaps): '
                  + ', '.join(f'{ticker} ({days} days)' for ticker,days in self.short_history_assets.items()) + '.')
        self.group_returns,self.universe_returns=self._return_frames(self.prices,self.all_assets)
        _,combined=self._return_frames(self.prices,self.assets)
        required=[column for column in self._required_columns(self.assets,self.prices) if column in combined]
        common=combined.dropna(subset=required)
        if len(common)<MIN_COMMON_RETURNS:
            raise ValueError(f'Only {len(common)} common returns; need at least {MIN_COMMON_RETURNS}.')
        self.uses_trading_calendar=self._uses_trading_calendar(self.assets,self.prices)
        self.trading_days_per_year=PERIODS_PER_YEAR[TRADFI] if self.uses_trading_calendar else PERIODS_PER_YEAR[CRYPTO]
        self.common_returns=common
        self.correlation=common[self.assets].corr(method='pearson')
        calendar='US trading days' if self.uses_trading_calendar else 'calendar days'
        print(f'Combined book: {len(common)} {calendar} from {common.index.min():%Y-%m-%d} to '
              f'{common.index.max():%Y-%m-%d}, annualized over {self.trading_days_per_year} days.')


# ----------------------------------------------------------------------------
# Correlation heatmap
# ----------------------------------------------------------------------------
#
# Dashboard call: `analysis.plot_correlations()`. Display the correlation matrix as an
# interactive Plotly heatmap.


class _CorrelationPlotMixin(_TableStyleMixin):
    def _heatmap(self, corr, title):
        rows, columns = len(corr.index), len(corr.columns)
        fig=go.Figure(go.Heatmap(
            z=corr.to_numpy(),x=corr.columns.tolist(),y=corr.index.tolist(),
            zmin=-1,zmax=1,colorscale='RdBu',reversescale=True,
            text=np.round(corr.to_numpy(),2),texttemplate='%{text:.2f}',
            hovertemplate='%{y} vs %{x}: %{z:.3f}<extra></extra>',
            showscale=False,
        ))
        fig.update_layout(title=title,template='plotly_white',
                          width=min(900,max(350,70*columns+250)),height=min(800,max(300,60*rows+150)))
        fig.update_yaxes(autorange='reversed')
        fig.show()

    def plot_correlations(self):
        members={group:[ticker for ticker in self.all_assets if self.asset_group.get(ticker,CRYPTO)==group]
                 for group in (CRYPTO,TRADFI)}
        for group,tickers in members.items():
            if len(tickers)<2:
                print(f'{group} correlations need at least two assets; held: {", ".join(tickers) or "none"}.')
                continue
            # Each pair uses its own shared dates on the group's calendar, so a recently listed
            # asset never shortens the other pairs.
            frame=self.group_returns[group][tickers]
            calendar='trading days' if group==TRADFI and self._trading_calendar(self.prices) is not None else 'days'
            self._heatmap(frame.corr(method='pearson',min_periods=MIN_COMMON_RETURNS),
                          f'{group} return correlations (pairwise over up to {frame.notna().sum().max()} {calendar}; '
                          f'benchmark {self._benchmark_for(group)})')
        if members[CRYPTO] and members[TRADFI]:
            frame=self.universe_returns[members[CRYPTO]+members[TRADFI]]
            corr=frame.corr(method='pearson',min_periods=MIN_COMMON_RETURNS).loc[members[CRYPTO],members[TRADFI]]
            self._heatmap(corr,'Crypto vs TradFi return correlations (pairwise, US trading days)')
            print('Cross-asset correlations sample 24/7 crypto prices at each US trading day\'s close, '
                  'so weekend crypto moves count in the next trading day.')

    def show_benchmark_exposures(self):
        """Each asset against its group's benchmark: correlation, beta with a 90% range, Sharpe."""
        rows=[]
        for ticker in self.all_assets:
            group=self.asset_group.get(ticker,CRYPTO)
            benchmark=self._benchmark_for(group)
            frame=self.group_returns[group]
            pair=frame[[ticker,benchmark]].dropna()
            periods=PERIODS_PER_YEAR[group] if benchmark==BENCHMARKS[group] else PERIODS_PER_YEAR[CRYPTO]
            stats=bootstrap_btc_regression(pair[ticker],pair[benchmark],periods_per_year=periods,samples=2000)
            other='SPY' if benchmark=='BTC' else 'BTC'
            other_corr=(self.universe_returns[[ticker,other]].dropna().corr().iloc[0,1]
                        if other in self.universe_returns and ticker in self.universe_returns else np.nan)
            rows.append({
                'Asset':ticker,'Group':group,'Source':self.asset_source.get(ticker,''),'Benchmark':benchmark,
                'Returns':len(pair),'Correlation':pair.corr().iloc[0,1],'Beta':stats['beta'],
                'Beta 90% range':(f"{stats['beta_low']:.2f} to {stats['beta_high']:.2f}"
                                  if np.isfinite(stats.get('beta_low',np.nan)) else 'n/a'),
                'Alpha (/yr)':stats['alpha']/100,'Volatility (/yr)':pair[ticker].std()*np.sqrt(periods),
                'Sharpe':stats['sharpe'],f'Corr to other benchmark':other_corr,
                'In Kelly':'yes' if ticker in self.assets else f'no ({self.short_history_assets.get(ticker,0)} days)',
            })
        table=pd.DataFrame(rows).sort_values(['Group','Beta'],ascending=[True,False]).set_index('Asset')
        print('Crypto assets are measured against BTC over every day (365-day annualization); TradFi assets '
              'against SPY over US trading days (252-day). Beta ranges come from 2,000 bootstrap resamples. '
              '"Corr to other benchmark" is crypto vs SPY or TradFi vs BTC, on US trading days.')
        self.show_table(table,{
            'Correlation':'{:.2f}','Beta':'{:.2f}','Alpha (/yr)':'{:.1%}','Volatility (/yr)':'{:.1%}',
            'Sharpe':'{:.2f}','Corr to other benchmark':'{:.2f}','Returns':'{:.0f}',
        })
        self.benchmark_exposures=table


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
    def _show_sleeves(self, portfolio_returns):
        """Split the Kelly book into its Crypto and TradFi sleeves, each against its benchmark."""
        weights=self.full_kelly.reindex(self.assets)
        rows=[]
        for group in (CRYPTO,TRADFI):
            members=[ticker for ticker in self.assets if self.asset_group.get(ticker,CRYPTO)==group]
            if not members:
                continue
            sleeve=self.common_returns[members]@weights[members]
            benchmark=self._benchmark_for(group)
            bench=self.common_returns[benchmark] if benchmark in self.common_returns else pd.Series(dtype=float)
            rows.append({
                'Sleeve':group,'Assets':len(members),'Net weight':weights[members].sum(),
                'Gross weight':weights[members].abs().sum(),'Benchmark':benchmark,
                'Beta to benchmark':sleeve.cov(bench)/bench.var(ddof=1) if len(bench) else np.nan,
                'Correlation to benchmark':sleeve.corr(bench) if len(bench) else np.nan,
                # Share of the book's variance this sleeve carries (the shares sum to 100% with cash).
                'Share of book variance':sleeve.cov(portfolio_returns)/portfolio_returns.var(ddof=1),
            })
        if len(rows)>1:
            print('Kelly sleeves: each group\'s part of the optimized book, measured against its own benchmark.')
            self.show_table(pd.DataFrame(rows).set_index('Sleeve'),{
                'Net weight':'{:.1%}','Gross weight':'{:.1%}','Beta to benchmark':'{:.2f}',
                'Correlation to benchmark':'{:.2f}','Share of book variance':'{:.1%}',
            })

    def analyze_performance(self):
        benchmark='BTC'
        if benchmark not in self.common_returns.columns:
            raise ValueError(f'Benchmark {benchmark} is not in the return data.')
        has_spy='SPY' in self.common_returns.columns and self.uses_trading_calendar
        periods=self.trading_days_per_year
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
        beta_spy=np.nan
        if has_spy:
            spy_returns=self.common_returns['SPY']
            beta_spy=portfolio_returns.cov(spy_returns)/spy_returns.var(ddof=1)
            # With both benchmarks, alpha is what remains after BTC and SPY together.
            design=np.column_stack([np.ones(len(excess)),benchmark_excess,spy_returns-self.risk_free_daily])
            coefficients,*_=np.linalg.lstsq(design,excess.to_numpy(),rcond=None)
            alpha=coefficients[0]*self.trading_days_per_year
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
        if has_spy:
            spy_value=pd.concat([
                pd.Series([initial_value],index=pd.DatetimeIndex([initial_date])),
                initial_value*(1+self.common_returns['SPY']).cumprod(),
            ])
            fig.add_trace(go.Scatter(x=spy_value.index,y=spy_value,name='SPY',line={'color':'#16a34a','width':2}))
        fig.update_layout(title='Since-inception value: constrained long/short full-Kelly vs. BTC'
                          +(' and SPY' if has_spy else ''),
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
            f'Sharpe (excess, sqrt {periods})':risk_metrics['Sharpe (annualized)'],
            'Sharpe (excess, sqrt 252)':risk_metrics['Sharpe (excess, 252d)'],
            'Sharpe (raw, sqrt 252)':risk_metrics['Sharpe (raw, 252d)'],
            f'Sortino ({periods}d)':risk_metrics['Sortino (annualized)'],
            'Beta (BTC)':beta,'Beta (SPY)':beta_spy,'Alpha':alpha,
            f'Arithmetic Return ({periods}d, linear)':risk_metrics['Arithmetic Return (annualized)'],
            'Hist Vol (daily)':risk_metrics['Hist Vol (daily)'],
            f'Hist Vol (annualized, {periods}d)':risk_metrics['Hist Vol (annualized)'],
            'Historical 1D VaR 95%':risk_metrics['Historical VaR 95%'],
            'Historical VaR 95% ($ / $100)':risk_metrics['Historical VaR 95%']*100,
            'Historical 1D VaR 99%':risk_metrics['Historical VaR 99%'],
            'Historical VaR 99% ($ / $100)':risk_metrics['Historical VaR 99%']*100,
            'Bootstrap 1D VaR 95%':risk_metrics['Bootstrap VaR 95%'],
            'Bootstrap VaR 95% ($ / $100)':risk_metrics['Bootstrap VaR 95%']*100,
            'Bootstrap VaR 95% (90% CI low)':risk_metrics['Bootstrap VaR 95% Low'],
            'Bootstrap VaR 95% (90% CI high)':risk_metrics['Bootstrap VaR 95% High'],
            'Bootstrap 1D VaR 99%':risk_metrics['Bootstrap VaR 99%'],
            'Bootstrap VaR 99% ($ / $100)':risk_metrics['Bootstrap VaR 99%']*100,
            'Bootstrap VaR 99% (90% CI low)':risk_metrics['Bootstrap VaR 99% Low'],
            'Bootstrap VaR 99% (90% CI high)':risk_metrics['Bootstrap VaR 99% High'],
            'Daily Return Observations':risk_metrics['Daily Return Observations'],
            'Historical VaR 99% Tail Observations':risk_metrics['Historical VaR 99% Tail Observations'],
            'Risk-Free Rate':risk_free_annual,'Max Drawdown':max_drawdown,
        }],index=['In-sample: weights fit and evaluated on same window'])
        # At 252-day annualization the main Sharpe already is the sqrt-252 one (same column).
        if not has_spy:
            metrics=metrics.drop(columns=['Beta (SPY)'])
        self.show_table(metrics,{
            f'Sharpe (excess, sqrt {periods})':'{:.3f}',
            'Sharpe (excess, sqrt 252)':'{:.3f}',
            'Sharpe (raw, sqrt 252)':'{:.3f}',
            f'Sortino ({periods}d)':'{:.3f}',
            'Beta (BTC)':'{:.3f}','Beta (SPY)':'{:.3f}','Alpha':'{:.2%}',
            f'Arithmetic Return ({periods}d, linear)':'{:.2%}',
            'Hist Vol (daily)':'{:.2%}',f'Hist Vol (annualized, {periods}d)':'{:.2%}',
            'Historical 1D VaR 95%':'{:.2%}',
            'Historical VaR 95% ($ / $100)':'${:,.2f}',
            'Historical 1D VaR 99%':'{:.2%}',
            'Historical VaR 99% ($ / $100)':'${:,.2f}',
            'Bootstrap 1D VaR 95%':'{:.2%}',
            'Bootstrap VaR 95% ($ / $100)':'${:,.2f}',
            'Bootstrap VaR 95% (90% CI low)':'{:.2%}',
            'Bootstrap VaR 95% (90% CI high)':'{:.2%}',
            'Bootstrap 1D VaR 99%':'{:.2%}',
            'Bootstrap VaR 99% ($ / $100)':'${:,.2f}',
            'Bootstrap VaR 99% (90% CI low)':'{:.2%}',
            'Bootstrap VaR 99% (90% CI high)':'{:.2%}',
            'Risk-Free Rate':'{:.2%}',
            'Max Drawdown':'{:.2%}',
        })
        calendar_text=('US trading days with 252-day annualization (crypto prices are sampled at each '
                       'trading day\'s close)' if self.uses_trading_calendar else 'daily observations with 365-day '
                       'crypto annualization')
        print(f"Risk estimates use {risk_metrics['Daily Return Observations']} {calendar_text}. "
              + ("Alpha is net of both BTC and SPY exposure. " if has_spy else "")
              + f"Historical 99% VaR has only "
              f"{risk_metrics['Historical VaR 99% Tail Observations']} observed tail "
              "observation(s); treat it as a weak tail estimate. Bootstrap VaR resamples "
              "observed daily returns, so it cannot exceed the largest loss in the sample; "
              "its range shows sampling uncertainty, not a worst case. Arithmetic return is "
              f"the daily mean multiplied by {periods}, not a forecast or compounded return. "
              "These are in-sample results: Kelly weights were optimized and evaluated "
              "on the same return window; use a walk-forward backtest for out-of-sample evidence.")
        self.portfolio_daily_returns=portfolio_returns
        self.portfolio_value=portfolio_value
        self.btc_value=btc_value
        self.performance_metrics=metrics
        self.beta=beta
        self.beta_spy=beta_spy
        self._show_sleeves(portfolio_returns)
        self.max_drawdown=max_drawdown
        self.portfolio_period=(self.common_returns.index.min(),self.common_returns.index.max())


# ----------------------------------------------------------------------------
# Factor model
# ----------------------------------------------------------------------------
#
# Dashboard calls: `analysis.fit_factor_model()` and `analysis.test_alpha_out_of_sample()`.
# Fit excess returns against BTC, BTC momentum, BTC realized volatility, a HL funding
# spread, and available CoinGecko cross-sectional factors; HL-only assets remain in the
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


CORE_FACTORS=['BTC market excess','SPY market excess','BTC momentum','BTC realized vol','Funding (low - high)']
STATE_FACTORS=('BTC realized vol',)


class _FactorModelMixin(_TableStyleMixin):
    def _funding_signal(self, assets=None, start=None):
        """Return each perp's lagged trailing 7-day average daily funding."""
        start=(self.common_returns.index.min() if start is None else start)-pd.Timedelta(days=10)
        daily={}
        for ticker in (self.assets if assets is None else assets):
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
        # Crypto factors are built from every-day data, then moved onto the book's calendar.
        daily_returns=_normalize_daily_index(self.returns)
        crypto_assets=[asset for asset in self.assets if self.asset_group.get(asset,CRYPTO)==CRYPTO]
        if CRYPTO in self.group_returns:
            daily_index=_normalize_daily_index(self.group_returns[CRYPTO]).dropna(subset=['BTC']).index
        else:
            daily_index=daily_returns['BTC'].dropna().index
        rf_daily=(1+self.risk_free_annual)**(1/PERIODS_PER_YEAR[CRYPTO])-1
        market_caps=pd.DataFrame(index=daily_index)
        volumes=pd.DataFrame(index=daily_index)
        market_assets=[]
        for asset in crypto_assets:
            cap_col,volume_col=f'{asset}_market_cap',f'{asset}_volume'
            if cap_col not in market or not market[cap_col].notna().any():
                continue
            market_caps[asset]=pd.to_numeric(market[cap_col],errors='coerce').reindex(daily_index).ffill(limit=2)
            if volume_col in market:
                volumes[asset]=pd.to_numeric(market[volume_col],errors='coerce').reindex(daily_index).ffill(limit=2)
            else:
                volumes[asset]=np.nan
            market_assets.append(asset)
        alt_assets=[asset for asset in market_assets if asset!='BTC']
        asset_returns=daily_returns.reindex(daily_index)
        factors=pd.DataFrame(index=daily_index)
        factors['BTC market excess']=asset_returns['BTC']-rf_daily
        btc_history=daily_returns['BTC'].dropna()
        factors['BTC momentum']=btc_momentum_factor(btc_history,rf_daily).reindex(daily_index)
        factors['BTC realized vol']=btc_realized_vol(
            btc_history,periods_per_year=PERIODS_PER_YEAR[CRYPTO]
        ).reindex(daily_index)
        funding_signal=self._funding_signal()
        funding_assets=[asset for asset in funding_signal.columns if asset in asset_returns]
        # Price-only spread, matching the portfolio returns, which also exclude funding payments.
        if len(funding_assets)>=4:
            factors['Funding (low - high)']=_long_short_factor(
                funding_signal.reindex(daily_index),asset_returns[funding_assets],long_low=True,
            )
        if alt_assets:
            lagged_caps=market_caps[alt_assets].shift(1)
            alt_weights=lagged_caps.div(lagged_caps.sum(axis=1),axis=0)
            alt_return=(asset_returns[alt_assets]*alt_weights).sum(axis=1,min_count=len(alt_assets))
            factors['Alt market - BTC']=alt_return-asset_returns['BTC']
        factor_notes=[]
        if len(funding_assets)<4:
            factor_notes.append(f'Funding factor omitted: {len(funding_assets)} perps have funding history; at least 4 are needed.')
        missing_market_assets=[asset for asset in crypto_assets if asset not in market_assets]
        if missing_market_assets:
            factor_notes.append(
                'CoinGecko cross-sectional factors exclude assets without market-cap history: '
                f'{", ".join(missing_market_assets)}.'
            )
        if not alt_assets:
            factor_notes.append('Alt market factor omitted: no selected assets have CoinGecko market-cap history.')
        if len(alt_assets)>=4:
            momentum=self.prices[alt_assets].pct_change(30).shift(1)
            momentum.index=pd.to_datetime(momentum.index,utc=True).normalize()
            momentum=momentum.reindex(daily_index)
            turnover=volumes[alt_assets].div(market_caps[alt_assets].where(market_caps[alt_assets]>0))
            lagged_turnover=turnover.shift(1).rolling(30,min_periods=20).mean()
            factors['Small - large']=_long_short_factor(lagged_caps[alt_assets],asset_returns[alt_assets],long_low=True)
            factors['Momentum']=_long_short_factor(momentum,asset_returns[alt_assets])
            if volumes[alt_assets].notna().any().any():
                factors['High - low turnover']=_long_short_factor(lagged_turnover,asset_returns[alt_assets])
            else:
                factor_notes.append('Turnover factor omitted: CoinGecko volume history is unavailable.')
        else:
            factor_notes.append(
                f'Size, momentum, and turnover spreads omitted: {len(alt_assets)} CoinGecko-priced '
                'altcoins; at least 4 are needed.'
            )
        factors=_normalize_daily_index(factors)
        if self.uses_trading_calendar:
            # Return factors compound over each trading day's period; the volatility state is
            # read at the trading day; SPY joins as the TradFi market factor.
            calendar=_normalize_daily_index(self.common_returns).index
            converted=pd.DataFrame(index=calendar)
            for name in factors.columns:
                if name in STATE_FACTORS:
                    converted[name]=factors[name].reindex(calendar,method='ffill',
                                                          tolerance=pd.Timedelta(days=MAX_CALENDAR_GAP_DAYS))
                elif name=='BTC market excess':
                    converted[name]=compound_to_calendar(asset_returns['BTC'],calendar)-self.risk_free_daily
                else:
                    converted[name]=compound_to_calendar(factors[name],calendar)
            converted['SPY market excess']=_normalize_daily_index(self.common_returns)['SPY']-self.risk_free_daily
            factors=converted
            factor_notes.append('Crypto factors are compounded onto US trading days; SPY market excess is the '
                                'TradFi market factor. Size, momentum and turnover spreads use crypto assets only.')
        factor_note=' '.join(factor_notes) or None
        portfolio_excess=_normalize_daily_index(
            (self.portfolio_daily_returns-self.risk_free_daily).rename('Portfolio excess')
        )
        core_factors=[name for name in CORE_FACTORS if name in factors]
        # Every factor built, before any is left out of this model, for the style models to reuse.
        self.factor_returns_all=factors.copy()
        model_data=pd.concat([portfolio_excess,factors],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        core_data=pd.concat([portfolio_excess,factors[core_factors]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        # CoinGecko market data covers only the latest year, so its factors can cut a long price
        # history short; drop them here when they would halve the sample (Fama-MacBeth still tests them).
        if len(factors.columns)>len(core_factors) and len(model_data)<max(60,0.5*len(core_data)):
            full_factor_observations=len(model_data)
            factors=factors[core_factors]
            model_data=core_data
            factor_notes.append(
                f'CoinGecko-based factors (alt market, size, momentum, turnover) cover only {full_factor_observations} '
                f'days against {len(core_data)} for the core factors; omitting them here to keep the full history '
                '(estimate_factor_premia() still tests them).'
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
        # Is each factor's own premium distinguishable from zero? (HAC t-stat of its mean.)
        premium_t=pd.Series({name:(np.nan if name in state_factors else newey_west_mean(model_data[name])[1])
                             for name in names})
        summary=pd.DataFrame({
            'Beta':fit_result.betas,'HAC t-stat':fit_result.t_stats,
            'Factor premium (%/yr)':annual_premiums,'Premium t-stat':premium_t,
            'Portfolio contribution (%/yr)':annual_contributions,
        },index=names)
        summary.loc['Alpha']=[np.nan,fit_result.alpha_t_stat,np.nan,np.nan,annual_alpha]
        fit=pd.Series({
            'Observations':fit_result.observations,'R-squared':fit_result.r_squared,
            'Adjusted R-squared':fit_result.adjusted_r_squared,
            'Annual alpha':annual_alpha,
            'BTC beta (factor model)':fit_result.betas['BTC market excess'],'BTC beta (single factor)':self.beta,
            **({'SPY beta (factor model)':fit_result.betas['SPY market excess'],'SPY beta (single factor)':self.beta_spy}
               if 'SPY market excess' in fit_result.betas else {}),
        },name='Factor model fit')
        print(f'Factor model: R²={fit_result.r_squared:.1%} | adjusted R²={fit_result.adjusted_r_squared:.1%} | '
              f'annual alpha={annual_alpha:.2%} | n={fit_result.observations} days')
        print(f'Single-factor BTC beta={self.beta:.3f} | factor-model BTC beta={summary.loc["BTC market excess","Beta"]:.3f}'
              + (f' | single-factor SPY beta={self.beta_spy:.3f} | factor-model SPY beta='
                 f'{summary.loc["SPY market excess","Beta"]:.3f}' if 'SPY market excess' in summary.index else ''))
        print('BTC realized vol beta is the daily return per one standard deviation of lagged 30-day volatility; '
              'alpha is at average volatility. Funding (low - high) is long low-funding and short high-funding '
              'held perps, by lagged 7-day average funding. Returns exclude funding payments and trading costs.')
        if factor_note:
            print(factor_note)
        self.show_table(summary,{
            'Beta':'{:.3f}','HAC t-stat':'{:.2f}',
            'Factor premium (%/yr)':'{:.2%}','Premium t-stat':'{:.2f}',
            'Portfolio contribution (%/yr)':'{:.2%}',
        })
        self.show_table(fit.to_frame())
        self.factor_returns=factors
        self.factor_model_summary=summary
        self.factor_model_fit=fit
        self.factor_model_dates=model_data.index

    def _style_factor_returns(self, calendar):
        """Equity style factors (Ken French, plus an ETF dividend proxy) and the crypto block.

        Equity styles apply to a book on US trading days; a crypto-only book uses the crypto
        block alone, because US equity factors have no weekend returns.
        """
        factors=pd.DataFrame(index=calendar)
        notes=[]
        risk_free=None
        if self.uses_trading_calendar:
            cache=self.data_dir/'french'
            us=load_french('us5',cache)
            momentum=load_french('momentum',cache)
            reversal=load_french('short_reversal',cache)
            industries=load_french('industries',cache)
            developed=load_french('developed_ex_us',cache)
            on=lambda series: series.reindex(calendar)
            factors['Market (US)']=on(us['Mkt-RF'])
            factors['Size (SMB)']=on(us['SMB'])
            factors['Value (HML)']=on(us['HML'])
            factors['Quality (RMW)']=on(us['RMW'])
            factors['Investment (CMA)']=on(us['CMA'])
            factors['Momentum (12-1m)']=on(momentum.iloc[:,0])
            factors['Short-term reversal (1m)']=on(reversal.iloc[:,0])
            factors['Domicile (developed ex-US - US)']=on(developed['Mkt-RF']-us['Mkt-RF'])
            factors['Sector: tech (BusEq - market)']=on(industries['BusEq']-(us['Mkt-RF']+us['RF']))
            risk_free=on(us['RF'])
            if DIVIDEND_ETF in self.prices and 'SPY' in self.prices:
                pair=calendar_prices(self.prices[[DIVIDEND_ETF,'SPY']].apply(pd.to_numeric,errors='coerce'),calendar)
                pair=pair.pct_change(fill_method=None)
                factors[f'Dividend ({DIVIDEND_ETF} - SPY)']=pair[DIVIDEND_ETF]-pair['SPY']
            else:
                notes.append(f'Dividend factor omitted: no {DIVIDEND_ETF} prices (IB was unreachable on refresh).')
            notes.append(f'Ken French data runs to {us.index.max():%Y-%m-%d} (updated monthly), so later days '
                         'are not yet in the regression.')
        else:
            notes.append('Crypto-only book on every day: US equity style factors are omitted.')
        crypto=getattr(self,'factor_returns_all',None)
        if crypto is None:
            crypto=getattr(self,'factor_returns',None)
        if crypto is None:
            raise RuntimeError('Run fit_factor_model() first: the style model reuses its crypto factors.')
        crypto=_normalize_daily_index(crypto)
        for name in ('BTC market excess','Alt market - BTC','BTC momentum','Funding (low - high)'):
            if name in crypto:
                factors[name]=crypto[name].reindex(calendar)
        return factors,risk_free,notes

    def fit_style_model(self, returns=None, label='Kelly model portfolio', factors=None):
        """Industry-standard multi-factor regression of a book's excess return.

        excess return = alpha + sum(beta_k x factor_k) + residual. Alpha is the intercept (the
        average return no factor explains); the specific return each period is alpha plus the
        residual, which is what a team is paid on. Betas use HAC t-stats; a factor that
        duplicates another in the sample is dropped.

        ``returns`` is an actual book's simple returns indexed by period end (daily or weekly);
        the daily factors are compounded into the same periods. Without it, the Kelly model
        portfolio is used (in-sample: its weights were fit on the same days).

        ``factors`` names an exact factor set (for example ``['Alt market - BTC']``); it is used
        as given, without the automatic coverage and core-factor trimming.
        """
        calendar=_normalize_daily_index(self.common_returns).index
        chosen=list(factors) if factors else None
        factors,risk_free,notes=self._style_factor_returns(calendar)
        if chosen:
            missing=[name for name in chosen if name not in factors]
            if missing:
                raise ValueError(f'Unknown or unavailable factor(s): {missing}; available: {list(factors.columns)}.')
            factors=factors[chosen]
            notes=[note for note in notes if 'Ken French' not in note]
        # French's risk-free rate where published, the desk's own after its last date.
        rf=(risk_free.fillna(self.risk_free_daily) if risk_free is not None
            else pd.Series(self.risk_free_daily,index=calendar))
        if returns is None:
            portfolio=_normalize_daily_index(self.portfolio_daily_returns).reindex(calendar)
            excess=(portfolio-rf).rename('Book excess')
            periods=self.trading_days_per_year
            unit='day'
        else:
            ends=pd.DatetimeIndex(pd.to_datetime(returns.index,utc=True).normalize()).unique().sort_values()
            returns=pd.Series(returns.to_numpy(),index=pd.to_datetime(returns.index,utc=True).normalize())
            returns=returns[~returns.index.duplicated(keep='last')].reindex(ends)
            # Median spacing in days (pandas 3 may store timestamps in seconds or microseconds).
            spacing=float(pd.Series(ends).diff().dt.total_seconds().median()/86_400) if len(ends)>1 else 1.0
            periods=int(round(365/spacing)) if spacing>1.5 else self.trading_days_per_year
            unit='week' if 5<=spacing<=9 else 'period' if spacing>1.5 else 'day'
            # Compound each daily factor into the book's periods; a period after a factor's last
            # published day is left empty rather than half-compounded.
            compounded=pd.DataFrame(index=ends)
            for name in factors.columns:
                daily=factors[name].dropna()
                series=compound_to_calendar(daily,ends)
                compounded[name]=series.where(ends<=daily.index.max()) if len(daily) else np.nan
            factors=compounded
            rf_daily=rf.dropna()
            rf=compound_to_calendar(rf_daily,ends).where(ends<=rf_daily.index.max())
            excess=(returns-rf.fillna(self.risk_free_daily*spacing)).rename('Book excess')
        excess=excess.replace([np.inf,-np.inf],np.nan)
        usable=excess.notna()
        coverage=factors[usable].notna().sum()
        # A factor covering under three quarters of the book's periods (the CoinGecko-based ones
        # cover one year) is left out rather than cutting the sample to its window.
        short=[] if chosen else [name for name in factors.columns if coverage[name]<0.75*coverage.max()]
        factors=factors.drop(columns=short)
        data=pd.concat([excess,factors],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        names=independent_factors(data[list(factors.columns)])
        dropped=[name for name in factors.columns if name not in names]
        core=['Market (US)','Size (SMB)','Value (HML)','Momentum (12-1m)','BTC market excess',
              'Alt market - BTC','Funding (low - high)']
        if not chosen and len(data)<10*len(names):
            # Under ten observations per factor: keep the core styles so the model is not fit to noise.
            names=[name for name in names if name in core]
            notes.append(f'Only {len(data)} {unit}s, so the model keeps the core factors: {", ".join(names)}.')
        if len(data)<len(names)+10:
            raise ValueError(f'Only {len(data)} {unit}s with every factor; too few for {len(names)} factors.')
        if short:
            notes.append('Left out for covering under three quarters of the sample: '+', '.join(short)+'.')
        fit=fit_factor_regression(data['Book excess'],data[names])
        explained=data[names]@fit.betas
        specific=data['Book excess']-explained  # alpha + residual, each period
        residual=specific-fit.alpha
        tracking=residual.std()*np.sqrt(periods)
        table=pd.DataFrame({
            'Beta':fit.betas,'HAC t-stat':fit.t_stats,
            'Factor return (/yr)':data[names].mean()*periods,
            'Contribution (/yr)':fit.betas*data[names].mean()*periods,
        })
        table.loc['Alpha (intercept)']=[np.nan,fit.alpha_t_stat,np.nan,fit.alpha*periods]
        attribution=pd.Series({
            **{name:fit.betas[name]*data[name].sum() for name in names},
            'Specific return (alpha + residual)':specific.sum(),
        },name='Return over sample')
        attribution.loc['Total excess return']=data['Book excess'].sum()
        summary=pd.Series({
            'Book':label,'Periods':f'{fit.observations} {unit}s','From':f'{data.index.min():%Y-%m-%d}','To':f'{data.index.max():%Y-%m-%d}',
            'R-squared':fit.r_squared,'Adjusted R-squared':fit.adjusted_r_squared,
            'Alpha (/yr)':fit.alpha*periods,'Alpha t-stat':fit.alpha_t_stat,
            'Residual volatility (/yr)':tracking,
            'Information ratio':fit.alpha*periods/tracking if tracking>0 else np.nan,
        },name='Style model')
        print(f'Style-factor model, {label}: {len(names)} factors, {fit.observations} {unit}s '
              f'({data.index.min():%Y-%m-%d} to {data.index.max():%Y-%m-%d}), R²={fit.r_squared:.1%}. '
              f'Alpha={fit.alpha*periods:.1%}/yr (t={fit.alpha_t_stat:.2f}); information ratio '
              f'{summary["Information ratio"]:.2f}.')
        print('Each beta is the book\'s exposure to a factor; the contribution is beta x the factor\'s average '
              'return. Alpha is the intercept: the part of the average return no factor explains. Specific '
              f'return is alpha plus each {unit}\'s residual; it is the return attributable to selection.')
        self.show_table(table,{'Beta':'{:.3f}','HAC t-stat':'{:.2f}','Factor return (/yr)':'{:.1%}',
                               'Contribution (/yr)':'{:.1%}'})
        self.show_table(summary.to_frame())
        print(f'Return attribution over the sample (sum of {dict(day="daily",week="weekly").get(unit,"periodic")} excess returns):')
        self.show_table(attribution.to_frame(),{'Return over sample':'{:.1%}'})
        fig=go.Figure()
        for series,line_name,color in ((data['Book excess'].cumsum(),'Book excess return','#2563eb'),
                                       (explained.cumsum(),'Explained by factors','#9ca3af'),
                                       (specific.cumsum(),'Specific (alpha + residual)','#16a34a')):
            fig.add_trace(go.Scatter(x=series.index,y=series,name=line_name,line={'color':color,'width':2}))
        fig.update_layout(title=f'Cumulative return attribution, {label}: factors vs specific',yaxis_tickformat='.0%',
                          template='plotly_white',hovermode='x unified')
        fig.show()
        if dropped:
            notes.append('Dropped because another factor duplicates them in this sample: '+', '.join(dropped)+'.')
        for note in notes:
            print(note)
        print(f'With {fit.observations} {unit}s and {len(names)} factors, treat |t| below about 2 as noise; '
              'betas are estimated from the book\'s returns, not from holdings\' characteristics.'
              + (' This is the Kelly model portfolio, fit on these same days, so its alpha is in-sample.'
                 if returns is None else ''))
        result={'model':table,'summary':summary,'attribution':attribution,'specific':specific}
        self.style_results=getattr(self,'style_results',{})
        self.style_results[label]=result
        if returns is None:
            self.style_model,self.style_summary=table,summary
            self.style_attribution,self.style_specific_returns=attribution,specific
        return result

    def hl_book_returns(self, frequency='W', desk_data_dir=None):
        """Time-weighted returns of the actual HL book: every configured wallet combined.

        Built from the saved HL value and PnL history, so deposits, withdrawals and transfers
        between wallets are not counted as performance. Only returns between consecutive days
        (or weeks) are kept, so a gap in the saved history is never read as one period.
        """
        from desk.hl.trade_analytics import time_weighted_index

        folder=Path(desk_data_dir) if desk_data_dir else DESK_DATA_DIR
        histories=[]
        for wallet in self.wallet_addresses:
            path=folder/f'hl_portfolio_history_{wallet.lower()}.csv'
            if path.exists():
                history=pd.read_csv(path)
                history['ts']=pd.to_datetime(history['ts'],utc=True,format='mixed')
                histories.append(history)
        index=time_weighted_index(histories)
        if index.empty:
            return pd.Series(dtype=float,name='HL book')
        index.index=pd.to_datetime(index.index,utc=True)
        rule,step=('1D',pd.Timedelta(days=1)) if frequency=='D' else ('W-SUN',pd.Timedelta(days=7))
        closes=index.resample(rule).last().dropna()
        closes.index=closes.index.normalize()
        returns=closes.pct_change()
        return returns[closes.index.to_series().diff().eq(step)].dropna().rename('HL book')

    def _book_frequency(self, daily, frequency):
        if frequency=='auto':
            return 'D' if len(daily)>=MIN_DAILY_BOOK_RETURNS else 'W'
        return frequency

    def fit_style_model_hl_book(self, frequency='auto', desk_data_dir=None):
        """The style model on the actual HL book (all configured wallets), net of cash flows.

        Weekly until MIN_DAILY_BOOK_RETURNS consecutive daily returns are saved (HL keeps
        detailed history for about a month, so daily history builds from run to run).
        """
        daily=self.hl_book_returns('D',desk_data_dir)
        frequency=self._book_frequency(daily,frequency)
        returns=daily if frequency=='D' else self.hl_book_returns('W',desk_data_dir)
        print(f'HL book: {len(daily)} consecutive daily returns saved (daily attribution from '
              f'{MIN_DAILY_BOOK_RETURNS}); using {"daily" if frequency=="D" else "weekly"} returns: {len(returns)} '
              f'from {returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d}.' if len(returns) else
              'HL book: no saved history yet; run the HL account notebooks first.')
        if returns.empty:
            return None
        return self.fit_style_model(returns,label='HL book (actual)')

    def ib_book_returns(self, frequency='W', ib_dir=None, refresh=True):
        """Time-weighted returns of the actual IB account, from IB Flex NAV history.

        With a Flex token and query (see desk/venues/ib_flex.py) the daily NAV and deposits come
        from IB itself; without them, the account snapshots saved by the desk overview are used,
        which cannot separate deposits from performance and are flagged as such.
        """
        from desk.venues import ib_flex

        folder=Path(ib_dir) if ib_dir else ib_data_dir()
        if refresh:
            for account in accounts_for('ib',load_accounts()):
                try:
                    ib_flex.update_nav_history(account,folder)
                except ib_flex.FlexNotConfigured as exc:
                    print(f'IB Flex not set up ({exc}); using saved snapshots.')
                except Exception as exc:
                    print(f'IB Flex history unavailable ({type(exc).__name__}: {exc}); using saved files.')
        nav_files=sorted(folder.glob('ib_nav_history_*.csv'))
        if nav_files:
            returns=pd.concat([ib_flex.time_weighted_returns(pd.read_csv(path)) for path in nav_files],axis=1)
            daily=returns.sum(axis=1,min_count=1).dropna() if returns.shape[1]>1 else returns.iloc[:,0].dropna()
            self.ib_book_source='IB Flex NAV, net of deposits and withdrawals'
        else:
            frames=[]
            for path in sorted(folder.glob('ib_account_history_*.csv')):
                snapshot=pd.read_csv(path)
                snapshot['ts']=pd.to_datetime(snapshot['ts'],utc=True,format='mixed')
                frames.append(snapshot.set_index('ts')['equity_usd'])
            if not frames:
                return pd.Series(dtype=float,name='IB book')
            equity=pd.concat(frames,axis=1).sum(axis=1)
            daily=equity.resample('1D').last().dropna()
            daily.index=daily.index.normalize()
            daily=daily.pct_change()[daily.index.to_series().diff().eq(pd.Timedelta(days=1))].dropna()
            self.ib_book_source='saved net-liquidation snapshots (deposits are NOT removed)'
        daily=daily.rename('IB book')
        daily.index=pd.to_datetime(daily.index,utc=True)
        if frequency=='D':
            return daily
        # Weeks end on Sunday, like the HL book, and need most of their trading days.
        weekly=(1+daily).resample('W-SUN').prod(min_count=1)-1
        counts=daily.resample('W-SUN').count()
        weekly=weekly[counts>=4]
        weekly.index=weekly.index.normalize()
        return weekly.rename('IB book')

    def fit_style_model_ib_book(self, frequency='auto', ib_dir=None, refresh=True):
        """The style model on the actual IB account, as its history builds."""
        daily=self.ib_book_returns('D',ib_dir,refresh)
        frequency=self._book_frequency(daily,frequency)
        returns=daily if frequency=='D' else self.ib_book_returns('W',ib_dir,refresh=False)
        if returns.empty:
            print('IB book: no history yet. Set up IB Flex (desk/venues/ib_flex.py) for up to a year of daily '
                  'NAV at once, or let daily snapshots build from desk overview runs.')
            return None
        print(f'IB book ({self.ib_book_source}): {len(daily)} daily returns; using '
              f'{"daily" if frequency=="D" else "weekly"} returns: {len(returns)}.')
        if len(returns)<20:
            print('Too little IB history for a factor regression yet.')
            return None
        return self.fit_style_model(returns,label='IB book (actual)')

    def _crypto_characteristics(self, assets, dates):
        """Lagged characteristics for crypto assets on ``dates``: known the day before each return."""
        prices=_normalize_daily_index(self.prices)[assets].reindex(dates)
        daily=_normalize_daily_index(self.returns).reindex(dates)
        returns=daily[assets]
        btc=daily['BTC']
        characteristics={
            'Momentum (30d)':prices.pct_change(30,fill_method=None).shift(1),
            'Volatility (30d)':returns.rolling(30,min_periods=20).std().shift(1),
            'BTC beta (60d)':(returns.rolling(60,min_periods=40).cov(btc)
                              .div(btc.rolling(60,min_periods=40).var(),axis=0)).shift(1),
        }
        notes=[]
        market_file=self.data_dir/'market_data.csv'
        if market_file.exists():
            market=pd.read_csv(market_file,parse_dates=['date'])
            market['date']=pd.to_datetime(market['date'],utc=True).dt.normalize()
            market=market.set_index('date').sort_index()
            caps=pd.DataFrame({asset:pd.to_numeric(market[f'{asset}_market_cap'],errors='coerce')
                               for asset in assets if f'{asset}_market_cap' in market}).reindex(dates)
            volumes=pd.DataFrame({asset:pd.to_numeric(market[f'{asset}_volume'],errors='coerce')
                                  for asset in caps.columns if f'{asset}_volume' in market}).reindex(dates)
            if len(caps.columns):
                characteristics['Size (log market cap)']=np.log(caps.where(caps>0)).shift(1).reindex(columns=assets)
                # The same size characteristic over the full price history, estimated before CoinGecko's year.
                full_caps,estimated,excluded=estimate_market_caps(caps,prices[list(caps.columns)])
                characteristics['Size (estimated, full history)']=(
                    np.log(full_caps.where(full_caps>0)).shift(1).reindex(columns=assets))
                notes.append(
                    'Size (estimated, full history) uses actual CoinGecko market caps for the latest year and, '
                    'before that, price x the earliest supply CoinGecko shows'
                    + (f' for {", ".join(estimated)}' if estimated else '') + '.'
                    + (' Left at actual market caps only, because their circulating supply changed more than '
                       f'{SUPPLY_DRIFT_LIMIT:.0%} in the observed year: '
                       + ', '.join(f'{asset} ({drift:+.0%})' for asset,drift in excluded.items()) + '.'
                       if excluded else '')
                    + ' If both size rows agree, the estimate is not distorting the result.'
                )
            if len(volumes.columns):
                turnover=volumes.div(caps.where(caps>0))
                characteristics['Turnover (30d)']=turnover.rolling(30,min_periods=20).mean().shift(1).reindex(columns=assets)
            notes.append(f'Size and turnover cover the {len(caps.columns)} assets with CoinGecko market data.')
        perps=[asset for asset in assets if asset in self.perp_asset_dex]
        funding=self._funding_signal(perps,start=dates.min())
        if len(funding.columns):
            characteristics['Funding (7d)']=funding.reindex(dates).reindex(columns=assets)
            notes.append(f'Funding covers the {len(funding.columns)} perps with funding history.')
        return returns,characteristics,notes

    def estimate_factor_premia(self, min_assets=8):
        """Cross-sectional premia with t-stats that respect correlation across assets.

        Fama-MacBeth: each day, regress every crypto asset's return on its standardized lagged
        characteristic; the premium is the average daily slope and its t-stat comes from the
        time series of slopes (Newey-West), so a day when all assets move together moves only
        that day's slope. Portfolio sort: the daily long top-half / short bottom-half spread
        for the same characteristic, with a Newey-West t-stat. Uses every crypto asset,
        including short-history ones, over the dates each has data.
        """
        assets=[asset for asset in self.all_assets if self.asset_group.get(asset,CRYPTO)==CRYPTO]
        if len(assets)<min_assets:
            print(f'Fama-MacBeth needs at least {min_assets} crypto assets; {len(assets)} are held.')
            return
        dates=_normalize_daily_index(self.group_returns[CRYPTO]).dropna(subset=['BTC']).index
        returns,characteristics,notes=self._crypto_characteristics(assets,dates)
        periods=PERIODS_PER_YEAR[CRYPTO]
        rows=[]
        for name,signal in characteristics.items():
            fm,_=fama_macbeth(returns,{name:signal},min_assets=min_assets)
            spread_mean,spread_t,spread_days=newey_west_mean(sort_spread(signal,returns,min_assets=min_assets))
            rows.append({
                'Characteristic':name,'FM premium (/yr per 1 sd)':fm.loc[name,'Premium (daily)']*periods,
                'FM t-stat':fm.loc[name,'t-stat'],'FM days':fm.loc[name,'Days'],'Avg assets':fm.loc[name,'Avg assets'],
                'Sort spread (/yr)':spread_mean*periods,'Sort t-stat':spread_t,'Sort days':spread_days,
            })
        table=pd.DataFrame(rows).set_index('Characteristic')
        # Joint regression on the characteristics most assets have, so they control for each other.
        broad=[name for name in table.index if table.loc[name,'Avg assets']>=0.7*len(assets)]
        joint=None
        if len(broad)>=2:
            joint,_=fama_macbeth(returns,{name:characteristics[name] for name in broad},min_assets=min_assets)
            joint=pd.DataFrame({'Joint FM premium (/yr per 1 sd)':joint['Premium (daily)']*periods,
                                'Joint t-stat':joint['t-stat'],'Days':joint['Days'],'Avg assets':joint['Avg assets']})
        print(f'Crypto cross-section: {len(assets)} assets, daily returns, characteristics lagged one day. A premium '
              'is the extra yearly return of an asset one standard deviation higher in that characteristic '
              '(Fama-MacBeth) or of the top half over the bottom half (sort). |t| of 2 or more is the usual '
              'bar for a premium distinguishable from zero.')
        self.show_table(table,{
            'FM premium (/yr per 1 sd)':'{:.1%}','FM t-stat':'{:.2f}','FM days':'{:.0f}','Avg assets':'{:.1f}',
            'Sort spread (/yr)':'{:.1%}','Sort t-stat':'{:.2f}','Sort days':'{:.0f}',
        })
        if joint is not None:
            print('Joint Fama-MacBeth (each premium controls for the others): '+', '.join(broad)+'.')
            self.show_table(joint,{'Joint FM premium (/yr per 1 sd)':'{:.1%}','Joint t-stat':'{:.2f}',
                                   'Days':'{:.0f}','Avg assets':'{:.1f}'})
        for note in notes:
            print(note)
        print(f'With about {len(assets)} assets the cross-section is small, so even these t-stats are noisy; '
              'treat |t| near 2 as suggestive, not settled.')
        self.factor_premia=table
        self.factor_premia_joint=joint

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


PORTFOLIO_HIGHLIGHTS_STATE='portfolio_highlights_state.json'
SIGNIFICANT_T=2.0


class _HighlightsMixin(_TableStyleMixin):
    """One summary of every section that has run, plus what changed since the last run."""

    def _highlight_metrics(self):
        """Key numbers compared from run to run."""
        metrics={'Assets analysed':len(self.all_assets),'Assets in Kelly':len(self.assets)}
        performance=getattr(self,'performance_metrics',None)
        if performance is not None:
            row=performance.iloc[0]
            sharpe=[column for column in performance.columns if column.startswith('Sharpe (excess, sqrt')]
            if sharpe:
                metrics['Kelly Sharpe (in-sample)']=float(row[sharpe[0]])
            for name in ('Beta (BTC)','Beta (SPY)'):
                if name in performance:
                    metrics[f'Kelly {name.lower()}']=float(row[name])
        for label,result in getattr(self,'style_results',{}).items():
            if '(actual)' in label:
                metrics[f'{label}: alpha (/yr)']=float(result['summary']['Alpha (/yr)'])
                metrics[f'{label}: alpha t-stat']=float(result['summary']['Alpha t-stat'])
        return {name:value for name,value in metrics.items() if value==value}

    def highlights_panel(self):
        """Reserve the highlights block here (top of the notebook); show_highlights() fills it in."""
        self._highlights_panel=HighlightsPanel()

    def show_highlights(self):
        """What stands out, what needs attention, and what changed since the last run.

        Shown in the panel from highlights_panel() when the notebook created one, otherwise here."""
        well,attention=[],[]
        groups={group:[ticker for ticker in self.all_assets if self.asset_group.get(ticker,CRYPTO)==group]
                for group in (CRYPTO,TRADFI)}
        well.append(('Universe','; '.join(f'{len(tickers)} {group} vs {BENCHMARKS[group]}'
                                           for group,tickers in groups.items() if tickers)
                     +f'; {len(self.assets)} with {self.min_history_days}+ days enter Kelly and factor sections.'))
        if len(self.short_history_assets):
            attention.append(('Short history','Left out of Kelly and factor sections: '+', '.join(
                f'{ticker} ({days}d)' for ticker,days in self.short_history_assets.items())+'.'))
        exposures=getattr(self,'benchmark_exposures',None)
        if exposures is not None and len(exposures):
            for group in (CRYPTO,TRADFI):
                rows=exposures[exposures['Group']==group]
                if len(rows):
                    top=rows['Beta'].idxmax()
                    calm=rows['Correlation'].abs().idxmin()
                    well.append((f'{group} exposure',
                                 f'Highest beta to {rows.loc[top,"Benchmark"]}: {top} {rows.loc[top,"Beta"]:.2f}; '
                                 f'least correlated: {calm} ({rows.loc[calm,"Correlation"]:+.2f}).'))
            wide=exposures[exposures['Volatility (/yr)']>1.5]
            if len(wide):
                attention.append(('Very volatile',', '.join(f'{ticker} {vol:.0%}/yr' for ticker,vol in
                                                             wide['Volatility (/yr)'].sort_values(ascending=False).items())+'.'))
        performance=getattr(self,'performance_metrics',None)
        if performance is not None:
            row=performance.iloc[0]
            sharpe=next((column for column in performance.columns if column.startswith('Sharpe (excess, sqrt')),None)
            text=f'Sharpe {row[sharpe]:.2f}' if sharpe else ''
            text+=''.join(f', {name.lower()} {row[name]:.2f}' for name in ('Beta (BTC)','Beta (SPY)') if name in performance)
            text+=f', max drawdown {row["Max Drawdown"]:.1%}' if 'Max Drawdown' in performance else ''
            attention.append(('Kelly portfolio',text+'. In-sample: the weights were fit on the same days.'))
        split=getattr(self,'alpha_split_results',None)
        if split is not None and len(split):
            forward=split.iloc[-2]
            verdict=alpha_verdict(forward['Annual alpha'],forward['Alpha t-stat'])
            (well if verdict.startswith('survives') else attention).append(
                ('Out-of-sample alpha',f'{forward["Annual alpha"]:+.0%}/yr (t {forward["Alpha t-stat"]:.2f}): {verdict}.'))
        summary=getattr(self,'factor_model_summary',None)
        if summary is not None:
            strong=summary.drop(index='Alpha',errors='ignore')
            strong=strong[strong['HAC t-stat'].abs()>=SIGNIFICANT_T]
            if len(strong):
                well.append(('Factor exposures','Significant: '+', '.join(
                    f'{name} {row["Beta"]:+.2f} (t {row["HAC t-stat"]:.1f})' for name,row in strong.iterrows())+'.'))
        premia=getattr(self,'factor_premia',None)
        if premia is not None and len(premia):
            best=premia['FM t-stat'].abs().idxmax()
            significant=premia[(premia['FM t-stat'].abs()>=SIGNIFICANT_T)|(premia['Sort t-stat'].abs()>=SIGNIFICANT_T)]
            if len(significant):
                well.append(('Cross-section premia','|t| >= 2: '+', '.join(
                    f'{name} (FM t {row["FM t-stat"]:.1f}, sort t {row["Sort t-stat"]:.1f})'
                    for name,row in significant.iterrows())+'.'))
            else:
                attention.append(('Cross-section premia',f'None clears |t| = 2; strongest is {best} '
                                  f'(FM t {premia.loc[best,"FM t-stat"]:.2f}).'))
        for label,result in getattr(self,'style_results',{}).items():
            fit,model=result['summary'],result['model'].drop(index='Alpha (intercept)',errors='ignore')
            strong=model[model['HAC t-stat'].abs()>=SIGNIFICANT_T]
            text=(f'Alpha {fit["Alpha (/yr)"]:+.0%}/yr (t {fit["Alpha t-stat"]:.2f}), R² {fit["R-squared"]:.0%}, '
                  f'{fit["Periods"]} to {fit["To"]}'
                  +('; significant: '+', '.join(f'{name} {row["Beta"]:+.2f}' for name,row in strong.iterrows())
                    if len(strong) else '')+'.')
            if '(actual)' not in label:
                attention.append((label,text+' In-sample, not a measure of skill.'))
            else:
                (well if fit['Alpha t-stat']>=SIGNIFICANT_T else attention).append((label,text))
        if getattr(self,'ib_book_source','').startswith('saved'):
            attention.append(('IB history','Built from snapshots, which cannot separate deposits from returns; '
                              'set up IB Flex for clean history.'))
        french=self.data_dir/'french'/FRENCH_FILES['us5']
        if french.exists():
            try:
                last=load_french('us5',self.data_dir/'french').index.max()
                lag=(pd.Timestamp.now(tz='UTC')-last).days
                if lag>20:
                    attention.append(('Factor data',f'Ken French data ends {last:%Y-%m-%d}; the last {lag} days are not '
                                      'in the style models until its next monthly update.'))
            except Exception:
                pass
        # Since the last run: key metrics and the asset list.
        path=self.data_dir/PORTFOLIO_HIGHLIGHTS_STATE
        try:
            baseline=json.loads(path.read_text())
        except (OSError,ValueError):
            baseline=None
        metrics=self._highlight_metrics()
        changes=[]
        if baseline:
            previous=baseline.get('metrics',{})
            for name,value in metrics.items():
                if name in previous and abs(value-previous[name])>1e-9:
                    changes.append({'Metric':name,'Last run':previous[name],'Now':value,'Change':value-previous[name]})
            added=sorted(set(self.all_assets)-set(baseline.get('assets',[])))
            removed=sorted(set(baseline.get('assets',[]))-set(self.all_assets))
            if added or removed:
                well.append(('Universe change',('Added: '+', '.join(added)+'. ' if added else '')
                             +('Dropped: '+', '.join(removed)+'.' if removed else '')))
        since=SinceLastRun(last_run=pd.Timestamp(baseline['ts']) if baseline else None,
                           changes=[(row['Metric'],f"{row['Last run']:,.3f}",f"{row['Now']:,.3f}",f"{row['Change']:+,.3f}")
                                    for row in changes])
        panel=getattr(self,'_highlights_panel',None)
        (panel.show if panel is not None else show_highlight_block)(well,attention,since)
        path.write_text(json.dumps({'ts':pd.Timestamp.now(tz='UTC').isoformat(),'metrics':metrics,
                                    'assets':list(self.all_assets)},indent=1))


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
        if getattr(self,'alpha_split_results',None) is not None:  # only if test_alpha_out_of_sample() ran
            self.alpha_split_results.to_csv(data/'alpha_split_results.csv',index=False)
        pd.DataFrame({
            'group':[self.asset_group.get(ticker,CRYPTO) for ticker in self.all_assets],
            'source':[self.asset_source.get(ticker,'') for ticker in self.all_assets],
            'benchmark':[self._benchmark_for(self.asset_group.get(ticker,CRYPTO)) for ticker in self.all_assets],
            'in_kelly':[ticker in self.assets for ticker in self.all_assets],
        },index=pd.Index(self.all_assets,name='asset')).to_csv(data/'asset_groups.csv')
        if getattr(self,'benchmark_exposures',None) is not None:
            self.benchmark_exposures.to_csv(data/'benchmark_exposures.csv')
        for label,result in getattr(self,'style_results',{}).items():
            slug=re.sub(r'[^a-z0-9]+','_',label.lower()).strip('_')
            result['model'].to_csv(data/f'style_model_{slug}.csv')
            result['summary'].to_csv(data/f'style_model_fit_{slug}.csv')
            result['attribution'].to_csv(data/f'style_attribution_{slug}.csv')
            result['specific'].rename('specific_return').to_csv(data/f'style_specific_returns_{slug}.csv',header=True)
        if getattr(self,'factor_premia',None) is not None:
            self.factor_premia.to_csv(data/'factor_premia.csv')
        if getattr(self,'factor_premia_joint',None) is not None:
            self.factor_premia_joint.to_csv(data/'factor_premia_joint.csv')
        print('Saved portfolio, factor, market, and return outputs to data/.')


# ----------------------------------------------------------------------------
# Analytics object
# ----------------------------------------------------------------------------
#
# The report notebook creates one `PortfolioAnalytics` instance and calls the named
# methods above.


class PortfolioAnalytics(
    _CalendarMixin, _MarketDataMixin, _PriceHistoryMixin, _ReturnAnalysisMixin, _CorrelationPlotMixin,
    _KellyMixin, _PerformanceMixin,
    _FactorModelMixin, _HighlightsMixin, _OutputMixin, _TableStyleMixin,
):
    def __init__(self, data_dir, wallet_addresses, assets=None, include_wallet_tokens=True, include_ib=True,
                 min_history_days=180, history_years=DEFAULT_HISTORY_YEARS):
        self.data_dir=Path(data_dir)
        self.configured_assets=list(dict.fromkeys(assets)) if assets else []
        if not isinstance(wallet_addresses, (list, tuple)) or not wallet_addresses:
            raise ValueError('wallet_addresses must be a non-empty list of wallet addresses.')
        if any(not isinstance(wallet, str) for wallet in wallet_addresses):
            raise ValueError('Every configured wallet address must be a string.')
        self.wallet_addresses=list(dict.fromkeys(wallet_addresses))
        for wallet in self.wallet_addresses:
            if not isinstance(wallet, str) or not re.fullmatch(r'0x[0-9a-fA-F]{40}', wallet):
                raise ValueError(f'Invalid wallet address: {wallet!r}')
        self.assets=[]
        # Every analysable asset; self.assets narrows to those with min_history_days of prices.
        self.all_assets=[]
        if not isinstance(min_history_days,int) or min_history_days<0:
            raise ValueError('min_history_days must be a non-negative whole number of days.')
        self.min_history_days=min_history_days
        if not isinstance(history_years,int) or not 1<=history_years<=20:
            raise ValueError('history_years must be a whole number from 1 to 20.')
        self.history_years=history_years
        self.history_start=pd.Timestamp.now(tz='UTC').normalize()-pd.DateOffset(years=history_years)
        self.short_history_assets=pd.Series(dtype=int)
        self.wallet_assets=[]
        # Wallet tokens on other chains and IB positions are added unless assets are given explicitly.
        self.include_wallet_tokens=include_wallet_tokens
        self.include_ib=include_ib
        self.asset_group={}
        self.asset_source={}
        self.wallet_token_keys={}
        self.benchmark='BTC'
        self.data_dir.mkdir(parents=True,exist_ok=True)
