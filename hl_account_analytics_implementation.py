'Reusable Hyperliquid account analytics and ' \
    'interactive reporting.'
from __future__ import annotations
import html
import re
import uuid
from pathlib import Path
from time import sleep
from types import SimpleNamespace
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from hl_trade_analytics import (
    MAX_RATE_LIMIT_RETRIES,
    _rate_limit_delay,
    add_expected_pnl_columns,
    read_cache_json,
    write_cache_json,
    implied_unrealized_history,
    apply_fill_fees,
    drop_inactive_weeks,
    drop_refreshed_daily_rows,
    inactive_periods,
    time_weighted_index,
    visible_month_starts,
)
from IPython.display import HTML, display
from plotly.subplots import make_subplots
API = "https://api.hyperliquid.xyz/info"
ACCOUNTING_TIMEZONE = "America/New_York"
HISTORY_COLUMNS = [
    "ts", "accountValue", "source", "asset", "balance", "open_position_value",
    "unrealized_pnl", "realized_pnl",
]


def display_interactive_table(frame, formatters=None, cell_style=None, bold_last=False,
    columns=None, blank_missing_cells=None):
    'Display a self-contained HTML table with text' \
        ' filtering and sortable columns.'
    formatters = formatters or {}
    table_id = f'interactive-{uuid.uuid4().hex}'
    headers = list(columns) if columns is not None else list(frame.columns)
    header_html = ''.join(
        f"<th tabindex='0' aria-sort='none'>{html.escape(str(column))}</th>"
        for column in headers
    )
    body_rows = []
    for row_index, row in enumerate(frame.to_dict(orient='records')):
        cells = []
        for column in headers:
            value = row[column]
            missing = pd.isna(value)
            if missing:
                visible = '' if blank_missing_cells and (row_index,
                    column) in blank_missing_cells else 'n/a'
                sort_value = ''
            else:
                formatter = formatters.get(column)
                visible = formatter(value) if formatter else str(value)
                visible = re.sub(r'(?<![\w.])\+(\$?)(?=\d|\.\d)', r'\1', visible)
                sort_value = str(value)
            style = cell_style(row_index, column, value, row) if cell_style else ''
            if not style and not missing and isinstance(value, (int, float, np.number)) \
                and not isinstance(value, (bool, np.bool_)):
                style = 'color:#188038;' if value > 0 else 'color:#c62828;' if value < 0 else ''
            weight = 'font-weight:700;' if bold_last and row_index == len(frame) - 1 else ''
            sort_attribute = html.escape(sort_value, quote=True)
            style_attribute = html.escape(style + weight, quote=True)
            cells.append(f"<td data-sort='{sort_attribute}' style='{style_attribute}'>{html.escape(str(visible))}</td>")
        body_rows.append('<tr>' + ''.join(cells) + '</tr>')
    markup = f"""<div id="{table_id}" class="interactive-table-wrap">
      <input class="interactive-table-filter" type="search" placeholder="Filter \
rows..." aria-label="Filter table rows">
      <table \
class="interactive-table"><thead><tr>{header_html}</tr></thead>\
<tbody>{''.join(body_rows)}</tbody></table>
    </div>
    <style>
      #{table_id} {{ overflow-x:auto; margin:0.5em 0 1em; color:#e5e7eb; }}
      #{table_id} input {{ margin-bottom:0.4em; padding:0.35em 0.5em; width:min(24em, \
100%); color:#e5e7eb; background:#1f2937; border:1px solid #4b5563; }}
      #{table_id} table {{ border-collapse:collapse; width:100%; font-size:0.92em; \
background:#000; }}
      #{table_id} th, #{table_id} td {{ border:1px solid #4b5563; padding:0.35em \
0.55em; text-align:left; white-space:nowrap; }}
      #{table_id} th {{ background:#000; color:#f9fafb; cursor:pointer; \
position:sticky; top:0; }}
      #{table_id} tbody tr, #{table_id} tbody tr:nth-child(even), #{table_id} tbody \
tr:hover {{ background:#000; }}
    </style>
    <script>
      (() => {{
        const root = document.getElementById('{table_id}');
        if (!root) return;
        const body = root.querySelector('tbody');
        const rows = Array.from(body.rows);
        root.querySelector('input').addEventListener('input', event => {{
          const query = event.target.value.toLocaleLowerCase();
          rows.forEach(row => row.hidden = \
!row.textContent.toLocaleLowerCase().includes(query));
        }});
        root.querySelectorAll('th').forEach((header, column) => {{
          header.addEventListener('click', () => {{
            const ascending = header.dataset.sortDir !== 'asc';
            root.querySelectorAll('th').forEach(item => {{ item.dataset.sortDir = ''; \
item.setAttribute('aria-sort', 'none'); }});
            header.dataset.sortDir = ascending ? 'asc' : 'desc';
            header.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
            rows.sort((left, right) => {{
              const a = left.cells[column].dataset.sort;
              const b = right.cells[column].dataset.sort;
              const numberA = Number(a.replace(/[,$%]/g, '').replace(/^\\+/, ''));
              const numberB = Number(b.replace(/[,$%]/g, '').replace(/^\\+/, ''));
              const bothNumeric = a.trim() !== '' && b.trim() !== '' && \
Number.isFinite(numberA) && Number.isFinite(numberB);
              const comparison = bothNumeric ? numberA - numberB : a.localeCompare(b, \
undefined, {{numeric:true, sensitivity:'base'}});
              return ascending ? comparison : -comparison;
            }}).forEach(row => body.appendChild(row));
          }});
        }});
      }})();
    </script>"""
    display(HTML(markup))


def post(payload):
    """Send one request to Hyperliquid's public info endpoint, retrying rate limits."""
    transport_failures = 0
    rate_limit_retries = 0
    while True:
        try:
            response = requests.post(API, json=payload, timeout=30)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            transport_failures += 1
            if transport_failures == 3:
                raise
            sleep(2 ** (transport_failures - 1))
            continue
        if response.status_code == 429 and rate_limit_retries < MAX_RATE_LIMIT_RETRIES:
            rate_limit_retries += 1
            sleep(_rate_limit_delay(response, rate_limit_retries))
            continue
        response.raise_for_status()
        return response.json()


def as_float(value, default=0.0):
    'Convert API number strings safely, returning ' \
        'a default for missing values.'
    return float(value) if value not in (None, '') else default

# ----------------------------------------------------------------------------
# Wallet API and raw data helpers
# ----------------------------------------------------------------------------
# Request helpers and wallet data fetchers used by the report.


def spot_balance_usd(user=None):
    'Value spot balances using available spot ' \
        'markets and USD stablecoin anchors.'
    spot_state = post({'type': 'spotClearinghouseState', 'user': user})
    metadata = post({'type': 'spotMeta'})
    mids = post({'type': 'allMids', 'dex': ''})
    tokens = {token.get('index'): token.get('name') for token in metadata.get('tokens', [])}
    rates = {}
    for market in metadata.get('universe', []):
        token_ids = market.get('tokens', [])
        if len(token_ids) != 2:
            continue
        market_index = market.get('index')
        mid = as_float(mids.get(f'@{market_index}'), default=0.0)
        if mid <= 0:
            continue
        base, quote = (tokens.get(token_id) for token_id in token_ids)
        if not base or not quote:
            continue
        rates.setdefault(base, []).append((quote, mid))
        rates.setdefault(quote, []).append((base, 1.0 / mid))
    usd_prices = {coin: 1.0 for coin in ('USDC', 'USDT', 'USDE', 'DAI', 'USDH')}
    changed = True
    while changed:
        changed = False
        for coin, edges in rates.items():
            if coin in usd_prices:
                continue
            for quote, coin_in_quote in edges:
                if quote in usd_prices:
                    usd_prices[coin] = coin_in_quote * usd_prices[quote]
                    changed = True
                    break
    total_usd, priced, excluded = (0.0, [], [])
    for balance in spot_state.get('balances', []):
        amount = as_float(balance.get('total'))
        if not amount:
            continue
        coin = balance.get('coin', '')
        price = usd_prices.get(coin)
        if price is None or price <= 0:
            excluded.append((coin, amount))
            continue
        value_usd = amount * price
        total_usd += value_usd
        priced.append((coin, amount, price, value_usd))
    return (total_usd, priced, excluded)


def load_saved_history(csv_path):
    'Load the local history CSV, or return an ' \
        'empty frame with its schema.'
    if not csv_path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    saved = pd.read_csv(csv_path)
    raw_ts = saved['ts'].astype(str)
    aware_timestamp = raw_ts.str.contains(r'(?:Z|[+-]\d{2}:\d{2})$', regex=True)
    local_ts = pd.to_datetime(raw_ts, format='mixed', utc=True).dt.tz_convert(ACCOUNTING_TIMEZONE)
    legacy_daily = (~aware_timestamp) & saved['source'].isin(
        ['perp_realized_daily', 'spot_realized_daily', 'funding_daily']
    )
    if legacy_daily.any():
        # Legacy daily totals were bucketed by UTC day. Their exact event split cannot be recovered;
        # map the midnight bucket label to its corresponding Eastern calendar date once.
        local_ts.loc[legacy_daily] = local_ts.loc[legacy_daily].dt.normalize()
    saved['ts'] = local_ts
    return saved


def fetch_all_time_history(user):
    "Fetch the account's allTime API history for " \
        'an initial/full refresh.'
    portfolio = post({'type': 'portfolio', 'user': user})
    periods = {item[0]: item[1] for item in portfolio if isinstance(item, (list,
                tuple)) and len(item) == 2 and isinstance(item[1], dict)}
    history = periods.get('allTime', {}).get('accountValueHistory', [])
    if not history:
        raise ValueError('No allTime account value history was returned.')
    return history


def fetch_portfolio_history(user):
    """Fetch the account's all-time value and cumulative PnL history."""
    portfolio = post({'type': 'portfolio', 'user': user})
    periods = {item[0]: item[1] for item in portfolio if isinstance(item, (list,
                tuple)) and len(item) == 2 and isinstance(item[1], dict)}
    all_time = periods.get('allTime', {})
    values = pd.DataFrame(all_time.get('accountValueHistory', []), columns=['time', 'accountValue'])
    pnl = pd.DataFrame(all_time.get('pnlHistory', []), columns=['time', 'pnl'])
    history = values.merge(pnl, on='time').astype({'accountValue': float, 'pnl': float})
    history['ts'] = pd.to_datetime(history['time'], unit='ms', utc=True).dt.tz_convert(
        ACCOUNTING_TIMEZONE
    )
    return history[['ts', 'accountValue', 'pnl']]


def fetch_perp_pnl_history(user):
    """Fetch the account's all-time cumulative perp PnL, including open positions."""
    portfolio = post({'type': 'portfolio', 'user': user})
    periods = {item[0]: item[1] for item in portfolio if isinstance(item, (list,
                tuple)) and len(item) == 2 and isinstance(item[1], dict)}
    pnl = pd.DataFrame(periods.get('perpAllTime', {}).get('pnlHistory', []), columns=['time', 'pnl'])
    pnl['pnl'] = pd.to_numeric(pnl['pnl'], errors='coerce')
    pnl['ts'] = pd.to_datetime(pnl['time'], unit='ms', utc=True)
    return pnl.dropna(subset=['pnl'])[['ts', 'pnl']]


def wallet_unrealized_history(user, data):
    """Rebuild one wallet's daily perp unrealized PnL from its fills and funding."""
    perp_fills, _ = split_realized_fills(data.realized_df)
    return implied_unrealized_history(
        fetch_perp_pnl_history(user),
        perp_fills if not perp_fills.empty else pd.DataFrame(columns=['ts', 'closedPnl']),
        data.funding_history_df if not data.funding_history_df.empty
            else pd.DataFrame(columns=['ts', 'fundingPnl']),
        ACCOUNTING_TIMEZONE,
    )


def fetch_pnl_windows(user):
    """Return the account's PnL over the last day and in total, including open positions."""
    portfolio = post({'type': 'portfolio', 'user': user})
    periods = {item[0]: item[1] for item in portfolio if isinstance(item, (list,
                tuple)) and len(item) == 2 and isinstance(item[1], dict)}
    def latest(period):
        history = periods.get(period, {}).get('pnlHistory', [])
        return float(history[-1][1]) if history else 0.0
    return latest('day'), latest('allTime')


def fetch_perp_states(user):
    'Fetch the clearinghouse state from main ' \
        'Hyperliquid and every listed perp DEX.'
    dexes = post({'type': 'perpDexs'})
    names = [''] + [dex['name'] for dex in dexes if dex and dex.get('name')]
    return {dex: post({'type': 'clearinghouseState', 'user': user, 'dex': dex}) for dex in names}


def fetch_user_fills(user, max_pages=5):
    """Fetch up to 10,000 most recent fills, paging backward in time."""
    fills = []
    end_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
    for _ in range(max_pages):
        batch = post({'type': 'userFillsByTime', 'user': user, 'startTime': 0,
                'endTime': end_ms, 'aggregateByTime': True})
        if not isinstance(batch, list) or not batch:
            break
        fills.extend(batch)
        times = [int(fill['time']) for fill in batch if fill.get('time') is not None]
        if len(batch) < 2000 or not times or len(fills) >= 10000:
            break
        next_end_ms = min(times) - 1
        if next_end_ms >= end_ms:
            break
        end_ms = next_end_ms
    return fills


def fetch_user_funding(user, start_ms=0):
    """Return signed USDC funding transfers, downloading only events not saved yet."""
    name = f'user_funding_{user.lower()}.json'
    events = read_cache_json(name) or []
    now_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
    latest = max((int(event['time']) for event in events), default=None)
    # Funding is paid hourly, so nothing new can exist within an hour of the last payment.
    if latest is None or now_ms - latest >= 3_600_000:
        fresh = _download_user_funding(user, 0 if latest is None else latest + 1)
        if fresh:
            events = events + fresh
            write_cache_json(name, events)
    return [event for event in events if int(event['time']) >= int(start_ms)]


def _download_user_funding(user, start_ms):
    'Fetch signed USDC funding transfers, paging ' \
        'through 500-event responses.'
    end_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
    cursor = max(0, int(start_ms))
    events = []
    page = 0
    while cursor <= end_ms:
        batch = post({'type': 'userFunding', 'user': user, 'startTime': cursor, 'endTime': end_ms})
        if not isinstance(batch, list) or not batch:
            break
        events.extend(batch)
        page += 1
        times = [int(event['time']) for event in batch if event.get('time') is not None]
        if len(batch) < 500 or not times:
            break
        next_start = max(times) + 1
        if next_start <= cursor:
            raise RuntimeError('Funding history pagination did not advance; ' \
                'stopping to avoid duplicate requests.')
        cursor = next_start
        if page % 10 == 0:
            sleep(2.6)
    return events


# ----------------------------------------------------------------------------
# Fill and account history normalization
# ----------------------------------------------------------------------------
# Normalize API events and merge them into the local account history cache.


def normalize_funding(events):
    'Parse signed USDC funding transfers; negative' \
        ' means paid, positive received.'
    rows = []
    for event in events:
        delta = event.get('delta', {}) or {}
        if event.get('time') is None or delta.get('usdc') is None:
            continue
        rows.append({'time': int(event['time']),
                'coin': str(delta.get('coin') or 'Unknown').strip(),
                'fundingPnl': as_float(delta.get('usdc'), default=float('nan'))})
    frame = pd.DataFrame(rows, columns=['time', 'coin', 'fundingPnl'])
    if not frame.empty:
        frame['ts'] = pd.to_datetime(frame['time'], unit='ms', utc=True)
        frame['day'] = frame['ts'].dt.tz_convert(ACCOUNTING_TIMEZONE).dt.normalize()
    return frame


def make_daily_funding_rows(funding_df):
    'Aggregate signed funding by Eastern calendar ' \
        'day and perp asset for history and attribution.'
    if funding_df.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    daily = funding_df.groupby(['day', 'coin'], as_index=False)['fundingPnl'].sum()
    daily = daily.rename(columns={'day': 'ts', 'coin': 'asset', 'fundingPnl': 'realized_pnl'})
    daily['source'] = 'funding_daily'
    for column in ['accountValue', 'balance', 'open_position_value', 'unrealized_pnl']:
        daily[column] = float('nan')
    return daily.reindex(columns=HISTORY_COLUMNS)


def normalize_fills(fills):
    """Create a consistent spot/perp fill table for PnL calculations."""
    rows = [{'time': int(fill['time']), 'coin': fill.get('coin', ''),
            'assetType': 'spot' if str(fill.get('coin', '')).startswith('@') else 'perp',
            'dir': fill.get('dir', ''), 'side': fill.get('side', ''),
            'size': abs(as_float(fill.get('sz'))),
            'notional': abs(as_float(fill.get('px')) * as_float(fill.get('sz'))),
            'closedPnl': as_float(fill.get('closedPnl')), 'fee': as_float(fill.get('fee'),
                default=float('nan')),
            'feeToken': str(fill.get('feeToken') or 'Unknown').strip()} for fill in \
                fills if fill.get('time') is not None]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = apply_fill_fees(frame)
        frame['ts'] = pd.to_datetime(frame['time'], unit='ms', utc=True)
        frame['day'] = frame['ts'].dt.tz_convert(ACCOUNTING_TIMEZONE).dt.normalize()
    return frame


def make_api_history_rows(history, saved):
    'Format new API points and skip timestamps ' \
        'already stored in the CSV.'
    frame = pd.DataFrame(history, columns=['ts', 'accountValue'])
    frame['ts'] = pd.to_datetime(frame['ts'], unit='ms', utc=True).dt.tz_convert(
        ACCOUNTING_TIMEZONE
    )
    frame['accountValue'] = frame['accountValue'].astype(float)
    frame['source'] = 'portfolio_allTime'
    for column in ['balance', 'open_position_value', 'unrealized_pnl', 'realized_pnl']:
        frame[column] = float('nan')
    if not saved.empty:
        previous = saved[saved['source'] == 'portfolio_allTime']
        last_ts = previous['ts'].max() if not previous.empty else pd.Timestamp.min
        frame = frame[frame['ts'] > last_ts]
    return frame


def make_live_snapshot(account_value, balance, position_value, unrealized_pnl):
    'Build one current snapshot using the live ' \
        'balance + open-position uPnL formula.'
    return pd.DataFrame([{'ts': pd.Timestamp.now(tz=ACCOUNTING_TIMEZONE),
                'accountValue': account_value, 'source': 'live_snapshot', 'balance': balance,
                'open_position_value': position_value, 'unrealized_pnl': unrealized_pnl,
                'realized_pnl': float('nan')}])


def make_daily_realized_rows(realized_df):
    'Aggregate fill PnL by day and asset class for' \
        ' persistent history.'
    daily_frames = []
    if not realized_df.empty:
        for asset_type, source_name in [('perp', 'perp_realized_daily'), ('spot',
                'spot_realized_daily')]:
            daily = realized_df[realized_df['assetType'] == asset_type].groupby('day',
                as_index=False)['closedPnl'].sum()
            if daily.empty:
                continue
            daily = daily.rename(columns={'day': 'ts', 'closedPnl': 'realized_pnl'})
            daily['source'] = source_name
            for column in ['accountValue', 'balance', 'open_position_value', 'unrealized_pnl']:
                daily[column] = float('nan')
            daily_frames.append(daily)
    if daily_frames:
        return pd.concat(daily_frames, ignore_index=True)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def merge_and_save_history(csv_path, *frames):
    'Combine all history sources, keep the latest ' \
        'row per timestamp/source, and save.'
    combined = pd.concat([frame.reindex(columns=HISTORY_COLUMNS) for frame in frames],
        ignore_index=True)
    combined['ts'] = pd.to_datetime(combined['ts'])
    combined = combined.drop_duplicates(subset=['ts', 'source', 'asset'], keep='last')
    combined = combined.sort_values('ts').reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    return combined


# ----------------------------------------------------------------------------
# Account / PnL
# ----------------------------------------------------------------------------
# Performance and risk calculations behind `show_account_pnl()`.
#
# Dashboard call: `analytics.show_account_pnl()`. Implementation method:
# `_AccountPnLMixin.show_account_pnl()`.


def fetch_btc_candles(start_ms, end_ms, interval='1d'):
    'Fetch BTC candles for a millisecond time ' \
        'range and requested interval.'
    return pd.DataFrame(post({'type': 'candleSnapshot', 'req': {'coin': 'BTC',
                    'interval': interval, 'startTime': start_ms, 'endTime': end_ms}}))


def historical_var(pnl_observations, account_value, confidence, min_observations):
    'Estimate loss VaR from the empirical ' \
        'lower-tail daily PnL quantile.'
    sample = pd.to_numeric(pnl_observations, errors='coerce').dropna()
    count = len(sample)
    if count < min_observations or account_value <= 0:
        return {'usd': float('nan'), 'pct': float('nan'), 'n': count, 'quantile_pnl': float('nan')}
    quantile_pnl = float(sample.quantile(1 - confidence))
    loss_usd = max(0.0, -quantile_pnl)
    return {'usd': loss_usd, 'pct': loss_usd / account_value * 100, 'n': count,
        'quantile_pnl': quantile_pnl}


def historical_volatility(pnl_observations, account_value, annualization_days=365):
    'Estimate daily and annualized volatility from' \
        ' observed daily PnL.'
    sample = pd.to_numeric(pnl_observations, errors='coerce').dropna()
    count = len(sample)
    if count < 2 or account_value <= 0:
        return {'daily_pct': float('nan'), 'annualized_pct': float('nan'), 'n': count}
    daily_returns_pct = sample / account_value * 100
    daily_vol_pct = float(daily_returns_pct.std(ddof=1))
    return {'daily_pct': daily_vol_pct,
        'annualized_pct': daily_vol_pct * annualization_days ** 0.5, 'n': count}


def calculate_account_metrics(chart_df, current_account_value, unrealized_pnl_total):
    'Build daily PnL/return series and risk ' \
        'metrics from saved account history.'
    account_series = chart_df.loc[chart_df['source'] == 'portfolio_allTime'].set_index( \
        'ts')['accountValue'].dropna()
    account_series.index = pd.to_datetime(account_series.index, utc=True).tz_convert(
        ACCOUNTING_TIMEZONE
    )
    daily_account = account_series.resample('1D').last().dropna()
    realized_sources = ['perp_realized_daily', 'spot_realized_daily', 'funding_daily']
    realized_history = chart_df.loc[chart_df['source'].isin(realized_sources), ['ts',
            'realized_pnl']].dropna()
    realized_series = realized_history.groupby('ts')['realized_pnl'].sum().sort_index()
    unrealized_series = chart_df.loc[chart_df['source'] == 'live_snapshot'].set_index( \
        'ts')['unrealized_pnl'].dropna()
    unrealized_series.index = pd.to_datetime(unrealized_series.index, utc=True).tz_convert(
        ACCOUNTING_TIMEZONE
    )
    daily_unrealized = unrealized_series.resample('1D').last().dropna()
    if not realized_series.empty or not daily_unrealized.empty:
        date_min = min((series.index.min() for series in [realized_series,
                    daily_unrealized] if not series.empty))
        date_max = max((series.index.max() for series in [realized_series,
                    daily_unrealized] if not series.empty))
        day_index = pd.date_range(date_min, date_max, freq='1D')
        daily_realized = realized_series.reindex(day_index, fill_value=0.0)
        daily_unrealized = daily_unrealized.reindex(day_index)
    else:
        day_index = pd.DatetimeIndex([])
        daily_realized = pd.Series(dtype=float, index=day_index)
        daily_unrealized = pd.Series(dtype=float, index=day_index)
    realized_avg_day = daily_realized.mean() if not daily_realized.empty else float('nan')
    u_change = daily_unrealized.dropna().diff().dropna()
    u_avg_day = u_change.mean() if not u_change.empty else float('nan')
    net_change = daily_realized.add(daily_unrealized.ffill().diff().fillna(0),
        fill_value=0).dropna()
    net_avg_day = net_change.mean() if not net_change.empty else float('nan')
    equity_for_returns = daily_account.reindex(net_change.index).ffill()
    daily_return_pct = (net_change / equity_for_returns.shift(1) * 100).replace([float('inf'),
            -float('inf')], float('nan')).dropna()
    avg_daily_return_pct = daily_return_pct.mean() if not daily_return_pct.empty else float('nan')
    max_daily_gain_pct = daily_return_pct.max() if not daily_return_pct.empty else float('nan')
    max_daily_loss_pct = daily_return_pct.min() if not daily_return_pct.empty else float('nan')
    monthly_return_pct = ((1 + daily_return_pct / 100).groupby(daily_return_pct.index. \
        to_period('M')).prod() - 1) * 100
    monthly_return_pct.index = monthly_return_pct.index.to_timestamp()
    weekly_return_pct = ((1 + daily_return_pct / 100).groupby(daily_return_pct.index. \
        to_period('W-SUN')).prod() - 1) * 100
    weekly_return_pct.index = weekly_return_pct.index.to_timestamp()
    cumulative_pnl = daily_realized.cumsum() + daily_unrealized.ffill().fillna(0)
    pnl_dd_usd = (cumulative_pnl - cumulative_pnl.cummax()).min() if not cumulative_pnl. \
        empty else float('nan')
    if not daily_account.empty:
        account_peak = daily_account.cummax()
        account_dd_usd = (daily_account - account_peak).min()
        positive = daily_account > 0
        account_dd_pct = (daily_account[positive] / account_peak[positive] - 1).min() * \
            100 if positive.any() else float('nan')
    else:
        account_dd_usd = account_dd_pct = float('nan')
    live_account = chart_df.loc[chart_df['source'] == 'live_snapshot'].set_index('ts')[ \
        'accountValue'].dropna()
    live_account.index = pd.to_datetime(live_account.index, utc=True).tz_convert(
        ACCOUNTING_TIMEZONE
    )
    live_account_daily = live_account.resample('1D').last().dropna()
    realized_history = chart_df.loc[chart_df['source'].isin(['perp_realized_daily',
                'spot_realized_daily', 'funding_daily']), ['ts', 'realized_pnl']].dropna().copy()
    if not realized_history.empty:
        realized_history['day'] = pd.to_datetime(realized_history['ts']).dt.normalize()
        account_realized_daily = realized_history.groupby('day')['realized_pnl'].sum()
    else:
        account_realized_daily = pd.Series(dtype=float)
    account_unrealized_series = chart_df.loc[chart_df['source'] == 'live_snapshot']. \
        set_index('ts')['unrealized_pnl'].dropna()
    account_unrealized_series.index = pd.to_datetime(
        account_unrealized_series.index, utc=True
    ).tz_convert(ACCOUNTING_TIMEZONE)
    account_unrealized_daily = account_unrealized_series.resample('1D').last().dropna()
    account_unrealized_change = account_unrealized_daily.diff()
    consecutive_days = account_unrealized_daily.index.tz_localize(None).to_series(
        index=account_unrealized_daily.index
    ).diff().eq(
        pd.Timedelta(days=1)
    )
    account_unrealized_change = account_unrealized_change[consecutive_days].dropna()
    observed_days = account_realized_daily.index.union(account_unrealized_change.index). \
        sort_values()
    observed_daily_pnl = account_realized_daily.reindex(observed_days,
        fill_value=0.0).add(account_unrealized_change.reindex(observed_days, fill_value=0.0))
    var_95 = historical_var(observed_daily_pnl, current_account_value, 0.95, min_observations=20)
    var_99 = historical_var(observed_daily_pnl, current_account_value, 0.99, min_observations=100)
    hist_vol = historical_volatility(observed_daily_pnl, current_account_value,
        annualization_days=365)
    alpha_annual = beta = sharpe_annual = float('nan')
    btc_sample_n = 0
    btc_df = pd.DataFrame()
    try:
        starts = [index.min() for index in [day_index, live_account_daily.index] if len(index)]
        if starts:
            start_ms = int(min(starts).timestamp() * 1000)
            end_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
            btc_df = fetch_btc_candles(start_ms, end_ms, interval='1d')
            if not btc_df.empty:
                btc_df['day'] = pd.to_datetime(btc_df['t'], unit='ms',
                    utc=True).dt.tz_convert(ACCOUNTING_TIMEZONE).dt.normalize()
                btc_close = btc_df.assign(close=btc_df['c'].astype(float)).set_index( \
                    'day')['close'].sort_index()
                btc_returns = btc_close.pct_change()
                equity = daily_account.reindex(net_change.index).ffill()
                portfolio_returns = net_change / equity.shift(1)
                compare = pd.concat([portfolio_returns.rename('pnl'),
                        btc_returns.reindex(portfolio_returns.index).rename('btc')],
                            axis=1).dropna()
                btc_sample_n = len(compare)
                if btc_sample_n >= 2 and compare['btc'].var() > 0:
                    beta = compare['pnl'].cov(compare['btc']) / compare['btc'].var()
                    alpha_annual = (compare['pnl'].mean() - beta * compare['btc'].mean() \
                        ) * 365 * 100
                    pnl_std = compare['pnl'].std()
                    sharpe_annual = compare['pnl'].mean() / pnl_std * 365 ** 0.5 if \
                        pnl_std else float('nan')
    except Exception as exc:
        print(f'BTC metrics unavailable: {type(exc).__name__}')
    return SimpleNamespace(**{'account_series': account_series,
            'daily_account': daily_account, 'realized_series': realized_series,
            'daily_realized': daily_realized, 'daily_unrealized': daily_unrealized,
            'day_index': day_index, 'realized_avg_day': realized_avg_day, 'u_avg_day': u_avg_day,
            'net_change': net_change, 'net_avg_day': net_avg_day,
            'daily_return_pct': daily_return_pct, 'avg_daily_return_pct': avg_daily_return_pct,
            'max_daily_gain_pct': max_daily_gain_pct, 'max_daily_loss_pct': max_daily_loss_pct,
            'monthly_return_pct': monthly_return_pct, 'weekly_return_pct': weekly_return_pct,
            'cumulative_pnl': cumulative_pnl, 'pnl_dd_usd': pnl_dd_usd,
            'account_dd_usd': account_dd_usd, 'account_dd_pct': account_dd_pct,
            'live_account': live_account, 'btc_df': btc_df, 'alpha_annual': alpha_annual,
            'beta': beta, 'sharpe_annual': sharpe_annual, 'btc_sample_n': btc_sample_n,
            'current_account_value': current_account_value,
            'unrealized_pnl_total': unrealized_pnl_total, 'observed_daily_pnl': observed_daily_pnl,
            'var_95': var_95, 'var_99': var_99, 'historical_volatility': hist_vol})


def fetch_latest_portfolio_value(user=None):
    'Return the last allTime accountValueHistory ' \
        'value, or NaN if absent.'
    response = post({'type': 'portfolio', 'user': user})
    periods = {item[0]: item[1] for item in response if isinstance(item, (list,
                tuple)) and len(item) == 2}
    history = periods.get('allTime', {}).get('accountValueHistory', [])
    return float(history[-1][1]) if history else float('nan')


def print_account_metrics(metrics, new_api_rows, fill_count, portfolio_value, realized_1d=None):
    'Display account performance and risk measures' \
        ' as an interactive table.'
    account = metrics.daily_account
    account_value = float(account.iloc[-1]) if not account.empty else float('nan')
    account_change_pct = account.pct_change().iloc[-1] * 100 if len(account) > 1 else float('nan')
    realized_today = (
        float(realized_1d) if realized_1d is not None
        else float(metrics.daily_realized.iloc[-1]) if not metrics.daily_realized.empty else 0.0
    )
    realized_pct = realized_today / account_value * 100 if pd.notna(account_value) and \
        account_value else float('nan')
    valid_unrealized = metrics.daily_unrealized.dropna()
    u_change_today = valid_unrealized.diff().iloc[-1] if len(valid_unrealized) > 1 else float('nan')
    unrealized_pct = metrics.unrealized_pnl_total / account_value * 100 if pd.notna( \
        account_value) and account_value else float('nan')
    def money(value):
        return f'${value:,.2f}' if pd.notna(value) else 'n/a'
    def percent(value):
        return f'{value:.2f}%' if pd.notna(value) else 'n/a'
    portfolio_value_text = (
        f'${portfolio_value:,.2f}'
        if isinstance(portfolio_value, (int, float)) and pd.notna(portfolio_value)
        else 'n/a' if isinstance(portfolio_value, (int, float)) else str(portfolio_value)
    )
    account_history = metrics.daily_account
    drawdown_sample = (
        f'{len(account_history):,} daily observations '
        f'({account_history.index.min().date()} to {account_history.index.max().date()})'
        if not account_history.empty else 'n/a'
    )
    rows = [('Account value',
            f'${account_value:,.2f}' if pd.notna(account_value) else 'n/a'), ('Portfolio value',
            portfolio_value_text), ('Account 1D (ET)', percent(account_change_pct)), ('Realized today (ET)',
            money(realized_today)), ('Realized today (% of account value)', percent(realized_pct)),
        ('Unrealized PnL', money(metrics.unrealized_pnl_total)),
        ('Unrealized PnL (% of account value)', percent(unrealized_pct)),
        ('Unrealized 1D change (ET)', money(u_change_today)), ('Average realized PnL / day',
            money(metrics.realized_avg_day)), ('Average unrealized PnL change / day',
            money(metrics.u_avg_day)), ('--- RISK ---', ''), ('BTC alpha / year',
            percent(metrics.alpha_annual)), ('BTC beta',
            f'{metrics.beta:.2f}' if pd.notna(metrics.beta) else 'n/a'), ('Annualized Sharpe',
            f'{metrics.sharpe_annual:.2f}' if pd.notna(metrics.sharpe_annual) else 'n/a'),
        ('Matched BTC observations', str(metrics.btc_sample_n)), ('Average daily account PnL',
            money(metrics.net_avg_day)), ('Average daily account return',
            percent(metrics.avg_daily_return_pct)), ('Maximum daily gain',
            percent(metrics.max_daily_gain_pct)), ('Maximum daily loss',
            percent(metrics.max_daily_loss_pct)), ('Account maximum drawdown',
            f'{money(metrics.account_dd_usd)} ({percent(metrics.account_dd_pct)})'),
        ('Account drawdown sample', drawdown_sample),
        ('PnL maximum drawdown', money(metrics.pnl_dd_usd)), ('1D historical VaR 95%',
            f"{money(metrics.var_95['usd'])} ({percent(metrics.var_95['pct'])}; n={metrics.var_95['n']})"),
        ('1D historical VaR 99%',
            f"{money(metrics.var_99['usd'])} ({percent(metrics.var_99['pct'])}; n={metrics.var_99['n']})"),
        ('Historical volatility',
            f"{percent(metrics.historical_volatility['daily_pct'])}/day; {percent(metrics.historical_volatility['annualized_pct'])}/year (n={metrics.historical_volatility['n']})"),
        ('New history rows', str(new_api_rows)),
        ('Available realized fills', f'{fill_count:,}' + (' (10,000-fill API cap)' if fill_count >= 10000 else ''))]
    table = pd.DataFrame(rows, columns=['Metric', 'Value'])
    def sign_style(row_index, column, value, row):
        if column != 'Value' or not value:
            return ''
        text = str(value).strip()
        if row['Metric'] == 'BTC beta':
            # Green when the account moves less than BTC, including when it moves opposite.
            return 'color:#188038;font-weight:600;' if text != 'n/a' and float(text) < 1 else ''
        if row['Metric'] == 'Portfolio value' or row['Metric'].startswith('1D historical VaR'):
            if re.search(r'\$\s*\d', text) and '$-' not in text:
                return 'color:#188038;font-weight:600;'
        if text.startswith('-') or '$-' in text or '(-' in text:
            return 'color:#c62828;font-weight:600;'
        try:
            if float(text.replace('$', '').replace('%', '').replace(',', '').replace('(',
                    '').replace(')', '')) > 0:
                return 'color:#188038;font-weight:600;'
        except ValueError:
            return ''
        return ''
    print('=== ACCOUNT / PNL AND RISK ===')
    display_interactive_table(table, cell_style=sign_style)


class _AccountPnLMixin:
    def show_account_pnl(self):
        data = self._require_data()
        try:
            portfolio_value = fetch_latest_portfolio_value(self.wallet_address)
        except requests.exceptions.RequestException as exc:
            portfolio_value = f'n/a (API error: {type(exc).__name__})'
        print_account_metrics(
            self.account_metrics, data.history_rows_added, len(data.realized_fills), portfolio_value
        )


# ----------------------------------------------------------------------------
# Account charts
# ----------------------------------------------------------------------------
# Account value, daily PnL, cumulative PnL, and return-distribution charts.
#
# Dashboard call: `analytics.show_account_charts()`. Implementation method:
# `_AccountChartsMixin.show_account_charts()`.


def clip_inactive_periods(fig, periods, axis_selector=None):
    """Collapse inactive periods on a figure's date axes and label each one."""
    if not periods:
        return
    times = pd.concat([
        pd.to_datetime(pd.Series(trace.x), utc=True, errors='coerce')
        for trace in fig.data if trace.type == 'scatter' and trace.x is not None
    ]).dropna().dt.tz_localize(None)
    fig.update_xaxes(
        rangebreaks=[{'bounds': [start, end]} for start, end in periods],
        tickmode='array', tickformat='%b %Y',
        tickvals=visible_month_starts(times.min(), times.max(), periods),
        **(axis_selector or {}),
    )
    for start, end in periods:
        fig.add_annotation(
            x=end, xref='x', y=1, yref='y domain', yanchor='bottom', xanchor='left', showarrow=False,
            text=f'Inactive {start:%b %Y} to {end:%b %Y} ({(end-start).days} days hidden)',
            font={'size': 11, 'color': '#52514e'},
        )


def plot_account_dashboard(metrics):
    'Show account history and PnL distribution in ' \
        'linked Plotly subplots.'
    # Only the three time panels share an axis; the histogram's x-axis is a return.
    fig = make_subplots(rows=4, cols=1, vertical_spacing=0.08,
        subplot_titles=('Account Value', 'Daily Realized and Unrealized PnL', 'Cumulative PnL',
            'Weekly Return Distribution'), row_heights=[0.28, 0.28, 0.24, 0.2])
    if not metrics.daily_account.empty:
        fig.add_trace(go.Scatter(x=metrics.daily_account.index,
                y=metrics.daily_account.values, name='Account Value', line={'color': '#6c757d',
                    'width': 2}), row=1, col=1)
    if not metrics.daily_realized.empty:
        fig.add_trace(go.Scatter(x=metrics.day_index, y=metrics.daily_realized.values,
                name='Realized', mode='lines+markers', line={'color': '#457b9d'}), row=2, col=1)
    if not metrics.daily_unrealized.empty:
        fig.add_trace(go.Scatter(x=metrics.day_index, y=metrics.daily_unrealized.values,
                name='Unrealized', mode='lines+markers', connectgaps=True,
                line={'color': '#e76f51'}), row=2, col=1)
    if not metrics.cumulative_pnl.empty:
        fig.add_trace(go.Scatter(x=metrics.cumulative_pnl.index,
                y=metrics.cumulative_pnl.values, name='Cumulative PnL', line={'color': '#8e44ad',
                    'width': 2}), row=3, col=1)
    periods = inactive_periods(pd.Index(metrics.day_index)[
        np.asarray(metrics.daily_realized.values) != 0
    ]) if not metrics.daily_realized.empty else []
    returns = drop_inactive_weeks(metrics.weekly_return_pct, periods)
    if len(returns) < len(metrics.weekly_return_pct):
        print(f'Weekly return chart omits {len(metrics.weekly_return_pct) - len(returns)} '
              'inactive weeks; statistics still include them.')
    if not returns.empty:
        display_returns = returns[returns >= -100]
        excluded = len(returns) - len(display_returns)
        if not display_returns.empty:
            fig.add_trace(go.Histogram(x=display_returns.values, name='Weekly returns',
                    marker_color='#457b9d', opacity=0.8, xbins={'size': 2.5},
                        showlegend=False), row=4, col=1)
            fig.add_vline(x=float(display_returns.mean()), line_dash='dash',
                line_color='#e76f51', annotation_text='Average', row=4, col=1)
            fig.add_vline(x=0, line_color='#555', row=4, col=1)
            if excluded:
                print(f'Weekly return chart omits {excluded} return(s) below -100%; \
                    other data and stats are unchanged.')
        else:
            fig.add_annotation(text='No valid monthly returns to display', x=0.5, y=0.5,
                xref='x4 domain', yref='y4 domain', showarrow=False)
    else:
        fig.add_annotation(text='Need matched weekly return observations', x=0.5, y=0.5,
            xref='x4 domain', yref='y4 domain', showarrow=False)
    fig.add_hline(y=0, line_color='#777', row=2, col=1)
    fig.add_hline(y=0, line_color='#777', row=3, col=1)
    fig.update_yaxes(title_text='USD', row=1, col=1)
    fig.update_yaxes(title_text='USD', row=2, col=1)
    fig.update_yaxes(title_text='USD', row=3, col=1)
    fig.update_yaxes(title_text='Weeks', row=4, col=1)
    fig.update_xaxes(title_text='Weekly return (%)', row=4, col=1)
    fig.update_xaxes(matches='x3', showticklabels=False, row=1, col=1)
    fig.update_xaxes(matches='x3', showticklabels=False, row=2, col=1)
    clip_inactive_periods(fig, periods, {'selector': lambda axis: axis.anchor != 'y4'})
    fig.update_layout(height=1000, template='plotly_white', barmode='overlay',
        hovermode='x unified', showlegend=False, margin={'l': 65, 'r': 25, 't': 95, 'b': 50})
    fig.show()


class _AccountChartsMixin:
    def show_account_charts(self):
        data = self._require_data()
        chart_metrics = SimpleNamespace(**vars(self.account_metrics))
        # Chart only: the risk statistics keep using observed live snapshots.
        chart_metrics.daily_unrealized = self.account_metrics.daily_unrealized.combine_first(
            wallet_unrealized_history(self.wallet_address, data).reindex(self.account_metrics.day_index)
        )
        plot_account_dashboard(chart_metrics)


# ----------------------------------------------------------------------------
# Account vs BTC
# ----------------------------------------------------------------------------
# Time-weighted account-vs-BTC index chart.
#
# Dashboard call: `analytics.show_account_vs_btc()`. Implementation method:
# `_AccountVsBTCMixin.show_account_vs_btc()`.


def plot_value_added_index(metrics, chart_df, account_index):
    """Plot the time-weighted account index against BTC."""
    btc_daily = metrics.btc_df
    account_points = pd.DataFrame({'ts': account_index.index, 'accountValue': account_index.values})
    realized_rows = chart_df.loc[
        chart_df['source'].isin(['perp_realized_daily','spot_realized_daily','funding_daily'])
        & (chart_df['realized_pnl'].fillna(0) != 0),
        'ts',
    ]
    periods = inactive_periods(
        pd.to_datetime(realized_rows,utc=True).dt.tz_convert(ACCOUNTING_TIMEZONE)
    )
    if not account_points.empty and not btc_daily.empty:
        account_points = account_points.rename(columns={'ts':'time','accountValue':'Account'})
        account_points['time'] = pd.to_datetime(account_points['time'],utc=True).dt.tz_convert(
            ACCOUNTING_TIMEZONE
        )
        btc_points = pd.DataFrame({
            'time':pd.to_datetime(btc_daily['t'],unit='ms',utc=True).dt.tz_convert(ACCOUNTING_TIMEZONE),
            'BTC':pd.to_numeric(btc_daily['c'],errors='coerce'),
        }).dropna().sort_values('time').drop_duplicates('time',keep='last')
        merge_time_dtype = pd.DatetimeTZDtype(unit='ns', tz=ACCOUNTING_TIMEZONE)
        account_points['time'] = account_points['time'].astype(merge_time_dtype)
        btc_points['time'] = btc_points['time'].astype(merge_time_dtype)
        matched = pd.merge_asof(
            account_points.sort_values('time'),btc_points,on='time',direction='backward'
        )
        matched = matched.dropna(subset=['BTC'])
        matched = matched[(matched['Account'] > 0) & (matched['BTC'] > 0)]
        matched = matched.drop_duplicates('time',keep='last')
        if not matched.empty:
            index = matched.set_index('time')[['Account', 'BTC']]
            index = index.div(index.iloc[0]).mul(100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=index.index, y=index['Account'], name='Account',
                    mode='lines+markers', line={'color': '#457b9d'}))
            fig.add_trace(go.Scatter(x=index.index, y=index['BTC'], name='BTC (daily)',
                    mode='lines+markers', line={'color': '#e76f51'}))
            fig.add_hline(y=100, line_dash='dash', line_color='#777')
            fig.update_layout(title='Value Added Index ($100 start): Account (time-weighted) vs BTC',
                yaxis_title='Value of $100', template='plotly_white', height=480,
                    hovermode='x unified')
            clip_inactive_periods(fig, periods)
            fig.show()
            print('The account line is a time-weighted return: deposits, withdrawals, and '
                  'transfers do not move it. BTC is shown on the same dates.')
        else:
            print('Value Added Index unavailable: no matching daily account and BTC values.')
    else:
        print('Value Added Index unavailable: need daily account history and BTC candles.')


class _AccountVsBTCMixin:
    def show_account_vs_btc(self):
        data = self._require_data()
        account_index = time_weighted_index([fetch_portfolio_history(self.wallet_address)])
        plot_value_added_index(self.account_metrics, data.chart_df, account_index)


# ----------------------------------------------------------------------------
# Traded volume vs realized PnL chart
# ----------------------------------------------------------------------------
# Daily fill volume compared with realized PnL by market type.
#
# Dashboard call: `analytics.show_volume_chart()`. Implementation method:
# `_VolumeChartMixin.show_volume_chart()`.


def plot_volume_pnl(realized_df):
    """Plot daily traded volume versus realized PnL by market type."""
    if realized_df.empty:
        print("No fill volume or realized PnL data is available.")
        return
    volume_data = realized_df.copy()
    volume_data["day"] = pd.to_datetime(volume_data["day"], errors="coerce",
        utc=True).dt.tz_localize(None)
    volume_data["notional"] = pd.to_numeric(volume_data["notional"], errors="coerce")
    volume_data["closedPnl"] = pd.to_numeric(volume_data["closedPnl"], errors="coerce")
    daily_chart = volume_data.dropna(subset=["day", "assetType"]).groupby(
        ["day", "assetType"], as_index=False
    ).agg(volume=("notional", "sum"), realized_pnl=("closedPnl", "sum"),
        fills=("closedPnl", "size"))
    market_types = [kind for kind in ["perp", "spot"] if (daily_chart["assetType"] == kind).any()]
    if not market_types:
        print("No fill volume or realized PnL data is available.")
        return
    fig = make_subplots(rows=1, cols=len(market_types),
        subplot_titles=[kind.title() for kind in market_types], horizontal_spacing=0.12)
    colors = {"perp": "#457b9d", "spot": "#e76f51"}
    for column, asset_type in enumerate(market_types, start=1):
        part = daily_chart.loc[daily_chart["assetType"] == asset_type]
        fig.add_trace(go.Scatter(
            x=part["volume"], y=part["realized_pnl"], mode="markers", name=asset_type.title(),
            marker={"color": colors[asset_type], "size": 9, "opacity": 0.75},
            customdata=part[["day", "fills"]],
            hovertemplate='Date: %{customdata[0]}<br>Volume: ' \
                '$%{x:,.2f}<br>Realized PnL: ' \
                '$%{y:,.2f}<br>Fills: ' \
                '%{customdata[1]}<extra></extra>',
        ), row=1, col=column)
        if len(part) >= 2 and part["volume"].nunique() > 1:
            slope, intercept = np.polyfit(part["volume"], part["realized_pnl"], 1)
            x_line = np.linspace(part["volume"].min(), part["volume"].max(), 100)
            fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept,
                    mode="lines", name=f"{asset_type.title()} trend", line={"color": "#f4a261",
                        "dash": "dash"}, showlegend=False), row=1, col=column)
        fig.add_hline(y=0, line_color="#777", line_width=1, row=1, col=column)
        fig.update_xaxes(title_text="Daily traded volume ($)", row=1, col=column)
        fig.update_yaxes(title_text="Daily realized PnL ($)", row=1, col=column)
    fig.update_layout(height=520, width=max(760, 650 * len(market_types)),
        template="plotly_white", title="Daily Traded Volume vs Realized PnL", showlegend=False)
    fig.show()


class _VolumeChartMixin:
    def show_volume_chart(self):
        data = self._require_data()
        plot_volume_pnl(data.realized_df)


# ----------------------------------------------------------------------------
# Open positions
# ----------------------------------------------------------------------------
# Fetching, margin/funding calculations, and interactive table formatting.
#
# Dashboard call: `analytics.show_open_positions()`. Implementation method:
# `_OpenPositionsMixin.show_open_positions()`.


def fetch_market_contexts(dexes):
    'Fetch current hourly funding and mark prices ' \
        'for each position DEX.'
    contexts = {}
    for dex in dexes:
        payload = {'type': 'metaAndAssetCtxs'}
        if dex:
            payload['dex'] = dex
        response = post(payload)
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise ValueError(f"Unexpected metaAndAssetCtxs response for DEX {dex or 'Hyperliquid'}")
        metadata, asset_contexts = response
        universe = metadata.get('universe', [])
        if len(universe) != len(asset_contexts):
            raise ValueError(
                f'Mismatched market metadata and asset contexts for DEX '
                f'{dex or "Hyperliquid"}'
            )
        dex_name = dex or 'Hyperliquid'
        for asset, context in zip(universe, asset_contexts):
            coin = asset.get('name')
            if coin:
                contexts[dex_name, coin] = context
    return contexts


def current_position_fees(coin, current_size, fills, fills_complete, return_usd=False):
    'Sum fees from the latest flat-to-open ' \
        'transition; return n/a if fills are ' \
        'incomplete.'
    def result(display, fee_totals=None):
        fee_usd = fee_totals.get('USDC', float('nan')) if fee_totals and set(fee_totals) == {'USDC'} else float('nan')
        return (display, fee_usd) if return_usd else display
    required = {'coin', 'assetType', 'time', 'side', 'size', 'fee', 'feeToken'}
    if (
        fills is None
        or fills.empty
        or not required.issubset(fills.columns)
        or current_size == 0
    ):
        return result('n/a')
    coin_names = fills['coin'].fillna('').astype(str)
    perp_mask = fills['assetType'] == 'perp'
    trades = fills.loc[perp_mask & (coin_names == str(coin))]
    if trades.empty:
        namespaced_mask = perp_mask & coin_names.str.endswith(f':{coin}')
        namespaces = coin_names.loc[namespaced_mask].str.split(':', n=1).str[0].unique()
        if not namespaced_mask.any():
            return result('n/a (no matching perp fills)')
        if len(namespaces) != 1:
            return result('n/a (ambiguous namespaced fills)')
        trades = fills.loc[namespaced_mask]
    trades = trades.sort_values('time', ascending=False, kind='stable')
    if trades.empty:
        return result('n/a')
    current_sign = np.sign(current_size)
    position_after = current_size
    fee_totals = {}
    for fill_index, fill in enumerate(trades.itertuples(index=False)):
        size = as_float(fill.size, default=float('nan'))
        side = str(fill.side).upper()
        if not np.isfinite(size) or size <= 0 or side not in {'B', 'A'}:
            return result('n/a')
        signed_size = size if side == 'B' else -size
        position_before = position_after - signed_size
        tolerance = max(1e-10, size * 1e-08)
        if np.sign(position_after) != current_sign:
            return result('n/a')
        reached_open = abs(position_before) <= tolerance
        if np.sign(position_before) not in (0, current_sign):
            fee_share = abs(position_after) / size
            reached_open = True
        else:
            fee_share = 1.0
        fee = as_float(fill.fee, default=float('nan'))
        token = str(fill.feeToken or 'Unknown').strip() or 'Unknown'
        if not np.isfinite(fee):
            return result('n/a')
        fee_totals[token] = fee_totals.get(token, 0.0) + fee * fee_share
        if reached_open:
            if fill_index == len(trades) - 1 and (not fills_complete):
                return result('n/a (entry may be outside available fills)')
            formatted_fees = (
                f'-${abs(amount):,.2f}'
                if token == 'USDC'
                else f'-{abs(amount):,.2f} {token}'
                for token, amount in sorted(fee_totals.items())
            )
            return result(' | '.join(formatted_fees), fee_totals)
        position_after = position_before
    return result('n/a (entry outside available fills)')


def fetch_position_rows(user=None, fills=None, fills_complete=False, states=None):
    'Return all open positions across the main ' \
        'perp DEX and HIP-3 DEXes.'
    states = fetch_perp_states(user) if states is None else states
    positions = [(dex, item) for dex,
        state in states.items() for item in state.get('assetPositions', [])]
    market_contexts = fetch_market_contexts({dex for dex, _ in positions})
    total_available_margin = sum((max(0.0,
                as_float((state.get('marginSummary') or {}).get('accountValue'),
                    default=0.0)) for state in states.values()))
    rows = []
    for dex, item in positions:
        position = item['position']
        size = as_float(position.get('szi'))
        value = as_float(position.get('positionValue'))
        implied_mark = abs(value) / abs(size) if size else 0.0
        dex_name = dex or 'Hyperliquid'
        market_context = market_contexts.get((dex_name, position['coin']), {})
        mark = as_float(market_context.get('markPx'), default=implied_mark)
        oracle = as_float(market_context.get('oraclePx'), default=mark)
        liq_price = as_float(position.get('liquidationPx'))
        if liq_price > 0 and mark > 0:
            liq_distance = (mark - liq_price) / mark * 100 if size > 0 else (liq_price - \
                mark) / mark * 100
        else:
            liq_distance = float('nan')
        funding_rate = as_float(market_context.get('funding'), default=float('nan'))
        funding_rate_pct = funding_rate * 100 if np.isfinite(funding_rate) else float('nan')
        funding_apr_pct = funding_rate * 24 * 365 * 100 if np.isfinite(funding_rate) \
            else float('nan')
        funding_usd = -as_float((position.get('cumFunding') or {}).get('sinceOpen'))
        margin = as_float(position.get('marginUsed'))
        margin_type = {'isolated': 'Isolated',
            'cross': 'Cross'}.get((position.get('leverage') or {}).get('type'), 'Unknown')
        fees, fees_usd = current_position_fees(
            position['coin'], size, fills, fills_complete, return_usd=True
        )
        rows.append({'Coin': position['coin'], 'Dex': dex, 'Pos': value, 'Margin': margin,
                'Margin Type': margin_type, 'ROE': as_float(position.get('returnOnEquity')) * 100,
                'uPnL': as_float(position.get('unrealizedPnl')), 'Funding': funding_usd,
                'Funding (1h)': funding_rate_pct, 'Funding APR': funding_apr_pct,
                'Fees': fees, 'Fees USD': fees_usd,
                'Margin %': margin / total_available_margin * 100 if \
                    total_available_margin else float('nan'),
                'Entry': as_float(position.get('entryPx')), 'Mark': mark, 'Oracle': oracle,
                'Liq Px': liq_price,
                'Liq %': liq_distance, 'Side': 'LONG' if size > 0 else 'SHORT',
                'Lev.': abs(value) / margin if margin else float('nan'), 'Sz': size})
    return (states, rows)


def summarize_positions(rows, user=None):
    'Calculate total perp exposure, PnL, and ' \
        'account-wide margin statistics.'
    position_value = sum((abs(row['Pos']) for row in rows))
    unrealized_pnl = sum((row['uPnL'] for row in rows))
    margin_used = sum((row['Margin'] for row in rows))
    funding = sum((row['Funding'] for row in rows))
    balance, priced, excluded = spot_balance_usd(user)
    return SimpleNamespace(position_value=position_value, unrealized_pnl=unrealized_pnl,
        margin_used=margin_used, funding=funding, balance=balance,
        current_account_value=balance + unrealized_pnl,
        roe=unrealized_pnl / margin_used * 100 if margin_used else 0.0,
        leverage=position_value / margin_used if margin_used else float('nan'),
        margin_pct=margin_used / balance * 100 if balance else float('nan'), priced_spot=priced,
        unpriced_spot=excluded)


def display_positions(rows, account_value=float('nan')):
    """Render the position table, sorted by absolute notional value."""
    if not rows:
        print('No open positions.')
        return
    rows, expected_pnl_warnings = add_expected_pnl_columns(rows)
    frame = pd.DataFrame(rows).sort_values('Pos', key=lambda values: values.abs(),
        ascending=False).reset_index(drop=True)
    position_value = pd.to_numeric(frame['Pos'], errors='coerce').abs()
    equity = pd.to_numeric(frame['Account value'], errors='coerce') if 'Account value' in frame else pd.Series(
        account_value, index=frame.index, dtype=float
    )
    funding = pd.to_numeric(frame['Funding'], errors='coerce')
    fees_usd = pd.to_numeric(frame['Fees USD'], errors='coerce')
    frame['Funding / Equity (%)'] = (funding / equity * 100).where(equity > 0)
    frame['Funding / Position (%)'] = (funding / position_value * 100).where(position_value > 0)
    frame['Fees / Equity (%)'] = (fees_usd.abs() / equity * 100).where(equity > 0)
    frame['Fees / Position (%)'] = (fees_usd.abs() / position_value * 100).where(position_value > 0)
    frame['Position / Account (%)'] = (position_value / equity * 100).where(equity > 0)
    frame = frame.rename(columns={'Funding': 'Funding (since open)'})
    position_columns = ['Coin', 'Dex', 'Liq %', 'ROE', 'uPnL', 'Funding (since open)',
        'Funding / Equity (%)', 'Funding / Position (%)', 'Funding (1h)', 'Funding APR',
        'Expected Price PnL (5d)', 'Expected Funding PnL (5d)',
        'Expected PnL incl. funding (5d)',
        'Expected Price Return (5d)', 'Expected Funding Return (5d)',
        'Entry', 'Mark', 'Pos', 'Margin', 'Margin Type', 'Margin %', 'Fees',
        'Fees / Equity (%)', 'Fees / Position (%)', 'Position / Account (%)',
        'Liq Px', 'Lev.', 'Sz']
    usd = lambda value: f'-${abs(value):,.2f}' if value < 0 else f'${value:,.2f}'
    usd_whole = lambda value: f'-${abs(value):,.0f}' if value < 0 else f'${value:,.0f}'
    price = lambda value: f'{value:,.8g}'
    def position_style(row_index, column, value, row):
        if column == 'Coin':
            return 'color:#c62828;font-weight:600;' if row['Side'] == 'SHORT' else \
                'color:#188038;font-weight:600;'
        if column == 'Margin Type':
            return {'Isolated': 'color:#c62828;font-weight:600;',
                'Cross': 'color:#188038;font-weight:600;'}.get(value, '')
        if column == 'Mark':
            in_profit = value > row['Entry'] if row['Side'] == 'LONG' else value < row['Entry']
            return 'color:#188038;font-weight:600;' if in_profit else \
                'color:#c62828;font-weight:600;'
        if column == 'Fees':
            return 'color:#c62828;'
        if column == 'Liq %' and pd.notna(value):
            return 'color:#c62828;' if value <= 10 else 'color:#d97706;' if value <= 25 \
                else 'color:#188038;'
        if column in {'Funding (1h)', 'Funding APR'} and pd.notna(value):
            if value == 0:
                return ''
            account_pays = (value > 0 and row['Side'] == 'LONG') or \
                (value < 0 and row['Side'] == 'SHORT')
            return 'color:#c62828;font-weight:600;' if account_pays else \
                'color:#188038;font-weight:600;'
        if column in {'Fees / Equity (%)', 'Fees / Position (%)', 'Fees'}:
            return 'color:#c62828;'
        if column in {'uPnL', 'ROE', 'Funding (since open)', 'Funding / Equity (%)',
                'Funding / Position (%)', 'Expected Price PnL (5d)',
                'Expected Funding PnL (5d)', 'Expected PnL incl. funding (5d)',
                'Expected Price Return (5d)', 'Expected Funding Return (5d)'} and isinstance(value, (int, float)):
            return 'color:#188038;' if value > 0 else 'color:#c62828;' if value < 0 else ''
        return ''
    display_interactive_table(frame, formatters={'Sz': lambda value: f'{value:,.2f}',
            'Entry': price, 'Mark': price, 'Pos': usd_whole,
            'Liq %': lambda value: f'{value:,.2f}%' if pd.notna(value) else 'n/a', 'uPnL': usd,
            'ROE': lambda value: f'{value:.2f}%', 'Lev.': lambda value: f'{value:.2f}x',
            'Liq Px': price, 'Margin': usd_whole, 'Funding (since open)': usd,
            'Funding (1h)': lambda value: f'{value:.6f}%' if pd.notna(value) else 'n/a',
            'Funding APR': lambda value: f'{value:.2f}%' if pd.notna(value) else 'n/a',
            'Funding / Equity (%)': lambda value: f'{value:.3f}%' if pd.notna(value) else 'n/a',
            'Funding / Position (%)': lambda value: f'{value:.3f}%' if pd.notna(value) else 'n/a',
            'Expected Price PnL (5d)': usd,
            'Expected Funding PnL (5d)': usd,
            'Expected PnL incl. funding (5d)': usd,
            'Expected Price Return (5d)': lambda value: f'{value:.3f}%' if pd.notna(value) else 'n/a',
            'Expected Funding Return (5d)': lambda value: f'{value:.3f}%' if pd.notna(value) else 'n/a',
            'Fees / Equity (%)': lambda value: f'{value:.3f}%' if pd.notna(value) else 'n/a',
            'Fees / Position (%)': lambda value: f'{value:.3f}%' if pd.notna(value) else 'n/a',
            'Position / Account (%)': lambda value: f'{value:.2f}%' if pd.notna(value) else 'n/a',
            'Margin %': lambda value: f'{value:.2f}%' if pd.notna(value) else 'n/a'},
        cell_style=position_style, columns=position_columns)
    print('Funding and fees are cumulative since the position was opened. Fee ratios are '
          'shown only for complete USDC-denominated fees; otherwise they are n/a.')
    if expected_pnl_warnings:
        print('Expected-PnL history warnings: ' + '; '.join(expected_pnl_warnings))


class _OpenPositionsMixin:
    def show_open_positions(self):
        self._require_data()
        self._load_positions()
        display_positions(self.positions, self.data.current_account_value)
        self.position_summary = summarize_positions(self.positions, self.wallet_address)
        summary = self.position_summary
        position_summary = pd.DataFrame([
            ("Position value", f"${summary.position_value:,.0f}"),
            ("uPnL", f"${summary.unrealized_pnl:,.2f}"),
            ("ROE", f"{summary.roe:,.2f}%"),
            ("Margin", f"${summary.margin_used:,.0f}"),
            ("Margin % bal",
                        f"{summary.margin_pct:,.2f}%" if pd.notna(summary.margin_pct) else "n/a"),
            ("Leverage (Pos / Margin)",
                        f"{summary.leverage:,.2f}x" if pd.notna(summary.leverage) else "n/a"),
            ("Funding (open positions, since open)", f"${summary.funding:,.2f}"),
        ], columns=["Metric", "Value"])
        def position_summary_style(*cell):
            _, column, value, row = cell
            if column != "Value" or not isinstance(value, str):
                return ""
            if row["Metric"] in {"Position value", "Margin"}:
                return "color:#188038;font-weight:600;"
            if row["Metric"] in {"uPnL", "ROE",
                "Funding (open positions, since open)"} and (value.startswith("-") or "$-" in value):
                return "color:#c62828;font-weight:600;"
            if row["Metric"] in {"uPnL", "ROE", "Funding (open positions, since open)"}:
                numeric = float(value.replace("$", "").replace("%", "").replace(",", ""))
                return "color:#188038;font-weight:600;" if numeric > 0 else ""
            return ""
        display_interactive_table(position_summary, cell_style=position_summary_style)


# ----------------------------------------------------------------------------
# Realized PnL statistics
# ----------------------------------------------------------------------------
# Fill PnL, fees, funding, liquidation, and win/loss statistics.
#
# Dashboard call: `analytics.show_realized_pnl()`. Implementation method:
# `_RealizedPnLMixin.show_realized_pnl()`.


def split_realized_fills(realized_df):
    """Separate the normalized fill table into spot and perp frames."""
    if realized_df.empty:
        empty = pd.DataFrame()
        return (empty.copy(), empty.copy())
    return (realized_df[realized_df['assetType'] == 'perp'].copy(),
        realized_df[realized_df['assetType'] == 'spot'].copy())


def realized_pnl_stats(frame, funding_history=None, now=None):
    'Summarize fee-net fill PnL plus signed ' \
        'funding, with trade stats based on fills.'
    now = now or pd.Timestamp.now(tz=ACCOUNTING_TIMEZONE)
    quarter_start_month = (now.month - 1) // 3 * 3 + 1
    quarter_start = pd.Timestamp(year=now.year, month=quarter_start_month, day=1, tz=ACCOUNTING_TIMEZONE)
    year_start = pd.Timestamp(year=now.year, month=1, day=1, tz=ACCOUNTING_TIMEZONE)
    if frame.empty:
        frame = pd.DataFrame(columns=['closedPnl', 'grossPnl', 'notional', 'ts'])
    if funding_history is None or funding_history.empty:
        funding_history = pd.DataFrame(columns=['fundingPnl', 'ts'])
    else:
        funding_history = funding_history.copy()
        funding_history['ts'] = pd.to_datetime(funding_history['ts'], utc=True)
    empty_result = {'total': 0.0, 'funding': 0.0, '1d': 0.0, '7d': 0.0, '30d': 0.0,
        'qtd': 0.0, 'ytd': 0.0, 'wins': 0, 'losses': 0, 'win_p': float('nan'),
        'loss_p': float('nan'), 'avg_win': float('nan'), 'avg_loss': float('nan'),
        'avg_win_pct': float('nan'), 'avg_loss_pct': float('nan'), 'ev': float('nan'), 'fills': 0}
    if frame.empty and funding_history.empty:
        return empty_result
    def combined_pnl_since(start_time=None):
        fill_part = frame
        funding_part = funding_history
        if start_time is not None:
            if not fill_part.empty:
                fill_part = fill_part.loc[fill_part['ts'] >= start_time]
            if not funding_part.empty:
                funding_part = funding_part.loc[funding_part['ts'] >= start_time]
        fill_pnl = fill_part['closedPnl'].sum() if not fill_part.empty else 0.0
        funding_pnl = funding_part['fundingPnl'].sum() if not funding_part.empty else 0.0
        return float(fill_pnl + funding_pnl)
    # Opening fills pay a fee too, so only fills with exchange-reported PnL count as closes.
    closes = frame.loc[frame['grossPnl'] != 0]
    wins = closes.loc[closes['closedPnl'] > 0]
    losses = closes.loc[closes['closedPnl'] < 0]
    win_returns = (wins['closedPnl'] / wins['notional'] * 100).replace([float('inf'),
            -float('inf')], float('nan')).dropna()
    loss_returns = (losses['closedPnl'] / losses['notional'] * 100).replace([float('inf'),
            -float('inf')], float('nan')).dropna()
    closed_count = len(wins) + len(losses)
    win_probability = len(wins) / closed_count if closed_count else float('nan')
    loss_probability = len(losses) / closed_count if closed_count else float('nan')
    average_win = wins['closedPnl'].mean() if not wins.empty else 0.0
    average_loss = losses['closedPnl'].mean() if not losses.empty else 0.0
    expected_value = win_probability * average_win + loss_probability * average_loss if \
        closed_count else float('nan')
    return {'total': combined_pnl_since(),
        'funding': float(funding_history['fundingPnl'].sum()) if not funding_history.empty else 0.0,
        '1d': combined_pnl_since(now - pd.Timedelta(days=1)),
        '7d': combined_pnl_since(now - pd.Timedelta(days=7)),
        '30d': combined_pnl_since(now - pd.Timedelta(days=30)),
        'qtd': combined_pnl_since(quarter_start), 'ytd': combined_pnl_since(year_start),
        'wins': len(wins), 'losses': len(losses), 'win_p': win_probability,
        'loss_p': loss_probability, 'avg_win': average_win, 'avg_loss': average_loss,
        'avg_win_pct': win_returns.mean() if not win_returns.empty else float('nan'),
        'avg_loss_pct': loss_returns.mean() if not loss_returns.empty else float('nan'),
        'ev': expected_value, 'fills': len(frame)}


def liquidation_stats(perp_frame, funding_history=None):
    'Count liquidation closes and calculate their ' \
        'share of closes and perp PnL.'
    if perp_frame.empty:
        return {'liquidations': 0, 'closes': 0, 'close_pct': 0.0, 'pnl': 0.0,
            'pnl_pct': float('nan')}
    directions = perp_frame['dir'].fillna('').astype(str).str.lower()
    close_mask = directions.str.contains('close|liquidat') | (perp_frame['grossPnl'] != 0)
    dedupe_columns = ['time', 'coin', 'dir']
    if 'wallet' in perp_frame.columns:
        dedupe_columns.insert(0, 'wallet')
    closes = perp_frame.loc[close_mask].drop_duplicates(subset=dedupe_columns)
    liquidations = perp_frame.loc[directions.str.contains('liquidat')].drop_duplicates(
        subset=dedupe_columns)
    funding_pnl = funding_history['fundingPnl'].sum() if funding_history is not None and \
        (not funding_history.empty) else 0.0
    total_pnl = perp_frame['closedPnl'].sum() + funding_pnl
    liquidation_pnl = liquidations['closedPnl'].sum()
    return {'liquidations': len(liquidations), 'closes': len(closes),
        'close_pct': len(liquidations) / len(closes) * 100 if len(closes) else 0.0,
        'pnl': liquidation_pnl,
            'pnl_pct': liquidation_pnl / total_pnl * 100 if total_pnl else float('nan')}


def perp_close_counts(perp_frame):
    """Count distinct recorded perp closes by asset."""
    if perp_frame.empty:
        return pd.Series(dtype='int64', name='Closes')
    directions = perp_frame['dir'].fillna('').astype(str).str.lower()
    close_mask = directions.str.contains('close|liquidat') | (perp_frame['grossPnl'] != 0)
    dedupe_columns = ['time', 'coin', 'dir']
    if 'wallet' in perp_frame.columns:
        dedupe_columns.insert(0, 'wallet')
    closes = perp_frame.loc[close_mask].drop_duplicates(
        subset=dedupe_columns).groupby('coin').size()
    closes.name = 'Closes'
    return closes


def print_realized_stats(frame, label, funding_history=None, account_value=float('nan')):
    'Display realized PnL summary metrics with account-value shares.'
    stats = realized_pnl_stats(frame, funding_history=funding_history)
    rows = [
        ('Realized PnL', stats['total'], 'money', True),
        ('24H', stats['1d'], 'money', True),
        ('7D', stats['7d'], 'money', True),
        ('30D', stats['30d'], 'money', True),
        ('QTD', stats['qtd'], 'money', True),
        ('YTD', stats['ytd'], 'money', True),
        ('W', stats['wins'], 'count', False),
        ('Win %', stats['win_p'] * 100, 'percent', False),
        ('Avg W', stats['avg_win'], 'money', False),
        ('L', stats['losses'], 'count', False),
        ('Loss %', stats['loss_p'] * 100, 'percent', False),
        ('Avg L', stats['avg_loss'], 'money', False),
        ('Avg W % of notional', stats['avg_win_pct'], 'percent', False),
        ('Avg L % of notional', stats['avg_loss_pct'], 'percent', False),
        ('EV / close', stats['ev'], 'money', False),
        ('Fills', stats['fills'], 'count', False),
    ]
    def format_stat(value, kind):
        if pd.isna(value):
            return 'n/a'
        if kind == 'money':
            return f'${value:,.2f}'
        if kind == 'percent':
            return f'{value:.2f}%'
        return f'{value:,.0f}'
    def account_value_pct(value, include_pct):
        if not include_pct or not np.isfinite(account_value) or \
            account_value == 0 or pd.isna(value):
            return 'n/a'
        return f'{value / account_value * 100:.2f}%'
    def format_stat_with_account_pct(value, kind, include_pct):
        formatted = format_stat(value, kind)
        account_pct = account_value_pct(value, include_pct)
        return f'{formatted} ({account_pct})' if account_pct != 'n/a' else formatted
    table = pd.DataFrame([
        (metric, format_stat_with_account_pct(value, kind, include_pct))
        for metric, value, kind, include_pct in rows
    ], columns=['Metric', 'Value'])
    def stats_style(row_index, column, value, row):
        if column != 'Value' or value == 'n/a':
            return ''
        if row['Metric'] in {'W', 'Win %', 'Avg W', 'Avg W % of notional'}:
            return 'color:#188038;font-weight:600;'
        if row['Metric'] in {'L', 'Loss %', 'Avg L', 'Avg L % of notional'}:
            return 'color:#c62828;font-weight:600;'
        return 'color:#c62828;' if str(value).startswith('-') or '$-' in str(value) else \
            'color:#188038;'
    print(f'=== REALIZED {label} ===')
    display_interactive_table(table, cell_style=stats_style)


def perp_pnl_breakdown(perp_fills, funding_history=None):
    """Break perp net PnL into gross trading, fee cost, and funding contributions.
    Hyperliquid reports fill fees as positive amounts, and normalize_fills has
    already subtracted USDC fees from closedPnl. Therefore gross trading PnL is
    closedPnl plus fees.
    Dollar reconciliation is only valid when all reported perp fees are USDC.
    """
    closed_pnl = float(perp_fills['closedPnl'].sum()) if not perp_fills.empty else 0.0
    funding_pnl = float(funding_history['fundingPnl'].sum()) if funding_history is not \
        None and (not funding_history.empty) else 0.0
    net_pnl = closed_pnl + funding_pnl
    fee_rows = perp_fills.dropna(subset=['fee']).copy() if 'fee' in perp_fills else pd.DataFrame()
    if not fee_rows.empty:
        fee_rows['feeToken'] = fee_rows['feeToken'].fillna('Unknown').astype(str).str.strip()
        fee_rows.loc[fee_rows['feeToken'] == '', 'feeToken'] = 'Unknown'
        nonzero_tokens = set(fee_rows.loc[fee_rows['fee'].abs() > 1e-12, 'feeToken'])
    else:
        nonzero_tokens = set()
    fees_are_usd = not nonzero_tokens or nonzero_tokens <= {'USDC'}
    if fees_are_usd:
        fees_paid = float(fee_rows['fee'].sum()) if not fee_rows.empty else 0.0
        rows = [('Trading', closed_pnl + fees_paid), ('Fees', -fees_paid), ('Funding',
                funding_pnl), ('Net perp PnL', net_pnl)]
        data = pd.DataFrame(rows, columns=['Component', 'Amount ($)'])
        data['Contribution (% of |net PnL|)'] = data['Amount ($)'] / abs(net_pnl) * 100 \
            if abs(net_pnl) > 1e-12 else float('nan')
        data.loc[data['Component'] == 'Net perp PnL',
            'Contribution (% of |net PnL|)'] = float('nan')
        return (data, None)
    data = pd.DataFrame([('Trading', float('nan')), ('Fees', float('nan')), ('Funding',
                funding_pnl), ('Net perp PnL', net_pnl)], columns=['Component', 'Amount ($)'])
    data['Contribution (% of |net PnL|)'] = data['Amount ($)'] / abs(net_pnl) * 100 if \
        abs(net_pnl) > 1e-12 else float('nan')
    data.loc[data['Component'] == 'Net perp PnL', 'Contribution (% of |net PnL|)'] = float('nan')
    return (data, ', '.join(sorted(nonzero_tokens)))


def breakdown_style(row_index, column, value, row):
    if column == 'Amount ($)':
        color_value = value
    elif column == 'Contribution (% of |net PnL|)':
        color_value = row['Amount ($)']
    else:
        return ''
    if pd.isna(color_value):
        return ''
    return 'color:#188038;' if color_value > 0 else 'color:#c62828;' if color_value < 0 else ''


class _RealizedPnLMixin:
    def show_realized_pnl(self):
        data = self._require_data()
        perp_df, spot_df = split_realized_fills(data.realized_df)
        print_realized_stats(spot_df, "SPOT", account_value=data.current_account_value)
        print_realized_stats(perp_df, "PERP", funding_history=data.funding_history_df,
            account_value=data.current_account_value)
        print("\n=== PERP PNL BREAKDOWN ===")
        perp_breakdown, non_usd_fee_tokens = perp_pnl_breakdown(perp_df, data.funding_history_df)
        net_row_index = perp_breakdown.index[perp_breakdown["Component"] == "Net perp PnL"][0]
        display_interactive_table(perp_breakdown, formatters={
            "Amount ($)": lambda value: f"${value:,.2f}",
                    "Contribution (% of |net PnL|)": lambda value: f"{value:.2f}%",
        }, cell_style=breakdown_style, bold_last=True,
            blank_missing_cells={(net_row_index, "Contribution (% of |net PnL|)")})
        if non_usd_fee_tokens:
            print('Trading and fee dollar contributions are n/a ' \
                'because fee amounts use non-USDC token(s): '
                  f"{non_usd_fee_tokens}. Net PnL still reconciles from closedPnl plus \
                      signed funding.")
        else:
            print('Trading is gross of fees; fees are shown as a' \
                ' negative contribution. Percentages use the ' \
                'absolute net PnL as denominator, so their ' \
                'signs match the dollar contributions.')
        liquidations = liquidation_stats(perp_df, funding_history=data.funding_history_df)
        liquidation_share = f"{liquidations['pnl_pct']:.2f}% of perp PnL" if pd.notna( \
            liquidations["pnl_pct"]) else "share n/a"
        liquidation_pnl = liquidations['pnl']
        print(f"Liquidations: {liquidations['liquidations']} of {liquidations['closes']} closes "
              f"({liquidations['close_pct']:.2f}%) | liquidation PnL: ${liquidation_pnl:,.2f} ({liquidation_share})")
        print('Realized PnL is fill closedPnl less USDC ' \
            'trading fees, plus signed perp funding; ' \
            'funding paid is negative.')
        print('Fill PnL, fees, win/loss, and liquidation ' \
            'stats use available fills (API cap: 10,000); ' \
            'funding history is stored separately.')


# ----------------------------------------------------------------------------
# Traded volume vs PnL
# ----------------------------------------------------------------------------
# Turnover efficiency and correlation summary helpers.
#
# Dashboard call: `analytics.show_volume_summary()`. Implementation method:
# `_VolumeSummaryMixin.show_volume_summary()`.


def build_daily_volume_pnl(fills):
    'Aggregate fill notional and realized PnL by ' \
        'UTC day and asset type.'
    if fills.empty:
        return pd.DataFrame(columns=['day', 'assetType', 'volume', 'realized_pnl', 'fills'])
    data = fills.copy()
    data['day'] = pd.to_datetime(data['day'], errors='coerce', utc=True).dt.tz_localize(None)
    data['notional'] = pd.to_numeric(data['notional'], errors='coerce')
    data['closedPnl'] = pd.to_numeric(data['closedPnl'], errors='coerce')
    data = data.dropna(subset=['day', 'assetType'])
    return data.groupby(['day', 'assetType'], as_index=False).agg(volume=('notional',
            'sum'), realized_pnl=('closedPnl', 'sum'), fills=('closedPnl',
            'size')).sort_values('day').reset_index(drop=True)


def summarize_volume_pnl(daily_data):
    'Summarize turnover, PnL efficiency, and daily' \
        ' correlations by market.'
    rows = []
    labels = {'perp': 'Perps', 'spot': 'Spot'}
    for asset_type in ['perp', 'spot']:
        part = daily_data.loc[daily_data['assetType'] == asset_type].copy()
        if part.empty:
            continue
        total_volume = part['volume'].sum()
        total_pnl = part['realized_pnl'].sum()
        pnl_ratio = total_pnl / total_volume if total_volume else float('nan')
        rows.append({'Market': labels.get(asset_type, asset_type.title()),
                'Volume ($)': total_volume, 'Realized PnL excl. funding (trading + fees) ($)': total_pnl,
                'Realized PnL / Volume (%)': pnl_ratio * 100,
                    'PnL / $1k volume ($)': pnl_ratio * 1000,
                'Active days': len(part), 'Daily Pearson r': part['volume'].corr(part[ \
                    'realized_pnl'],
                    method='pearson') if len(part) >= 2 else float('nan'),
                'Daily Spearman r': part['volume'].corr(part['realized_pnl'],
                    method='spearman') if len(part) >= 2 else float('nan')})
    return pd.DataFrame(rows)


class _VolumeSummaryMixin:
    def show_volume_summary(self):
        data = self._require_data()
        daily_volume_pnl = build_daily_volume_pnl(data.realized_df)
        volume_pnl_summary = summarize_volume_pnl(daily_volume_pnl)
        print("=== TRADED VOLUME VS REALIZED PNL ===")
        if volume_pnl_summary.empty:
            print("No fill volume or realized PnL data is available.")
            return
        def volume_style(row_index, column, value, row):
            if column in {"Realized PnL excl. funding (trading + fees) ($)", "Realized PnL / Volume (%)",
                "PnL / $1k volume ($)"} and pd.notna(value):
                return "color:#188038;" if value > 0 else "color:#c62828;"
            return ""
        display_interactive_table(volume_pnl_summary, formatters={
            "Volume ($)": lambda value: f"${value:,.2f}",
            "Realized PnL excl. funding (trading + fees) ($)": lambda value: f"${value:,.2f}",
            "Realized PnL / Volume (%)": lambda value: f"{value:.2f}%",
            "PnL / $1k volume ($)": lambda value: f"${value:,.2f}",
            "Daily Pearson r": lambda value: f"{value:.3f}",
            "Daily Spearman r": lambda value: f"{value:.3f}",
        }, cell_style=volume_style)
        print('Volume is one-sided fill notional (price ? ' \
            'size), counted once per fill. Realized PnL is' \
            ' Hyperliquid closedPnl less USDC ' \
            'trading fees; funding is excluded.')
        print('Correlation describes co-movement, not ' \
            'causation. Daily observations with no fills ' \
            'are omitted.')


# ----------------------------------------------------------------------------
# PnL attribution
# ----------------------------------------------------------------------------
# Realized, open-position, and total PnL contributions by asset.
#
# Dashboard call: `analytics.show_pnl_attribution()`. Implementation method:
# `_PnLAttributionMixin.show_pnl_attribution()`.


def build_pnl_attribution(realized_df, open_positions, funding_history=None):
    'Combine realized fill PnL and current perp ' \
        'uPnL by market/asset.'
    parts = []
    perp_fills = realized_df.loc[realized_df['assetType'] == 'perp'] \
        if not realized_df.empty else pd.DataFrame()
    close_counts = perp_close_counts(perp_fills)
    if not close_counts.empty:
        close_rows = close_counts.rename_axis('Asset').reset_index()
        close_rows['Type'] = 'Perp'
        close_rows['Realized'] = 0.0
        close_rows['uPnL'] = 0.0
        parts.append(close_rows[['Type', 'Asset', 'Realized', 'uPnL']])
    if funding_history is not None and (not funding_history.empty):
        funding = funding_history.groupby('coin', as_index=False)['fundingPnl'].sum()
        funding = funding.rename(columns={'coin': 'Asset', 'fundingPnl': 'Realized'})
        funding['Type'] = 'Perp'
        funding['uPnL'] = 0.0
        parts.append(funding[['Type', 'Asset', 'Realized', 'uPnL']])
    if not realized_df.empty:
        realized = realized_df.groupby(['assetType', 'coin'], as_index=False)['closedPnl'].sum()
        realized = realized.rename(columns={'assetType': 'Type', 'coin': 'Asset',
                'closedPnl': 'Realized'})
        realized['Type'] = realized['Type'].str.title()
        realized['uPnL'] = 0.0
        parts.append(realized[['Type', 'Asset', 'Realized', 'uPnL']])
    if open_positions:
        unrealized = pd.DataFrame(open_positions).groupby('Coin', as_index=False)['uPnL'].sum()
        unrealized = unrealized.rename(columns={'Coin': 'Asset'})
        unrealized['Type'] = 'Perp'
        unrealized['Realized'] = 0.0
        parts.append(unrealized[['Type', 'Asset', 'Realized', 'uPnL']])
    if not parts:
        return (pd.DataFrame(), 0.0)
    attribution = pd.concat(parts, ignore_index=True).groupby(['Type', 'Asset'],
        as_index=False)[['Realized', 'uPnL']].sum()
    if attribution['Asset'].astype(str).str.startswith('@').any():
        try:
            metadata = post({'type': 'spotMeta'})
            names = {}
            for index, market in enumerate(metadata.get('universe', [])):
                asset_id = f"@{market.get('index', index)}"
                names[asset_id] = market.get('name', asset_id)
            attribution['Asset'] = attribution['Asset'].map(lambda asset: names.get(asset, asset))
        except Exception as exc:
            print(f'Spot labels remain as API IDs: {type(exc).__name__}')
    attribution['Closes'] = 0
    perp_rows = attribution['Type'] == 'Perp'
    attribution.loc[perp_rows, 'Closes'] = attribution.loc[perp_rows, 'Asset'].map(
        close_counts).fillna(0).astype(int)
    attribution = attribution[(attribution['Realized'] != 0) | (attribution['uPnL'] != 0) |
        (attribution['Closes'] > 0)]
    attribution['Total PnL'] = attribution['Realized'] + attribution['uPnL']
    realized_total = attribution['Realized'].sum()
    unrealized_total = attribution['uPnL'].sum()
    net_pnl = attribution['Total PnL'].sum()
    realized_gross = attribution['Realized'].abs().sum()
    unrealized_gross = attribution['uPnL'].abs().sum()
    total_gross = attribution['Total PnL'].abs().sum()
    attribution['Realized (% gross)'] = attribution['Realized'] / realized_gross * 100 \
        if realized_gross > 1e-12 else float('nan')
    attribution['uPnL (% gross)'] = attribution['uPnL'] / unrealized_gross * 100 if \
        unrealized_gross > 1e-12 else float('nan')
    attribution['Total (% gross)'] = attribution['Total PnL'] / total_gross * 100 if \
        total_gross > 1e-12 else float('nan')
    attribution['_abs_pnl'] = attribution['Total PnL'].abs()
    attribution = attribution.sort_values('_abs_pnl',
        ascending=False).drop(columns='_abs_pnl').reset_index(drop=True)
    total_row = pd.DataFrame([{'Type': '', 'Asset': 'Net/total', 'Closes': int(
                attribution['Closes'].sum()), 'Realized': realized_total,
                'Realized (% gross)': float('nan'), 'uPnL': unrealized_total,
                'uPnL (% gross)': float('nan'), 'Total PnL': net_pnl,
                'Total (% gross)': float('nan')}])
    columns = ['Type', 'Asset', 'Closes', 'Realized', 'Realized (% gross)', 'uPnL',
        'uPnL (% gross)', 'Total PnL', 'Total (% gross)']
    attribution = pd.concat([attribution, total_row], ignore_index=True).reindex(columns=columns)
    return (attribution, net_pnl)


class _PnLAttributionMixin:
    def show_pnl_attribution(self):
        data = self._require_data()
        pnl_attribution, net_attributed_pnl = build_pnl_attribution(
            data.realized_df, self._load_positions(), funding_history=data.funding_history_df
        )
        print("=== PNL ATTRIBUTION ===")
        if pnl_attribution.empty:
            print("No realized or open-position PnL to attribute.")
            return
        def attribution_style(row_index, column, value, row):
            if column in {"Realized", "uPnL", "Total PnL", "Realized (% gross)",
                "uPnL (% gross)", "Total (% gross)"} and pd.notna(value):
                return "color:#188038;" if value > 0 else "color:#c62828;" if value < 0 else ""
            return ""
        display_interactive_table(pnl_attribution, formatters={
            "Closes": lambda value: f"{value:,.0f}",
            "Realized": lambda value: f"${value:,.2f}", "uPnL": lambda value: f"${value:,.2f}",
            "Total PnL": lambda value: f"${value:,.2f}",
                    "Realized (% gross)": lambda value: f"{value:.2f}%",
            "uPnL (% gross)": lambda value: f"{value:.2f}%",
                    "Total (% gross)": lambda value: f"{value:.2f}%",
        }, cell_style=attribution_style, bold_last=True, blank_missing_cells={
            (len(pnl_attribution) - 1, column) for column in ("Realized (% gross)",
                        "uPnL (% gross)", "Total (% gross)")
        })
        print(f"Net attributed PnL: ${net_attributed_pnl:,.2f} | percentages are signed \
            shares of gross absolute PnL in each column, so their signs match the dollar \
                contributions.")
        print('Scope: available realized fills (API cap: ' \
            '10,000) + saved perp funding + current open ' \
            'perp uPnL; spot unrealized PnL is not ' \
            'included.')


# ----------------------------------------------------------------------------
# Wallet setup and dashboard facade
# ----------------------------------------------------------------------------
#
# Wallet data collection and shared state used by the report method mixins.


def collect_wallet_data(wallet_address, data_dir="data", refresh_full_history=False):
    'Fetch current wallet data and merge it into ' \
        "that wallet's history CSV."
    if not isinstance(wallet_address, str) or not re.fullmatch(r"0x[0-9a-fA-F]{40}",
        wallet_address):
        raise ValueError("wallet_address must be a 0x-prefixed 40-hex-character address")
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    csv_path = data_path / f"hl_account_history_{wallet_address.lower()}.csv"
    saved = load_saved_history(csv_path)
    history = fetch_all_time_history(wallet_address) if saved.empty or refresh_full_history else []
    history_rows_added = 0
    current_states = fetch_perp_states(wallet_address)
    balance_total, priced_spot, unpriced_spot = spot_balance_usd(wallet_address)
    unrealized_pnl_total = sum(
        as_float(item.get("position", {}).get("unrealizedPnl"))
        for state in current_states.values()
        for item in state.get("assetPositions", [])
    )
    position_value_total = sum(
        abs(as_float(item.get("position", {}).get("positionValue")))
        for state in current_states.values()
        for item in state.get("assetPositions", [])
    )
    current_account_value = balance_total + unrealized_pnl_total
    realized_fills = fetch_user_fills(wallet_address)
    realized_df = normalize_fills(realized_fills)
    # Rebuild daily fill and funding rows from the API so a wrong saved row cannot persist.
    funding_events = fetch_user_funding(wallet_address)
    funding_df = normalize_funding(funding_events)
    if not funding_df.empty:
        saved = drop_refreshed_daily_rows(saved, ['funding_daily'], funding_df['day'])
    if not realized_df.empty:
        saved = drop_refreshed_daily_rows(
            saved, ['perp_realized_daily', 'spot_realized_daily'], realized_df['day']
        )
    daily_funding_new = make_daily_funding_rows(funding_df)
    new_history = make_api_history_rows(history, saved)
    history_rows_added = len(new_history)
    live_snapshot = make_live_snapshot(current_account_value, balance_total,
        position_value_total, unrealized_pnl_total)
    daily_realized_new = make_daily_realized_rows(realized_df)
    chart_df = merge_and_save_history(csv_path, saved, new_history, live_snapshot,
        daily_realized_new, daily_funding_new)
    funding_history_df = chart_df.loc[
        chart_df["source"] == "funding_daily", ["ts", "asset", "realized_pnl"]
    ].rename(columns={"asset": "coin", "realized_pnl": "fundingPnl"}).copy()
    funding_history_df["ts"] = pd.to_datetime(funding_history_df["ts"], utc=True)
    return SimpleNamespace(
        wallet_address=wallet_address, csv_path=csv_path, history_rows_added=history_rows_added,
        chart_df=chart_df, current_states=current_states, balance_total=balance_total,
        priced_spot=priced_spot, unpriced_spot=unpriced_spot,
        unrealized_pnl_total=unrealized_pnl_total, position_value_total=position_value_total,
        current_account_value=current_account_value, realized_fills=realized_fills,
        realized_df=realized_df, funding_df=funding_df, funding_history_df=funding_history_df,
    )

class _AnalyticsCore:
    def __init__(self, wallet_address, data_dir="data"):
        if not isinstance(wallet_address, str) or not re.fullmatch(r"0x[0-9a-fA-F]{40}",
            wallet_address):
            raise ValueError("wallet_address must be a 0x-prefixed 40-hex-character address")
        self.wallet_address = wallet_address
        self.data_dir = data_dir
        self.data = None
        self.account_metrics = None
        self.positions = []
        self.position_summary = None
        self._positions_loaded = False
    def fetch_data(self, refresh_full_history=False):
        'Fetch current account inputs and persist ' \
            'fresh wallet-specific history.'
        self.data = None
        self.account_metrics = None
        self.positions = []
        self.position_summary = None
        self._positions_loaded = False
        self.data = collect_wallet_data(self.wallet_address, self.data_dir, refresh_full_history)
        self.account_metrics = calculate_account_metrics(
            self.data.chart_df, self.data.current_account_value, self.data.unrealized_pnl_total
        )
        return self.data
    def _require_data(self):
        if self.data is None:
            raise RuntimeError("Call fetch_data() before requesting an account report.")
        return self.data
    def _load_positions(self):
        data = self._require_data()
        if not self._positions_loaded:
            _, self.positions = fetch_position_rows(
                self.wallet_address, data.realized_df,
                    fills_complete=len(data.realized_fills) < 10000,
                    states=data.current_states,
            )
            self._positions_loaded = True
        return self.positions

class HyperliquidAnalytics(
    _AccountPnLMixin, _AccountChartsMixin, _AccountVsBTCMixin,
    _VolumeChartMixin, _OpenPositionsMixin, _RealizedPnLMixin,
    _VolumeSummaryMixin, _PnLAttributionMixin, _AnalyticsCore,
):
    """Public facade used by the dashboard notebook."""
