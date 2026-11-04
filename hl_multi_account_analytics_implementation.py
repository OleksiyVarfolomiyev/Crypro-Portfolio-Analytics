"""Consolidated Hyperliquid analytics implementation.

Methods used by `hl_multi_account_analytics.ipynb`. It reuses the single-wallet
implementation and combines data from each configured wallet.
"""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re

from hl_account_analytics_implementation import (
    ACCOUNTING_TIMEZONE,
    HISTORY_COLUMNS,
    HyperliquidAnalytics,
    breakdown_style,
    clip_inactive_periods,
    build_daily_volume_pnl,
    build_pnl_attribution,
    calculate_account_metrics,
    display_interactive_table,
    fetch_pnl_windows,
    fetch_portfolio_history,
    post,
    fetch_user_funding,
    liquidation_stats,
    perp_pnl_breakdown,
    plot_account_dashboard,
    plot_value_added_index,
    plot_volume_pnl,
    print_account_metrics,
    print_realized_stats,
    split_realized_fills,
    summarize_volume_pnl,
    wallet_unrealized_history,
)


# ----------------------------------------------------------------------------
# Shared aggregation helpers
# ----------------------------------------------------------------------------
#
# Aggregate historical account value over dates available for every wallet, and prepare a
# combined live snapshot and realized/funding history.


from hl_highlights import build_highlights,load_state,make_state,position_changes,save_state
from hl_trade_analytics import (
    add_expected_pnl_columns,
    adverse_funding_cost_per_unit,
    consolidate_current_funding,
    fetch_collateral_equity,
    fetch_daily_candles,
    fetch_funding_history,
    inactive_periods,
    time_weighted_index,
    net_funding_apr_history,
    summarize_current_funding,
    true_range,
    wilder_atr,
)

def _combine_wallet_data(wallet_analytics):
    """Combine account inputs while keeping the full common history window."""
    account_series = []
    realized_frames = []
    fills = []
    funding_frames = []
    for wallet, analytics in wallet_analytics.items():
        data = analytics.data
        chart = data.chart_df.copy()
        chart['ts'] = pd.to_datetime(chart['ts'], utc=True).dt.tz_convert(ACCOUNTING_TIMEZONE)
        history = chart.loc[chart['source'] == 'portfolio_allTime'].set_index('ts')['accountValue'].dropna()
        daily = history.resample('1D').last().dropna()
        if not daily.empty:
            account_series.append(daily.rename(wallet))
        realized = chart.loc[chart['source'].isin([
            'perp_realized_daily','spot_realized_daily','funding_daily'
        ])].copy()
        if not realized.empty:
            realized['wallet'] = wallet
            realized['ts'] = realized['ts'].dt.normalize()
            realized_frames.append(realized)
        frame = data.realized_df.copy()
        if not frame.empty:
            frame['wallet'] = wallet
            fills.append(frame)
        funding = data.funding_history_df.copy()
        if not funding.empty:
            funding['wallet'] = wallet
            funding_frames.append(funding)

    if account_series:
        daily_accounts = pd.concat(account_series, axis=1, join='inner').dropna()
        if daily_accounts.empty:
            raise ValueError('The configured wallets have no overlapping all-time account history.')
        combined_account_history = daily_accounts.sum(axis=1).rename('accountValue')
        oldest_wallet_history = min(account_series, key=lambda series: series.index.min())
        oldest_wallet = oldest_wallet_history.name
        shared_history_start = daily_accounts.index.min()
        chart_account_history = pd.concat([
            oldest_wallet_history.loc[oldest_wallet_history.index < shared_history_start],
            combined_account_history,
        ]).sort_index().rename('accountValue')
        account_history = combined_account_history.rename_axis('ts').reset_index()
        account_history['source'] = 'portfolio_allTime'
        for column in ['balance','open_position_value','unrealized_pnl','realized_pnl','asset']:
            account_history[column] = float('nan') if column != 'asset' else ''
        account_history = account_history[HISTORY_COLUMNS]
    else:
        account_history = pd.DataFrame(columns=HISTORY_COLUMNS)
        chart_account_history = pd.Series(dtype=float, name='accountValue')
        oldest_wallet = None
        shared_history_start = None

    if realized_frames:
        realized_history = pd.concat(realized_frames, ignore_index=True)
        realized_history['asset'] = realized_history['asset'].fillna('')
        realized_history = realized_history.groupby(
            ['ts','source','asset'],as_index=False,dropna=False
        )['realized_pnl'].sum()
        for column in ['accountValue','balance','open_position_value','unrealized_pnl']:
            realized_history[column] = float('nan')
        realized_history = realized_history[HISTORY_COLUMNS]
    else:
        realized_history = pd.DataFrame(columns=HISTORY_COLUMNS)

    total_balance = sum(analytics.data.balance_total for analytics in wallet_analytics.values())
    total_unrealized = sum(analytics.data.unrealized_pnl_total for analytics in wallet_analytics.values())
    total_position_value = sum(analytics.data.position_value_total for analytics in wallet_analytics.values())
    current_value = total_balance + total_unrealized
    snapshot = pd.DataFrame([{
        'ts':pd.Timestamp.now(tz=ACCOUNTING_TIMEZONE),
        'accountValue':current_value,'source':'live_snapshot','asset':'',
        'balance':total_balance,'open_position_value':total_position_value,
        'unrealized_pnl':total_unrealized,'realized_pnl':float('nan'),
    }])[HISTORY_COLUMNS]
    chart_df = pd.concat([account_history,realized_history,snapshot],ignore_index=True)
    chart_df['ts'] = pd.to_datetime(chart_df['ts'])
    chart_df = chart_df.sort_values('ts').reset_index(drop=True)
    chart_display_df = chart_df.copy()
    if oldest_wallet is not None:
        early_account_history = chart_account_history.loc[
            chart_account_history.index < shared_history_start
        ].rename_axis('ts').reset_index()
        if not early_account_history.empty:
            early_account_history['source'] = 'portfolio_allTime'
            for column in ['balance','open_position_value','unrealized_pnl','realized_pnl','asset']:
                early_account_history[column] = float('nan') if column != 'asset' else ''
            chart_display_df = pd.concat([
                chart_df,
                early_account_history[HISTORY_COLUMNS],
            ], ignore_index=True).sort_values('ts').reset_index(drop=True)
    realized_fills = pd.concat(fills,ignore_index=True) if fills else pd.DataFrame()
    if not realized_fills.empty:
        realized_fills['ts'] = pd.to_datetime(realized_fills['ts'],utc=True)
        realized_fills['day'] = pd.to_datetime(realized_fills['day'],utc=True)
    now_et=pd.Timestamp.now(tz=ACCOUNTING_TIMEZONE)
    cutoff=now_et.normalize()
    today_fill_pnl=0.0
    if not realized_fills.empty:
        today_fill_pnl=float(realized_fills.loc[realized_fills['ts']>=cutoff,'closedPnl'].sum())
    today_funding_pnl=0.0
    for analytics in wallet_analytics.values():
        raw_funding=analytics.data.funding_df
        if raw_funding is not None and not raw_funding.empty:
            funding_ts=pd.to_datetime(raw_funding['ts'],utc=True)
            today_funding_pnl+=float(
                raw_funding.loc[funding_ts>=cutoff,'fundingPnl'].sum()
            )
    today_realized_et=today_fill_pnl+today_funding_pnl
    funding_history = pd.concat(funding_frames,ignore_index=True) if funding_frames else pd.DataFrame(
        columns=['ts','coin','fundingPnl','wallet']
    )
    if not funding_history.empty:
        funding_history['ts'] = pd.to_datetime(funding_history['ts'],utc=True)
        funding_history = funding_history.groupby(['ts','coin'],as_index=False)['fundingPnl'].sum()
    return SimpleNamespace(
        chart_df=chart_df,chart_display_df=chart_display_df,
        chart_account_history=chart_account_history,oldest_wallet=oldest_wallet,
        shared_history_start=shared_history_start,
        history_rows_added=sum(a.data.history_rows_added for a in wallet_analytics.values()),
        realized_fills=sum((a.data.realized_fills for a in wallet_analytics.values()),[]),
        realized_df=realized_fills,funding_history_df=funding_history,
        current_account_value=current_value,balance_total=total_balance,
        unrealized_pnl_total=total_unrealized,position_value_total=total_position_value,
        realized_1d=today_realized_et,
    )


def _display_consolidated_positions(rows):
    if not rows:
        print('No open positions across the configured wallets.')
        return
    rows, expected_pnl_warnings = add_expected_pnl_columns(rows)
    frame=pd.DataFrame(rows).sort_values('Pos',key=lambda values:values.abs(),ascending=False).reset_index(drop=True)
    position_value = pd.to_numeric(frame['Pos'], errors='coerce').abs()
    equity = pd.to_numeric(frame['Account value'], errors='coerce')
    funding = pd.to_numeric(frame['Funding'], errors='coerce')
    fees_usd = pd.to_numeric(frame['Fees USD'], errors='coerce')
    frame['Fees / Equity (%)'] = (fees_usd.abs() / equity * 100).where(equity > 0)
    frame['Fees / Position (%)'] = (fees_usd.abs() / position_value * 100).where(position_value > 0)
    frame['Position / Account (%)'] = (position_value / equity * 100).where(equity > 0)
    frame = frame.rename(columns={'Funding': 'Funding (since open)','Coin':'Ticker'})
    # Split across two tables so each fits on screen; both repeat the identifying columns.
    identifiers=['Wallet','Ticker','Side']
    pnl_columns=identifiers+['Liq %','ROE','uPnL','Funding (since open)',
             'Funding (1h)','Funding APR',
             'Expected Price PnL (5d)','Expected Funding PnL (5d)',
             'Expected PnL incl. funding (5d)',
             'Expected Price Return (5d)','Expected Funding Return (5d)']
    position_columns=identifiers+['Entry','Mark','Pos','Margin','Margin Type','Margin %','Fees',
             'Fees / Equity (%)','Fees / Position (%)','Position / Account (%)',
             'Liq Px','Lev.','Sz']
    usd=lambda value:f'-${abs(value):,.2f}' if value<0 else f'${value:,.2f}'
    def style(row_index,column,value,row):
        if column=='Side' and value in {'LONG','SHORT'}:
            return 'color:#188038;font-weight:600;' if value=='LONG' else \
                'color:#c62828;font-weight:600;'
        if column=='Mark' and pd.notna(value) and pd.notna(row['Entry']):
            out_of_money=(row['Side']=='LONG' and value<row['Entry']) or \
                (row['Side']=='SHORT' and value>row['Entry'])
            return 'color:#c62828;font-weight:600;' if out_of_money else ''
        if column in {'Funding (1h)','Funding APR'} and pd.notna(value):
            if value==0: return ''
            pays=(value>0 and row['Side']=='LONG') or (value<0 and row['Side']=='SHORT')
            return 'color:#c62828;font-weight:600;' if pays else 'color:#188038;font-weight:600;'
        if column == 'Liq %' and pd.notna(value):
            return 'color:#c62828;' if value <= 10 else \
                'color:#d97706;' if value <= 25 else 'color:#188038;'
        if column in {'Fees','Fees / Equity (%)','Fees / Position (%)'}:
            return 'color:#c62828;'
        if column in {'uPnL','ROE','Funding (since open)','Expected Price PnL (5d)',
                'Expected Funding PnL (5d)','Expected PnL incl. funding (5d)',
                'Expected Price Return (5d)','Expected Funding Return (5d)'} and pd.notna(value):
            return 'color:#188038;' if value>0 else 'color:#c62828;' if value<0 else ''
        if column=='Fees': return 'color:#c62828;'
        return ''
    formatters={
        'Sz':lambda v:f'{v:,.2f}','Pos':lambda v:f'${v:,.0f}','Margin':lambda v:f'${v:,.0f}',
        'Liq %':lambda v:f'{v:.2f}%' if pd.notna(v) else 'n/a',
        'ROE':lambda v:f'{v:.2f}%','uPnL':usd,'Funding (since open)':usd,
        'Fees / Equity (%)':lambda v:f'{v:.3f}%' if pd.notna(v) else 'n/a',
        'Fees / Position (%)':lambda v:f'{v:.3f}%' if pd.notna(v) else 'n/a',
        'Position / Account (%)':lambda v:f'{v:.2f}%' if pd.notna(v) else 'n/a',
        'Funding (1h)':lambda v:f'{v:.6f}%' if pd.notna(v) else 'n/a',
        'Funding APR':lambda v:f'{v:.2f}%' if pd.notna(v) else 'n/a',
        'Expected Price PnL (5d)':usd,
        'Expected Funding PnL (5d)':usd,
        'Expected PnL incl. funding (5d)':usd,
        'Expected Price Return (5d)':lambda v:f'{v:.3f}%' if pd.notna(v) else 'n/a',
        'Expected Funding Return (5d)':lambda v:f'{v:.3f}%' if pd.notna(v) else 'n/a',
        'Entry':lambda v:f'{v:,.8g}','Mark':lambda v:f'{v:,.8g}',
        'Liq Px':lambda v:f'{v:,.5g}' if pd.notna(v) else 'n/a','Lev.':lambda v:f'{v:.2f}x',
        'Margin %':lambda v:f'{v:.2f}%' if pd.notna(v) else 'n/a',
    }
    display_interactive_table(frame,columns=pnl_columns,cell_style=style,formatters=formatters)
    display_interactive_table(frame,columns=position_columns,cell_style=style,formatters=formatters)
    print('Funding and fees are cumulative since the position was opened. Fee ratios are '
          'shown only for complete USDC-denominated fees; otherwise they are n/a.')
    if expected_pnl_warnings:
        print('Expected-PnL history warnings: ' + '; '.join(expected_pnl_warnings))

def _display_consolidated_funding(rows):
    funding_rows = consolidate_current_funding(rows)
    if not funding_rows:
        return
    print('=== ACCOUNT NET EXPOSURE AND FUNDING ===')
    account_summary = pd.DataFrame([summarize_current_funding(rows)])
    account_columns = [
        'Net Direction', 'Long Positions', 'Short Positions',
        'Long Notional ($)', 'Short Notional ($)', 'Gross Notional ($)',
        'Net Notional ($)', 'Net Exposure / Gross (%)',
        'Net Funding Rate / 1h (%) (covered)',
        'Net Funding APR (%) (simple, covered)',
        'Estimated Net Funding PnL / hour ($)',
        'Estimated Net Funding PnL / day ($)',
        'Funding-Covered Notional ($)', 'Funding Coverage (%)',
    ]
    display_interactive_table(account_summary, columns=account_columns, formatters={
        'Long Notional ($)': lambda value: f'${value:,.2f}',
        'Short Notional ($)': lambda value: f'${value:,.2f}',
        'Gross Notional ($)': lambda value: f'${value:,.2f}',
        'Net Notional ($)': lambda value: f'${value:,.2f}',
        'Net Exposure / Gross (%)': lambda value: f'{value:.2f}%' if pd.notna(value) else 'n/a',
        'Net Funding Rate / 1h (%) (covered)': lambda value: f'{value:.6f}%' if pd.notna(value) else 'n/a',
        'Net Funding APR (%) (simple, covered)': lambda value: f'{value:.2f}%' if pd.notna(value) else 'n/a',
        'Estimated Net Funding PnL / hour ($)': lambda value: f'${value:,.4f}',
        'Estimated Net Funding PnL / day ($)': lambda value: f'${value:,.2f}',
        'Funding-Covered Notional ($)': lambda value: f'${value:,.2f}',
        'Funding Coverage (%)': lambda value: f'{value:.1f}%' if pd.notna(value) else 'n/a',
    })
    print('Net direction compares long and short USD notionals; it is not delta- or beta-adjusted. '
          'Net funding rate is estimated net funding cash flow divided by funding-covered gross '
          'notional. Positive cash flow means estimated receipts. APR assumes current hourly '
          'rates persist; it is not a forecast.')
    print('=== FUNDING BY ASSET AND SIDE ===')
    frame = pd.DataFrame(funding_rows)
    columns = [
        'Coin', 'Side', 'Positions', 'Size', 'Position Value ($)',
        'Current Funding / 1h (%)', 'Funding APR (%)',
        'Estimated Funding PnL / hour ($)',
        'Estimated Funding PnL / day ($)', 'Funding Coverage (%)',
    ]

    def style(row_index, column, value, row):
        if column == 'Side':
            return 'color:#188038;font-weight:600;' if value == 'LONG' else \
                'color:#c62828;font-weight:600;'
        if column in {'Estimated Funding PnL / hour ($)',
                      'Estimated Funding PnL / day ($)'} and pd.notna(value):
            return 'color:#188038;' if value > 0 else \
                'color:#c62828;' if value < 0 else ''
        return ''

    display_interactive_table(frame, columns=columns, cell_style=style, formatters={
        'Size': lambda value: f'{value:,.4g}',
        'Position Value ($)': lambda value: f'${value:,.2f}',
        'Current Funding / 1h (%)': lambda value: f'{value:.6f}%' if pd.notna(value) else 'n/a',
        'Funding APR (%)': lambda value: f'{value:.2f}%' if pd.notna(value) else 'n/a',
        'Estimated Funding PnL / hour ($)': lambda value: f'${value:,.4f}',
        'Estimated Funding PnL / day ($)': lambda value: f'${value:,.2f}',
        'Funding Coverage (%)': lambda value: f'{value:.1f}%' if pd.notna(value) else 'n/a',
    })
    print('Funding rates are current hourly rates, weighted by position notional across wallets. '
          'Positive rates are paid by longs and received by shorts; estimated cash flows '
          'assume the current rate stays constant. APR is simple annualization, not a forecast.')


def _fetch_daily_atr(coin, period):
    if not isinstance(period, int) or period <= 0:
        raise ValueError('ATR_PERIOD must be a positive integer.')
    dex = ''
    base_coin = coin
    if ':' in coin:
        dex, base_coin = coin.split(':', 1)
    candles = fetch_daily_candles(base_coin, dex, lookback_days=max(period * 10 + 30, 365))
    available_ranges = len(true_range(candles))
    if available_ranges == 0:
        raise RuntimeError(f'No completed daily true-range observations available for {coin}.')
    period_used = min(period, available_ranges)
    return wilder_atr(candles, period_used), period_used


def _aggregate_open_risk(rows, atr_period, atr_stop_multiple):
    """Sum stop risk plus the adverse five-day funding reserve per asset and side."""
    if not rows:
        return 0.0
    if not np.isfinite(atr_stop_multiple) or atr_stop_multiple <= 0:
        raise ValueError('ATR_STOP_MULTIPLE must be a positive finite value.')
    grouped = {}
    for row in rows:
        key = (str(row['Coin']), str(row['Side']).upper())
        if key[1] not in {'LONG', 'SHORT'}:
            raise ValueError(f'Unsupported position side for {key[0]}: {key[1]}')
        grouped.setdefault(key, []).append(row)
    open_risk_total = 0.0
    atr_by_asset = {}
    funding_by_asset = {}
    short_history = set()
    for (coin, side), positions in sorted(grouped.items()):
        direction = 1 if side == 'LONG' else -1
        quantities = [abs(float(position['Sz'])) for position in positions]
        marks = [float(position['Mark']) for position in positions]
        oracles = [float(position.get('Oracle', position['Mark'])) for position in positions]
        total_quantity = sum(quantities)
        if total_quantity <= 0 or not all(np.isfinite(value) for value in quantities + marks + oracles):
            raise ValueError(f'Cannot calculate open risk for {coin} {side}: invalid size or mark.')
        current_mark = sum(quantity * mark for quantity, mark in zip(quantities, marks)) / total_quantity
        current_oracle = sum(quantity * oracle for quantity, oracle in zip(quantities, oracles)) / total_quantity
        if coin not in atr_by_asset:
            atr_by_asset[coin] = _fetch_daily_atr(coin, atr_period)
        atr, atr_period_used = atr_by_asset[coin]
        if atr_period_used < atr_period:
            short_history.add(coin)
        stop = current_mark - direction * atr_stop_multiple * atr
        dex = coin.split(':', 1)[0] if ':' in coin else ''
        base_coin = coin.split(':', 1)[-1]
        funding_key = (base_coin, dex)
        if funding_key not in funding_by_asset:
            funding_by_asset[funding_key] = fetch_funding_history(
                base_coin, dex, lookback_days=20
            )
        funding_reserve_per_unit = adverse_funding_cost_per_unit(
            funding_by_asset[funding_key], side, current_oracle,
            horizon_hours=5 * 24, percentile=0.95,
        )
        open_risk_total += sum(
            quantity * max(direction * (mark - stop), 0.0)
            for quantity, mark in zip(quantities, marks)
        ) + total_quantity * funding_reserve_per_unit
    if short_history:
        print(f'Limited daily history; ATR uses all completed candles available (shorter than '
              f'{atr_period} days) for: {", ".join(sorted(short_history))}.')
    return open_risk_total


# ----------------------------------------------------------------------------
# Wallet data collection
# ----------------------------------------------------------------------------
#
# Dashboard call: `analytics.fetch_data()`. Fetch each wallet once, combine daily account
# history over the shared date range, and pool fills and funding.


class _MultiWalletDataMixin:
    def fetch_data(self, refresh_full_history=False):
        self.wallet_analytics={}
        for wallet in self.wallet_addresses:
            account=HyperliquidAnalytics(wallet,self.data_dir)
            account.fetch_data(refresh_full_history=refresh_full_history)
            self.wallet_analytics[wallet]=account
        self.data=_combine_wallet_data(self.wallet_analytics)
        self._highlights_baseline_loaded=False
        self.collateral_equity=sum(
            fetch_collateral_equity(
                wallet.lower(),
                sum(float(state['marginSummary']['accountValue'])
                    for state in account.data.current_states.values()),
            )
            for wallet,account in self.wallet_analytics.items()
        )
        self.account_metrics=calculate_account_metrics(
            self.data.chart_df,self.data.current_account_value,self.data.unrealized_pnl_total
        )
        self.positions=[]
        self._positions_loaded=False
        fill_count=sum(len(a.data.realized_fills) for a in self.wallet_analytics.values())
        print(f'Loaded {len(self.wallet_addresses)} wallets; {fill_count:,} available fills; '
              f'combined current account value ${self.data.current_account_value:,.2f}.')
        common=self.data.chart_df.loc[self.data.chart_df['source']=='portfolio_allTime','ts']
        if not common.empty:
            print(f'Combined historical account value uses shared daily coverage in '
                  f'{ACCOUNTING_TIMEZONE} from {common.min().date()} through {common.max().date()}.')
        return self.data

    def _require_data(self):
        if self.data is None:
            raise RuntimeError('Call fetch_data() before requesting a consolidated report.')
        return self.data

    def _load_positions(self):
        self._require_data()
        if not self._positions_loaded:
            self.positions=[]
            for wallet,account in self.wallet_analytics.items():
                rows=account._load_positions()
                for row in rows:
                    self.positions.append({
                        **row,'Wallet':wallet[-4:],
                        'Account value':account.data.current_account_value,
                    })
            self._positions_loaded=True
        return self.positions


# ----------------------------------------------------------------------------
# Account reports
# ----------------------------------------------------------------------------
#
# Dashboard calls mirror the single-wallet account, chart, and BTC comparison sections;
# values and time series are consolidated.


class _MultiWalletAccountMixin:
    def show_account_pnl(self):
        data=self._require_data()
        print_account_metrics(
            self.account_metrics,data.history_rows_added,len(data.realized_fills),
            f'${data.current_account_value:,.2f} (combined live wallet value)',
            realized_1d=data.realized_1d,
        )

    def show_account_charts(self):
        data=self._require_data()
        chart_metrics=SimpleNamespace(**vars(self.account_metrics))
        chart_metrics.daily_account=data.chart_account_history
        # Chart only: the risk statistics keep using observed live snapshots.
        unrealized_by_wallet=pd.concat([
            wallet_unrealized_history(wallet,account.data).rename(wallet)
            for wallet,account in self.wallet_analytics.items()
        ],axis=1).sort_index().ffill().fillna(0.0)
        chart_metrics.daily_unrealized=self.account_metrics.daily_unrealized.combine_first(
            unrealized_by_wallet.sum(axis=1).reindex(self.account_metrics.day_index)
        )
        if data.oldest_wallet is not None:
            oldest_metrics=self.wallet_analytics[data.oldest_wallet].account_metrics
            early_returns=oldest_metrics.daily_return_pct.loc[
                oldest_metrics.daily_return_pct.index < data.shared_history_start
            ]
            chart_daily_returns=pd.concat([
                early_returns,self.account_metrics.daily_return_pct,
            ]).sort_index()
            if not chart_daily_returns.empty:
                weekly_returns=((1+chart_daily_returns/100).groupby(
                    chart_daily_returns.index.tz_localize(None).to_period('W-SUN')
                ).prod()-1)*100
                weekly_returns.index=weekly_returns.index.to_timestamp()
                chart_metrics.weekly_return_pct=weekly_returns
        plot_account_dashboard(chart_metrics)

    def show_account_vs_btc(self):
        data=self._require_data()
        account_index=time_weighted_index([
            fetch_portfolio_history(wallet) for wallet in self.wallet_addresses
        ])
        plot_value_added_index(self.account_metrics,data.chart_display_df,account_index)


# ----------------------------------------------------------------------------
# Open positions
# ----------------------------------------------------------------------------
#
# Dashboard call: `analytics.show_open_positions()`. Show per-wallet positions, consolidated
# current funding rates and side-aware funding estimates grouped by asset/side, stop-based
# sizing, and an aggregate account summary.


class _MultiWalletHighlightsMixin:
    def _positions_missing_stops(self,rows):
        missing=[]
        # Position rows identify a wallet by its last four characters.
        orders={wallet[-4:]:[] for wallet in self.wallet_addresses}
        for wallet in self.wallet_addresses:
            for dex in {row['Dex'] for row in rows if row['Wallet']==wallet[-4:]}:
                orders[wallet[-4:]].extend(post({'type':'frontendOpenOrders','user':wallet,'dex':dex}))
        for row in rows:
            stops=[
                order for order in orders[row['Wallet']]
                if order.get('coin')==row['Coin'] and order.get('reduceOnly')
                and 'stop' in str(order.get('orderType','')).lower()
            ]
            # A position-level stop has size zero because it always closes the whole position.
            covered=any(order.get('isPositionTpsl') for order in stops) or \
                sum(float(order['sz']) for order in stops)>=abs(float(row['Sz']))*0.999
            if not covered:
                missing.append(f"{row['Coin']} ({row['Wallet']})")
        return missing

    def show_highlights(self):
        data=self._require_data()
        rows=self._load_positions()
        now=pd.Timestamp.now(tz=ACCOUNTING_TIMEZONE)
        # Keep the same baseline if this cell is rerun without fetching new data.
        if not self._highlights_baseline_loaded:
            self._highlights_baseline=load_state(self.data_dir)
            self._highlights_baseline_loaded=True
        baseline=self._highlights_baseline
        windows=[fetch_pnl_windows(wallet) for wallet in self.wallet_addresses]
        pnl_24h,cumulative_pnl=(sum(values) for values in zip(*windows))
        margin_capacity=self.collateral_equity*(1-self.margin_reserve_pct/100)
        well,attention=build_highlights(
            now,baseline,rows,data.realized_df,data.funding_history_df,cumulative_pnl,pnl_24h,
            data.current_account_value,
            _aggregate_open_risk(rows,self.position_atr_period,self.position_atr_stop_multiple),
            self.collateral_equity*self.aggregate_risk_pct/100,
            margin_capacity-sum(row['Margin'] for row in rows),margin_capacity,
            self._positions_missing_stops(rows),
        )
        if baseline is None:
            print('No earlier run saved yet, so "Since last run" starts from the next run.')
        else:
            print(f"Last run: {pd.Timestamp(baseline['ts']).tz_convert(ACCOUNTING_TIMEZONE):%Y-%m-%d %H:%M %Z}.")
            for change,names in position_changes(baseline,rows).items():
                if names:
                    print(f"{change}: {', '.join(names)}.")
        for title,table in (('WHAT WENT WELL',well),('NEEDS ATTENTION',attention)):
            print(f'=== {title} ===')
            if table:
                display_interactive_table(pd.DataFrame(table),columns=['Window','Update','Detail'])
            else:
                print('Nothing to report.')
        save_state(self.data_dir,make_state(now,cumulative_pnl,rows))


class _MultiWalletPositionsMixin:
    def show_funding_apr_chart(self):
        events=[
            event for wallet in self.wallet_addresses
            for event in fetch_user_funding(wallet)
        ]
        history=net_funding_apr_history(events,timezone=ACCOUNTING_TIMEZONE)
        if history.empty:
            print('No funding history is available for the configured wallets.')
            return None
        periods=inactive_periods(history.index)
        filled=history.reindex(pd.date_range(history.index.min(),history.index.max(),freq='D'))
        fig=go.Figure()
        fig.add_trace(go.Scatter(
            x=filled.index,y=filled['apr_pct'],name='Daily',mode='lines',
            line={'color':'#b5b4ad','width':1},
            hovertemplate='Daily %{y:.2f}%<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=filled.index,y=filled['rolling_apr_pct'],name='7-day (notional-weighted)',mode='lines',
            line={'color':'#2a78d6','width':2},
            hovertemplate='7-day %{y:.2f}%<extra></extra>',
        ))
        fig.add_hline(y=0,line_color='#52514e',line_width=1)
        fig.update_layout(
            title='Net Funding APR (%) (simple, covered) over time, all wallets',
            template='plotly_white',height=480,hovermode='x unified',
            yaxis_title='Net funding APR (positive = received)',yaxis_ticksuffix='%',
            legend={'orientation':'h','y':1.02,'yanchor':'bottom','x':0,'xanchor':'left'},
            margin={'t':90},
        )
        fig.update_xaxes(showgrid=False)
        clip_inactive_periods(fig,periods)
        fig.update_yaxes(gridcolor='#ecebe6',zeroline=False)
        latest=history.iloc[-1]
        print(f"Latest 7-day net funding APR: {latest['rolling_apr_pct']:.2f}% "
              f"({history.index[-1]:%Y-%m-%d}, {ACCOUNTING_TIMEZONE} days). "
              f"Lifetime: {history['funding_pnl'].sum()/history['notional_hours'].sum()*24*365*100:.2f}% "
              f"on ${history['funding_pnl'].sum():,.2f} net funding.")
        fig.show()
        return history

    def show_open_positions(self):
        data=self._require_data()
        rows=self._load_positions()
        _display_consolidated_positions(rows)
        _display_consolidated_funding(rows)
        if not np.isfinite(self.margin_reserve_pct) or not 0 <= self.margin_reserve_pct < 100:
            raise ValueError('MARGIN_RESERVE_PCT must be at least 0 and less than 100.')
        if not np.isfinite(self.aggregate_risk_pct) or self.aggregate_risk_pct<=0:
            raise ValueError('AGGREGATE_RISK_PCT must be a positive finite percentage.')
        available_margin=self.collateral_equity*(1-self.margin_reserve_pct/100)
        aggregate_risk_cap=self.collateral_equity*self.aggregate_risk_pct/100
        open_risk=_aggregate_open_risk(rows,self.position_atr_period,self.position_atr_stop_multiple)
        position_value=sum(abs(row['Pos']) for row in rows)
        margin=sum(row['Margin'] for row in rows)
        funding=sum(row['Funding'] for row in rows)
        remaining_available_margin=(
            available_margin-margin if pd.notna(available_margin) else float('nan')
        )
        available_margin_used_pct=(
            margin/available_margin*100 if pd.notna(available_margin) and available_margin>0
            else float('nan')
        )
        remaining_available_margin_pct=(
            remaining_available_margin/available_margin*100
            if pd.notna(available_margin) and available_margin>0 else float('nan')
        )
        position_equity_pct=(
            position_value/data.current_account_value*100
            if data.current_account_value>0 else float('nan')
        )
        funding_equity_pct=(
            funding/data.current_account_value*100 if data.current_account_value>0 else float('nan')
        )
        funding_position_pct=(
            funding/position_value*100 if position_value>0 else float('nan')
        )
        summary=pd.DataFrame([
            ('Wallets',f'{len(self.wallet_addresses):,}'),
            ('Position value',f'${position_value:,.0f}'),
            ('uPnL',f'${data.unrealized_pnl_total:,.2f}'),
            ('ROE',f'{data.unrealized_pnl_total/margin*100:,.2f}%' if margin else 'n/a'),
            ('Margin',f'${margin:,.0f}'),
            ('Margin % bal',f'{margin/data.balance_total*100:,.2f}%' if data.balance_total else 'n/a'),
            (f'Available margin ({100-self.margin_reserve_pct:g}% of collateral)',
             f'${available_margin:,.2f}' if pd.notna(available_margin) else 'n/a'),
            (f'Available margin used ({self.margin_reserve_pct:g}% reserve)',
             f'{available_margin_used_pct:,.2f}%' if pd.notna(available_margin_used_pct) else 'n/a'),
            ('Available margin remaining',
             f'${remaining_available_margin:,.2f}' if pd.notna(remaining_available_margin) else 'n/a'),
            ('Available margin remaining (%)',
             f'{remaining_available_margin_pct:,.2f}%'
             if pd.notna(remaining_available_margin_pct) else 'n/a'),
            (f'Aggregate risk cap ({self.aggregate_risk_pct:g}% of collateral)',f'${aggregate_risk_cap:,.2f}'),
            ('Aggregate risk used',f'${open_risk:,.2f}'),
            ('Aggregate risk room',f'${aggregate_risk_cap-open_risk:,.2f}'),
            ('Position / Equity (%)',f'{position_equity_pct:,.2f}%'
             if pd.notna(position_equity_pct) else 'n/a'),
            ('Leverage (Pos / Margin)',f'{position_value/margin:,.2f}x' if margin else 'n/a'),
            ('Funding (open positions, since open)',f'${funding:,.2f}'),
            ('Funding / Equity (%)',f'{funding_equity_pct:,.3f}%' if pd.notna(funding_equity_pct) else 'n/a'),
            ('Funding / Position (%)',f'{funding_position_pct:,.3f}%' if pd.notna(funding_position_pct) else 'n/a'),
            ('Margin collateral',f'${self.collateral_equity:,.2f}'),
            ('Account value',f'${data.current_account_value:,.2f}'),
        ],columns=['Metric','Value'])
        def style(*cell):
            _,column,value,row=cell
            if column!='Value':
                return ''
            if row['Metric'] in {'Margin','Position value','Account value',
                                f'Available margin ({100-self.margin_reserve_pct:g}% of collateral)',
                                'Position / Equity (%)'}:
                return 'color:#188038;font-weight:600;'
            if row['Metric'] in {'Available margin remaining',
                                'Available margin remaining (%)','Aggregate risk room'}:
                if isinstance(value,str) and (value.startswith('-') or '$-' in value):
                    return 'color:#c62828;font-weight:600;'
                return 'color:#188038;font-weight:600;' if value!='n/a' else ''
            if row['Metric'] not in {'uPnL','ROE','Funding (open positions, since open)',
                                    'Funding / Equity (%)','Funding / Position (%)'}:
                return ''
            if isinstance(value,str) and (value.startswith('-') or '$-' in value):
                return 'color:#c62828;font-weight:600;'
            if isinstance(value,str) and value!='n/a':
                return 'color:#188038;font-weight:600;'
            return ''
        display_interactive_table(summary,cell_style=style)
        print('ROE is unrealized PnL divided by used margin; a negative ROE means the open positions '
              'are currently at an unrealized loss relative to the margin supporting them.')


# ----------------------------------------------------------------------------
# Realized PnL
# ----------------------------------------------------------------------------
#
# Dashboard call: `analytics.show_realized_pnl()`. Pool fills and funding from all
# configured wallets.


class _MultiWalletRealizedMixin:
    def show_realized_pnl(self):
        data=self._require_data()
        perp_df,spot_df=split_realized_fills(data.realized_df)
        print_realized_stats(spot_df,'SPOT',account_value=data.current_account_value)
        print_realized_stats(perp_df,'PERP',funding_history=data.funding_history_df,
                             account_value=data.current_account_value)
        print('\n=== PERP PNL BREAKDOWN ===')
        breakdown,non_usd_tokens=perp_pnl_breakdown(perp_df,data.funding_history_df)
        net_index=breakdown.index[breakdown['Component']=='Net perp PnL'][0]
        display_interactive_table(breakdown,formatters={
            'Amount ($)':lambda v:f'${v:,.2f}',
            'Contribution (% of |net PnL|)':lambda v:f'{v:.2f}%',
        },cell_style=breakdown_style,bold_last=True,
          blank_missing_cells={(net_index,'Contribution (% of |net PnL|)')})
        liquidations=liquidation_stats(perp_df,funding_history=data.funding_history_df)
        print(f"Liquidations: {liquidations['liquidations']} of {liquidations['closes']} closes "
              f"({liquidations['close_pct']:.2f}%) | liquidation PnL: ${liquidations['pnl']:,.2f}.")
        if non_usd_tokens:
            print(f'Trading/fee dollar contributions unavailable for fee token(s): {non_usd_tokens}.')


# ----------------------------------------------------------------------------
# Volume reports
# ----------------------------------------------------------------------------
#
# Dashboard calls: `analytics.show_volume_chart()` and `analytics.show_volume_summary()`.
# Pooled fill data feeds both outputs.


class _MultiWalletVolumeMixin:
    def show_volume_chart(self):
        data=self._require_data()
        plot_volume_pnl(data.realized_df)

    def show_volume_summary(self):
        data=self._require_data()
        summary=summarize_volume_pnl(build_daily_volume_pnl(data.realized_df))
        print('=== TRADED VOLUME VS REALIZED PNL (ALL WALLETS) ===')
        if summary.empty:
            print('No fill volume or realized PnL data is available.')
            return
        def style(row_index,column,value,row):
            if column in {'Realized PnL excl. funding (trading + fees) ($)','Realized PnL / Volume (%)','PnL / $1k volume ($)'} and pd.notna(value):
                return 'color:#188038;' if value>0 else 'color:#c62828;' if value<0 else ''
            return ''
        display_interactive_table(summary,formatters={
            'Volume ($)':lambda v:f'${v:,.2f}','Realized PnL excl. funding (trading + fees) ($)':lambda v:f'${v:,.2f}',
            'Realized PnL / Volume (%)':lambda v:f'{v:.2f}%',
            'PnL / $1k volume ($)':lambda v:f'${v:,.2f}',
            'Daily Pearson r':lambda v:f'{v:.3f}','Daily Spearman r':lambda v:f'{v:.3f}',
        },cell_style=style)


# ----------------------------------------------------------------------------
# PnL attribution
# ----------------------------------------------------------------------------
#
# Dashboard call: `analytics.show_pnl_attribution()`. Combine realized PnL, open positions,
# and funding by asset across wallets.


class _MultiWalletAttributionMixin:
    def show_pnl_attribution(self):
        data=self._require_data()
        positions=self._load_positions()
        attribution,net_pnl=build_pnl_attribution(
            data.realized_df,positions,funding_history=data.funding_history_df
        )
        print('=== CONSOLIDATED PNL ATTRIBUTION ===')
        if attribution.empty:
            print('No realized or open-position PnL to attribute.')
            return
        def style(row_index,column,value,row):
            if column in {'Realized','uPnL','Total PnL','Realized (% gross)',
                          'uPnL (% gross)','Total (% gross)'} and pd.notna(value):
                return 'color:#188038;' if value>0 else 'color:#c62828;' if value<0 else ''
            return ''
        display_interactive_table(attribution,formatters={
            'Closes':lambda v:f'{v:,.0f}','Realized':lambda v:f'${v:,.2f}',
            'uPnL':lambda v:f'${v:,.2f}','Total PnL':lambda v:f'${v:,.2f}',
            'Realized (% gross)':lambda v:f'{v:.2f}%','uPnL (% gross)':lambda v:f'{v:.2f}%',
            'Total (% gross)':lambda v:f'{v:.2f}%',
        },cell_style=style,bold_last=True,blank_missing_cells={
            (len(attribution)-1,column) for column in
            ('Realized (% gross)','uPnL (% gross)','Total (% gross)')
        })
        print(f'Net attributed PnL across wallets: ${net_pnl:,.2f}.')


# ----------------------------------------------------------------------------
# Consolidated analytics facade
# ----------------------------------------------------------------------------
#
# Create one object with the dashboard wallet list. It composes the report methods above.


class MultiWalletAnalytics(
    _MultiWalletHighlightsMixin,
    _MultiWalletDataMixin,_MultiWalletAccountMixin,_MultiWalletPositionsMixin,
    _MultiWalletRealizedMixin,_MultiWalletVolumeMixin,_MultiWalletAttributionMixin,
):
    def __init__(self,wallet_addresses,data_dir='data'):
        self.wallet_addresses=list(dict.fromkeys(wallet_addresses))
        if not self.wallet_addresses:
            raise ValueError('Configure at least one wallet address.')
        for wallet in self.wallet_addresses:
            if not isinstance(wallet,str) or not re.fullmatch(r'0x[0-9a-fA-F]{40}',wallet):
                raise ValueError(f'Invalid wallet address: {wallet!r}')
        self.data_dir=Path(data_dir)
        self.data_dir.mkdir(parents=True,exist_ok=True)
        self.position_risk_budget_pct=5.0
        self.position_atr_period=14
        self.position_atr_stop_multiple=1.25
        self.margin_reserve_pct=20.0
        self.aggregate_risk_pct=20.0
        self.data=None
        self._highlights_baseline=None
        self._highlights_baseline_loaded=False
        self.collateral_equity=None
        self.account_metrics=None
        self.positions=[]
        self._positions_loaded=False
