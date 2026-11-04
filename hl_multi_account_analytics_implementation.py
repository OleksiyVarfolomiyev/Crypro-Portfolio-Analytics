"""Consolidated Hyperliquid analytics implementation.

Methods used by `hl_multi_account_analytics.ipynb`. It reuses the single-wallet
implementation and combines data from each configured wallet.
"""

from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import re

from hl_account_analytics_implementation import (
    ACCOUNTING_TIMEZONE,
    HISTORY_COLUMNS,
    HyperliquidAnalytics,
    _HighlightsMixin,
    _OpenPositionsMixin,
    breakdown_style,
    build_daily_volume_pnl,
    build_pnl_attribution,
    calculate_account_metrics,
    display_interactive_table,
    display_pnl_attribution_tables,
    liquidation_stats,
    perp_pnl_breakdown,
    plot_account_dashboard,
    plot_value_added_index,
    plot_volume_pnl,
    print_account_metrics,
    print_realized_stats,
    rolling_24h_metrics,
    split_realized_fills,
    summarize_volume_pnl,
    volume_total_blank_cells,
    wallet_unrealized_history,
)


# ----------------------------------------------------------------------------
# Shared aggregation helpers
# ----------------------------------------------------------------------------
#
# Aggregate historical account value over dates available for every wallet, and prepare a
# combined live snapshot and realized/funding history.


from hl_trade_analytics import time_weighted_index


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
    current_value = sum(analytics.data.current_account_value for analytics in wallet_analytics.values())
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
    funding_events=[a.data.funding_df for a in wallet_analytics.values() if not a.data.funding_df.empty]
    funding_df=pd.concat(funding_events,ignore_index=True) if funding_events else pd.DataFrame(
        columns=['ts','coin','fundingPnl']
    )
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
        funding_df=funding_df,
    )


# ----------------------------------------------------------------------------
# Wallet data collection
# ----------------------------------------------------------------------------
#
# Dashboard call: `analytics.fetch_data()`. Fetch each wallet once, combine daily account
# history over the shared date range, and pool fills and funding.


class _MultiWalletDataMixin:
    def fetch_data(self):
        self.wallet_analytics={}
        for wallet in self.wallet_addresses:
            account=HyperliquidAnalytics(wallet,self.data_dir)
            account.fetch_data()
            self.wallet_analytics[wallet]=account
        self.data=_combine_wallet_data(self.wallet_analytics)
        self._highlights_baseline_loaded=False
        self.collateral_equity=sum(account.collateral_equity for account in self.wallet_analytics.values())
        self.account_metrics=calculate_account_metrics(
            self.data.chart_df,self.data.current_account_value,self.data.unrealized_pnl_total,
            time_weighted_index([account.portfolio_history for account in self.wallet_analytics.values()]),
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
            self.positions=[
                row for account in self.wallet_analytics.values() for row in account._load_positions()
            ]
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
            rolling_24h_metrics(self.wallet_addresses,data.realized_df,data.funding_df),
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
        plot_account_dashboard(chart_metrics)

    def show_account_vs_btc(self):
        data=self._require_data()
        plot_value_added_index(self.account_metrics,data.chart_display_df,self.account_metrics.account_index)




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
        print_realized_stats(perp_df,'PERP',funding_history=data.funding_history_df,
                             account_value=data.current_account_value,funding_events=data.funding_df)
        print_realized_stats(spot_df,'SPOT',account_value=data.current_account_value)
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
        },cell_style=style,bold_last=True,blank_missing_cells=volume_total_blank_cells(summary))


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
        tables,net_pnl=build_pnl_attribution(
            data.realized_df,positions,funding_history=data.funding_history_df
        )
        if not tables:
            print('=== CONSOLIDATED PNL ATTRIBUTION ===')
            print('No realized or open-position PnL to attribute.')
            return
        display_pnl_attribution_tables(tables,'CONSOLIDATED PNL ATTRIBUTION')
        print(f'Net attributed PnL across wallets: ${net_pnl:,.2f}. '
              'Spot unrealized PnL is not included.')


# ----------------------------------------------------------------------------
# Consolidated analytics facade
# ----------------------------------------------------------------------------
#
# Create one object with the dashboard wallet list. It composes the report methods above.


class MultiWalletAnalytics(
    _HighlightsMixin,
    _MultiWalletDataMixin,_MultiWalletAccountMixin,_OpenPositionsMixin,
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
        self.data=None
        self.collateral_equity=None
        self.account_metrics=None
        self.positions=[]
        self._positions_loaded=False
