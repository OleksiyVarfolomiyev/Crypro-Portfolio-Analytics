import importlib
import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Tests mock the API, so saved responses must not be read or written.
os.environ["HL_DISABLE_CACHE"] = "1"

# Modules live in the repository root and its execution/ and research/ folders.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "execution"), str(ROOT / "research")]

import hl_trade_analytics
from hl_backtest import BacktestSettings, backtest_signal
from hl_highlights import build_highlights, make_state, position_changes
from hl_market_research import (
    asset_class,
    CHANGE,
    FUNDING,
    OPEN_INTEREST,
    VOLUME,
    fetch_perp_markets,
    rank_markets,
    select_view,
    trending_markets,
)
from hl_trade_analytics import (
    adverse_funding_cost_per_unit,
    add_expected_pnl_columns,
    implied_unrealized_history,
    apply_fill_fees,
    consolidate_current_funding,
    drop_inactive_weeks,
    drop_refreshed_daily_rows,
    expected_position_pnl,
    expected_forward_return,
    fetch_collateral_equity,
    fetch_daily_candles,
    fetch_funding_history,
    inactive_periods,
    net_funding_apr_history,
    post_info,
    rolling_funding_sums,
    side_to_direction,
    summarize_current_funding,
    time_weighted_index,
    visible_month_starts,
    wilder_atr,
)
from hl_trade_execution import (
    AccountSnapshot,
    Market,
    Position,
    collect_snapshot,
    fetch_open_orders,
    plan_position_protection,
    _place_entry,
    plan_protection_ladder,
    plan_position_actions,
    fetch_markets,
    post_only_price,
    PositionAction,
    plan_open_position_protection,
    read_position_actions,
    _matching_protection_orders,
    _price_tick,
    _round_price,
    _submit_protection_order,
    size_trade,
)


def make_candles(count=30):
    times = pd.date_range("2025-01-01", periods=count, freq="D", tz="UTC")
    closes = 100 * (1.01 ** np.arange(count))
    return pd.DataFrame(
        {
            "time": times,
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
        }
    )


def make_funding(count=500, rate=0.00001):
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC"),
            "fundingRate": np.full(count, rate),
        }
    )


def load_account_implementation():
    """Import the account implementation, stubbing display packages when not installed."""
    stubs = {
        name: MagicMock()
        for name in ("plotly", "plotly.graph_objects", "plotly.subplots", "IPython", "IPython.display")
        if importlib.util.find_spec(name.split(".")[0]) is None
    }
    with patch.dict(sys.modules, stubs):
        return importlib.import_module("hl_account_analytics_implementation")


class RealizedPnlTests(unittest.TestCase):
    def setUp(self):
        self.account = load_account_implementation()

    def fills(self):
        def fill(time, direction, closed_pnl, fee, token="USDC"):
            return {
                "time": time,
                "coin": "BTC",
                "dir": direction,
                "side": "B",
                "px": "100",
                "sz": "1",
                "closedPnl": str(closed_pnl),
                "fee": str(fee),
                "feeToken": token,
            }

        return self.account.normalize_fills(
            [
                fill(1_790_000_000_000, "Open Long", 0, 0.4),
                fill(1_790_000_100_000, "Close Long", 10, 0.5),
                fill(1_790_000_200_000, "Open Long", 0, 0.4),
                fill(1_790_000_300_000, "Close Long", -3, 0.5),
                # Before fees this is a win; after fees it is a loss.
                fill(1_790_000_400_000, "Open Long", 0, 0.1),
                fill(1_790_000_500_000, "Close Long", 0.2, 0.3),
            ]
        )

    def test_post_retries_rate_limits_then_returns_data(self):
        limited = MagicMock(status_code=429, headers={"Retry-After": "0"})
        success = MagicMock(status_code=200)
        success.json.return_value = {"ok": True}
        with (
            patch.object(self.account.requests, "post", side_effect=[limited, success]) as post,
            patch.object(self.account, "sleep") as sleep,
        ):
            self.assertEqual(self.account.post({"type": "meta"}), {"ok": True})
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.0)

    def test_realized_pnl_subtracts_fees_on_opening_and_closing_fills(self):
        stats = self.account.realized_pnl_stats(self.fills())
        self.assertAlmostEqual(stats["total"], 7.2 - 2.2)

    def test_opening_fees_do_not_count_as_losing_closes(self):
        stats = self.account.realized_pnl_stats(self.fills())
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["losses"], 2)
        self.assertEqual(int(self.account.perp_close_counts(self.fills()).sum()), 3)

    def test_breakdown_reconciles_gross_trading_fees_and_funding(self):
        funding = pd.DataFrame({"fundingPnl": [-1.0]})
        breakdown, non_usd = self.account.perp_pnl_breakdown(self.fills(), funding)
        amounts = breakdown.set_index("Component")["Amount ($)"]
        self.assertIsNone(non_usd)
        self.assertAlmostEqual(amounts["Trading"], 7.2)
        self.assertAlmostEqual(amounts["Fees"], -2.2)
        self.assertAlmostEqual(amounts["Net perp PnL"], 7.2 - 2.2 - 1.0)

    def test_non_usdc_fee_is_not_subtracted_from_usdc_pnl(self):
        fills = pd.DataFrame(
            {"closedPnl": [5.0, 5.0, 5.0], "fee": [1.0, 1.0, float("nan")], "feeToken": ["USDC", "HYPE", "USDC"]}
        )
        self.assertEqual(apply_fill_fees(fills)["closedPnl"].tolist(), [4.0, 5.0, 5.0])

    def test_fresh_data_replaces_saved_rows_but_keeps_older_and_other_sources(self):
        day = lambda text: pd.Timestamp(text, tz="America/New_York")
        saved = pd.DataFrame(
            {
                "ts": [day("2026-01-01"), day("2026-01-05"), day("2026-01-05")],
                "source": ["funding_daily", "funding_daily", "perp_realized_daily"],
                "realized_pnl": [1.0, 12.8, 2.0],
            }
        )
        days = pd.Series([day("2026-01-03")])
        kept = drop_refreshed_daily_rows(saved, ["funding_daily"], days)
        self.assertEqual(kept["realized_pnl"].tolist(), [1.0, 2.0])
        self.assertTrue(drop_refreshed_daily_rows(saved, ["funding_daily"], days.iloc[0:0]).equals(saved))


class TimeWeightedIndexTests(unittest.TestCase):
    @staticmethod
    def history(values, pnl):
        times = pd.date_range("2026-01-01", periods=len(values), freq="D", tz="America/New_York")
        return pd.DataFrame({"ts": times, "accountValue": values, "pnl": pnl})

    def test_deposit_does_not_count_as_return(self):
        index = time_weighted_index([self.history([0, 100, 110, 1110, 1221], [0, 0, 10, 10, 121])])
        self.assertEqual(len(index), 5)
        self.assertAlmostEqual(index.iloc[2], 1.10)
        self.assertAlmostEqual(index.iloc[3], 1.10)
        self.assertAlmostEqual(index.iloc[4], 1.21)

    def test_withdrawal_does_not_count_as_loss(self):
        index = time_weighted_index([self.history([100, 110, 60], [0, 10, 10])])
        self.assertAlmostEqual(index.iloc[-1], 1.10)

    def test_pnl_in_a_deposit_interval_uses_the_larger_balance(self):
        index = time_weighted_index([self.history([100, 1110], [0, 10])])
        self.assertAlmostEqual(index.iloc[-1], 1 + 10 / 1100)

    def test_transfer_between_wallets_cancels(self):
        one = self.history([100, 50, 55], [0, 0, 5])
        two = self.history([100, 150, 150], [0, 0, 0])
        self.assertAlmostEqual(time_weighted_index([one, two]).iloc[-1], 1 + 5 / 200)

    def test_wallets_with_different_timestamps_are_aligned(self):
        one = self.history([100, 110], [0, 10])
        two = self.history([200, 200], [0, 0])
        two["ts"] = two["ts"] + pd.Timedelta(hours=6)
        self.assertAlmostEqual(time_weighted_index([one, two]).iloc[-1], 1 + 10 / 300)

    def test_no_funded_history_returns_empty(self):
        self.assertTrue(time_weighted_index([]).empty)
        self.assertTrue(time_weighted_index([self.history([0, 0], [0, 0])]).empty)


class InactivePeriodTests(unittest.TestCase):
    def test_inactive_periods_finds_only_gaps_over_threshold(self):
        days = pd.to_datetime(["2026-01-01", "2026-01-31", "2026-03-03", "2026-03-04"])
        self.assertEqual(
            inactive_periods(days),
            [(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-03"))],
        )

    def test_inactive_periods_handles_timezones_duplicates_and_no_data(self):
        days = pd.DatetimeIndex(
            ["2026-06-01 09:00", "2026-01-01 09:00", "2026-01-01 15:00"], tz="America/New_York"
        )
        self.assertEqual(
            inactive_periods(days),
            [(pd.Timestamp("2026-01-02"), pd.Timestamp("2026-06-01"))],
        )
        self.assertEqual(inactive_periods(pd.DatetimeIndex([])), [])

    def test_visible_month_starts_skips_hidden_months_and_stays_in_range(self):
        periods = [(pd.Timestamp("2025-08-26"), pd.Timestamp("2026-09-26"))]
        months = visible_month_starts(pd.Timestamp("2025-07-15"), pd.Timestamp("2026-11-02"), periods)
        self.assertEqual(
            [month.strftime("%Y-%m") for month in months],
            ["2025-08", "2026-10", "2026-11"],
        )

    def test_drop_inactive_weeks_keeps_weeks_touching_active_days(self):
        weeks = pd.date_range("2026-01-05", periods=5, freq="7D")
        returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=weeks)
        periods = [(pd.Timestamp("2026-01-12"), pd.Timestamp("2026-01-30"))]
        kept = drop_inactive_weeks(returns, periods)
        self.assertEqual(kept.tolist(), [1.0, 4.0, 5.0])
        self.assertTrue(drop_inactive_weeks(returns, []).equals(returns))


class TradeAnalyticsTests(unittest.TestCase):
    def test_net_funding_apr_history_matches_current_funding_formula(self):
        def event(day, usdc, rate, kind="funding"):
            time_ms = int(pd.Timestamp(day, tz="America/New_York").timestamp() * 1000)
            return {"time": time_ms, "delta": {"type": kind, "usdc": str(usdc), "fundingRate": str(rate)}}

        # A $1,000 short receiving 0.01%/h and a $1,000 long paying 0.005%/h.
        events = [
            event("2026-01-01 12:00", 0.1, 0.0001),
            event("2026-01-01 12:00", -0.05, 0.00005),
            event("2026-01-01 12:00", 0.0, 0.0),
            event("2026-01-01 12:00", 5, 0.1, kind="other"),
            event("2026-01-10 12:00", -0.1, 0.0001),
        ]
        history = net_funding_apr_history(events)

        summary = summarize_current_funding(
            [
                {"Coin": "A", "Side": "SHORT", "Pos": 1_000, "Sz": 1, "Funding (1h)": 0.01},
                {"Coin": "B", "Side": "LONG", "Pos": 1_000, "Sz": 1, "Funding (1h)": 0.005},
            ]
        )
        first = history.loc["2026-01-01"]
        self.assertAlmostEqual(first["apr_pct"], summary["Net Funding APR (%) (simple, covered)"])
        self.assertAlmostEqual(first["notional_hours"], 2_000)
        # Nine days apart, so the 7-day window does not reach back to the first event.
        self.assertAlmostEqual(history.loc["2026-01-10", "rolling_apr_pct"], -87.6)
        self.assertTrue(net_funding_apr_history([]).empty)

    def test_consolidated_current_funding_is_notional_weighted_and_side_aware(self):
        rows = [
            {"Coin": "BTC", "Side": "LONG", "Pos": 1000, "Sz": 10, "Funding (1h)": 0.01},
            {"Coin": "BTC", "Side": "LONG", "Pos": 3000, "Sz": 30, "Funding (1h)": 0.02},
            {"Coin": "BTC", "Side": "SHORT", "Pos": -2000, "Sz": -20, "Funding (1h)": 0.02},
        ]
        long_row, short_row = consolidate_current_funding(rows)
        self.assertEqual((long_row["Coin"], long_row["Side"]), ("BTC", "LONG"))
        self.assertAlmostEqual(long_row["Current Funding / 1h (%)"], 0.0175)
        self.assertAlmostEqual(long_row["Funding APR (%)"], 153.3)
        self.assertAlmostEqual(long_row["Estimated Funding PnL / hour ($)"], -0.7)
        self.assertAlmostEqual(long_row["Estimated Funding PnL / day ($)"], -16.8)
        self.assertEqual(long_row["Funding Coverage (%)"], 100)
        self.assertEqual((short_row["Coin"], short_row["Side"]), ("BTC", "SHORT"))
        self.assertAlmostEqual(short_row["Estimated Funding PnL / hour ($)"], 0.4)

    def test_consolidated_current_funding_reports_rate_coverage(self):
        rows = [
            {"Coin": "ETH", "Side": "LONG", "Pos": 250, "Sz": 1, "Funding (1h)": -0.01},
            {"Coin": "ETH", "Side": "LONG", "Pos": 750, "Sz": 3, "Funding (1h)": float("nan")},
        ]
        [result] = consolidate_current_funding(rows)
        self.assertEqual(result["Funding Coverage (%)"], 25)
        self.assertEqual(result["Estimated Funding PnL / hour ($)"], 0.025)

    def test_account_funding_summary_reports_direction_and_net_carry(self):
        rows = [
            {"Coin": "BTC", "Side": "LONG", "Pos": 1000, "Funding (1h)": 0.01},
            {"Coin": "ETH", "Side": "LONG", "Pos": 500, "Funding (1h)": -0.02},
            {"Coin": "SOL", "Side": "SHORT", "Pos": -1200, "Funding (1h)": 0.03},
        ]
        summary = summarize_current_funding(rows)
        self.assertEqual(summary["Net Direction"], "NET LONG")
        self.assertEqual(summary["Long Positions"], 2)
        self.assertEqual(summary["Short Positions"], 1)
        self.assertEqual(summary["Long Notional ($)"], 1500)
        self.assertEqual(summary["Short Notional ($)"], 1200)
        self.assertEqual(summary["Net Notional ($)"], 300)
        self.assertEqual(summary["Gross Notional ($)"], 2700)
        self.assertAlmostEqual(summary["Estimated Net Funding PnL / hour ($)"], 0.36)
        self.assertAlmostEqual(summary["Estimated Net Funding PnL / day ($)"], 8.64)
        self.assertAlmostEqual(
            summary["Net Funding Rate / 1h (%) (covered)"],
            0.36 / 2700 * 100,
        )
        self.assertEqual(summary["Funding Coverage (%)"], 100)

    def test_account_funding_summary_exposes_missing_rate_coverage(self):
        summary = summarize_current_funding(
            [
                {"Coin": "BTC", "Side": "LONG", "Pos": 1000, "Funding (1h)": 0.01},
                {"Coin": "ETH", "Side": "SHORT", "Pos": -1000, "Funding (1h)": float("nan")},
            ]
        )
        self.assertEqual(summary["Net Direction"], "FLAT")
        self.assertEqual(summary["Funding-Covered Notional ($)"], 1000)
        self.assertEqual(summary["Funding Coverage (%)"], 50)

    def test_info_request_retries_rate_limit_using_retry_after(self):
        rate_limited = unittest.mock.Mock(
            status_code=429,
            ok=False,
            headers={"Retry-After": "3"},
        )
        successful = unittest.mock.Mock(status_code=200, ok=True, json=lambda: {"ok": True})
        with (
            patch("hl_trade_analytics.requests.post", side_effect=[rate_limited, successful]),
            patch("hl_trade_analytics.time.sleep") as sleep,
        ):
            self.assertEqual(post_info({"type": "meta"}), {"ok": True})
        sleep.assert_called_once_with(3.0)

    def test_info_request_reports_rate_limit_after_retry_limit(self):
        rate_limited = unittest.mock.Mock(status_code=429, ok=False, headers={})
        with (
            patch("hl_trade_analytics.requests.post", return_value=rate_limited),
            patch("hl_trade_analytics.time.sleep"),
            self.assertRaisesRegex(RuntimeError, r"HTTP 429"),
        ):
            post_info({"type": "meta"})

    def test_side_direction_is_explicit(self):
        self.assertEqual(side_to_direction("long"), 1)
        self.assertEqual(side_to_direction("SHORT"), -1)
        with self.assertRaises(ValueError):
            side_to_direction("hold")

    def test_funding_stress_is_directional_and_requires_contiguous_hours(self):
        long_funding = make_funding(rate=0.0001)
        short_funding = make_funding(rate=-0.0001)
        self.assertAlmostEqual(
            adverse_funding_cost_per_unit(long_funding, "LONG", 100),
            1.2,
        )
        self.assertAlmostEqual(
            adverse_funding_cost_per_unit(short_funding, "SHORT", 100),
            1.2,
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            rolling_funding_sums(make_funding(count=130).drop(index=50), 120)

    def test_contiguous_funding_accepts_exchange_timestamp_jitter(self):
        funding = make_funding(count=130)
        jitter = np.where(np.arange(len(funding)) % 2, 500, -500)
        funding["time"] += pd.to_timedelta(jitter, unit="ms")
        self.assertFalse(rolling_funding_sums(funding, 120).empty)

    def test_hip3_history_requests_keep_dex_qualified_market_symbol(self):
        candle_time = int((pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)).timestamp() * 1000)
        funding_time = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        responses = [
            [{"t": candle_time, "o": "100", "h": "102", "l": "99", "c": "101"}],
            [{"time": funding_time, "fundingRate": "0.0001"}],
        ]
        with patch("hl_trade_analytics.post_info", side_effect=responses) as post:
            fetch_daily_candles("AAPL", dex="xyz")
            fetch_funding_history("AAPL", dex="xyz")
        self.assertEqual(post.call_args_list[0].args[0]["req"]["coin"], "xyz:AAPL")
        self.assertEqual(post.call_args_list[1].args[0]["coin"], "xyz:AAPL")

    def test_atr_and_forward_expected_pnl(self):
        candles = make_candles()
        funding = make_funding(rate=-0.00001)
        self.assertGreater(wilder_atr(candles, period=14), 0)
        mean_return, observations = expected_forward_return(candles["close"], 5)
        self.assertAlmostEqual(mean_return, 1.01**5 - 1, places=10)
        self.assertEqual(observations, len(candles) - 5)
        expected = expected_position_pnl(candles, funding, "LONG", 2, 100, 100, 5)
        self.assertAlmostEqual(expected.price_pnl, 200 * mean_return)
        self.assertGreater(expected.funding_pnl, 0)
        self.assertAlmostEqual(expected.net_pnl, expected.price_pnl + expected.funding_pnl)

    def test_position_rows_receive_expected_price_and_funding_columns(self):
        row = {
            "Coin": "BTC",
            "Dex": "",
            "Side": "LONG",
            "Sz": 2.0,
            "Mark": 100.0,
            "Oracle": 100.0,
        }
        with (
            patch("hl_trade_analytics.fetch_daily_candles", return_value=make_candles()),
            patch("hl_trade_analytics.fetch_funding_history", return_value=make_funding(rate=-0.00001)),
        ):
            enriched, warnings = add_expected_pnl_columns([row])
        self.assertFalse(warnings)
        self.assertIn("Expected Price PnL (5d)", enriched[0])
        self.assertIn("Expected Funding PnL (5d)", enriched[0])
        self.assertIn("Expected PnL incl. funding (5d)", enriched[0])
        notional = abs(float(row["Sz"])) * float(row["Mark"])
        self.assertAlmostEqual(
            enriched[0]["Expected Price Return (5d)"],
            enriched[0]["Expected Price PnL (5d)"] / notional * 100.0,
        )
        self.assertAlmostEqual(
            enriched[0]["Expected Funding Return (5d)"],
            enriched[0]["Expected Funding PnL (5d)"] / notional * 100.0,
        )

    def test_implied_unrealized_subtracts_realized_up_to_each_day(self):
        pnl = pd.DataFrame({
            "ts": pd.to_datetime(["2026-01-01 12:00", "2026-01-02 06:00", "2026-01-02 18:00"], utc=True),
            "pnl": [10.0, 40.0, 30.0],
        })
        fills = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01 13:00"], utc=True), "closedPnl": [25.0]})
        funding = pd.DataFrame({"ts": pd.to_datetime(["2026-01-02 01:00"], utc=True), "fundingPnl": [-5.0]})
        implied = implied_unrealized_history(pnl, fills, funding)
        self.assertEqual(implied.tolist(), [10.0, 10.0])
        mixed_precision = pnl.assign(ts=pnl["ts"].astype("datetime64[ms, UTC]"))
        self.assertEqual(
            implied_unrealized_history(mixed_precision, fills, funding).tolist(), [10.0, 10.0]
        )

    def test_market_research_matches_app_formulas_and_tabs(self):
        def context(mid, previous="100", volume="1000"):
            return {"midPx": mid, "prevDayPx": previous, "markPx": "110", "funding": "0.0001",
                    "openInterest": "2", "dayNtlVlm": volume}
        responses = {
            "perpCategories": [["xyz:GOLD", "Commodities"], ["xyz:BTC", "crypto"]],
            "perpDexs": [None, {"name": "xyz"}],
        }
        metas = {
            "": ([{"name": "BTC"}, {"name": "OLD", "isDelisted": True}, {"name": "NOPX"}],
                 [context("110"), context("1"), context(None)]),
            "xyz": ([{"name": "xyz:GOLD"}, {"name": "xyz:BTC"}], [context("90"), context("95")]),
        }
        def post(payload):
            if payload["type"] == "metaAndAssetCtxs":
                meta, contexts = metas[payload["dex"]]
                return [{"universe": meta}, contexts]
            return responses[payload["type"]]
        with patch("hl_market_research.post_info", side_effect=post):
            markets = fetch_perp_markets()
        btc = markets.set_index("Ticker").loc["BTC"]
        self.assertAlmostEqual(btc[CHANGE], 10.0)
        self.assertAlmostEqual(btc[FUNDING], 0.08)
        self.assertAlmostEqual(btc[OPEN_INTEREST], 220.0)
        self.assertEqual(select_view(markets, "crypto")["Ticker"].tolist(), ["BTC", "NOPX"])
        self.assertEqual(select_view(markets, "hip3")["Ticker"].tolist(), ["xyz:GOLD", "xyz:BTC"])
        self.assertEqual(select_view(markets, "tradfi")["Ticker"].tolist(), ["xyz:GOLD"])
        self.assertEqual(rank_markets(markets, CHANGE, top=False)["Ticker"].tolist()[0], "xyz:GOLD")
        self.assertNotIn("NOPX", rank_markets(markets, CHANGE, top=True)["Ticker"].tolist())

    def test_trending_lists_biggest_loser_and_three_gainers_above_volume_floor(self):
        markets = pd.DataFrame({
            "Ticker": ["A", "B", "C", "D", "E", "THIN"],
            CHANGE: [-9.0, -1.0, 1.0, 2.0, 3.0, 50.0],
            VOLUME: [4e7, 4e7, 4e7, 4e7, 4e7, 2e7],
        })
        self.assertEqual(trending_markets(markets)["Ticker"].tolist(), ["E", "D", "C", "A"])

    def test_highlights_split_good_news_from_items_needing_attention(self):
        now = pd.Timestamp("2026-01-10 12:00", tz="UTC")
        def position(coin, size, upnl, liquidation=80.0):
            return {"Wallet": "w", "Coin": coin, "Side": "SHORT", "Sz": size, "uPnL": upnl, "Liq %": liquidation}
        baseline = make_state(now - pd.Timedelta(hours=2), 100.0,
                              [position("KEEP", -10, 5.0), position("GONE", -3, 1.0), position("CUT", -9, 2.0)])
        positions = [position("KEEP", -10, 9.0, 20.0), position("CUT", -3, -4.0), position("NEW", -1, 0.0)]
        fills = pd.DataFrame({
            "ts": [now - pd.Timedelta(hours=1), now - pd.Timedelta(hours=1), now - pd.Timedelta(hours=1),
                   now - pd.Timedelta(hours=30)],
            "coin": ["GONE", "CUT", "NEW", "OLD"],
            "closedPnl": [8.0, -3.0, -0.1, 50.0],
            "grossPnl": [8.5, -2.5, 0.0, 50.5],
        })
        funding = pd.DataFrame({"ts": [now - pd.Timedelta(hours=1)], "fundingPnl": [-2.0]})
        well, attention = build_highlights(
            now, baseline, positions, fills, funding, 90.0, 12.0, 1000.0, 700.0, 600.0, 5.0, 100.0, ["NEW (w)"],
        )
        updates = lambda rows, window: {row["Update"] for row in rows if row["Window"] == window}
        self.assertEqual(updates(well, "Last 24h"), {"PnL including open positions", "Closed at a profit"})
        self.assertIn("Closed at a loss", updates(attention, "Since last run"))
        self.assertIn("PnL including open positions", updates(attention, "Since last run"))
        self.assertIn("Funding paid", updates(attention, "Last 24h"))
        self.assertIn("Biggest open-position gains", updates(well, "Since last run"))
        self.assertIn("Biggest open-position losses", updates(attention, "Since last run"))
        self.assertEqual(updates(attention, "Now"), {
            "Aggregate risk over cap", "Positions missing full stop coverage",
            "Available margin low", "Within 25% of liquidation",
        })
        self.assertFalse(any("NEW" in row["Detail"] for row in attention if row["Update"] == "Closed at a loss"))
        self.assertFalse(any("OLD" in row["Detail"] for row in well if row["Window"] == "Since last run"))
        changes = position_changes(baseline, positions)
        self.assertEqual(changes["Opened"], ["NEW short (w)"])
        self.assertEqual(changes["Closed"], ["GONE short (w)"])
        self.assertEqual(changes["Resized"], ["CUT short (w)"])

    def test_asset_class_gives_each_market_one_group(self):
        self.assertEqual(asset_class("", None), "Crypto")
        self.assertEqual(asset_class(None, None), "Crypto")
        self.assertEqual(asset_class("xyz", "Stocks"), "Tradfi")
        self.assertEqual(asset_class("flx", "crypto"), "HIP-3")
        self.assertEqual(asset_class("xyz", None), "HIP-3")

    def test_cached_candles_and_funding_download_only_what_is_new(self):
        now = pd.Timestamp.now(tz="UTC").floor("h")
        today = now.normalize()
        def candle(day):
            return {"t": int(day.timestamp() * 1000), "o": "1", "h": "2", "l": "1", "c": "1.5"}
        def post(payload):
            calls.append(payload)
            if payload["type"] == "candleSnapshot":
                low, high = payload["req"]["startTime"], payload["req"]["endTime"]
                days = pd.date_range(today - pd.Timedelta(days=60), today, freq="D")
                return [candle(day) for day in days if low <= day.timestamp() * 1000 <= high]
            hours = pd.date_range(now - pd.Timedelta(days=40), now, freq="h")
            return [
                {"coin": "BTC", "fundingRate": "0.0001", "premium": "0", "time": int(hour.timestamp() * 1000)}
                for hour in hours if payload["startTime"] <= hour.timestamp() * 1000 <= payload["endTime"]
            ]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            hl_trade_analytics, "CACHE_DIR", Path(directory)
        ), patch("hl_trade_analytics.post_info", side_effect=post):
            calls = []
            first = hl_trade_analytics.fetch_daily_candles("BTC", lookback_days=20)
            self.assertEqual(len(calls), 1)
            pd.testing.assert_frame_equal(first, hl_trade_analytics.fetch_daily_candles("BTC", lookback_days=20))
            self.assertEqual(len(calls), 1)
            self.assertEqual(first["time"].max(), today - pd.Timedelta(days=1))
            self.assertLess(len(hl_trade_analytics.fetch_daily_candles("BTC", lookback_days=10)), len(first))
            self.assertEqual(len(calls), 1)
            longer = hl_trade_analytics.fetch_daily_candles("BTC", lookback_days=40)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[-1]["req"]["startTime"] < calls[0]["req"]["startTime"], True)
            self.assertGreater(len(longer), len(first))

            calls.clear()
            funding = hl_trade_analytics.fetch_funding_history("BTC", lookback_days=5)
            self.assertEqual(len(calls), 1)
            again = hl_trade_analytics.fetch_funding_history("BTC", lookback_days=5)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(again), len(funding))
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(again["time"]))

    def test_size_respects_zero_risk_headroom_and_hip3_leverage_cap(self):
        candles = make_candles()
        funding = make_funding()
        market = Market("xyz:AAPL", "xyz", 100, 100, 2, 10)
        blocked = size_trade(
            "xyz:AAPL", "xyz", "LONG", market, candles, funding,
            equity=1_000, risk_headroom=0, available_margin=500,
            maker_fee_rate=0.0001, taker_fee_rate=0.0004,
        )
        self.assertEqual(blocked.quantity, 0)
        self.assertEqual(blocked.leverage, 3)
        self.assertIn("20% equity cap", blocked.block_reason)

    def test_position_risk_budget_and_margin_limit_scale_quantity(self):
        candles = make_candles()
        funding = make_funding()
        market = Market("BTC", "", 100, 100, 3, 10)
        planned = size_trade(
            "BTC", "", "LONG", market, candles, funding,
            equity=1_000, risk_headroom=200, available_margin=10,
            maker_fee_rate=0.0001, taker_fee_rate=0.0004,
        )
        self.assertLessEqual(planned.risk_budget, 40)
        self.assertLessEqual(planned.quantity * market.mark_price / planned.leverage, 10)

    def test_post_only_entry_price_never_crosses_book(self):
        book = {
            "levels": [
                [{"px": "100", "sz": "5", "n": 1}],
                [{"px": "102", "sz": "5", "n": 1}],
            ]
        }
        with patch("hl_trade_execution.post_info", return_value=book):
            buy_price = post_only_price("BTC", "", "LONG", 3)
            sell_price = post_only_price("BTC", "", "SHORT", 3)
        self.assertLess(buy_price, 102)
        self.assertGreater(sell_price, 100)

    def test_order_prices_obey_five_significant_digits_and_market_decimal_cap(self):
        rounded = _round_price(1.80148, size_decimals=1)
        self.assertEqual(rounded, 1.8015)
        self.assertAlmostEqual(rounded / _price_tick(rounded, 1), round(rounded / _price_tick(rounded, 1)))
        self.assertEqual(_round_price(123456, size_decimals=0), 123460)

    def test_stop_trigger_price_is_numeric_for_sdk_serialization(self):
        class FakeExchange:
            def order(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return {"response": {"data": {"statuses": [{"resting": {"oid": 123}}]}}}

        exchange = FakeExchange()
        oid = _submit_protection_order(
            exchange,
            "BTC",
            "LONG",
            0.1,
            95.0,
            "stop",
        )
        self.assertEqual(oid, 123)
        order_type = exchange.args[4]
        self.assertEqual(order_type["trigger"]["triggerPx"], 95.0)
        self.assertIsInstance(order_type["trigger"]["triggerPx"], float)

    def test_take_profit_protection_order_is_post_only_and_reduce_only(self):
        class FakeExchange:
            def order(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return {"response": {"data": {"statuses": [{"resting": {"oid": 124}}]}}}

        exchange = FakeExchange()
        oid = _submit_protection_order(
            exchange,
            "BTC",
            "SHORT",
            1.0,
            90.0,
            "take-profit",
        )

        self.assertEqual(oid, 124)
        self.assertTrue(exchange.args[1])
        self.assertEqual(exchange.args[4], {"limit": {"tif": "Alo"}})
        self.assertTrue(exchange.kwargs["reduce_only"])

    def test_protection_ladder_splits_size_and_ends_at_mark_anchored_atr_levels(self):
        market = Market("BTC", "", 100.0, 100.0, 2, 10)
        position = Position(
            wallet="0x" + "1" * 40,
            coin="BTC",
            dex="",
            quantity=-9.0,
            entry_price=110.0,
            mark_price=100.0,
            oracle_price=100.0,
            margin_used=50.0,
            market=market,
            leverage=3,
        )
        open_orders = [
            {
                "coin": "BTC",
                "side": "B",
                "orderType": "Stop Limit",
                "triggerPx": "130",
                "reduceOnly": False,
                "oid": 101,
            },
            {
                "coin": "BTC",
                "side": "B",
                "orderType": "Limit",
                "limitPx": "80",
                "reduceOnly": True,
                "oid": 102,
            },
        ]

        plan = plan_protection_ladder(position, open_orders, make_candles(30))

        self.assertEqual(plan.side, "SHORT")
        self.assertEqual(plan.existing_stop_order_ids, (101,))
        self.assertEqual(plan.existing_target_order_ids, (102,))
        self.assertEqual(len(plan.stop_steps), 3)
        self.assertEqual(len(plan.target_steps), 3)
        self.assertAlmostEqual(sum(step.quantity for step in plan.stop_steps), 9.0)
        self.assertAlmostEqual(sum(step.quantity for step in plan.target_steps), 9.0)
        self.assertEqual(
            [step.quantity for step in plan.stop_steps],
            [step.quantity for step in plan.target_steps],
        )
        self.assertGreater(plan.stop_steps[0].price, position.mark_price)
        self.assertGreater(plan.stop_steps[-1].price, plan.stop_steps[0].price)
        self.assertLess(plan.target_steps[0].price, position.mark_price)
        self.assertLess(plan.target_steps[-1].price, plan.target_steps[0].price)
        self.assertEqual(
            plan.stop_steps[-1].price,
            _round_price(position.mark_price + plan.atr * 1.25, market.size_decimals),
        )
        self.assertEqual(
            plan.target_steps[-1].price,
            _round_price(position.mark_price - plan.atr * 1.25, market.size_decimals),
        )

    def test_protection_ladder_reduces_steps_to_meet_exchange_minimum_order_value(self):
        scale = 0.03
        candles = make_candles(30)
        for column in ("open", "high", "low", "close"):
            candles[column] *= scale
        market = Market("LIT", "", 3.12, 3.12, 0, 10)
        position = Position(
            wallet="0x" + "1" * 40,
            coin="LIT",
            dex="",
            quantity=-9.0,
            entry_price=3.25,
            mark_price=3.12,
            oracle_price=3.12,
            margin_used=10.0,
            market=market,
            leverage=3,
        )

        plan = plan_protection_ladder(position, [], candles)

        self.assertEqual(len(plan.stop_steps), 2)
        self.assertEqual(len(plan.target_steps), 2)
        self.assertAlmostEqual(sum(step.quantity for step in plan.stop_steps), 9.0)
        self.assertAlmostEqual(sum(step.quantity for step in plan.target_steps), 9.0)
        self.assertTrue(all(step.quantity * step.price >= 10.0 for step in plan.stop_steps))
        self.assertTrue(all(step.quantity * step.price >= 10.0 for step in plan.target_steps))

    def test_protection_ladder_quantities_are_exact_at_size_precision(self):
        candles = make_candles(30)
        for column in ("open", "high", "low", "close"):
            candles[column] *= 0.0001
        market = Market("USUAL", "", 0.01, 0.01, 1, 3)
        position = Position(
            wallet="0x" + "1" * 40,
            coin="USUAL",
            dex="",
            quantity=-21449.8,
            entry_price=0.01,
            mark_price=0.01,
            oracle_price=0.01,
            margin_used=72.0,
            market=market,
            leverage=3,
        )

        plan = plan_protection_ladder(position, [], candles)

        for step in (*plan.stop_steps, *plan.target_steps):
            self.assertEqual(step.quantity, round(step.quantity, 1))
        self.assertAlmostEqual(sum(step.quantity for step in plan.stop_steps), 21449.8)

    def _run_entry(self, statuses, responses=None):
        exchange = MagicMock()
        exchange.order.side_effect = responses or [
            {"response": {"data": {"statuses": [{"resting": {"oid": index}}]}}}
            for index in range(1, 10)
        ]
        plan = SimpleNamespace(coin="USUAL", dex="", side="SHORT", quantity=21449.7)
        market = Market("USUAL", "", 0.01, 0.01, 1, 3)
        module = "hl_trade_execution"
        with patch(f"{module}.post_only_price", return_value=0.01), \
                patch(f"{module}.time.sleep"), \
                patch(f"{module}._order_status", side_effect=statuses):
            return exchange, _place_entry(exchange, "0x" + "1" * 40, plan, market)

    @staticmethod
    def _order_status_response(state, original, remaining):
        return {
            "status": "order",
            "order": {"order": {"origSz": str(original), "sz": str(remaining)}, "status": state},
        }

    def test_entry_chase_places_only_the_unfilled_remainder_after_partial_fill(self):
        status = self._order_status_response
        exchange, (quantity, entry) = self._run_entry([
            status("open", 21449.7, 16992.5),
            status("canceled", 21449.7, 16992.5),
            status("filled", 16992.5, 0),
        ])

        self.assertEqual([call.args[2] for call in exchange.order.call_args_list], [21449.7, 16992.5])
        self.assertAlmostEqual(quantity, 21449.7)
        self.assertAlmostEqual(entry, 0.01)

    def test_entry_chase_does_not_replace_order_when_cancel_is_unconfirmed(self):
        status = self._order_status_response
        exchange, (quantity, _) = self._run_entry([status("open", 21449.7, 16992.5)] * 31)

        self.assertEqual(exchange.order.call_count, 1)
        self.assertAlmostEqual(quantity, 4457.2)

    def test_entry_chase_waits_out_status_lag_after_cancel(self):
        status = self._order_status_response
        exchange, (quantity, _) = self._run_entry([
            status("open", 21449.7, 16992.5),
            status("open", 21449.7, 16992.5),
            status("canceled", 21449.7, 16992.5),
            status("filled", 16992.5, 0),
        ])

        self.assertEqual([call.args[2] for call in exchange.order.call_args_list], [21449.7, 16992.5])
        self.assertAlmostEqual(quantity, 21449.7)

    def test_entry_chase_returns_earlier_fills_when_a_reprice_is_rejected(self):
        status = self._order_status_response
        exchange, (quantity, _) = self._run_entry(
            [status("open", 21449.7, 16992.5), status("canceled", 21449.7, 16992.5)],
            [
                {"response": {"data": {"statuses": [{"resting": {"oid": 1}}]}}},
                {"response": {"data": {"statuses": [{"error": "Post only would have matched"}]}}},
            ],
        )

        self.assertEqual(exchange.order.call_count, 2)
        self.assertAlmostEqual(quantity, 4457.2)

    def test_fetch_markets_does_not_duplicate_dex_prefix(self):
        response = [
            {"universe": [{"name": "xyz:AAPL", "szDecimals": 2, "maxLeverage": 3}]},
            [{"markPx": "100", "oraclePx": "99"}],
        ]
        with patch("hl_trade_execution.post_info", return_value=response):
            markets = fetch_markets(["xyz"])
        self.assertIn("xyz:AAPL", markets)
        self.assertEqual(markets["xyz:AAPL"].dex, "xyz")
        self.assertNotIn("xyz:xyz:AAPL", markets)

    def test_fetch_open_orders_collects_default_and_dex_orders(self):
        wallet = "0x" + "1" * 40
        default_order = {"oid": 1, "coin": "BTC"}
        xyz_order = {"oid": 2, "coin": "xyz:AAPL"}
        responses = [[default_order], [xyz_order]]

        with (
            patch("hl_trade_execution.get_dexes", return_value=["", "xyz"]),
            patch("hl_trade_execution.post_info", side_effect=responses) as post_info_mock,
        ):
            orders = fetch_open_orders(wallet)

        self.assertEqual(orders, [default_order, xyz_order])
        self.assertEqual(
            [call.args[0] for call in post_info_mock.call_args_list],
            [
                {"type": "frontendOpenOrders", "user": wallet},
                {"type": "frontendOpenOrders", "user": wallet, "dex": "xyz"},
            ],
        )

    def test_open_position_protection_only_adds_uncovered_exit_quantity(self):
        market = Market("BTC", "", 100, 100, 3, 10)
        position = Position(
            wallet="0x" + "1" * 40,
            coin="BTC",
            dex="",
            quantity=1.0,
            entry_price=100.0,
            mark_price=100.0,
            oracle_price=100.0,
            margin_used=50.0,
            market=market,
        )
        orders = [
            {
                "coin": "BTC",
                "oid": 1,
                "side": "A",
                "reduceOnly": True,
                "orderType": "Stop Market",
                "triggerPx": "95",
                "sz": "0.4",
            },
            {
                "coin": "BTC",
                "oid": 2,
                "side": "A",
                "reduceOnly": True,
                "orderType": "Limit",
                "limitPx": "105",
                "sz": "0.25",
            },
            {
                "coin": "BTC",
                "oid": 3,
                "side": "A",
                "reduceOnly": False,
                "orderType": "Limit",
                "triggerPx": "0.0",
                "sz": "1",
            },
        ]
        plan = plan_position_protection(position, orders, make_candles(), make_funding())
        self.assertAlmostEqual(plan.stop_quantity_to_add, 0.6)
        self.assertAlmostEqual(plan.target_quantity_to_add, 0.75)
        self.assertEqual(plan.stop_order_ids, (1,))
        self.assertEqual(plan.target_order_ids, (2,))
        self.assertLess(plan.stop_price, position.entry_price)
        self.assertGreater(plan.target_price, position.entry_price)

    def test_existing_stop_limit_and_reduce_only_target_cover_position(self):
        market = Market("BTC", "", 100, 100, 3, 10)
        position = Position(
            wallet="0x" + "1" * 40,
            coin="BTC",
            dex="",
            quantity=-1.0,
            entry_price=100,
            mark_price=100,
            oracle_price=100,
            margin_used=50,
            market=market,
        )
        orders = [
            {
                "coin": "BTC",
                "oid": 10,
                "side": "B",
                "reduceOnly": False,
                "orderType": "Stop Limit",
                "triggerPx": "110",
                "sz": "1",
            },
            {
                "coin": "BTC",
                "oid": 11,
                "side": "B",
                "reduceOnly": True,
                "orderType": "Limit",
                "triggerPx": "0.0",
                "limitPx": "90",
                "sz": "1",
            },
        ]
        plan = plan_position_protection(position, orders, make_candles(), make_funding())
        self.assertEqual(plan.stop_quantity_to_add, 0)
        self.assertEqual(plan.target_quantity_to_add, 0)
        self.assertEqual(plan.stop_order_ids, (10,))
        self.assertEqual(plan.target_order_ids, (11,))

    def test_protection_plan_blocks_only_position_with_insufficient_history(self):
        market_a = Market("BTC", "", 100, 100, 3, 10)
        market_b = Market("JIO", "xyz", 100, 100, 3, 10)
        positions = (
            Position("0x" + "1" * 40, "BTC", "", 1, 100, 100, 100, 50, market_a),
            Position("0x" + "1" * 40, "xyz:JIO", "xyz", 1, 100, 100, 100, 50, market_b),
        )
        with (
            patch(
                "hl_trade_execution.fetch_daily_candles",
                side_effect=[make_candles(), ValueError("ATR(14) requires more history")],
            ),
            patch(
                "hl_trade_execution.fetch_funding_history",
                return_value=make_funding(),
            ),
        ):
            plans = plan_open_position_protection(positions, [])
        self.assertEqual(len(plans), 2)
        self.assertFalse(plans[0].stop_block_reason)
        self.assertFalse(plans[0].target_block_reason)
        self.assertIn("requires more history", plans[1].stop_block_reason)
        self.assertIn("requires more history", plans[1].target_block_reason)
        self.assertEqual(plans[1].stop_quantity_to_add, 1)

    def test_open_position_protection_plans_full_size_when_exits_are_absent(self):
        market = Market("BTC", "", 100, 100, 3, 10)
        position = Position(
            wallet="0x" + "1" * 40,
            coin="BTC",
            dex="",
            quantity=-0.5,
            entry_price=100.0,
            mark_price=100.0,
            oracle_price=100.0,
            margin_used=25.0,
            market=market,
        )
        plan = plan_position_protection(position, [], make_candles(), make_funding())
        self.assertEqual(plan.side, "SHORT")
        self.assertAlmostEqual(plan.stop_quantity_to_add, 0.5)
        self.assertAlmostEqual(plan.target_quantity_to_add, 0.5)
        self.assertGreater(plan.stop_price, position.entry_price)
        self.assertLess(plan.target_price, position.entry_price)
        self.assertFalse(plan.stop_block_reason)
        self.assertFalse(plan.target_block_reason)

    def test_adverse_funding_tightens_exchange_stop_distance(self):
        market = Market("BTC", "", 100, 100, 3, 10)
        position = Position(
            wallet="0x" + "1" * 40,
            coin="BTC",
            dex="",
            quantity=1.0,
            entry_price=100,
            mark_price=100,
            oracle_price=100,
            margin_used=25,
            market=market,
        )
        no_funding = plan_position_protection(position, [], make_candles(), make_funding(rate=0))
        adverse_funding = plan_position_protection(
            position, [], make_candles(), make_funding(rate=0.0001)
        )
        self.assertGreater(adverse_funding.stop_price, no_funding.stop_price)

    def test_reduce_only_limit_beyond_mark_counts_as_target_on_losing_position(self):
        market = Market("BTC", "", 80, 80, 3, 10)
        position = Position(
            wallet="0x" + "1" * 40,
            coin="BTC",
            dex="",
            quantity=3,
            entry_price=100,
            mark_price=80,
            oracle_price=80,
            margin_used=80,
            market=market,
        )
        def limit(oid, price):
            return {
                "oid": oid,
                "coin": "BTC",
                "side": "A",
                "sz": "1",
                "limitPx": str(price),
                "orderType": "Limit",
                "reduceOnly": True,
            }

        orders = [limit(1, 85), limit(2, 90), limit(3, 75)]
        _, targets = _matching_protection_orders(position, orders)
        self.assertEqual([order["oid"] for order in targets], [1, 2])

    def _snapshot_with_abstraction(self, abstraction):
        wallet = "0x" + "1" * 40
        state = {
            "marginSummary": {"accountValue": "400", "totalMarginUsed": "390"},
            "assetPositions": [],
        }
        balances = {
            "balances": [
                {"coin": "USDC", "total": "1000", "hold": "390"},
                {"coin": "HYPE", "total": "10", "hold": "0"},
            ]
        }
        with (
            patch("hl_trade_execution.fetch_user_state", return_value=state),
            patch("hl_trade_execution.parse_fee_rates", return_value=(0.0001, 0.0004)),
            patch("hl_trade_analytics.post_info", side_effect=[abstraction, balances]),
        ):
            return collect_snapshot([wallet], [""], {})

    def test_unified_account_equity_is_usdc_collateral_without_double_counting(self):
        snapshot = self._snapshot_with_abstraction("unifiedAccount")
        self.assertEqual(snapshot.equity, 1_000)
        self.assertEqual(snapshot.target_equity, 1_000)
        self.assertEqual(snapshot.margin_used, 390)

    def test_non_unified_account_equity_stays_perp_account_value(self):
        snapshot = self._snapshot_with_abstraction("default")
        self.assertEqual(snapshot.equity, 400)
        self.assertEqual(snapshot.target_equity, 400)

    def test_unified_account_rejects_invalid_usdc_balance(self):
        responses = ["unifiedAccount", {"balances": [{"coin": "USDC", "total": "bad"}]}]
        with patch("hl_trade_analytics.post_info", side_effect=responses):
            with self.assertRaises(ValueError):
                fetch_collateral_equity("0x" + "1" * 40, 400)

    def test_target_action_uses_per_position_risk_cap(self):
        market = Market("BTC", "", 100, 100, 3, 10)
        snapshot = AccountSnapshot(
            equity=1_000,
            margin_used=0,
            target_equity=1_000,
            target_margin_used=0,
            positions=(),
            maker_fee_rate=0.0001,
            taker_fee_rate=0.0004,
            risk_wallets=("0x" + "1" * 40,),
        )
        action = PositionAction("open_target", "BTC", "", "LONG")
        with (
            patch("hl_trade_execution.fetch_daily_candles", return_value=make_candles()),
            patch("hl_trade_execution.fetch_funding_history", return_value=make_funding()),
        ):
            [plan] = plan_position_actions([action], {"BTC": market}, snapshot, [])
        self.assertFalse(plan.block_reason)
        self.assertGreater(plan.target_quantity, 0)
        self.assertLessEqual(plan.entry_plan.risk_budget, 40)
        self.assertAlmostEqual(plan.quantity_delta, plan.target_quantity)

    def test_target_action_reduces_oversized_existing_position_to_protocol_cap(self):
        market = Market("BTC", "", 100, 100, 3, 10)
        position = Position(
            wallet="0x" + "1" * 40,
            coin="BTC",
            dex="",
            quantity=30,
            entry_price=100,
            mark_price=100,
            oracle_price=100,
            margin_used=750,
            market=market,
            leverage=4,
        )
        snapshot = AccountSnapshot(
            equity=1_000,
            margin_used=750,
            target_equity=1_000,
            target_margin_used=750,
            positions=(position,),
            maker_fee_rate=0.0001,
            taker_fee_rate=0.0004,
            risk_wallets=(position.wallet,),
        )
        action = PositionAction("target", "BTC", "", "")
        add_action = PositionAction("adjust_margin", "BTC", "", "", margin_usd=1)
        with (
            patch("hl_trade_execution.fetch_daily_candles", return_value=make_candles()),
            patch("hl_trade_execution.fetch_funding_history", return_value=make_funding()),
        ):
            [plan] = plan_position_actions([action], {"BTC": market}, snapshot, [])
        with (
            patch("hl_trade_execution.fetch_daily_candles", return_value=make_candles()),
            patch("hl_trade_execution.fetch_funding_history", return_value=make_funding()),
        ):
            [add_plan] = plan_position_actions([add_action], {"BTC": market}, snapshot, [])
        self.assertFalse(plan.block_reason)
        self.assertLess(plan.target_quantity, position.quantity)
        self.assertLess(plan.quantity_delta, 0)
        self.assertIn("exceeds risk/margin target", add_plan.block_reason)
        self.assertEqual(add_plan.quantity_delta, 0)

    def test_margin_adjustment_reduces_existing_position_without_reversing(self):
        market = Market("BTC", "", 100, 100, 3, 10)
        position = Position(
            wallet="0x" + "1" * 40,
            coin="BTC",
            dex="",
            quantity=0.5,
            entry_price=100,
            mark_price=100,
            oracle_price=100,
            margin_used=12.5,
            market=market,
            leverage=4,
        )
        snapshot = AccountSnapshot(
            equity=1_000,
            margin_used=12.5,
            target_equity=1_000,
            target_margin_used=12.5,
            positions=(position,),
            maker_fee_rate=0.0001,
            taker_fee_rate=0.0004,
            risk_wallets=(position.wallet,),
        )
        action = PositionAction("adjust_margin", "BTC", "", "", margin_usd=-100)
        with (
            patch("hl_trade_execution.fetch_daily_candles", return_value=make_candles()),
            patch("hl_trade_execution.fetch_funding_history", return_value=make_funding()),
        ):
            [plan] = plan_position_actions([action], {"BTC": market}, snapshot, [])
        self.assertFalse(plan.block_reason)
        self.assertEqual(plan.target_quantity, 0)
        self.assertEqual(plan.quantity_delta, -0.5)

    def test_position_action_csv_parses_margin_and_target_modes(self):
        content = (
            "action,coin,side,dex,margin_usd\n"
            "open_margin,BTC,LONG,,100\n"
            "adjust_margin,ETH,,,-25\n"
            "target,SOL,SHORT,,\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actions.csv"
            path.write_text(content, encoding="utf-8")
            actions = read_position_actions(path)
        self.assertEqual([action.action for action in actions], ["open_margin", "adjust_margin", "target"])
        self.assertEqual(actions[0].margin_usd, 100)
        self.assertEqual(actions[1].margin_usd, -25)
        self.assertEqual(actions[2].side, "SHORT")

    def test_close_action_on_flat_market_is_blocked_without_side_error(self):
        market = Market("BTC", "", 100, 100, 3, 10)
        snapshot = AccountSnapshot(
            equity=1_000,
            margin_used=0,
            target_equity=1_000,
            target_margin_used=0,
            positions=(),
            maker_fee_rate=0.0001,
            taker_fee_rate=0.0004,
            risk_wallets=("0x" + "1" * 40,),
        )
        action = PositionAction("close", "BTC", "", "")
        with (
            patch("hl_trade_execution.fetch_daily_candles", return_value=make_candles()),
            patch("hl_trade_execution.fetch_funding_history", return_value=make_funding()),
        ):
            [plan] = plan_position_actions([action], {"BTC": market}, snapshot, [])
        self.assertIn("no open position", plan.block_reason)

    def test_backtest_assumes_stop_first_when_both_levels_touch(self):
        candles = make_candles(30)
        candles.loc[14, "high"] = candles.loc[14, "open"] * 1.1
        candles.loc[14, "low"] = candles.loc[14, "open"] * 0.9
        settings = BacktestSettings(
            starting_equity=1_000,
            horizon_days=5,
            atr_period=14,
            stop_atr_multiple=1.25,
            reward_risk_ratio=1.0,
            maker_fee_bps=1.0,
            taker_fee_bps=4.0,
        )
        trades, skipped = backtest_signal(candles, make_funding(), "LONG", settings)
        self.assertFalse(skipped)
        self.assertTrue(trades)
        self.assertEqual(trades[0]["exit_reason"], "stop")
        self.assertLess(trades[0]["net_pnl"], 0)


if __name__ == "__main__":
    unittest.main()
