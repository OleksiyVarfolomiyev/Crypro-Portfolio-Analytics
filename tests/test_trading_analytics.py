import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

from hyperliquid_backtest import BacktestSettings, backtest_signal
from hyperliquid_trade_analytics import (
    adverse_funding_cost_per_unit,
    add_expected_pnl_columns,
    consolidate_current_funding,
    expected_position_pnl,
    expected_forward_return,
    fetch_daily_candles,
    fetch_funding_history,
    post_info,
    rolling_funding_sums,
    side_to_direction,
    summarize_current_funding,
    wilder_atr,
)
from hyperliquid_trade_execution import (
    AccountSnapshot,
    Market,
    Position,
    plan_position_protection,
    plan_position_actions,
    fetch_markets,
    post_only_price,
    PositionAction,
    plan_open_position_protection,
    read_position_actions,
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


class TradeAnalyticsTests(unittest.TestCase):
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
            patch("hyperliquid_trade_analytics.requests.post", side_effect=[rate_limited, successful]),
            patch("hyperliquid_trade_analytics.time.sleep") as sleep,
        ):
            self.assertEqual(post_info({"type": "meta"}), {"ok": True})
        sleep.assert_called_once_with(3.0)

    def test_info_request_reports_rate_limit_after_retry_limit(self):
        rate_limited = unittest.mock.Mock(status_code=429, ok=False, headers={})
        with (
            patch("hyperliquid_trade_analytics.requests.post", return_value=rate_limited),
            patch("hyperliquid_trade_analytics.time.sleep"),
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
        with patch("hyperliquid_trade_analytics.post_info", side_effect=responses) as post:
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
            patch("hyperliquid_trade_analytics.fetch_daily_candles", return_value=make_candles()),
            patch("hyperliquid_trade_analytics.fetch_funding_history", return_value=make_funding(rate=-0.00001)),
        ):
            enriched, warnings = add_expected_pnl_columns([row])
        self.assertFalse(warnings)
        self.assertIn("Expected Price PnL (5d)", enriched[0])
        self.assertIn("Expected Funding PnL (5d)", enriched[0])
        self.assertIn("Expected PnL incl. funding (5d)", enriched[0])

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
        with patch("hyperliquid_trade_execution.post_info", return_value=book):
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

    def test_fetch_markets_does_not_duplicate_dex_prefix(self):
        response = [
            {"universe": [{"name": "xyz:AAPL", "szDecimals": 2, "maxLeverage": 3}]},
            [{"markPx": "100", "oraclePx": "99"}],
        ]
        with patch("hyperliquid_trade_execution.post_info", return_value=response):
            markets = fetch_markets(["xyz"])
        self.assertIn("xyz:AAPL", markets)
        self.assertEqual(markets["xyz:AAPL"].dex, "xyz")
        self.assertNotIn("xyz:xyz:AAPL", markets)

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
                "hyperliquid_trade_execution.fetch_daily_candles",
                side_effect=[make_candles(), ValueError("ATR(14) requires more history")],
            ),
            patch(
                "hyperliquid_trade_execution.fetch_funding_history",
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
            patch("hyperliquid_trade_execution.fetch_daily_candles", return_value=make_candles()),
            patch("hyperliquid_trade_execution.fetch_funding_history", return_value=make_funding()),
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
            patch("hyperliquid_trade_execution.fetch_daily_candles", return_value=make_candles()),
            patch("hyperliquid_trade_execution.fetch_funding_history", return_value=make_funding()),
        ):
            [plan] = plan_position_actions([action], {"BTC": market}, snapshot, [])
        with (
            patch("hyperliquid_trade_execution.fetch_daily_candles", return_value=make_candles()),
            patch("hyperliquid_trade_execution.fetch_funding_history", return_value=make_funding()),
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
            patch("hyperliquid_trade_execution.fetch_daily_candles", return_value=make_candles()),
            patch("hyperliquid_trade_execution.fetch_funding_history", return_value=make_funding()),
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
            patch("hyperliquid_trade_execution.fetch_daily_candles", return_value=make_candles()),
            patch("hyperliquid_trade_execution.fetch_funding_history", return_value=make_funding()),
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
