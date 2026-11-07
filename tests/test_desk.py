import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "execution"), str(ROOT / "research")]
# Keep tests off real accounts: no IB or wallet in accounts.json is ever contacted.
os.environ["DESK_ACCOUNTS_FILE"] = str(ROOT / "tests" / "no_accounts.json")
# IB records from tests go to a throwaway folder, never the real one next to the repository.
os.environ["DESK_IB_DATA_DIR"] = __import__("tempfile").mkdtemp(prefix="desk-ib-")

import ib_trade  # noqa: E402
from desk import risk  # noqa: E402
from desk.accounts import Account, load_accounts, parse_accounts  # noqa: E402
from desk.venues import ib as ib_venue  # noqa: E402
from desk.venues.base import VenuePosition  # noqa: E402


def value(tag, amount, currency="USD"):
    return SimpleNamespace(tag=tag, value=str(amount), currency=currency, account="U1")


def contract(symbol="AAPL", sec_type="STK", currency="USD", con_id=1, multiplier="", local=""):
    return SimpleNamespace(symbol=symbol, secType=sec_type, currency=currency, conId=con_id,
                           multiplier=multiplier, localSymbol=local or symbol, exchange="SMART",
                           primaryExchange="NASDAQ", lastTradeDateOrContractMonth="")


class FakeBracket:
    """Iterates parent, take-profit, stop-loss, like ib_async's BracketOrder."""

    def __init__(self, parent, take_profit, stop_loss):
        self.parent, self.takeProfit, self.stopLoss = parent, take_profit, stop_loss

    def __iter__(self):
        return iter((self.parent, self.takeProfit, self.stopLoss))


class FakeIB:
    """Just enough of ib_async.IB for snapshots and bracket submission."""

    def __init__(self, base="USD", rates=None, portfolio=(), summary=None):
        rates = rates or {"USD": 1.0}
        self._values = [value("ExchangeRate", rate, currency) for currency, rate in rates.items()]
        self._portfolio = list(portfolio)
        self._summary = summary or {"NetLiquidation": 10_000, "TotalCashValue": 6_000,
                                    "InitMarginReq": 2_000, "AvailableFunds": 8_000}
        self.base = base
        self.placed = []
        self.cancelled = []
        self.desk_errors = []
        self.next_id = 100

    def managedAccounts(self):
        return ["U1"]

    def accountValues(self, account=""):
        return self._values

    def accountSummary(self, account=""):
        return [value(tag, amount, self.base) for tag, amount in self._summary.items()]

    def portfolio(self, account=""):
        return self._portfolio

    def sleep(self, seconds):
        pass

    @property
    def client(self):
        def next_id():
            self.next_id += 1
            return self.next_id
        return SimpleNamespace(getReqId=next_id)

    def placeOrder(self, contract, order):
        price = order.auxPrice if order.orderType in ("STP", "STP LMT") else order.lmtPrice
        self.placed.append((order.orderType, order.action, order.totalQuantity, price, order.tif, order.outsideRth))
        trade = self.trades.get(order.orderId) if hasattr(self, "trades") else None
        if trade is None:
            trade = SimpleNamespace(order=order, log=[], orderStatus=SimpleNamespace(
                status="Submitted", filled=0.0, avgFillPrice=0.0), isDone=lambda: False)
            self.trades = getattr(self, "trades", {})
            self.trades[order.orderId] = trade
        return trade

    def cancelOrder(self, order):
        self.cancelled.append(order.orderId)


class AccountsTests(unittest.TestCase):
    def test_parses_both_venues_and_names(self):
        accounts = parse_accounts([
            {"venue": "hl", "id": "0x7532dA34edA8f3e383bC480d50e7A7d3b12c07f2"},
            {"venue": "IB", "id": "U1234567", "port": 7497},
        ])
        self.assertEqual(accounts[0].name, "HL 07f2")
        self.assertEqual((accounts[1].venue, accounts[1].port, accounts[1].name), ("ib", 7497, "IB U1234567"))

    def test_rejects_bad_entries(self):
        for item in ({"venue": "binance", "id": "x"}, {"venue": "hl", "id": "0x12"},
                     {"venue": "ib", "id": "u-1"}):
            with self.assertRaises(ValueError):
                parse_accounts([item])
        with self.assertRaises(ValueError):
            parse_accounts([{"venue": "ib", "id": "U1"}, {"venue": "ib", "id": "U1"}])

    def test_missing_file_means_no_accounts(self):
        self.assertEqual(load_accounts(ROOT / "tests" / "does_not_exist.json"), [])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "accounts.json"
            path.write_text(json.dumps([{"venue": "ib", "id": ""}]))
            self.assertEqual(load_accounts(path)[0].name, "IB")


class IBSnapshotTests(unittest.TestCase):
    def test_fx_rates_convert_a_euro_base_account_to_usd(self):
        base, usd_per = ib_venue.fx_rates_to_usd([value("ExchangeRate", 1.0, "EUR"),
                                              value("ExchangeRate", 0.9, "USD")])
        self.assertEqual(base, "EUR")
        self.assertAlmostEqual(usd_per["EUR"], 1 / 0.9)
        self.assertAlmostEqual(usd_per["USD"], 1.0)

    def test_snapshot_values_positions_in_usd(self):
        items = [
            SimpleNamespace(contract=contract("AAPL"), position=10, marketPrice=200.0, marketValue=2000,
                            averageCost=180.0, unrealizedPNL=200.0, realizedPNL=0, account="U1"),
            SimpleNamespace(contract=contract("ES", "FUT", con_id=2, multiplier="50", local="ESZ6"),
                            position=-1, marketPrice=6000.0, marketValue=-300000, averageCost=301000.0,
                            unrealizedPNL=1000.0, realizedPNL=0, account="U1"),
            SimpleNamespace(contract=contract("SAP", currency="EUR", con_id=3), position=5,
                            marketPrice=100.0, marketValue=500, averageCost=90.0,
                            unrealizedPNL=50.0, realizedPNL=0, account="U1"),
        ]
        ib = FakeIB(rates={"USD": 1.0, "EUR": 1.1}, portfolio=items)
        snap = ib_venue.snapshot_from_ib(ib, Account("ib", ""))
        self.assertEqual(snap.account, "U1")
        self.assertEqual(snap.equity_usd, 10_000)
        by_symbol = {position.symbol: position for position in snap.positions}
        self.assertEqual(by_symbol["ESZ6"].asset_class, "futures")
        self.assertAlmostEqual(by_symbol["ESZ6"].avg_price, 6020.0)
        self.assertAlmostEqual(by_symbol["ESZ6"].notional_usd, -300_000)
        self.assertAlmostEqual(by_symbol["SAP"].notional_usd, 550.0)
        self.assertAlmostEqual(snap.unrealized_usd, 200 + 1000 + 55)

    def test_front_month_skips_contracts_about_to_expire(self):
        def detail(expiry):
            return SimpleNamespace(realExpirationDate=expiry,
                                   contract=SimpleNamespace(lastTradeDateOrContractMonth=expiry))
        chosen = ib_venue.front_month([detail("20261218"), detail("20261108"), detail("20270319")],
                                  today=pd.Timestamp("2026-11-06"))
        self.assertEqual(chosen.realExpirationDate, "20261218")

    def test_round_to_tick(self):
        self.assertEqual(ib_venue.round_to_tick(101.237, 0.05, "down"), 101.2)
        self.assertEqual(ib_venue.round_to_tick(101.201, 0.05, "up"), 101.25)
        self.assertEqual(ib_venue.round_to_tick(6012.30, 0.25), 6012.25)


class IBRiskTests(unittest.TestCase):
    def test_position_risk_uses_atr_or_a_conservative_fallback(self):
        position = VenuePosition("ib", "U1", "AAPL", "equity", 100, 180, 200)
        with_atr, basis = risk.ib_position_risk(position, atr=4.0)
        self.assertAlmostEqual(with_atr, 100 * 4.0 * 1.25 + 2 * 1.0)
        self.assertIn("ATR", basis)
        fallback, basis = risk.ib_position_risk(position, atr=None)
        self.assertAlmostEqual(fallback, 100 * 200 * 0.10 + 2.0)
        self.assertIn("no price history", basis)

    def test_unreachable_ib_is_reported_not_raised(self):
        accounts = [Account("ib", "U1")]
        with patch.object(ib_venue, "connect", side_effect=ib_venue.IBConnectionError("refused")):
            book = risk.other_venue_book("hl", accounts)
        self.assertEqual(book.entries, [])
        self.assertEqual(book.unreachable, ["IB U1: refused"])

    def test_one_failing_wallet_does_not_hide_the_others(self):
        good, bad = "0x" + "1" * 40, "0x" + "2" * 40
        accounts = [Account("hl", good, "good"), Account("hl", bad, "bad"),
                    Account("ib", "U1")]

        def entry(wallets, label=""):
            if wallets == [bad]:
                raise ConnectionError("down")
            return risk.BookEntry("hl", label, 1_000.0, 100.0, positions=2)
        with patch.object(risk, "hl_book_entry", side_effect=entry), \
                patch.object(risk, "ib_book_entry", side_effect=KeyError("unexpected")):
            book = risk.other_venue_book("ib", accounts)
            both = risk.other_venue_book("other", accounts)
        self.assertEqual([entry.label for entry in book.entries], ["good"])
        self.assertEqual(book.unreachable, ["bad: ConnectionError: down"])
        # An unexpected IB error type is still reported, not raised.
        self.assertEqual(len(both.unreachable), 2)

    def test_broken_accounts_file_is_unreachable_not_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "accounts.json"
            path.write_text("[{\"venue\": \"nowhere\"}]")
            with patch.dict(os.environ, {"DESK_ACCOUNTS_FILE": str(path)}):
                book = risk.other_venue_book("hl")
        self.assertEqual(book.entries, [])
        self.assertIn("accounts.json could not be read", book.unreachable[0])


class BracketSizingTests(unittest.TestCase):
    def size(self, **overrides):
        args = dict(side="LONG", reference=100.0, bid=99.99, ask=100.01, atr=2.0, tick=0.01, step=1.0,
                    multiplier=1.0, fx_to_usd=1.0, risk_budget=200.0, risk_headroom=1_000.0,
                    margin_room_usd=10_000.0, margin_per_unit_usd=0.0, commission_per_unit_usd=0.005)
        args.update(overrides)
        return ib_trade.size_bracket(**args)

    def test_long_bracket_levels_and_risk_limited_size(self):
        sizing = self.size()
        self.assertEqual((sizing.stop_price, sizing.target_price), (97.5, 102.5))
        self.assertEqual(sizing.entry_limit, 100.03)
        # Risk per share: limit 100.03 to stop 97.50, plus two commissions.
        self.assertEqual(sizing.quantity, 78)
        self.assertLessEqual(sizing.risk_usd, 200.0)
        self.assertEqual(sizing.block_reason, "")

    def test_short_bracket_mirrors_levels(self):
        sizing = self.size(side="SHORT")
        self.assertEqual((sizing.stop_price, sizing.target_price, sizing.entry_limit), (102.5, 97.5, 99.97))

    def test_desk_risk_cap_blocks_and_margin_limits(self):
        blocked = self.size(risk_headroom=-5.0)
        self.assertEqual(blocked.quantity, 0)
        self.assertIn("equity cap", blocked.block_reason)
        margin_limited = self.size(margin_room_usd=1_000.0, margin_per_unit_usd=50.0)
        self.assertEqual(margin_limited.quantity, 20)

    def test_futures_multiplier_and_fx_scale_risk(self):
        sizing = self.size(reference=6000.0, bid=5999.75, ask=6000.25, atr=60.0, tick=0.25,
                           multiplier=50.0, risk_budget=10_000.0, risk_headroom=10_000.0,
                           commission_per_unit_usd=2.5)
        # One contract risks about 75 points x $50.
        self.assertEqual(sizing.quantity, 2)
        euro = self.size(fx_to_usd=1.1)
        self.assertLess(euro.quantity, self.size().quantity)

    def test_reads_ib_signals_with_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "signals.csv"
            path.write_text("symbol,side,sec_type,exchange\naapl,long,,\nES,SHORT,FUT,CME\n")
            signals = ib_trade.read_signals(path)
        self.assertEqual(signals[0].spec, ib_venue.ContractSpec("AAPL"))
        self.assertEqual((signals[1].spec.sec_type, signals[1].spec.exchange, signals[1].side), ("FUT", "CME", "SHORT"))


class BracketSubmissionTests(unittest.TestCase):
    def plan(self):
        return ib_trade.IBPlan(symbol="AAPL", side="LONG", asset_class="equity", reference=100.0,
                               stop_price=97.5, target_price=102.5, atr=2.0, quantity=10, entry_limit=100.03,
                               tick=0.01, contract=contract())

    def live_quote(self, live=True):
        return ib_venue.Quote(bid=99.99, ask=100.01, last=100.0, close=99.0, live=live)

    def test_full_fill_leaves_gtc_stop_and_target(self):
        ib = FakeIB()
        original = ib.placeOrder

        def place(contract_, order):
            trade = original(contract_, order)
            if order.orderType == "LMT" and order.action == "BUY":
                trade.orderStatus.filled, trade.orderStatus.avgFillPrice = 10.0, 100.02
                trade.orderStatus.status = "Filled"
                trade.isDone = lambda: True
            return trade
        ib.placeOrder = place
        with patch.object(ib_venue, "quote", return_value=self.live_quote()):
            message = ib_trade.submit_bracket(ib, self.plan(), "U1")
        self.assertIn("filled 10", message)
        self.assertEqual([entry[4] for entry in ib.placed], ["DAY", "GTC", "GTC"])
        self.assertEqual(ib.placed[0][3], 100.03)
        # Entry, target and a stop-limit stop, all able to work in extended hours.
        self.assertEqual([entry[0] for entry in ib.placed], ["LMT", "LMT", "STP LMT"])
        self.assertTrue(all(entry[5] for entry in ib.placed))
        self.assertEqual(ib.placed[2][3], 97.5)
        self.assertEqual(ib.cancelled, [])

    def test_rth_only_keeps_plain_stops_in_regular_hours(self):
        ib = FakeIB()
        with patch.object(ib_venue, "quote", return_value=self.live_quote()), \
                patch.object(ib_trade, "extended_hours", False), patch.object(ib_trade, "ENTRY_TIMEOUT_SECONDS", 0):
            ib_trade.submit_bracket(ib, self.plan(), "U1")
        self.assertEqual([entry[0] for entry in ib.placed[:3]], ["LMT", "LMT", "STP"])
        self.assertFalse(any(entry[5] for entry in ib.placed[:3]))

    def test_partial_fill_cancels_rest_and_resizes_exits(self):
        ib = FakeIB()
        original = ib.placeOrder

        def place(contract_, order):
            trade = original(contract_, order)
            if order.orderType == "LMT" and order.action == "BUY":
                trade.orderStatus.filled = 4.0
            return trade
        ib.placeOrder = place
        with patch.object(ib_venue, "quote", return_value=self.live_quote()), \
                patch.object(ib_trade, "ENTRY_TIMEOUT_SECONDS", 0):
            message = ib_trade.submit_bracket(ib, self.plan(), "U1")
        self.assertIn("filled 4", message)
        self.assertEqual(len(ib.cancelled), 1)
        self.assertEqual([entry[2] for entry in ib.placed[3:]], [4.0, 4.0])

    def test_no_live_quote_or_price_through_stop_sends_nothing(self):
        ib = FakeIB()
        with patch.object(ib_venue, "quote", return_value=self.live_quote(live=False)):
            self.assertIn("no real-time quote", ib_trade.submit_bracket(ib, self.plan(), "U1"))
        with patch.object(ib_venue, "quote", return_value=ib_venue.Quote(97.0, 97.1, 97.0, 99.0, True)):
            self.assertIn("through the stop", ib_trade.submit_bracket(ib, self.plan(), "U1"))
        self.assertEqual(ib.placed, [])


class HLDeskRiskTests(unittest.TestCase):
    def test_other_venue_risk_and_equity_enter_hl_sizing(self):
        import hl_trade as executor

        snapshot = executor.AccountSnapshot(
            equity=1_000.0, margin_used=0.0, target_equity=1_000.0, target_margin_used=0.0,
            positions=(), maker_fee_rate=0.0, taker_fee_rate=0.0, risk_wallets=("0x" + "1" * 40,))
        _, base_risk = executor.plan_trades([], {}, snapshot)
        with_ib = replace(snapshot, external_equity=1_000.0, external_risk=150.0)
        _, desk_risk = executor.plan_trades([], {}, with_ib)
        _, own_risk = executor.plan_trades([], {}, with_ib, include_external=False)
        self.assertEqual((base_risk, desk_risk, own_risk), (0.0, 150.0, 0.0))
        self.assertEqual(with_ib.desk_equity, 2_000.0)
        output = io.StringIO()
        with redirect_stdout(output):
            executor.print_plan(with_ib, [], desk_risk)
        self.assertIn("Desk equity (all venues): $2,000.00", output.getvalue())
        self.assertIn("Aggregate stop-risk cap: $400.00", output.getvalue())

    def test_live_run_stops_when_another_account_is_unreachable(self):
        import argparse

        import hl_trade as executor

        snapshot = executor.AccountSnapshot(1_000.0, 0.0, 1_000.0, 0.0, (), 0.0, 0.0, ("0x" + "1" * 40,))
        book = risk.DeskBook(unreachable=["IB: refused"])
        with patch("desk.risk.other_venue_book", return_value=book), redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                executor.attach_other_venues(snapshot, argparse.Namespace(live=True, ignore_unreachable=False))
            dry = executor.attach_other_venues(snapshot, argparse.Namespace(live=False, ignore_unreachable=False))
            forced = executor.attach_other_venues(snapshot, argparse.Namespace(live=True, ignore_unreachable=True))
        self.assertEqual((dry.external_risk, forced.external_risk), (0.0, 0.0))


class OverviewStorageTests(unittest.TestCase):
    def test_ib_fills_merge_by_execution_id(self):
        from desk.overview import merge_ib_fills

        def fills(*ids):
            return pd.DataFrame([{"exec_id": exec_id, "time": pd.Timestamp("2026-11-0%d" % (index + 1), tz="UTC"),
                                  "symbol": "AAPL", "quantity": 1.0} for index, exec_id in enumerate(ids)])
        with tempfile.TemporaryDirectory() as folder:
            merge_ib_fills(fills("a", "b"), Path(folder), "U1")
            merged = merge_ib_fills(fills("b", "c"), Path(folder), "U1")
        self.assertEqual(sorted(merged["exec_id"]), ["a", "b", "c"])




class OnchainWalletTests(unittest.TestCase):
    WALLET = "0x" + "a" * 40

    def response(self, payload, status=200):
        return SimpleNamespace(status_code=status, json=lambda: payload)

    def test_snapshot_values_priced_tokens_and_skips_spam_and_keyless_chains(self):
        from desk.venues import onchain

        usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        spam = "0x" + "5" * 40

        def get(url, params=None, headers=None):
            if url.endswith(f"/addresses/{self.WALLET}"):
                return self.response({"coin_balance": str(10 ** 18)})
            if url.endswith("/token-balances"):
                return self.response([
                    {"value": "25000000", "token": {"type": "ERC-20", "decimals": "6", "symbol": "USDC", "address_hash": usdc}},
                    {"value": "1" + "0" * 24, "token": {"type": "ERC-20", "decimals": "18", "symbol": "www.claim.top", "address_hash": spam}},
                ])
            if url.startswith(onchain.LLAMA_URL):
                return self.response({"coins": {
                    "coingecko:ethereum": {"price": 2000.0, "confidence": 0.99},
                    f"ethereum:{usdc}": {"price": 1.0, "confidence": 0.99},
                }})
            raise AssertionError(url)
        with patch.object(onchain, "_get", side_effect=get), patch.object(onchain, "etherscan_key", return_value=""):
            snapshot, notes = onchain.wallet_snapshot(Account("hl", self.WALLET, "HL aaaa"),
                                                      chains=["ethereum", "arbitrum"])
        self.assertEqual(snapshot.venue, "onchain")
        self.assertEqual(sorted(position.symbol for position in snapshot.positions), ["ETH (ethereum)", "USDC (ethereum)"])
        self.assertAlmostEqual(snapshot.equity_usd, 2025.0)
        self.assertTrue(any("arbitrum skipped" in note for note in notes))
        self.assertTrue(any("1 unpriced" in note for note in notes))

    def test_one_failing_chain_does_not_stop_the_others(self):
        from desk.venues import onchain

        def get(url, params=None, headers=None):
            if "optimism" in url:
                raise ConnectionError("down")
            if url.startswith(onchain.LLAMA_URL):
                return self.response({"coins": {"coingecko:ethereum": {"price": 2000.0, "confidence": 0.99}}})
            if url.endswith(f"/addresses/{self.WALLET}"):
                return self.response({"coin_balance": str(10 ** 17)})
            return self.response([])
        with patch.object(onchain, "_get", side_effect=get), patch.object(onchain, "etherscan_key", return_value=""):
            snapshot, notes = onchain.wallet_snapshot(Account("hl", self.WALLET), chains=["ethereum", "optimism"])
        self.assertAlmostEqual(snapshot.equity_usd, 200.0)
        self.assertTrue(any(note.startswith("optimism unavailable") for note in notes))

    def test_activity_labels_the_hl_bridge_and_drops_unpriced_transfers(self):
        from desk.venues import onchain

        bridge = "0x2Df1c51E09aECF9cacB7bc98cB1742757f163dF7"
        usdc = "0xaf88d065e77c8cc2239327c5edb3a432268e5831"
        transfers = [
            {"timeStamp": "1793900000", "hash": "0x1", "logIndex": "3", "from": self.WALLET, "to": bridge,
             "tokenSymbol": "USDC", "tokenDecimal": "6", "contractAddress": usdc, "value": "500000000"},
            {"timeStamp": "1793800000", "hash": "0x2", "logIndex": "1", "from": "0x" + "9" * 40, "to": self.WALLET,
             "tokenSymbol": "SCAM", "tokenDecimal": "18", "contractAddress": "0x" + "8" * 40, "value": "1"},
        ]

        def etherscan(chain, key, **params):
            return transfers if params["action"] == "tokentx" else []
        prices = {f"arbitrum:{usdc}": 1.0}
        with patch.object(onchain, "_etherscan", side_effect=etherscan), \
                patch.object(onchain, "fetch_prices", return_value=prices), \
                patch.object(onchain, "etherscan_key", return_value="KEY"):
            frame, notes = onchain.wallet_activity(Account("hl", self.WALLET), pd.Timestamp("2026-01-01", tz="UTC"),
                                                   chains=["arbitrum"])
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual((row["direction"], row["amount"], row["value_usd"], row["label"]), ("out", 500.0, 500.0, "HL bridge"))
        self.assertTrue(any("1 transfer(s) of unpriced" in note for note in notes))

    def test_accounts_enable_onchain_for_hl_wallets_only(self):
        accounts = parse_accounts([
            {"venue": "hl", "id": "0x" + "1" * 40},
            {"venue": "hl", "id": "0x" + "2" * 40, "onchain": False},
            {"venue": "ib", "id": "U1234567"},
        ])
        self.assertEqual([account.onchain for account in accounts], [True, False, False])


class CounterpartyLabelTests(unittest.TestCase):
    def test_desk_wallets_bridge_mints_and_contract_names(self):
        from desk.venues import onchain

        other_wallet, router = "0x" + "b" * 40, "0x" + "c" * 40
        frame = pd.DataFrame({
            "chain": ["arbitrum"] * 4,
            "counterparty": [other_wallet.upper().replace("0X", "0x"), onchain.NATIVE, router, "0x" + "d" * 40],
            "value_usd": [100.0, 50.0, 25.0, 0.5],
            "label": ["", "", "", ""],
        })
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(onchain, "NAME_CACHE", Path(folder) / "names.json"), \
                patch.object(onchain, "etherscan_key", return_value="KEY"), \
                patch.object(onchain, "_etherscan", return_value=[{"ContractName": "SwapRouter"}]) as lookup:
            onchain.label_counterparties(frame, {other_wallet: "HL bbbb"})
            again = frame.assign(label="")
            onchain.label_counterparties(again, {other_wallet: "HL bbbb"})
        self.assertEqual(list(frame["label"]), ["Desk wallet HL bbbb", "Mint/burn", "SwapRouter", ""])
        # Only the one contract worth naming is looked up, and only once thanks to the cache.
        self.assertEqual(lookup.call_count, 1)


class BaseChainTests(unittest.TestCase):
    WALLET = "0x" + "a" * 40

    def setUp(self):
        # Never reach a real Alchemy key from the Keychain; tests that want one patch it in.
        from desk.venues import onchain

        patcher = patch.object(onchain, "alchemy_key", return_value="")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_alchemy_reads_every_token_and_both_transfer_directions(self):
        from desk.venues import onchain

        usdc, spam = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913", "0x" + "5" * 40
        calls = []

        def alchemy(chain, batch):
            calls.extend(method for method, _ in batch)
            replies = []
            for method, params in batch:
                if method == "eth_getBalance":
                    replies.append(hex(10 ** 18))
                elif method == "alchemy_getTokenBalances":
                    replies.append({"tokenBalances": [{"contractAddress": usdc, "tokenBalance": hex(5 * 10 ** 6)},
                                                      {"contractAddress": spam, "tokenBalance": hex(10 ** 30)},
                                                      {"contractAddress": "0x" + "6" * 40, "tokenBalance": "0x0"}]})
                elif method == "alchemy_getTokenMetadata":
                    replies.append({"decimals": 6, "symbol": "USDC"} if params[0] == usdc else {"decimals": 18, "symbol": "SPAM"})
                else:  # alchemy_getAssetTransfers
                    query = params[0]
                    outgoing = "fromAddress" in query
                    replies.append({"transfers": [
                        {"category": "erc20", "hash": "0xabc" if outgoing else "0xdef", "uniqueId": "0xabc:log:7" if outgoing else "0xdef:log:2",
                         "from": self.WALLET if outgoing else "0x" + "1" * 40, "to": "0x" + "2" * 40 if outgoing else self.WALLET,
                         "asset": "USDC", "rawContract": {"address": usdc, "value": hex(3 * 10 ** 6), "decimal": "0x6"},
                         "metadata": {"blockTimestamp": "2026-11-01T00:00:00Z"}},
                        {"category": "external", "hash": "0xold", "uniqueId": "0xold:external", "from": self.WALLET, "to": "0x" + "3" * 40,
                         "value": 1.0, "asset": "ETH", "rawContract": {}, "metadata": {"blockTimestamp": "2025-01-01T00:00:00Z"}},
                    ]})
            return replies
        prices = {"coingecko:ethereum": 2000.0, f"base:{usdc}": 1.0}
        with patch.object(onchain, "alchemy_key", return_value="alchemy_test"), \
                patch.object(onchain, "_alchemy", side_effect=alchemy), \
                patch.object(onchain, "fetch_prices", return_value=prices), \
                patch.object(onchain, "NAME_CACHE", Path(tempfile.mkdtemp()) / "names.json"):
            snapshot, notes = onchain.wallet_snapshot(Account("hl", self.WALLET), chains=["base"])
            activity, activity_notes = onchain.wallet_activity(
                Account("hl", self.WALLET), pd.Timestamp("2026-06-01", tz="UTC"), chains=["base"])
        self.assertEqual(sorted(position.symbol for position in snapshot.positions), ["ETH (base)", "USDC (base)"])
        self.assertAlmostEqual(snapshot.equity_usd, 2_005.0)
        self.assertEqual(notes, ["1 unpriced token balance(s) ignored (mostly spam airdrops)"])
        # The 2025 transfer is older than the cutoff, so each direction stops there.
        self.assertEqual(sorted(zip(activity["direction"], activity["log_index"])), [("in", 2), ("out", 7)])
        self.assertEqual(set(activity["value_usd"]), {3.0})
        self.assertEqual(calls.count("alchemy_getAssetTransfers"), 2)

    def test_alchemy_sends_its_key_in_a_header_and_never_in_errors(self):
        from desk.venues import onchain

        captured = {}

        def post(url, json=None, headers=None, timeout=None):
            captured.update(url=url, headers=headers)
            body = {"jsonrpc": "2.0", "id": 0, "error": {"code": -32600, "message": "invalid key alchemy_secret"}}
            return SimpleNamespace(status_code=401, text=str(body), json=lambda: body, raise_for_status=lambda: None)
        with patch.object(onchain, "alchemy_key", return_value="alchemy_secret"), \
                patch.object(onchain.requests, "post", side_effect=post):
            with self.assertRaises(onchain.MissingKey) as caught:
                onchain._alchemy(onchain.CHAINS["base"], [("eth_blockNumber", [])])
        self.assertEqual(captured["url"], "https://base-mainnet.g.alchemy.com/v2")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer alchemy_secret")
        self.assertNotIn("alchemy_secret", str(caught.exception))

    def test_keyless_base_reads_native_and_well_known_tokens_by_rpc(self):
        from desk.venues import onchain

        usdc = onchain.WELL_KNOWN_TOKENS["base"][0]
        symbol = "0x" + (32).to_bytes(32, "big").hex() + (4).to_bytes(32, "big").hex() + b"USDC".hex().ljust(64, "0")

        def post(url, json=None, **kwargs):
            replies = []
            for request in json:
                if request["method"] == "eth_getBalance":
                    replies.append({"id": request["id"], "result": hex(2 * 10 ** 18)})
                    continue
                data = request["params"][0]["data"]
                to = request["params"][0]["to"]
                if to != usdc:
                    result = "0x" + "0" * 64 if data.startswith("0x70a08231") else "0x"
                elif data.startswith("0x70a08231"):
                    result = hex(250 * 10 ** 6)
                elif data == "0x95d89b41":
                    result = symbol
                else:
                    result = hex(6)
                replies.append({"id": request["id"], "result": result})
            return SimpleNamespace(status_code=200, headers={"content-type": "application/json"},
                                   raise_for_status=lambda: None, json=lambda: replies)
        prices = {"coingecko:ethereum": 2000.0, f"base:{usdc}": 1.0}
        with patch.object(onchain.requests, "post", side_effect=post), \
                patch.object(onchain.time, "sleep"), \
                patch.object(onchain, "blockscout_key", return_value=""), \
                patch.object(onchain, "fetch_prices", return_value=prices):
            snapshot, notes = onchain.wallet_snapshot(Account("hl", self.WALLET), chains=["base"])
        self.assertEqual(sorted(position.symbol for position in snapshot.positions), ["ETH (base)", "USDC (base)"])
        self.assertAlmostEqual(snapshot.equity_usd, 4_250.0)
        self.assertTrue(any("main tokens only" in note for note in notes))

    def test_base_uses_blockscout_pro_with_the_key_in_a_header(self):
        from desk.venues import onchain

        seen = []

        def get(url, params=None, headers=None):
            seen.append((url, dict(params or {}), dict(headers or {})))
            if url.endswith(f"/addresses/{self.WALLET}"):
                return SimpleNamespace(status_code=200, json=lambda: {"coin_balance": str(10 ** 18)})
            return SimpleNamespace(status_code=200, json=lambda: [])
        with patch.object(onchain, "_get", side_effect=get), \
                patch.object(onchain, "blockscout_key", return_value="proapi_test"), \
                patch.object(onchain, "BLOCKSCOUT_PRO_SPACING", 0), \
                patch.object(onchain, "fetch_prices", return_value={"coingecko:ethereum": 2000.0}):
            snapshot, notes = onchain.wallet_snapshot(Account("hl", self.WALLET), chains=["base"])
        self.assertAlmostEqual(snapshot.equity_usd, 2_000.0)
        self.assertTrue(all(url.startswith("https://api.blockscout.com/8453/api/v2/") for url, _, _ in seen))
        self.assertTrue(all(headers.get("Authorization") == "Bearer proapi_test" for _, _, headers in seen))
        self.assertTrue(all("apikey" not in params for _, params, _ in seen))
        self.assertEqual(notes, [])

    def test_paid_plan_reply_falls_back_to_rpc_and_never_leaks_keys(self):
        from desk.venues import onchain

        def get(url, params=None, headers=None):
            raise onchain.PaidPlanRequired("needs a paid plan (Featured chain 8453 requires Builder)")
        with patch.object(onchain, "_get", side_effect=get), \
                patch.object(onchain, "blockscout_key", return_value="proapi_secret"), \
                patch.object(onchain, "etherscan_key", return_value="ETHERSCANSECRET"), \
                patch.object(onchain, "BLOCKSCOUT_PRO_SPACING", 0), \
                patch.object(onchain, "rpc_holdings", return_value=[onchain.Holding("base", onchain.NATIVE, "ETH", 1.0)]), \
                patch.object(onchain, "fetch_prices", return_value={"coingecko:ethereum": 2000.0}):
            snapshot, notes = onchain.wallet_snapshot(Account("hl", self.WALLET), chains=["base"])
            leaked = onchain._chain_problem(onchain.CHAINS["arbitrum"],
                                            RuntimeError("500 for url ...?apikey=ETHERSCANSECRET&x=proapi_secret"))
        self.assertAlmostEqual(snapshot.equity_usd, 2_000.0)
        self.assertIn("main tokens only", notes[0])
        self.assertIn("Featured chain 8453", notes[0])
        self.assertNotIn("SECRET", leaked)
        self.assertNotIn("proapi_secret", leaked)

    def test_alchemy_failure_falls_back_to_keyless_base_balances(self):
        from desk.venues import onchain

        with patch.object(onchain, "alchemy_key", return_value="alchemy_test"), \
                patch.object(onchain, "blockscout_key", return_value=""), \
                patch.object(onchain, "alchemy_holdings",
                             side_effect=onchain.MissingKey("the Alchemy app does not have base-mainnet enabled")), \
                patch.object(onchain, "rpc_holdings", return_value=[onchain.Holding("base", onchain.NATIVE, "ETH", 1.0)]), \
                patch.object(onchain, "fetch_prices", return_value={"coingecko:ethereum": 2000.0}):
            snapshot, notes = onchain.wallet_snapshot(Account("hl", self.WALLET), chains=["base"])
        self.assertAlmostEqual(snapshot.equity_usd, 2_000.0)
        self.assertIn("main tokens only", notes[0])
        self.assertIn("base-mainnet enabled", notes[0])


class DeskCoverageTests(unittest.TestCase):
    def test_hl_wallets_missing_from_risk_wallets_still_count(self):
        first, second = "0x" + "1" * 40, "0x" + "2" * 40
        accounts = [Account("hl", first, "one"), Account("hl", second, "two")]

        def entry(wallets, label=""):
            return risk.BookEntry("hl", label, 1_000.0, 50.0, positions=1)
        with patch.object(risk, "hl_book_entry", side_effect=entry) as read:
            book = risk.other_venue_book("hl", accounts, already_counted=(first,))
            nothing_extra = risk.other_venue_book("hl", accounts, already_counted=(first, second.upper().replace("0X", "0x")))
        self.assertEqual([item.label for item in book.entries], ["two"])
        self.assertEqual(nothing_extra.entries, [])
        self.assertEqual(read.call_count, 1)


class IBAccountAnalyticsTests(unittest.TestCase):
    def test_book_returns_weight_today_values_over_net_liquidation(self):
        from desk.ib.account_analytics import book_returns, max_drawdown, simple_beta

        days = pd.date_range("2026-01-01", periods=4)
        returns = {"AAA": pd.Series([0.10, -0.05, 0.0, 0.02], index=days),
                   "BBB": pd.Series([0.0, 0.10, np.nan, 0.0], index=days)}
        book, notes = book_returns({"AAA": 500.0, "BBB": -250.0, "CCC": 100.0}, returns, equity=1_000.0)
        # Day 2: 500 x -5% + (-250) x 10% = -50 on 1,000 of equity.
        self.assertEqual(list(book.round(4)), [0.05, -0.05, 0.0, 0.01])
        self.assertTrue(any("CCC" in note for note in notes))
        self.assertTrue(any("BBB" in note for note in notes))
        self.assertAlmostEqual(max_drawdown(pd.Series([0.10, -0.50, 0.20])), -0.5)
        benchmark = pd.Series(np.linspace(-0.02, 0.02, 30))
        self.assertAlmostEqual(simple_beta(2 * benchmark, benchmark), 2.0)
        self.assertTrue(np.isnan(simple_beta(benchmark.head(5), benchmark.head(5))))


class PublicRpcTests(unittest.TestCase):
    def test_refused_calls_are_retried_and_never_silently_dropped(self):
        from desk.venues import onchain

        attempts = {"count": 0}

        def post(url, json=None, **kwargs):
            attempts["count"] += 1
            replies = []
            for call in json:
                # The first time each call is seen, the node refuses it.
                if call["id"] % 2 and attempts["count"] == 1:
                    replies.append({"id": call["id"], "error": {"code": -32011, "message": "request limit reached"}})
                else:
                    replies.append({"id": call["id"], "result": hex(call["id"])})
            return SimpleNamespace(status_code=200, headers={"content-type": "application/json"}, json=lambda: replies)
        calls = [{"jsonrpc": "2.0", "id": index, "method": "eth_blockNumber", "params": []} for index in range(5)]
        with patch.object(onchain.requests, "post", side_effect=post), patch.object(onchain.time, "sleep"):
            results = onchain._rpc_batch("https://rpc.example", calls)
        self.assertEqual(results, {index: hex(index) for index in range(5)})

        def always_refused(url, json=None, **kwargs):
            return SimpleNamespace(status_code=429, headers={"content-type": "text/plain"}, json=lambda: None)
        with patch.object(onchain.requests, "post", side_effect=always_refused), patch.object(onchain.time, "sleep"):
            with self.assertRaises(RuntimeError):
                onchain._rpc_batch("https://rpc.example", calls)

    def test_staked_vvv_is_valued_at_the_vvv_price(self):
        from desk.venues import onchain

        base = onchain.CHAINS["base"]
        staked, vvv = "0x321b7ff75154472b18edb199033ff4d116f340ff", "0xacfe6019ed1a7dc6f7b508c02d1b04ec88cc21bf"
        self.assertEqual(onchain.price_key(base, staked.upper().replace("0X", "0x")), f"base:{vvv}")
        with patch.object(onchain, "alchemy_key", return_value=""), \
                patch.object(onchain, "blockscout_key", return_value=""), \
                patch.object(onchain, "rpc_holdings", return_value=[onchain.Holding("base", staked, "sVVV", 100.0)]), \
                patch.object(onchain, "fetch_prices", return_value={f"base:{vvv}": 21.0, "coingecko:ethereum": 2000.0}):
            snapshot, _ = onchain.wallet_snapshot(Account("hl", "0x" + "a" * 40), chains=["base"])
        self.assertEqual(snapshot.positions[0].symbol, "sVVV (VVV staked on Venice, base)")
        self.assertAlmostEqual(snapshot.equity_usd, 2_100.0)


    def test_unclaimed_venice_rewards_are_read_and_shown(self):
        from desk.venues import onchain

        vvv = "0xacfe6019ed1a7dc6f7b508c02d1b04ec88cc21bf"
        with patch.object(onchain, "_rpc_batch", return_value={0: hex(4 * 10 ** 16)}) as rpc:
            rewards = onchain.staking_reward_holdings(onchain.CHAINS["base"], "0x" + "a" * 40)
        self.assertEqual(rewards, [onchain.Holding("base", f"rewards:{vvv}", "VVV", 0.04)])
        self.assertTrue(rpc.call_args[0][1][0]["params"][0]["data"].startswith("0x31d7a262"))
        self.assertEqual(onchain.staking_reward_holdings(onchain.CHAINS["ethereum"], "0x" + "a" * 40), [])
        with patch.object(onchain, "alchemy_key", return_value=""), \
                patch.object(onchain, "blockscout_key", return_value=""), \
                patch.object(onchain, "rpc_holdings", return_value=[]), \
                patch.object(onchain, "staking_reward_holdings", return_value=rewards), \
                patch.object(onchain, "fetch_prices", return_value={f"base:{vvv}": 21.0}):
            snapshot, _ = onchain.wallet_snapshot(Account("hl", "0x" + "a" * 40), chains=["base"])
        # Worth only $0.84, but rewards are shown despite the $1 dust threshold.
        self.assertEqual(snapshot.positions[0].symbol, "VVV (unclaimed staking rewards, base)")
        self.assertAlmostEqual(snapshot.equity_usd, 0.84)


class IBConnectionTests(unittest.TestCase):
    def test_read_only_connections_use_fresh_ids_and_execution_keeps_its_own(self):
        self.assertEqual(ib_venue.client_ids("execution"), [41])
        ids = ib_venue.client_ids("analytics")
        self.assertEqual(len(ids), 3)
        self.assertTrue(all(44_001 <= value <= 44_999 for value in ids))

    def test_blocking_ib_calls_move_off_a_running_event_loop(self):
        import asyncio
        import threading

        def where():
            return threading.current_thread().name

        self.assertEqual(ib_venue.run_off_event_loop(where), threading.current_thread().name)

        async def notebook_cell():
            return ib_venue.run_off_event_loop(where)
        self.assertTrue(asyncio.run(notebook_cell()).startswith("ib"))


class IBProtectionTests(unittest.TestCase):
    def resting(self, order_type, action, quantity, price=None, con_id=7, status="Submitted"):
        order = SimpleNamespace(orderType=order_type, action=action, totalQuantity=quantity,
                                lmtPrice=price if price is not None else ib_venue.UNSET)
        return SimpleNamespace(contract=SimpleNamespace(conId=con_id), order=order,
                               orderStatus=SimpleNamespace(status=status, filled=0))

    def test_whole_shares_only_and_fractional_remainder_reported(self):
        plan = ib_trade.plan_protection("GOOGL", 7, 3.25, mark=100.0, atr=4.0, tick=0.01, step=1.0, trades=[])
        self.assertEqual((plan.protectable_quantity, plan.stop_price, plan.target_price), (3.0, 95.0, 105.0))
        self.assertEqual((plan.stop_quantity_to_add, plan.target_quantity_to_add), (3.0, 3.0))
        self.assertIn("0.25 fractional", plan.notes[0])
        meta = ib_trade.plan_protection("META", 8, 0.75, mark=600.0, atr=20.0, tick=0.01, step=1.0, trades=[])
        self.assertIn("fractional", meta.block_reason)
        self.assertEqual(meta.stop_quantity_to_add, 0)

    def test_existing_orders_skip_or_block_but_never_stack(self):
        covered = [self.resting("STP", "SELL", 3), self.resting("LMT", "SELL", 3, price=105.0)]
        done = ib_trade.plan_protection("X", 7, 3, 100.0, 4.0, 0.01, 1.0, covered)
        self.assertEqual((done.block_reason, done.stop_quantity_to_add, done.target_quantity_to_add), ("", 0, 0))
        stop_only = ib_trade.plan_protection("X", 7, 3, 100.0, 4.0, 0.01, 1.0, [self.resting("STP", "SELL", 3)])
        self.assertIn("unlinked", stop_only.block_reason)
        ok_without_targets = ib_trade.plan_protection("X", 7, 3, 100.0, 4.0, 0.01, 1.0, [self.resting("STP", "SELL", 3)],
                                                      with_targets=False)
        self.assertEqual(ok_without_targets.block_reason, "")
        # Cancelled orders, other contracts, and buy-side orders do not count as protection.
        ignored = [self.resting("STP", "SELL", 3, status="Cancelled"), self.resting("STP", "SELL", 3, con_id=99),
                   self.resting("STP", "BUY", 3)]
        fresh = ib_trade.plan_protection("X", 7, 3, 100.0, 4.0, 0.01, 1.0, ignored)
        self.assertEqual(fresh.stop_quantity_to_add, 3)

    def test_ladder_splits_evenly_and_shrinks_for_small_positions(self):
        plan = ib_trade.plan_protection("NET", 7, 6, 274.0, 20.0, 0.01, 1.0, [], steps=3)
        self.assertEqual(plan.ladder, [(2.0, 265.66, 282.34), (2.0, 257.33, 290.67), (2.0, 249.0, 299.0)])
        self.assertAlmostEqual(plan.risk_usd, 2 * (8.34 + 16.67 + 25.0))
        uneven = ib_trade.ladder_steps(7, 1.0, 100.0, 5.0, 1, 0.01, 3)
        self.assertEqual([size for size, _, _ in uneven], [3.0, 2.0, 2.0])
        tiny = ib_trade.plan_protection("KYIV", 7, 1, 13.0, 0.5, 0.01, 1.0, [], steps=3)
        self.assertEqual(len(tiny.ladder), 1)
        self.assertIn("1 step(s) instead of 3", tiny.notes[0])
        # Steps under the $10 minimum are merged.
        cheap = ib_trade.ladder_steps(4, 1.0, 6.0, 1.5, 1, 0.01, 3)
        self.assertTrue(all(size * stop >= 10 for size, stop, _ in cheap))
        self.assertLess(len(cheap), 3)

    def test_each_ladder_step_is_a_linked_stop_target_pair(self):
        placed = []

        def place(contract, order):
            placed.append(order)
            return SimpleNamespace(order=order, log=[], orderStatus=SimpleNamespace(status="Submitted"))
        ib = SimpleNamespace(placeOrder=place, sleep=lambda seconds: None)
        plan = ib_trade.plan_protection("NET", 7, 6, 274.0, 20.0, 0.01, 1.0, [], steps=3)
        plan.contract = SimpleNamespace(conId=7)
        report = ib_trade.submit_protection(ib, plan, "U1")
        self.assertEqual(len(placed), 6)
        self.assertEqual([order.orderType for order in placed], ["STP LMT", "LMT"] * 3)
        self.assertTrue(all(order.action == "SELL" and order.tif == "GTC" and order.account == "U1" for order in placed))
        groups = [order.ocaGroup for order in placed]
        self.assertEqual(len(set(groups)), 3)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(sum(order.totalQuantity for order in placed if order.orderType == "STP LMT"), 6)
        self.assertEqual(report.count("Submitted"), 6)
        stops_only = ib_trade.plan_protection("NET", 7, 6, 274.0, 20.0, 0.01, 1.0, [], with_targets=False, steps=2)
        stops_only.contract = plan.contract
        placed.clear()
        ib_trade.submit_protection(ib, stops_only, "U1")
        self.assertEqual([order.orderType for order in placed], ["STP LMT", "STP LMT"])


class ExtendedHoursTests(unittest.TestCase):
    def test_session_state_from_ib_hours(self):
        details = SimpleNamespace(timeZoneId="US/Eastern",
                                  tradingHours="20261106:0400-20261106:2000;20261107:CLOSED",
                                  liquidHours="20261106:0930-20261106:1600;20261107:CLOSED")
        at = lambda text: pd.Timestamp(text, tz="US/Eastern")
        self.assertEqual(ib_venue.session_state(details, at("2026-11-06 12:00")), "regular")
        self.assertEqual(ib_venue.session_state(details, at("2026-11-06 05:00")), "extended")
        self.assertEqual(ib_venue.session_state(details, at("2026-11-06 19:59")), "extended")
        self.assertEqual(ib_venue.session_state(details, at("2026-11-06 20:30")), "closed")
        self.assertEqual(ib_venue.session_state(details, at("2026-11-07 12:00")), "closed")

    def test_stop_limit_and_close_prices(self):
        # Half an ATR (2.0) is capped at 1.5% of the stop, which IB accepts.
        self.assertEqual(ib_trade.stop_limit_price("SELL", 100.0, 4.0, 0.01), 98.5)
        self.assertEqual(ib_trade.stop_limit_price("BUY", 100.0, 4.0, 0.01), 101.5)
        self.assertEqual(ib_trade.stop_limit_price("SELL", 100.0, 1.0, 0.01), 99.5)
        # The NET and SWMR levels IB rejected now sit within the cap.
        self.assertGreaterEqual(ib_trade.stop_limit_price("SELL", 263.4, 19.74, 0.01), 263.4 * 0.985 - 0.01)
        self.assertGreaterEqual(ib_trade.stop_limit_price("SELL", 5.09, 1.405, 0.01), 5.09 * 0.985 - 0.01)
        self.assertEqual(ib_trade.stop_limit_price("SELL", 100.0, float("nan"), 0.01), 99.0)
        self.assertEqual(ib_trade.close_limit_price("SELL", 100.0, 0.05, 0.01), 95.0)
        self.assertEqual(ib_trade.close_limit_price("BUY", 100.0, 0.02, 0.01), 102.0)
        stop = ib_trade.exit_stop_order("SELL", 3, 95.0, 4.0, 0.01, "U1")
        self.assertEqual((stop.orderType, stop.auxPrice, stop.lmtPrice, stop.outsideRth, stop.tif),
                         ("STP LMT", 95.0, 93.57, True, "GTC"))


class ProtectionReplaceTests(unittest.TestCase):
    def resting(self, order_type, quantity, price=None, ref="desk-protect", status="PreSubmitted"):
        order = SimpleNamespace(orderType=order_type, action="SELL", totalQuantity=quantity, orderRef=ref,
                                lmtPrice=price if price is not None else ib_venue.UNSET)
        return SimpleNamespace(contract=SimpleNamespace(conId=7), order=order,
                               orderStatus=SimpleNamespace(status=status, filled=0), log=[])

    def test_replace_swaps_only_desk_orders(self):
        mine = [self.resting("STP", 3), self.resting("LMT", 3, price=105.0)]
        plan = ib_trade.plan_protection("X", 7, 3, 100.0, 4.0, 0.01, 1.0, mine, replace=True)
        self.assertEqual((plan.block_reason, len(plan.replace_trades), plan.stop_quantity_to_add), ("", 2, 3))
        manual = [self.resting("STP", 3, ref="")]
        kept = ib_trade.plan_protection("X", 7, 3, 100.0, 4.0, 0.01, 1.0, manual, replace=True)
        self.assertEqual(kept.replace_trades, [])
        self.assertIn("not placed by the desk", kept.notes[-1])

    def test_new_orders_are_sent_only_after_old_ones_cancel(self):
        old = [self.resting("STP", 3)]
        plan = ib_trade.plan_protection("X", 7, 3, 100.0, 4.0, 0.01, 1.0, old, replace=True)
        plan.contract = SimpleNamespace(conId=7)
        placed = []
        stuck = SimpleNamespace(cancelOrder=lambda order: None, sleep=lambda seconds: None,
                                placeOrder=lambda contract, order: placed.append(order))
        with patch.object(ib_trade, "CANCEL_WAIT_SECONDS", 0):
            message = ib_trade.submit_protection(stuck, plan, "U1")
        self.assertIn("nothing new was placed", message)
        self.assertEqual(placed, [])


class KillSwitchTests(unittest.TestCase):
    def target(self, symbol, quantity, session="regular", exits=()):
        return ib_trade.KillTarget(symbol, SimpleNamespace(conId=hash(symbol) % 1000, secType="STK"), quantity,
                                   float(int(abs(quantity))), 100.0, 0.01, 4.0, session, list(exits))

    def exit_trade(self):
        trade = SimpleNamespace(order=SimpleNamespace(orderId=1, action="SELL"),
                                orderStatus=SimpleNamespace(status="PreSubmitted", filled=0), log=[])
        trade.order.owner = trade  # lets the fake mark the right trade cancelled
        return trade

    def fake_ib(self, fills):
        """Fills: symbol -> attempt number at which its close fills (None = never)."""
        events = []

        class Fake:
            def cancelOrder(self, order):
                events.append(("cancel", order.orderId))
                if hasattr(order, "owner"):
                    order.owner.orderStatus.status = "Cancelled"
                for trade in trades.values():
                    if trade.order is order:
                        trade.orderStatus.status = "Cancelled"

            def placeOrder(self, contract, order):
                symbol = contract.symbol
                events.append(("place", symbol, order.orderType, order.lmtPrice if order.orderType == "LMT" else order.auxPrice))
                if order.orderType != "LMT" or order.orderRef != ib_trade.KILL_ORDER_REF:
                    return SimpleNamespace(order=order, orderStatus=SimpleNamespace(status="PreSubmitted", filled=0), log=[])
                trade = trades.get(symbol)
                if trade is None:
                    trade = SimpleNamespace(order=order, attempt=0, log=[],
                                            orderStatus=SimpleNamespace(status="Submitted", filled=0, avgFillPrice=0))
                    trades[symbol] = trade
                else:
                    trade.attempt += 1
                if fills.get(symbol) is not None and trade.attempt >= fills[symbol]:
                    trade.orderStatus.status, trade.orderStatus.filled = "Filled", order.totalQuantity
                    trade.orderStatus.avgFillPrice = order.lmtPrice
                return trade

            def sleep(self, seconds):
                pass

        trades = {}
        return Fake(), events, trades

    def run_kill(self, targets, fills):
        ib, events, trades = self.fake_ib(fills)
        for target in targets:
            target.contract.symbol = target.symbol
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(ib_trade, "KILL_SWITCH_FILE", Path(folder) / "KILL_SWITCH"), \
                patch.object(ib_trade, "kill_targets", return_value=(targets, [])), \
                patch.object(ib_venue, "quote", side_effect=RuntimeError("no data")), \
                patch.object(ib_trade, "KILL_ATTEMPT_SECONDS", 0), patch.object(ib_trade, "CANCEL_WAIT_SECONDS", 0), \
                redirect_stdout(io.StringIO()) as output:
            ok = ib_trade.run_kill_switch(ib, "U1")
            engaged = (Path(folder) / "KILL_SWITCH").exists()
        return ok, events, output.getvalue(), engaged

    def test_exits_cancelled_just_before_each_close_which_widens(self):
        stop = self.exit_trade()
        ok, events, output, engaged = self.run_kill([self.target("NET", 6, exits=[stop])], {"NET": 1})
        self.assertTrue(ok and engaged)
        self.assertEqual(events[0], ("cancel", 1))
        self.assertEqual(events[1], ("place", "NET", "LMT", 98.0))   # 2% through
        self.assertEqual(events[2], ("place", "NET", "LMT", 95.0))   # widened to 5%, then filled
        self.assertIn("NET: closed 6", output)

    def test_unfilled_close_is_cancelled_and_reprotected(self):
        ok, events, output, _ = self.run_kill([self.target("SWMR", 1)], {"SWMR": None})
        self.assertFalse(ok)
        self.assertEqual([event[3] for event in events if event[0] == "place" and event[2] == "LMT"], [98.0, 95.0, 90.0])
        self.assertEqual(events[-1][:3], ("place", "SWMR", "STP LMT"))
        self.assertIn("re-protected 1", output)

    def test_closed_session_queues_and_fractions_are_reported(self):
        ok, events, output, _ = self.run_kill([self.target("GOOGL", 3.25, session="closed"),
                                               self.target("META", 0.75)], {})
        self.assertFalse(ok)
        self.assertEqual([event for event in events if event[0] == "place"], [("place", "GOOGL", "LMT", 90.0)])
        self.assertIn("queued", output)
        self.assertIn("0.25 fractional", output)
        self.assertIn("META: 0.75 fractional", output)


class RejectedStopSafetyNetTests(unittest.TestCase):
    def run_with(self, rejected_types):
        placed, cancelled = [], []

        def place(contract, order):
            status = "Inactive" if order.orderType in rejected_types else "PreSubmitted"
            trade = SimpleNamespace(order=order, log=[], orderStatus=SimpleNamespace(status=status, filled=0))
            placed.append(trade)
            return trade

        def cancel(order):
            cancelled.append(order.orderType)
            for trade in placed:
                if trade.order is order:
                    trade.orderStatus.status = "Cancelled"
        ib = SimpleNamespace(placeOrder=place, cancelOrder=cancel, sleep=lambda seconds: None)
        plan = ib_trade.plan_protection("NET", 7, 2, 274.0, 20.0, 0.01, 1.0, [])
        plan.contract = SimpleNamespace(conId=7)
        with patch.object(ib_trade, "CANCEL_WAIT_SECONDS", 0):
            report = ib_trade.submit_protection(ib, plan, "U1")
        return placed, cancelled, report

    def test_rejected_stop_limit_falls_back_to_a_linked_plain_stop(self):
        placed, cancelled, report = self.run_with({"STP LMT"})
        self.assertEqual([trade.order.orderType for trade in placed], ["STP LMT", "LMT", "STP"])
        self.assertEqual(placed[2].order.ocaGroup, placed[1].order.ocaGroup)
        self.assertEqual(cancelled, [])
        self.assertIn("plain regular-hours stop", report)

    def test_target_is_cancelled_if_no_stop_is_accepted(self):
        placed, cancelled, report = self.run_with({"STP LMT", "STP"})
        self.assertEqual(cancelled, ["LMT"])
        self.assertIn("WARNING", report)


class ExternalNamesTests(unittest.TestCase):
    """Names owned by outside services must survive any internal renaming (hyperliquid -> hl)."""

    REQUIRED = {
        "research/portfolio_analytics_implementation.py": ["'HYPE': 'hyperliquid'", "'USDH': 'hyperliquid-usd'",
                                                           "https://api.hyperliquid.xyz/info"],
        "execution/hl_trade.py": ['KEYCHAIN_SERVICE = "Hyperliquid API wallet"', "https://api.hyperliquid.xyz",
                                  "from hyperliquid."],
        "requirements-trading.txt": ["hyperliquid-python-sdk"],
    }

    def test_external_identifiers_are_unchanged(self):
        for path, needles in self.REQUIRED.items():
            text = (ROOT / path).read_text()
            for needle in needles:
                self.assertIn(needle, text, f"{path} lost the external name {needle!r}")


class DeskAccountsTests(unittest.TestCase):
    def test_ib_nav_becomes_value_and_pnl_net_of_deposits(self):
        from desk.desk_accounts import ib_history

        nav = pd.DataFrame({"date": ["2026-01-05", "2026-01-06", "2026-01-07"],
                            "nav": [1000.0, 1010.0, 1520.1], "flows": [0.0, 0.0, 500.0]})
        history = ib_history(nav)
        self.assertEqual(list(history["accountValue"]), [1000.0, 1010.0, 1520.1])
        self.assertAlmostEqual(history["pnl"].iloc[-1], 20.1)  # the 500 deposit is not PnL
        self.assertEqual(history["ts"].iloc[0].hour, 21)

    def test_daily_returns_need_both_days_covered_by_every_account(self):
        from desk.desk_accounts import period_returns

        index = pd.Series([1.0, 1.01, 1.02, 1.05, 1.06],
                          index=pd.date_range("2026-01-01 12:00", periods=5, freq="D", tz="UTC"))
        valid = pd.DatetimeIndex(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"], utc=True))
        returns = period_returns(index, "1D", valid)
        # Jan 3 is missing for some account, so Jan 3 and Jan 4 returns are dropped.
        self.assertEqual([day.day for day in returns.index], [2, 5])

    def test_lifetime_pnl_before_the_window_is_not_a_loss_inside_it(self):
        from unittest.mock import patch
        import desk.desk_accounts as module

        times = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
        hl = pd.DataFrame({"ts": times, "accountValue": [100.0, 101.0, 102.0, 103.0], "pnl": [-500.0, -499.0, -498.0, -497.0]})
        desk = module.DeskAccounts.__new__(module.DeskAccounts)
        desk.histories, desk.windowed = {"HL": hl}, set()
        start = times[0]
        part = hl[hl["ts"] >= start].copy()
        part["pnl"] = part["pnl"] - part["pnl"].iloc[0]
        index = module.time_weighted_index([part])
        self.assertGreater(index.iloc[-1], 1.0)  # +1 a day on ~100, not a -500 loss
        self.assertAlmostEqual(index.iloc[-1], (101 / 100) * (102 / 101) * (103 / 102), places=6)


class OverviewSectionsTests(unittest.TestCase):
    def overview(self):
        from desk.overview import DeskOverview
        from desk.venues.base import VenueAccount

        overview = DeskOverview([Account("hl", "0x" + "1" * 40)], data_dir=tempfile.mkdtemp())
        hl = VenueAccount("hl", "0x1", "HL 1111", 1000.0, 1000.0, 300.0, 700.0, 50.0, positions=(
            VenuePosition("hl", "0x1", "ETH", "crypto", -1.0, 2100.0, 2000.0, unrealized_usd=100.0),
            VenuePosition("hl", "0x1", "SOL", "crypto", 10.0, 105.0, 100.0, unrealized_usd=-50.0)))
        wallet = VenueAccount("onchain", "0x1", "HL 1111 wallet", 2112.0, 2112.0, 0.0, 2112.0, float("nan"), positions=(
            VenuePosition("onchain", "0x1", "sVVV (VVV staked on Venice, base)", "crypto spot", 100.0, float("nan"), 21.12,
                          unrealized_usd=float("nan")),))
        overview.snapshots = [hl, wallet]
        return overview

    def test_metrics_holdings_and_pnl_sections(self):
        from unittest.mock import patch
        import desk.overview as module

        overview = self.overview()
        tables = {}
        capture = lambda frame, **kwargs: tables.__setitem__(kwargs.get("title"), frame)
        with patch.object(module, "display_interactive_table", side_effect=capture), redirect_stdout(io.StringIO()):
            overview.show_desk_metrics()
            overview.show_wallet_holdings()
            overview.show_unrealized_pnl()
        metrics = dict(zip(tables["DESK METRICS"]["Metric"], tables["DESK METRICS"]["Value"]))
        self.assertEqual(metrics["Desk value (all accounts)"], "$3,112.00")
        self.assertEqual(metrics["Wallet holdings outside HL"], "$2,112.00")
        self.assertIn("3.00x", metrics["Gross exposure"])  # 2,000 + 1,000 gross on 1,000 trading equity
        holdings = tables["WALLET HOLDINGS OUTSIDE HL"]
        self.assertEqual((holdings.iloc[0]["Token"], holdings.iloc[0]["Chain"], holdings.iloc[0]["Note"]),
                         ("sVVV", "base", "VVV staked on Venice"))
        movers = tables["LARGEST WINNERS AND LOSERS (TOP 5 EACH)"]
        self.assertEqual(list(movers["Symbol"]), ["ETH", "SOL"])
        self.assertAlmostEqual(movers.iloc[0]["% of position"], 100 / 2100)


class HistoryTimeoutTests(unittest.TestCase):
    def test_history_requests_fail_fast_after_ib_stops_answering(self):
        from desk.venues import ib as ib_venue

        class Contract:
            secType, symbol, localSymbol = "STK", "SPY", "SPY"

        class FakeIB:
            def __init__(self):
                self.desk_errors, self.calls = [], 0

            def reqHistoricalData(self, contract, **kwargs):
                self.calls += 1
                self.desk_errors.append("162: Historical Market Data Service error message:API historical data query cancelled: 4")
                return []

        ib = FakeIB()
        with self.assertRaises(ib_venue.HistoryUnavailable):
            ib_venue.daily_bars(ib, Contract())
        with self.assertRaises(ib_venue.HistoryUnavailable):
            ib_venue.adjusted_daily_closes(ib, Contract())
        self.assertEqual(ib.calls, 1)  # the second request never waited on IB

    def test_no_data_for_one_contract_does_not_stop_other_requests(self):
        from desk.venues import ib as ib_venue

        class Contract:
            secType, symbol, localSymbol = "STK", "XYZ", "XYZ"

        class FakeIB:
            desk_errors = ["162: Historical Market Data Service error message:HMDS query returned no data"]

            def reqHistoricalData(self, contract, **kwargs):
                return []

        ib = FakeIB()
        with self.assertRaises(ValueError) as caught:
            ib_venue.daily_bars(ib, Contract())
        self.assertNotIsInstance(caught.exception, ib_venue.HistoryUnavailable)
        self.assertFalse(getattr(ib, "desk_history_down", False))


class AdjustedHistoryFallbackTests(unittest.TestCase):
    def test_adjusted_timeout_falls_back_to_trade_prices(self):
        from types import SimpleNamespace

        from desk.venues import ib as ib_venue

        class Contract:
            secType, symbol, localSymbol = "STK", "SPY", "SPY"

        class FakeIB:
            def __init__(self):
                self.desk_errors, self.asked = [], []

            def sleep(self, seconds):
                pass

            def reqHistoricalData(self, contract, whatToShow, **kwargs):
                self.asked.append(whatToShow)
                if whatToShow == "ADJUSTED_LAST":
                    self.desk_errors.append("162: Historical Market Data Service error message:API historical data query cancelled: 4")
                    return []
                return [SimpleNamespace(date=f"2026-11-0{day}", close=700.0 + day) for day in (2, 3, 4)]

        ib = FakeIB()
        closes = ib_venue.adjusted_daily_closes(ib, Contract())
        self.assertEqual(ib.asked, ["ADJUSTED_LAST", "TRADES"])
        self.assertEqual(len(closes), 3)
        self.assertFalse(getattr(ib, "desk_history_down", False))


if __name__ == "__main__":
    unittest.main()
