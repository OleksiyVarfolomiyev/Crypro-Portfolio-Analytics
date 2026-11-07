import tempfile
import unittest
from pathlib import Path

import pandas as pd

from desk.hl.backtest import backtest_highlights
from desk.hl.market_research import CHANGE, FUNDING, VOLUME, market_highlights
from desk.notebook_highlights import HighlightsState, render


class HighlightsStateTests(unittest.TestCase):
    def test_compares_numbers_and_items_with_the_last_run(self):
        path = Path(tempfile.mkdtemp()) / "state.json"
        first = HighlightsState(path).compare({"Desk value": (1000.0, "usd")}, {"Positions": ["AAPL long", "QNT short"]})
        self.assertIsNone(first.last_run)
        second = HighlightsState(path).compare({"Desk value": (1100.0, "usd"), "Sharpe": (float("nan"), "num")},
                                               {"Positions": ["AAPL long", "IONQ short"]})
        self.assertEqual(second.changes, [("Desk value", "$1,000.00", "$1,100.00", "+$100.00")])
        self.assertIn("Positions added: IONQ short.", second.notes)
        self.assertIn("Positions removed: QNT short.", second.notes)

    def test_baseline_is_kept_when_the_same_state_object_runs_again(self):
        path = Path(tempfile.mkdtemp()) / "state.json"
        HighlightsState(path).compare({"Value": (1.0, "num")})
        state = HighlightsState(path)
        state.compare({"Value": (2.0, "num")})
        again = state.compare({"Value": (2.0, "num")})
        self.assertEqual(again.changes[0][3], "+1.00")

    def test_render_escapes_text_and_lists_both_sections(self):
        block = render([("Good", "a < b")], [], None)
        self.assertIn("a &lt; b", block)
        self.assertIn("NEEDS ATTENTION", block)
        self.assertIn("Nothing to report.", block)


class ReportHighlightTests(unittest.TestCase):
    def test_market_highlights_flag_extreme_funding(self):
        universe = pd.DataFrame({"Ticker": ["A", "B", "C"], CHANGE: [5.0, -3.0, 1.0],
                                 FUNDING: [0.08, -0.001, 0.01], VOLUME: [1e8, 5e7, 1e7]})
        well, attention = market_highlights(universe, "crypto")
        self.assertIn("1 markets; A +0.0800", dict(attention)["Funding beyond ±0.05%/8h"])
        self.assertIn("67% of 3 markets are up", dict(well)["Breadth"])

    def test_backtest_highlights_split_pnl_and_flag_losing_signals(self):
        trades = pd.DataFrame({
            "coin": ["BTC", "BTC", "ETH", "ETH"], "side": ["LONG"] * 4,
            "entry_time": ["2026-01-01", "2026-01-08", "2026-01-01", "2026-01-08"],
            "exit_reason": ["target", "stop", "stop", "stop"],
            "price_pnl": [100.0, -40.0, -30.0, -30.0], "funding_pnl": [-1.0] * 4, "fees": [2.0] * 4,
            "net_pnl": [97.0, -43.0, -33.0, -33.0]})
        well, attention = backtest_highlights(trades, [{"coin": "ETH", "reason": "gap"}], 10_000)
        self.assertIn("Worst signal", dict(attention))
        self.assertIn("ETH long", dict(attention)["Worst signal"])
        self.assertIn("1 entries skipped", dict(attention)["Skipped entries"])
        self.assertIn("stop 75%", dict(well)["Exits"])


if __name__ == "__main__":
    unittest.main()
