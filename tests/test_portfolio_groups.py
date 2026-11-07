import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "execution"), str(ROOT / "research")]
# Keep tests off real accounts: no IB or wallet in accounts.json is ever contacted.
os.environ["DESK_ACCOUNTS_FILE"] = str(ROOT / "tests" / "no_accounts.json")
# IB records from tests go to a throwaway folder, never the real one next to the repository.
os.environ["DESK_IB_DATA_DIR"] = __import__("tempfile").mkdtemp(prefix="desk-ib-")

import portfolio_analytics_implementation as portfolio  # noqa: E402


def days(start, count):
    return pd.date_range(start, periods=count, freq="D", tz="UTC")


class CalendarTests(unittest.TestCase):
    def test_trading_day_prices_take_the_last_close_within_four_days(self):
        prices = pd.DataFrame({"X": [10.0, 11.0, 12.0, 13.0]}, index=days("2026-11-05", 4))  # Thu..Sun
        calendar = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-20"], tz="UTC")  # Fri, Mon, much later
        sampled = portfolio.calendar_prices(prices, calendar)
        self.assertEqual(sampled.loc["2026-11-06", "X"], 11.0)
        self.assertEqual(sampled.loc["2026-11-09", "X"], 13.0)   # Sunday's close carries into Monday
        self.assertTrue(np.isnan(sampled.loc["2026-11-20", "X"]))  # too stale

    def test_daily_returns_compound_into_each_trading_day(self):
        returns = pd.Series([0.10, 0.10, -0.50, 0.20], index=days("2026-11-06", 4))  # Fri..Mon
        calendar = pd.DatetimeIndex(["2026-11-06", "2026-11-09"], tz="UTC")
        compounded = portfolio.compound_to_calendar(returns, calendar)
        self.assertTrue(np.isnan(compounded.iloc[0]))  # first period has no start
        self.assertAlmostEqual(compounded.loc["2026-11-09"], 1.10 * 0.50 * 1.20 - 1)

    def test_staked_vvv_is_analysed_as_vvv(self):
        staking = "0x321b7ff75154472b18edb199033ff4d116f340ff"
        self.assertEqual(portfolio.wallet_token_ticker("sVVV (VVV staked on Venice, base)", "base", staking), "VVV")
        self.assertEqual(portfolio.wallet_token_ticker("DPI (ethereum)", "ethereum", "0x1494"), "DPI")


class GroupFrameTests(unittest.TestCase):
    def analysis(self):
        folder = tempfile.mkdtemp()
        analysis = portfolio.PortfolioAnalytics(folder, ["0x" + "1" * 40], assets=["ETH"])
        analysis.asset_group = {"ETH": portfolio.CRYPTO, "NVDA": portfolio.TRADFI}
        return analysis

    def prices(self):
        everyday = days("2026-01-01", 120)
        trading = everyday[everyday.dayofweek < 5]
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(index=everyday)
        frame["BTC"] = 100 * np.cumprod(1 + rng.normal(0, 0.02, len(everyday)))
        frame["ETH"] = 50 * np.cumprod(1 + rng.normal(0, 0.03, len(everyday)))
        frame["SPY"] = pd.Series(400 * np.cumprod(1 + rng.normal(0, 0.01, len(trading))), index=trading)
        frame["NVDA"] = pd.Series(150 * np.cumprod(1 + rng.normal(0, 0.02, len(trading))), index=trading)
        return frame

    def test_each_group_gets_its_benchmark_and_calendar(self):
        analysis, prices = self.analysis(), self.prices()
        groups, combined = analysis._return_frames(prices, ["ETH", "NVDA"])
        self.assertEqual(set(groups[portfolio.CRYPTO].columns), {"ETH", "BTC"})
        self.assertEqual(set(groups[portfolio.TRADFI].columns), {"NVDA", "SPY"})
        # Crypto keeps weekends; TradFi and the combined book use only trading days.
        self.assertTrue((groups[portfolio.CRYPTO].index.dayofweek >= 5).any())
        self.assertFalse((groups[portfolio.TRADFI].index.dayofweek >= 5).any())
        self.assertFalse((combined.index.dayofweek >= 5).any())
        self.assertEqual(set(combined.columns), {"ETH", "NVDA", "BTC", "SPY"})
        # Monday's ETH return in the combined book covers Friday close to Monday close.
        friday, monday = pd.Timestamp("2026-01-02", tz="UTC"), pd.Timestamp("2026-01-05", tz="UTC")
        self.assertAlmostEqual(combined.loc[monday, "ETH"], prices.loc[monday, "ETH"] / prices.loc[friday, "ETH"] - 1)

    def test_crypto_only_book_stays_on_every_day(self):
        analysis, prices = self.analysis(), self.prices()
        _, combined = analysis._return_frames(prices, ["ETH"])
        self.assertTrue((combined.index.dayofweek >= 5).any())
        self.assertNotIn("SPY", combined.columns)

    def test_without_spy_tradfi_falls_back_to_btc(self):
        analysis, prices = self.analysis(), self.prices().drop(columns="SPY")
        groups, combined = analysis._return_frames(prices, ["ETH", "NVDA"])
        self.assertEqual(set(groups[portfolio.TRADFI].columns), {"NVDA", "BTC"})
        self.assertFalse(analysis._uses_trading_calendar(["ETH", "NVDA"], prices))


if __name__ == "__main__":
    unittest.main()


class MinimumHistoryTests(unittest.TestCase):
    def test_short_history_assets_leave_kelly_but_stay_in_the_universe(self):
        folder = tempfile.mkdtemp()
        analysis = portfolio.PortfolioAnalytics(folder, ["0x" + "1" * 40], assets=["OLD", "NEW"], min_history_days=180)
        everyday = days("2026-01-01", 300)
        rng = np.random.default_rng(5)
        prices = pd.DataFrame(index=everyday)
        for name in ("BTC", "OLD", "NEW"):
            prices[name] = 100 * np.cumprod(1 + rng.normal(0, 0.02, len(everyday)))
        prices.loc[everyday[:240], "NEW"] = np.nan  # listed 60 days ago
        analysis.all_assets = ["OLD", "NEW"]
        analysis.asset_group = {"OLD": portfolio.CRYPTO, "NEW": portfolio.CRYPTO}
        analysis.prices = prices
        with redirect_stdout(io.StringIO()) as output:
            analysis.calculate_returns()
        self.assertEqual(analysis.assets, ["OLD"])
        self.assertEqual(analysis.all_assets, ["OLD", "NEW"])
        self.assertEqual(list(analysis.short_history_assets.index), ["NEW"])
        # The Kelly sample keeps OLD's full history instead of NEW's 60 days.
        self.assertGreater(len(analysis.common_returns), 250)
        self.assertIn("NEW (59 days)", output.getvalue())
        with self.assertRaises(ValueError):
            portfolio.PortfolioAnalytics(folder, ["0x" + "1" * 40], min_history_days=-1)


class CrossSectionalPremiaTests(unittest.TestCase):
    def panel(self, premium, common_scale=0.03, seed=1, periods=300, assets=12):
        rng = np.random.default_rng(seed)
        dates = days("2025-01-01", periods)
        signal = pd.DataFrame(rng.normal(size=(periods, assets)), index=dates)
        common = rng.normal(0, common_scale, size=(periods, 1))
        returns = pd.DataFrame(common + premium * signal.to_numpy() + rng.normal(0, 0.01, (periods, assets)), index=dates)
        return signal, returns

    def test_fama_macbeth_recovers_a_premium_despite_a_common_shock(self):
        from factor_model import fama_macbeth

        signal, returns = self.panel(premium=0.002)
        summary, daily = fama_macbeth(returns, {"signal": signal})
        self.assertAlmostEqual(summary.loc["signal", "Premium (daily)"], 0.002, delta=0.0005)
        self.assertGreater(summary.loc["signal", "t-stat"], 5)
        self.assertEqual(summary.loc["signal", "Days"], 300)

    def test_no_premium_gives_a_small_t_stat_and_thin_days_are_skipped(self):
        from factor_model import fama_macbeth, newey_west_mean, sort_spread

        signal, returns = self.panel(premium=0.0, seed=2)
        summary, _ = fama_macbeth(returns, {"signal": signal})
        self.assertLess(abs(summary.loc["signal", "t-stat"]), 3)
        thin = signal.copy()
        thin.iloc[:100, 5:] = np.nan  # only 5 assets on the first 100 days
        summary, _ = fama_macbeth(returns, {"signal": thin}, min_assets=8)
        self.assertEqual(summary.loc["signal", "Days"], 200)
        _, t_stat, n = newey_west_mean(sort_spread(signal, returns))
        self.assertEqual(n, 300)
        self.assertLess(abs(t_stat), 3)

    def test_newey_west_matches_the_plain_t_stat_without_lags(self):
        from factor_model import newey_west_mean

        values = pd.Series(np.random.default_rng(4).normal(0.1, 1.0, 500))
        mean, t_stat, n = newey_west_mean(values, lags=0)
        self.assertAlmostEqual(t_stat, mean / (values.std(ddof=0) / np.sqrt(n)))


class HistoryBackfillTests(unittest.TestCase):
    def analysis(self, folder):
        analysis = portfolio.PortfolioAnalytics(folder, ["0x" + "1" * 40], assets=["ETH"], history_years=2)
        analysis.assets = ["ETH"]
        analysis.perp_asset_dex = {}
        analysis.wallet_token_keys = {}
        return analysis

    def test_backfill_fills_only_missing_dates_and_remembers_the_source_floor(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            analysis = self.analysis(folder)
            recent = days(str((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)).date()), 20)
            prices = pd.DataFrame({"ETH": 2000.0, "BTC": 60000.0}, index=recent)
            listed = analysis.history_start + pd.Timedelta(days=200)  # coin lists after the window start
            older = pd.Series(1500.0, index=pd.date_range(listed, recent[0] + pd.Timedelta(days=5), freq="D"))
            calls = []

            def history(key, start, end=None):
                calls.append(key)
                return older if key == "coingecko:ethereum" else pd.Series(dtype=float)
            ids = {"ETH": "ethereum", "BTC": "bitcoin"}
            with patch.object(analysis, "_defillama_history", side_effect=history):
                filled, added = analysis._backfill(prices, ids, [])
            # Overlapping dates keep the primary source's 2000; only older dates take 1500.
            self.assertTrue((filled.loc[recent, "ETH"] == 2000.0).all())
            self.assertEqual(filled["ETH"].dropna().index.min(), listed)
            self.assertEqual(added["ETH"], len(older) - 6)
            floor = json.loads((Path(folder) / portfolio.HISTORY_FLOOR_FILE).read_text())
            self.assertEqual(floor["ETH"], listed.strftime("%Y-%m-%d"))
            # A second run does not ask DefiLlama for ETH again: its listing date is known.
            calls.clear()
            with patch.object(analysis, "_defillama_history", side_effect=history):
                analysis._backfill(filled, ids, [])
            self.assertNotIn("coingecko:ethereum", calls)

    def test_defillama_pages_start_at_midnight_and_cover_the_range(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        requested = []

        def get(url, params=None, timeout=None):
            requested.append(params)
            start = pd.Timestamp(params["start"], unit="s", tz="UTC")
            points = [{"timestamp": int((start + pd.Timedelta(days=offset)).timestamp()), "price": 1.0 + offset}
                      for offset in range(params["span"])]
            return SimpleNamespace(ok=True, json=lambda: {"coins": {"coingecko:x": {"prices": points}}})
        with tempfile.TemporaryDirectory() as folder, patch.object(portfolio.requests, "get", side_effect=get):
            series = self.analysis(folder)._defillama_history(
                "coingecko:x", pd.Timestamp("2024-01-01 19:48", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"))
        self.assertTrue(all(pd.Timestamp(page["start"], unit="s").hour == 0 for page in requested))
        self.assertTrue(all(page["span"] <= portfolio.DEFILLAMA_PAGE_DAYS for page in requested))
        self.assertEqual(len(requested), 2)  # 731 days in pages of at most 500
        self.assertTrue(series.index.is_monotonic_increasing and not series.index.has_duplicates)
        self.assertEqual(series.index.min(), pd.Timestamp("2023-12-31", tz="UTC"))  # a midnight point closes the prior day


class EstimatedMarketCapTests(unittest.TestCase):
    def test_caps_extend_with_the_earliest_supply_and_skip_unlocking_tokens(self):
        dates = days("2024-01-01", 500)
        prices = pd.DataFrame({"STEADY": np.linspace(1.0, 3.0, 500), "UNLOCK": np.linspace(1.0, 2.0, 500)}, index=dates)
        observed = dates[-300:]  # CoinGecko's window
        caps = pd.DataFrame(index=dates, columns=["STEADY", "UNLOCK"], dtype=float)
        caps.loc[observed, "STEADY"] = prices.loc[observed, "STEADY"] * np.linspace(1_000.0, 1_100.0, 300)  # +10% supply
        caps.loc[observed, "UNLOCK"] = prices.loc[observed, "UNLOCK"] * np.linspace(1_000.0, 3_000.0, 300)  # +200%
        full, estimated, excluded = portfolio.estimate_market_caps(caps, prices)
        self.assertEqual(estimated, ["STEADY"])
        self.assertEqual(list(excluded), ["UNLOCK"])
        # 30-day medians at each end of a 1,000 -> 3,000 ramp over 300 days.
        expected = (1_000 + 2_000 * 284.5 / 299) / (1_000 + 2_000 * 14.5 / 299) - 1
        self.assertAlmostEqual(excluded["UNLOCK"], expected, places=6)
        early = dates[0]
        # Before the window: price x the supply at the window's start (~1,000), not today's 1,100.
        self.assertAlmostEqual(full.loc[early, "STEADY"] / prices.loc[early, "STEADY"], 1_000.0, delta=10)
        self.assertTrue(full.loc[dates[:200], "UNLOCK"].isna().all())
        pd.testing.assert_series_equal(full.loc[observed, "STEADY"], caps.loc[observed, "STEADY"])


class StyleModelTests(unittest.TestCase):
    def test_french_csv_first_daily_table_in_decimals(self):
        text = ("This file was created by CMPT_ME_BEME_RETS_DAILY\n\n"
                ",Mkt-RF,SMB,RF\n20260929,  1.00, -0.50, 0.02\n20260930,-99.99,  0.25, 0.02\n\n"
                " Equal Weighted Returns\n,Mkt-RF,SMB,RF\n20260929, 9.0, 9.0, 9.0\n")
        frame = portfolio.parse_french_csv(text)
        self.assertEqual(list(frame.columns), ["Mkt-RF", "SMB", "RF"])
        self.assertEqual(len(frame), 2)  # the second (equal-weighted) table is ignored
        self.assertAlmostEqual(frame.loc["2026-09-29", "Mkt-RF"], 0.01)
        self.assertTrue(np.isnan(frame.loc["2026-09-30", "Mkt-RF"]))  # -99.99 means missing

    def test_regression_recovers_betas_alpha_and_attribution_adds_up(self):
        from unittest.mock import patch

        rng = np.random.default_rng(11)
        dates = days("2025-01-01", 400)
        btc = pd.Series(rng.normal(0, 0.03, 400), index=dates)
        momentum = pd.Series(rng.normal(0, 0.02, 400), index=dates)
        alpha = 0.001
        book = alpha + 0.6 * btc - 0.3 * momentum + rng.normal(0, 0.005, 400)
        analysis = portfolio.PortfolioAnalytics(tempfile.mkdtemp(), ["0x" + "1" * 40], assets=["ETH"])
        analysis.common_returns = pd.DataFrame({"ETH": book}, index=dates)
        analysis.portfolio_daily_returns = pd.Series(book, index=dates)
        analysis.factor_returns = pd.DataFrame({"BTC market excess": btc, "BTC momentum": momentum})
        analysis.uses_trading_calendar = False
        analysis.trading_days_per_year = 365
        analysis.risk_free_daily = 0.0
        with patch.object(portfolio.go.Figure, "show"), patch.object(portfolio, "display"), \
                redirect_stdout(io.StringIO()):
            analysis.fit_style_model()
        table = analysis.style_model
        self.assertAlmostEqual(table.loc["BTC market excess", "Beta"], 0.6, delta=0.03)
        self.assertAlmostEqual(table.loc["BTC momentum", "Beta"], -0.3, delta=0.03)
        self.assertAlmostEqual(analysis.style_summary["Alpha (/yr)"], alpha * 365, delta=0.15)
        self.assertGreater(analysis.style_summary["Alpha t-stat"], 2)
        parts = analysis.style_attribution.drop("Total excess return")
        self.assertAlmostEqual(parts.sum(), analysis.style_attribution["Total excess return"])


class ActualBookTests(unittest.TestCase):
    def test_hl_book_returns_net_of_deposits_and_only_consecutive_periods(self):
        with tempfile.TemporaryDirectory() as folder:
            wallet = "0x" + "a" * 40
            times = list(pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")) + [pd.Timestamp("2026-01-09", tz="UTC")]
            # +1% PnL a day on 1,000; a 1,000 deposit on day 3 adds value but no PnL.
            values = [1000, 1010, 2020, 2040.2, 2060.6, 2081.2]
            pnl = [0, 10, 20, 40.2, 60.6, 81.2]
            pd.DataFrame({"ts": times, "accountValue": values, "pnl": pnl}).to_csv(
                Path(folder) / f"hl_portfolio_history_{wallet}.csv", index=False)
            analysis = portfolio.PortfolioAnalytics(folder, [wallet], assets=["ETH"])
            daily = analysis.hl_book_returns("D", folder)
        self.assertEqual(len(daily), 4)  # the 4-day gap to Jan 9 is not one return
        # A deposit is not performance: a naive value change would read day 3 as +100%. The index
        # treats it as arriving at the day's start, so that day's 10 of PnL is on 2,010.
        self.assertLess(daily.max(), 0.011)
        self.assertAlmostEqual(daily.loc["2026-01-03"], 10 / 2010, places=6)

    def test_flex_statement_parses_nav_and_deposits_into_twr(self):
        from desk.venues import ib_flex

        xml = ("<FlexQueryResponse><FlexStatements><FlexStatement accountId='U123'>"
               "<EquitySummaryInBase>"
               "<EquitySummaryByReportDateInBase reportDate='20260105' total='1000'/>"
               "<EquitySummaryByReportDateInBase reportDate='20260106' total='1010'/>"
               "<EquitySummaryByReportDateInBase reportDate='20260107' total='1520.1'/>"
               "</EquitySummaryInBase><CashTransactions>"
               "<CashTransaction type='Deposits/Withdrawals' amount='500' fxRateToBase='1' reportDate='20260107'/>"
               "<CashTransaction type='Dividends' amount='3' fxRateToBase='1' reportDate='20260107'/>"
               "</CashTransactions></FlexStatement></FlexStatements></FlexQueryResponse>")
        account, history = ib_flex.parse_nav_history(xml)
        self.assertEqual(account, "U123")
        self.assertEqual(list(history["flows"]), [0.0, 0.0, 500.0])  # dividends are return, not a flow
        returns = ib_flex.time_weighted_returns(history)
        self.assertAlmostEqual(returns.iloc[0], 0.01)
        self.assertAlmostEqual(returns.iloc[1], 0.01)

    def test_flex_polls_until_ready_and_hides_the_token(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from desk.venues import ib_flex

        replies = iter([
            "<FlexStatementResponse><Status>Success</Status><ReferenceCode>42</ReferenceCode><Url>https://x</Url></FlexStatementResponse>",
            "<FlexStatementResponse><ErrorCode>1019</ErrorCode><ErrorMessage>in progress</ErrorMessage></FlexStatementResponse>",
            "<FlexQueryResponse><FlexStatements/></FlexQueryResponse>",
        ])
        get = lambda url, params=None, timeout=None, headers=None: SimpleNamespace(
            text=next(replies), raise_for_status=lambda: None)
        with patch.object(ib_flex.requests, "get", side_effect=get):
            text = ib_flex.fetch_statement("123456", "999", sleep=lambda seconds: None)
        self.assertIn("FlexQueryResponse", text)

        def fails(url, params=None, timeout=None, headers=None):
            raise ib_flex.requests.ConnectionError(f"failed for {url}?t=123456")
        with patch.object(ib_flex.requests, "get", side_effect=fails):
            with self.assertRaises(RuntimeError) as caught:
                ib_flex.fetch_statement("123456", "999", sleep=lambda seconds: None)
        self.assertNotIn("123456", str(caught.exception))


class WeeklyStyleModelTests(unittest.TestCase):
    def test_weekly_book_is_annualized_over_52_and_keeps_core_factors_when_short(self):
        from unittest.mock import patch

        rng = np.random.default_rng(21)
        dates = days("2025-01-01", 420)
        btc = pd.Series(rng.normal(0, 0.03, 420), index=dates)
        momentum = pd.Series(rng.normal(0, 0.02, 420), index=dates)
        weekly_btc = (1 + btc).resample("W-SUN").prod() - 1
        weeks = weekly_btc.index[1:-1]
        book = 0.002 + 0.5 * weekly_btc.loc[weeks] + rng.normal(0, 0.01, len(weeks))
        analysis = portfolio.PortfolioAnalytics(tempfile.mkdtemp(), ["0x" + "1" * 40], assets=["ETH"])
        analysis.common_returns = pd.DataFrame({"ETH": btc}, index=dates)
        analysis.factor_returns = pd.DataFrame({"BTC market excess": btc, "BTC momentum": momentum})
        analysis.uses_trading_calendar = False
        analysis.trading_days_per_year = 365
        analysis.risk_free_daily = 0.0
        with patch.object(portfolio.go.Figure, "show"), patch.object(portfolio, "display"), \
                redirect_stdout(io.StringIO()):
            result = analysis.fit_style_model(pd.Series(book.to_numpy(), index=weeks), label="weekly test")
        self.assertIn("weeks", result["summary"]["Periods"])
        self.assertAlmostEqual(result["model"].loc["BTC market excess", "Beta"], 0.5, delta=0.08)
        # Annualized over 52 weeks (about 0.10), not 365 periods (about 0.73): one standard error is about 0.07.
        self.assertAlmostEqual(result["summary"]["Alpha (/yr)"], 0.002 * 52, delta=0.15)


class PortfolioHighlightsTests(unittest.TestCase):
    def test_highlights_summarize_sections_and_report_changes_since_last_run(self):
        from unittest.mock import patch

        folder = tempfile.mkdtemp()
        analysis = portfolio.PortfolioAnalytics(folder, ["0x" + "1" * 40], assets=["ETH"])
        analysis.all_assets, analysis.assets = ["ETH", "NEW", "NVDA"], ["ETH", "NVDA"]
        analysis.asset_group = {"ETH": portfolio.CRYPTO, "NEW": portfolio.CRYPTO, "NVDA": portfolio.TRADFI}
        analysis.short_history_assets = pd.Series({"NEW": 40})
        analysis.performance_metrics = pd.DataFrame([{"Sharpe (excess, sqrt 252)": 1.5, "Beta (BTC)": 0.4,
                                                      "Beta (SPY)": 0.8, "Max Drawdown": -0.2}])
        analysis.factor_premia = pd.DataFrame({"FM t-stat": [-1.9, 0.3], "Sort t-stat": [-1.8, 0.1]},
                                              index=["Funding (7d)", "Momentum (30d)"])
        shown = []
        with patch.object(portfolio, "display", side_effect=lambda frame: shown.append(frame.data)), \
                redirect_stdout(io.StringIO()) as first:
            analysis.show_highlights()
        text = " ".join(" ".join(frame["Highlight"]) for frame in shown)
        self.assertIn("1 TradFi vs SPY", text)
        self.assertIn("NEW (40d)", text)
        self.assertIn("strongest is Funding (7d)", text)
        self.assertIn("No earlier run saved yet", first.getvalue())
        # A second run reports what moved.
        analysis.performance_metrics.loc[0, "Beta (BTC)"] = 0.6
        analysis.all_assets = ["ETH", "NVDA"]
        shown.clear()
        with patch.object(portfolio, "display", side_effect=lambda frame: shown.append(frame.data)), \
                patch.object(analysis, "show_table") as table, redirect_stdout(io.StringIO()):
            analysis.show_highlights()
        changes = table.call_args[0][0]
        self.assertAlmostEqual(changes.loc["Kelly beta (btc)", "Change"], 0.2)
        self.assertIn("Dropped: NEW", " ".join(" ".join(frame["Highlight"]) for frame in shown))


class StyleResultKeyTests(unittest.TestCase):
    def test_each_book_keeps_its_own_style_result(self):
        from unittest.mock import patch

        rng = np.random.default_rng(31)
        dates = days("2025-01-01", 300)
        btc = pd.Series(rng.normal(0, 0.03, 300), index=dates)
        analysis = portfolio.PortfolioAnalytics(tempfile.mkdtemp(), ["0x" + "1" * 40], assets=["ETH"])
        analysis.common_returns = pd.DataFrame({"ETH": btc}, index=dates)
        analysis.portfolio_daily_returns = 0.5 * btc + rng.normal(0, 0.01, 300)
        analysis.factor_returns = pd.DataFrame({"BTC market excess": btc,
                                                "BTC momentum": pd.Series(rng.normal(0, 0.02, 300), index=dates)})
        analysis.uses_trading_calendar, analysis.trading_days_per_year, analysis.risk_free_daily = False, 365, 0.0
        with patch.object(portfolio.go.Figure, "show"), patch.object(portfolio, "display"), redirect_stdout(io.StringIO()):
            analysis.fit_style_model()
            analysis.fit_style_model(pd.Series(0.3 * btc.to_numpy(), index=dates), label="HL book (actual)")
        self.assertEqual(set(analysis.style_results), {"Kelly model portfolio", "HL book (actual)"})
        self.assertEqual(analysis.style_results["HL book (actual)"]["summary"]["Book"], "HL book (actual)")


class ExactFactorSetTests(unittest.TestCase):
    def test_a_chosen_factor_is_used_alone_even_with_short_coverage(self):
        from unittest.mock import patch

        rng = np.random.default_rng(41)
        dates = days("2025-01-01", 400)
        alt = pd.Series(rng.normal(0, 0.03, 400), index=dates)
        alt.iloc[:250] = np.nan  # covers well under three quarters of the sample
        btc = pd.Series(rng.normal(0, 0.03, 400), index=dates)
        analysis = portfolio.PortfolioAnalytics(tempfile.mkdtemp(), ["0x" + "1" * 40], assets=["ETH"])
        analysis.common_returns = pd.DataFrame({"ETH": btc}, index=dates)
        book = pd.Series(0.4 * alt.fillna(0).to_numpy() + rng.normal(0, 0.005, 400), index=dates)
        analysis.factor_returns_all = pd.DataFrame({"BTC market excess": btc, "Alt market - BTC": alt})
        analysis.uses_trading_calendar, analysis.trading_days_per_year, analysis.risk_free_daily = False, 365, 0.0
        with patch.object(portfolio.go.Figure, "show"), patch.object(portfolio, "display"), redirect_stdout(io.StringIO()):
            result = analysis.fit_style_model(book, label="alt only", factors=["Alt market - BTC"])
            with self.assertRaises(ValueError):
                analysis.fit_style_model(book, label="bad", factors=["Not a factor"])
        self.assertEqual(list(result["model"].index), ["Alt market - BTC", "Alpha (intercept)"])
        self.assertAlmostEqual(result["model"].loc["Alt market - BTC", "Beta"], 0.4, delta=0.06)  # 150 days: one s.e. is about 0.015
