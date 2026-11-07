import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Modules live in the repository root and its execution/ and research/ folders.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "execution"), str(ROOT / "research")]
# Keep tests off real accounts: no IB or wallet in accounts.json is ever contacted.
os.environ["DESK_ACCOUNTS_FILE"] = str(ROOT / "tests" / "no_accounts.json")
# IB records from tests go to a throwaway folder, never the real one next to the repository.
os.environ["DESK_IB_DATA_DIR"] = __import__("tempfile").mkdtemp(prefix="desk-ib-")

from factor_model import (
    alpha_verdict,
    btc_momentum_factor,
    btc_realized_vol,
    fit_factor_regression,
    independent_factors,
    split_in_half,
)


class FactorModelTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2026-01-01", periods=200, freq="D", tz="UTC")
        self.rng = np.random.default_rng(7)

    def test_regression_recovers_known_alpha_and_beta(self):
        market = pd.Series(self.rng.normal(0, 0.03, 200), index=self.dates)
        portfolio = 0.002 - 0.8 * market
        fit = fit_factor_regression(portfolio, market.to_frame("BTC"))
        self.assertAlmostEqual(fit.alpha, 0.002)
        self.assertAlmostEqual(fit.betas["BTC"], -0.8)
        self.assertEqual(fit.observations, 200)

    def test_state_factor_keeps_alpha_at_its_average_level(self):
        state = pd.Series(self.rng.normal(50, 10, 200), index=self.dates)
        portfolio = 0.002 + 0.0001 * (state - state.mean())
        fit = fit_factor_regression(portfolio, state.to_frame("Vol"), state_factors=("Vol",))
        self.assertAlmostEqual(fit.alpha, 0.002)
        self.assertAlmostEqual(fit.betas["Vol"], 0.0001 * state.std())

    def test_momentum_and_vol_do_not_use_the_same_day_return(self):
        returns = pd.Series(self.rng.normal(0, 0.02, 200), index=self.dates)
        changed = returns.copy()
        changed.iloc[100] = -0.5
        for build in (lambda r: btc_momentum_factor(r, 0.0, 30) / r, lambda r: btc_realized_vol(r, 30)):
            pd.testing.assert_series_equal(build(returns).iloc[:101], build(changed).iloc[:101], check_names=False)

    def test_momentum_follows_the_trailing_sign(self):
        returns = pd.Series(0.01, index=self.dates)
        returns.iloc[100:] = -0.01
        factor = btc_momentum_factor(returns, 0.0, 30)
        self.assertTrue(np.isnan(factor.iloc[29]))
        self.assertAlmostEqual(factor.iloc[50], 0.01)
        self.assertAlmostEqual(factor.iloc[100], -0.01)
        self.assertAlmostEqual(factor.iloc[150], 0.01)

    def test_independent_factors_skip_a_duplicate(self):
        market = pd.Series(self.rng.normal(0, 0.03, 200), index=self.dates)
        other = pd.Series(self.rng.normal(0, 0.03, 200), index=self.dates)
        frame = pd.DataFrame({"BTC": market, "Momentum": market, "Other": other})
        self.assertEqual(independent_factors(frame), ["BTC", "Other"])

    def test_split_keeps_the_earlier_half_first(self):
        first, second = split_in_half(self.dates[::-1])
        self.assertLess(first.max(), second.min())
        self.assertEqual(len(first) + len(second), 200)

    def test_verdict_needs_positive_and_significant_alpha(self):
        self.assertTrue(alpha_verdict(-0.1, 3.0).startswith("does not survive"))
        self.assertTrue(alpha_verdict(0.1, 1.0).startswith("not established"))
        self.assertTrue(alpha_verdict(0.1, 2.5).startswith("survives"))


if __name__ == "__main__":
    unittest.main()
