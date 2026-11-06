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

from portfolio_risk import bootstrap_var, calculate_portfolio_risk_metrics


class PortfolioRiskMetricTests(unittest.TestCase):
    def test_metrics_use_excess_return_and_configured_calendar_annualization(self):
        returns = pd.Series([0.02, 0.01, -0.02, 0.03, -0.01])
        risk_free_daily = 0.0001
        metrics = calculate_portfolio_risk_metrics(
            returns,
            risk_free_daily,
            periods_per_year=365,
        )
        self.assertAlmostEqual(
            metrics["Sharpe (annualized)"],
            (returns.mean() - risk_free_daily) / returns.std(ddof=1) * np.sqrt(365),
        )
        self.assertAlmostEqual(
            metrics["Arithmetic Return (annualized)"],
            returns.mean() * 365,
        )
        self.assertAlmostEqual(
            metrics["Sharpe (raw, 252d)"],
            returns.mean() / returns.std(ddof=1) * np.sqrt(252),
        )
        self.assertAlmostEqual(
            metrics["Sharpe (excess, 252d)"],
            (returns.mean() - risk_free_daily)
            / returns.std(ddof=1)
            * np.sqrt(252),
        )
        self.assertEqual(metrics["Daily Return Observations"], 5)

    def test_historical_and_bootstrap_var_come_from_observed_returns(self):
        returns = pd.Series([-0.04, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10])
        metrics = calculate_portfolio_risk_metrics(returns, risk_free_daily=0.0)
        self.assertAlmostEqual(
            metrics["Historical VaR 99%"],
            max(0, -returns.quantile(0.01)),
        )
        self.assertNotIn("Normal VaR 99%", metrics)
        for level in ("95%", "99%"):
            low = metrics[f"Bootstrap VaR {level} Low"]
            high = metrics[f"Bootstrap VaR {level} High"]
            self.assertLessEqual(low, metrics[f"Bootstrap VaR {level}"])
            self.assertLessEqual(metrics[f"Bootstrap VaR {level}"], high)
            # Resampling cannot produce a loss beyond the largest observed one.
            self.assertLessEqual(high, 0.04)
        self.assertEqual(metrics["Historical VaR 99% Tail Observations"], 1)

    def test_bootstrap_var_recovers_known_quantile_and_repeats_exactly(self):
        returns = pd.Series(np.random.default_rng(1).standard_t(df=3, size=1_000) * 0.02)
        first = bootstrap_var(returns, 0.95)
        self.assertEqual(first, bootstrap_var(returns, 0.95))
        self.assertAlmostEqual(first["var"], -returns.quantile(0.05), delta=0.003)
        self.assertLess(first["low"], first["var"])
        self.assertGreater(first["high"], first["var"])

    def test_bootstrap_var_floors_an_all_gain_sample_at_zero(self):
        result = bootstrap_var(pd.Series([0.01, 0.02, 0.03]), 0.95)
        self.assertEqual(result, {"var": 0.0, "low": 0.0, "high": 0.0})

    def test_rejects_insufficient_or_nonfinite_returns(self):
        with self.assertRaisesRegex(ValueError, "At least two"):
            calculate_portfolio_risk_metrics(pd.Series([0.01]), risk_free_daily=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            calculate_portfolio_risk_metrics(
                pd.Series([0.01, np.inf]),
                risk_free_daily=0.0,
            )


if __name__ == "__main__":
    unittest.main()
