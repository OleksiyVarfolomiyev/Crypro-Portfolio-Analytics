import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Modules live in the repository root and its execution/ and research/ folders.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "execution"), str(ROOT / "research")]

from portfolio_risk import calculate_portfolio_risk_metrics


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

    def test_empirical_var_and_normal_var_are_distinguished(self):
        returns = pd.Series([-0.04, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10])
        metrics = calculate_portfolio_risk_metrics(returns, risk_free_daily=0.0)
        self.assertAlmostEqual(
            metrics["Historical VaR 99%"],
            max(0, -returns.quantile(0.01)),
        )
        self.assertAlmostEqual(
            metrics["Normal VaR 99%"],
            max(0, -(returns.mean() - 2.326347874 * returns.std(ddof=1))),
            places=8,
        )
        self.assertAlmostEqual(
            metrics["Normal VaR 99% (zero mean)"],
            2.326347874 * returns.std(ddof=1),
            places=8,
        )
        self.assertEqual(metrics["Historical VaR 99% Tail Observations"], 1)

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
