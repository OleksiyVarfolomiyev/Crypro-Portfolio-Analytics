"""Transparent risk statistics for daily crypto-portfolio returns."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def calculate_portfolio_risk_metrics(
    returns: pd.Series,
    risk_free_daily: float,
    periods_per_year: int = 365,
) -> dict[str, float | int]:
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    if not np.isfinite(risk_free_daily):
        raise ValueError("risk_free_daily must be finite.")

    daily_returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(daily_returns) < 2 or not np.isfinite(daily_returns.to_numpy()).all():
        raise ValueError("At least two finite daily portfolio returns are required.")

    daily_mean = float(daily_returns.mean())
    daily_vol = float(daily_returns.std(ddof=1))
    excess = daily_returns - risk_free_daily
    downside_deviation = float(np.sqrt(np.square(np.minimum(excess, 0.0)).mean()))
    sharpe = (
        float(excess.mean()) / daily_vol * np.sqrt(periods_per_year)
        if daily_vol > 0
        else float("nan")
    )
    sortino = (
        float(excess.mean()) / downside_deviation * np.sqrt(periods_per_year)
        if downside_deviation > 0
        else float("nan")
    )

    quantile_95 = float(daily_returns.quantile(0.05))
    quantile_99 = float(daily_returns.quantile(0.01))
    normal = NormalDist(mu=daily_mean, sigma=daily_vol)
    normal_var_95 = max(0.0, -normal.inv_cdf(0.05))
    normal_var_99 = max(0.0, -normal.inv_cdf(0.01))

    return {
        "Sharpe (annualized)": sharpe,
        "Sortino (annualized)": sortino,
        "Arithmetic Return (annualized)": daily_mean * periods_per_year,
        "Hist Vol (daily)": daily_vol,
        "Hist Vol (annualized)": daily_vol * np.sqrt(periods_per_year),
        "Historical VaR 95%": max(0.0, -quantile_95),
        "Historical VaR 99%": max(0.0, -quantile_99),
        "Normal VaR 95%": normal_var_95,
        "Normal VaR 99%": normal_var_99,
        "Daily Return Observations": int(len(daily_returns)),
        "Historical VaR 99% Tail Observations": int((daily_returns <= quantile_99).sum()),
        "Largest Daily Loss": float(daily_returns.min()),
    }
