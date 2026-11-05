"""Transparent risk statistics for daily crypto-portfolio returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hl_trade_analytics import bootstrap_var


def calculate_portfolio_risk_metrics(
    returns: pd.Series,
    risk_free_daily: float,
    periods_per_year: int = 365,
) -> dict[str, float | int]:
    """Return annualized performance, volatility, and value-at-risk statistics.

    ``returns`` holds simple daily returns as fractions; non-numeric and missing
    values are dropped. ``risk_free_daily`` is on the same daily scale. VaR values
    are positive loss fractions, floored at zero when the quantile is a gain.
    Ratios are NaN when their denominator is zero.
    """
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
    # Target semideviation: gains count as zero but stay in the denominator.
    downside_deviation = float(np.sqrt(np.square(np.minimum(excess, 0.0)).mean()))
    sharpe = (
        float(excess.mean()) / daily_vol * np.sqrt(periods_per_year)
        if daily_vol > 0
        else float("nan")
    )
    # 252-day variants are for comparison with conventions from equity markets.
    sharpe_252_raw = (
        daily_mean / daily_vol * np.sqrt(252)
        if daily_vol > 0
        else float("nan")
    )
    sharpe_252_excess = (
        float(excess.mean()) / daily_vol * np.sqrt(252)
        if daily_vol > 0
        else float("nan")
    )
    sortino = (
        float(excess.mean()) / downside_deviation * np.sqrt(periods_per_year)
        if downside_deviation > 0
        else float("nan")
    )

    # Left-tail return quantiles: the 5% and 1% worst-day cutoffs.
    quantile_95 = float(daily_returns.quantile(0.05))
    quantile_99 = float(daily_returns.quantile(0.01))
    # Resampling keeps the sample's fat tails, which a normal model would understate.
    bootstrap_95 = bootstrap_var(daily_returns, 0.95)
    bootstrap_99 = bootstrap_var(daily_returns, 0.99)

    return {
        "Sharpe (annualized)": sharpe,
        "Sharpe (excess, 252d)": sharpe_252_excess,
        "Sharpe (raw, 252d)": sharpe_252_raw,
        "Sortino (annualized)": sortino,
        "Arithmetic Return (annualized)": daily_mean * periods_per_year,
        "Hist Vol (daily)": daily_vol,
        "Hist Vol (annualized)": daily_vol * np.sqrt(periods_per_year),
        "Historical VaR 95%": max(0.0, -quantile_95),
        "Historical VaR 99%": max(0.0, -quantile_99),
        "Bootstrap VaR 95%": bootstrap_95["var"],
        "Bootstrap VaR 95% Low": bootstrap_95["low"],
        "Bootstrap VaR 95% High": bootstrap_95["high"],
        "Bootstrap VaR 99%": bootstrap_99["var"],
        "Bootstrap VaR 99% Low": bootstrap_99["low"],
        "Bootstrap VaR 99% High": bootstrap_99["high"],
        "Daily Return Observations": int(len(daily_returns)),
        # Few tail observations mean the 99% historical estimate is unreliable.
        "Historical VaR 99% Tail Observations": int((daily_returns <= quantile_99).sum()),
        "Largest Daily Loss": float(daily_returns.min()),
    }
