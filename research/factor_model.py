"""Factor regressions and factor series for the crypto portfolio notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HAC_MAX_LAGS = 7
SIGNIFICANT_T_STAT = 2.0


@dataclass(frozen=True)
class FactorFit:
    """One regression's estimates; ``alpha`` and betas are per day, as estimated."""

    names: list[str]
    alpha: float
    alpha_t_stat: float
    betas: pd.Series
    t_stats: pd.Series
    r_squared: float
    adjusted_r_squared: float
    observations: int


def btc_momentum_factor(
    btc_returns: pd.Series, risk_free_daily: float, lookback_days: int = 30
) -> pd.Series:
    """Return BTC's excess return, long after a positive trailing return and short after a negative one.

    The signal is lagged one day, so each day's position uses only earlier returns.
    """
    trailing = np.log1p(btc_returns).rolling(lookback_days, min_periods=lookback_days).sum().shift(1)
    return np.sign(trailing) * (btc_returns - risk_free_daily)


def btc_realized_vol(
    btc_returns: pd.Series, window_days: int = 30, periods_per_year: int = 365
) -> pd.Series:
    """Return annualized trailing BTC volatility, lagged one day."""
    return btc_returns.rolling(window_days, min_periods=window_days).std().shift(1) * np.sqrt(periods_per_year)


def fit_factor_regression(
    dependent: pd.Series, factors: pd.DataFrame, state_factors: tuple[str, ...] = ()
) -> FactorFit:
    """Fit OLS with Newey-West t-stats on the dates where every series is available.

    ``state_factors`` are conditions rather than returns, so they are standardized
    within this sample. The intercept is then alpha at their average level, and each
    beta is the daily return per one standard deviation.
    """
    data = pd.concat([dependent.rename("__dependent__"), factors], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    names = list(factors.columns)
    if len(data) <= len(names) + 1:
        raise ValueError(f"Only {len(data)} observations for {len(names)} factors.")
    for name in state_factors:
        scale = data[name].std()
        if not scale > 0:
            raise ValueError(f"State factor {name} does not vary in this sample.")
        data[name] = (data[name] - data[name].mean()) / scale
    y = data["__dependent__"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(data)), data[names].to_numpy(dtype=float)])
    coefficients, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < X.shape[1]:
        raise ValueError("Factor model is rank-deficient; check factor correlations or history coverage.")
    residuals = y - X @ coefficients
    r_squared = 1.0 - (residuals @ residuals) / np.sum((y - y.mean()) ** 2)
    adjusted = 1.0 - (1.0 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1])
    scores = X * residuals[:, None]
    meat = scores.T @ scores
    lags = min(HAC_MAX_LAGS, len(y) - 1)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    bread = np.linalg.pinv(X.T @ X)
    standard_errors = np.sqrt(np.maximum(np.diag(bread @ meat @ bread), 0.0))
    t_stats = np.divide(coefficients, standard_errors, out=np.full_like(coefficients, np.nan), where=standard_errors > 0)
    return FactorFit(
        names=names,
        alpha=float(coefficients[0]),
        alpha_t_stat=float(t_stats[0]),
        betas=pd.Series(coefficients[1:], index=names),
        t_stats=pd.Series(t_stats[1:], index=names),
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted),
        observations=len(y),
    )


def independent_factors(factors: pd.DataFrame) -> list[str]:
    """Return factor names in order, skipping any that earlier factors already explain.

    A short window can make factors identical; BTC momentum equals BTC's own return
    when the trailing sign never changes.
    """
    data = factors.replace([np.inf, -np.inf], np.nan).dropna()
    kept: list[str] = []
    for name in factors.columns:
        candidate = np.column_stack([np.ones(len(data)), data[kept + [name]].to_numpy(dtype=float)])
        if np.linalg.matrix_rank(candidate) == candidate.shape[1]:
            kept.append(name)
    return kept


def split_in_half(dates: pd.Index) -> tuple[pd.Index, pd.Index]:
    """Split sorted dates into an earlier and a later half."""
    ordered = dates.sort_values()
    middle = len(ordered) // 2
    return ordered[:middle], ordered[middle:]


def alpha_verdict(alpha: float, t_stat: float) -> str:
    """Describe one out-of-sample alpha estimate."""
    if alpha <= 0:
        return "does not survive: alpha is zero or negative"
    if t_stat < SIGNIFICANT_T_STAT:
        return "not established: alpha is positive but not distinguishable from zero"
    return "survives: alpha is positive and statistically significant"
