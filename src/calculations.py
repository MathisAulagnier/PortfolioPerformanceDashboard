from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def calculate_dca_portfolio(
    returns: pd.DataFrame | pd.Series,
    weights: list[float] | float,
    initial_capital: float,
    dca_enabled: bool,
    dca_amount: float,
    dca_frequency: str,
):
    """Calcule l'évolution du portefeuille (ou benchmark) avec/sans DCA.

    returns: DataFrame des rendements journaliers (colonnes = actifs) ou Series.
    weights: liste des poids (somme=1) ou 1.0 si Series.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame("asset")
        if not isinstance(weights, (int, float)):
            weights = 1.0
        weights = [float(weights)]

    if returns.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    w = np.array(weights, dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1)

    portfolio_returns = (returns * w).sum(axis=1)
    portfolio_returns = portfolio_returns.dropna()

    portfolio_value = pd.Series(index=portfolio_returns.index, dtype=float)
    total_invested = pd.Series(index=portfolio_returns.index, dtype=float)

    current_value = float(initial_capital)
    cumulative_invested = float(initial_capital)

    portfolio_value.iloc[0] = current_value
    total_invested.iloc[0] = cumulative_invested

    dca_dates = set()
    if dca_enabled:
        current_date = portfolio_returns.index[0]
        end_date = portfolio_returns.index[-1]
        while current_date <= end_date:
            available_dates = portfolio_returns.index[portfolio_returns.index >= current_date]
            if len(available_dates) > 0:
                dca_dates.add(available_dates[0])
            current_date += relativedelta(months=1) if dca_frequency == "Mensuelle" else relativedelta(years=1)

    for i in range(1, len(portfolio_returns)):
        date = portfolio_returns.index[i]
        if dca_enabled and date in dca_dates:
            current_value += float(dca_amount)
            cumulative_invested += float(dca_amount)
        current_value *= (1 + float(portfolio_returns.iloc[i]))
        portfolio_value.iloc[i] = current_value
        total_invested.iloc[i] = cumulative_invested

    return portfolio_value, total_invested, portfolio_returns


def calculate_metrics_with_dca(
    portfolio_returns: pd.Series,
    portfolio_value: pd.Series,
    total_invested: pd.Series,
    risk_free_rate: float = 0.0,
):
    if portfolio_returns.empty or portfolio_returns.isnull().all():
        return 0.0, 0.0, 0.0, 0.0, 0.0

    simple_return = (portfolio_value.iloc[-1] / total_invested.iloc[-1]) - 1
    twr = (1 + portfolio_returns).prod() - 1
    num_days = len(portfolio_returns)
    if num_days == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    annualized_return = ((1 + twr) ** (252 / num_days)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / (volatility + 1e-10)
    return float(simple_return), float(annualized_return), float(volatility), float(sharpe_ratio), float(twr)


def calculate_advanced_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
):
    if portfolio_returns.empty or benchmark_returns.empty:
        return {"sortino_ratio": 0.0, "alpha": 0.0, "beta": 0.0}

    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty or len(aligned.columns) < 2:
        return {"sortino_ratio": 0.0, "alpha": 0.0, "beta": 0.0}

    p = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]

    daily_rf = risk_free_rate / 252
    neg = p[p < daily_rf] - daily_rf
    downside_dev = neg.std() * np.sqrt(252) if len(neg) > 0 else 1e-10
    p_ann = (1 + p.mean()) ** 252 - 1
    sortino = (p_ann - risk_free_rate) / downside_dev

    beta = 0.0
    if b.var() > 1e-10:
        cov = np.cov(p, b)[0, 1]
        beta = cov / b.var()

    b_ann = (1 + b.mean()) ** 252 - 1
    alpha = p_ann - (risk_free_rate + beta * (b_ann - risk_free_rate))
    return {"sortino_ratio": float(sortino), "alpha": float(alpha), "beta": float(beta)}


def calculate_drawdown_series(portfolio_value: pd.Series) -> pd.Series:
    if portfolio_value.empty or portfolio_value.isnull().all():
        return pd.Series(dtype=float)
    peak = portfolio_value.expanding(min_periods=1).max()
    return (portfolio_value / peak) - 1


def calculate_max_drawdown(portfolio_value: pd.Series) -> float:
    if portfolio_value.empty or portfolio_value.isnull().all():
        return 0.0
    return float(calculate_drawdown_series(portfolio_value).min())


def calculate_holding_period_analysis(portfolio_returns: pd.Series, max_horizon_years: int = 20) -> dict:
    if portfolio_returns.empty or len(portfolio_returns) < 252:
        return {}
    results: dict = {}
    cumulative = (1 + portfolio_returns).cumprod()
    horizons = []
    for months in [6, 12, 18, 24, 30, 36]:
        if months <= max_horizon_years * 12:
            horizons.append((months / 12, f"{months}m"))
    for years in range(4, max_horizon_years + 1):
        horizons.append((years, f"{years}a"))

    for horizon_years, label in horizons:
        horizon_days = int(horizon_years * 252)
        if horizon_days >= len(cumulative):
            continue
        window_returns = []
        for i in range(len(cumulative) - horizon_days):
            start_v = cumulative.iloc[i]
            end_v = cumulative.iloc[i + horizon_days]
            window_returns.append((end_v / start_v) - 1)
        if not window_returns:
            continue
        arr = np.array(window_returns)
        pos = arr[arr > 0]
        results[horizon_years] = {
            "label": label,
            "probability_positive": len(pos) / len(arr) * 100,
            "avg_return": float(np.mean(arr)),
            "median_return": float(np.median(arr)),
            "avg_positive_return": float(np.mean(pos)) if len(pos) > 0 else 0.0,
            "avg_negative_return": float(np.mean(arr[arr <= 0])) if np.any(arr <= 0) else 0.0,
            "percentile_10": float(np.percentile(arr, 10)),
            "percentile_90": float(np.percentile(arr, 90)),
            "num_periods": int(len(arr)),
            "annualized_avg_return": ((1 + float(np.mean(arr))) ** (1 / horizon_years)) - 1 if horizon_years > 0 else float(np.mean(arr)),
        }
    return results
