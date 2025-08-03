# calculations.py
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def calculate_dca_portfolio(returns, weights, initial_capital, dca_enabled, dca_amount, dca_frequency):
    """Calcule l'évolution du portefeuille avec ou sans DCA."""
    if returns.empty:
        return pd.Series(), pd.Series(), pd.Series()
    
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_value = pd.Series(index=returns.index, dtype=float)
    total_invested = pd.Series(index=returns.index, dtype=float)
    
    current_value = initial_capital
    cumulative_invested = initial_capital
    portfolio_value.iloc[0] = current_value
    total_invested.iloc[0] = cumulative_invested
    
    dca_dates = set()
    if dca_enabled:
        current_date = returns.index[0]
        end_date = returns.index[-1]
        while current_date <= end_date:
            available_dates = returns.index[returns.index >= current_date]
            if len(available_dates) > 0:
                dca_dates.add(available_dates[0])
            increment = relativedelta(months=1) if dca_frequency == 'Mensuelle' else relativedelta(years=1)
            current_date += increment
            
    for i in range(1, len(returns)):
        date = returns.index[i]
        if dca_enabled and date in dca_dates:
            current_value += dca_amount
            cumulative_invested += dca_amount
        current_value *= (1 + portfolio_returns.iloc[i])
        portfolio_value.iloc[i] = current_value
        total_invested.iloc[i] = cumulative_invested
        
    return portfolio_value, total_invested, portfolio_returns

def calculate_metrics_with_dca(portfolio_returns, portfolio_value, total_invested, risk_free_rate=0.0):
    """Calcule les métriques de performance en tenant compte du DCA."""
    if portfolio_returns.empty or portfolio_returns.isnull().all():
        return 0, 0, 0, 0, 0
    
    twr = (1 + portfolio_returns).prod() - 1
    num_days = len(portfolio_returns)
    if num_days == 0: return 0, 0, 0, 0, 0
    
    simple_return = (portfolio_value.iloc[-1] / total_invested.iloc[-1]) - 1
    annualized_return = ((1 + twr) ** (252 / num_days)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / (volatility + 1e-10)
    
    return simple_return, annualized_return, volatility, sharpe_ratio, twr

def calculate_advanced_metrics(portfolio_returns, benchmark_returns, risk_free_rate=0.0):
    """Calcule les métriques de performance avancées."""
    if portfolio_returns.empty or benchmark_returns.empty:
        return {'sortino_ratio': 0, 'alpha': 0, 'beta': 0}
    
    aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    if aligned_data.empty or len(aligned_data.columns) < 2:
        return {'sortino_ratio': 0, 'alpha': 0, 'beta': 0}
    
    portfolio_aligned = aligned_data.iloc[:, 0]
    benchmark_aligned = aligned_data.iloc[:, 1]
    
    # Ratio de Sortino
    daily_risk_free = risk_free_rate / 252
    negative_returns = portfolio_aligned[portfolio_aligned < daily_risk_free] - daily_risk_free
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 1e-10
    portfolio_annual_return = ((1 + portfolio_aligned.mean()) ** 252) - 1
    sortino_ratio = (portfolio_annual_return - risk_free_rate) / downside_deviation
    
    # Alpha et Bêta
    covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1] if benchmark_aligned.var() > 1e-10 else 0
    beta = covariance / benchmark_aligned.var() if benchmark_aligned.var() > 1e-10 else 0
    benchmark_annual_return = ((1 + benchmark_aligned.mean()) ** 252) - 1
    alpha = portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
    
    return {'sortino_ratio': sortino_ratio, 'alpha': alpha, 'beta': beta}

def calculate_drawdown_series(portfolio_value):
    """Calcule la série temporelle des drawdowns."""
    if portfolio_value.empty or portfolio_value.isnull().all():
        return pd.Series()
    peak = portfolio_value.expanding(min_periods=1).max()
    return (portfolio_value / peak) - 1

def calculate_max_drawdown(portfolio_value):
    """Calcule le Drawdown Maximal."""
    if portfolio_value.empty or portfolio_value.isnull().all(): return 0
    drawdown_series = calculate_drawdown_series(portfolio_value)
    return drawdown_series.min() if not drawdown_series.empty else 0

def calculate_holding_period_analysis(portfolio_returns, max_horizon_years=20):
    """Calcule l'analyse de l'horizon de placement."""
    if portfolio_returns.empty or len(portfolio_returns) < 252:
        return {}
        
    results = {}
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    horizons = [(m / 12, f"{m}m") for m in [6, 12, 18, 24, 30, 36] if m <= max_horizon_years * 12]
    horizons.extend([(y, f"{y}a") for y in range(4, max_horizon_years + 1)])
    
    for horizon_years, label in horizons:
        horizon_days = int(horizon_years * 252)
        if horizon_days >= len(cumulative_returns): continue
        
        period_returns = np.array([(cumulative_returns.iloc[i + horizon_days] / cumulative_returns.iloc[i]) - 1 for i in range(len(cumulative_returns) - horizon_days)])
        if len(period_returns) == 0: continue

        results[horizon_years] = {
            'label': label,
            'probability_positive': (period_returns > 0).sum() / len(period_returns) * 100,
            'annualized_avg_return': ((1 + np.mean(period_returns)) ** (1/horizon_years)) - 1 if horizon_years > 0 else np.mean(period_returns)
        }
    return results