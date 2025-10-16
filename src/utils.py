from typing import Any, Dict


def build_metrics_for_ai(
    portfolio_simple: float,
    portfolio_annual: float,
    portfolio_vol: float,
    portfolio_sharpe: float,
    portfolio_drawdown: float,
    benchmark_total: float,
    benchmark_annual: float,
    benchmark_vol: float,
    benchmark_sharpe: float,
    benchmark_drawdown: float,
) -> Dict[str, Any]:
    """Return a compact dictionary of metrics used to build AI prompts.

    Kept small and typed to make unit testing trivial.
    """
    return {
        "portfolio_simple": portfolio_simple,
        "portfolio_annual": portfolio_annual,
        "portfolio_vol": portfolio_vol,
        "portfolio_sharpe": portfolio_sharpe,
        "portfolio_drawdown": portfolio_drawdown,
        "portfolio_max_dd": portfolio_drawdown,
        "benchmark_total": benchmark_total,
        "benchmark_annual": benchmark_annual,
        "benchmark_vol": benchmark_vol,
        "benchmark_sharpe": benchmark_sharpe,
        "benchmark_drawdown": benchmark_drawdown,
    }
