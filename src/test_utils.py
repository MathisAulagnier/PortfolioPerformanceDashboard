from utils import build_metrics_for_ai


def test_build_metrics_for_ai_basic():
    d = build_metrics_for_ai(
        portfolio_simple=0.1,
        portfolio_annual=0.08,
        portfolio_vol=0.12,
        portfolio_sharpe=1.2,
        portfolio_drawdown=-0.15,
        benchmark_total=0.05,
        benchmark_annual=0.04,
        benchmark_vol=0.10,
        benchmark_sharpe=0.8,
        benchmark_drawdown=-0.12,
    )

    expected_keys = {
        "portfolio_simple",
        "portfolio_annual",
        "portfolio_vol",
        "portfolio_sharpe",
        "portfolio_drawdown",
        "portfolio_max_dd",
        "benchmark_total",
        "benchmark_annual",
        "benchmark_vol",
        "benchmark_sharpe",
        "benchmark_drawdown",
    }

    assert set(d.keys()) == expected_keys
    assert d["portfolio_simple"] == 0.1
    assert d["portfolio_max_dd"] == d["portfolio_drawdown"]
