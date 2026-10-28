import pytest

import tragent.quant as quant
from tragent.quant.models import (
    BacktestCosts,
    MacdConfig,
    MovingAverageConfig,
    RsiConfig,
    strategy_name,
    strategy_parameters,
)


def test_strategy_configs_have_stable_names_and_parameters() -> None:
    moving_average = MovingAverageConfig(short_window=5, long_window=20)
    rsi = RsiConfig(rsi_window=14, entry_threshold=30, exit_threshold=70)
    macd = MacdConfig(fast_period=12, slow_period=26, signal_period=9)

    assert strategy_name(moving_average) == "moving_average"
    assert strategy_name(rsi) == "rsi"
    assert strategy_name(macd) == "macd"
    assert strategy_parameters(moving_average) == {"short_window": 5, "long_window": 20}
    assert strategy_parameters(rsi) == {
        "rsi_window": 14,
        "entry_threshold": 30.0,
        "exit_threshold": 70.0,
    }
    assert strategy_parameters(macd) == {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    }


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MovingAverageConfig(short_window=1, long_window=20),
        lambda: MovingAverageConfig(short_window=20, long_window=20),
        lambda: RsiConfig(rsi_window=1, entry_threshold=30, exit_threshold=70),
        lambda: RsiConfig(rsi_window=14, entry_threshold=70, exit_threshold=30),
        lambda: RsiConfig(rsi_window=14, entry_threshold=float("nan"), exit_threshold=70),
        lambda: MacdConfig(fast_period=26, slow_period=12, signal_period=9),
        lambda: MacdConfig(fast_period=12, slow_period=26, signal_period=1),
    ],
)
def test_strategy_configs_reject_invalid_parameters(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_backtest_costs_default_to_five_plus_five_bps() -> None:
    costs = BacktestCosts()

    assert costs.transaction_cost_bps == 5.0
    assert costs.slippage_bps == 5.0
    assert costs.total_bps == 10.0
    assert costs.cost_rate == pytest.approx(0.001)


@pytest.mark.parametrize(
    "costs",
    [
        {"transaction_cost_bps": -1},
        {"slippage_bps": float("inf")},
        {"transaction_cost_bps": 5_000, "slippage_bps": 5_000},
    ],
)
def test_backtest_costs_reject_invalid_values(costs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BacktestCosts(**costs)


def test_package_exports_the_stable_quant_api() -> None:
    expected_names = {
        "MarketDataDownloadError",
        "MarketDataProvider",
        "YFinanceProvider",
        "align_market_data",
        "calculate_asset_statistics",
        "generate_trade_signals",
        "run_training_evaluation",
        "run_final_evaluation",
        "analyze_drawdown",
    }

    assert expected_names <= set(quant.__all__)
    assert all(hasattr(quant, name) for name in expected_names)
