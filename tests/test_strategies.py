"""Deterministic tests for strategy indicators and transaction signals."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from tragent.quant import strategies
from tragent.quant.models import MacdConfig, MovingAverageConfig, RsiConfig
from tragent.quant.strategies import (
    accumulate_target_tranches,
    calculate_macd,
    calculate_rsi,
    generate_trade_signals,
    macd_trade_signals,
    moving_average_trade_signals,
    required_warmup_bars,
    rsi_trade_signals,
    validate_training_length,
)


def close_series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index, name="close", dtype=float)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (lambda: MovingAverageConfig(short_window=1, long_window=3), "short_window"),
        (lambda: MovingAverageConfig(short_window=3, long_window=3), "long_window"),
        (
            lambda: RsiConfig(rsi_window=1, entry_threshold=30, exit_threshold=70),
            "rsi_window",
        ),
        (
            lambda: RsiConfig(rsi_window=2, entry_threshold=70, exit_threshold=30),
            "entry_threshold",
        ),
        (
            lambda: MacdConfig(fast_period=1, slow_period=3, signal_period=2),
            "fast_period",
        ),
        (
            lambda: MacdConfig(fast_period=3, slow_period=3, signal_period=2),
            "slow_period",
        ),
        (
            lambda: MacdConfig(fast_period=2, slow_period=3, signal_period=1),
            "signal_period",
        ),
    ],
)
def test_strategy_configs_reject_invalid_parameters(
    factory: Callable[[], object],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        factory()


def test_required_warmup_bars_and_training_boundary() -> None:
    configs = [
        (MovingAverageConfig(short_window=2, long_window=5), 5),
        (RsiConfig(rsi_window=5, entry_threshold=30, exit_threshold=70), 6),
        (MacdConfig(fast_period=2, slow_period=5, signal_period=3), 7),
    ]

    for config, warmup in configs:
        assert required_warmup_bars(config) == warmup
        validate_training_length(config, warmup + 2)
        with pytest.raises(ValueError, match=str(warmup + 2)):
            validate_training_length(config, warmup + 1)


def test_moving_average_signals_use_full_window_and_strict_comparison() -> None:
    config = MovingAverageConfig(short_window=2, long_window=3)

    signals = moving_average_trade_signals(close_series([1, 2, 3, 2, 1]), config)
    equal_signals = moving_average_trade_signals(close_series([2, 2, 2]), config)

    assert signals.tolist() == [0, 0, 1, 0, -1]
    assert equal_signals.tolist() == [0, 0, 0]


def test_calculate_rsi_uses_wilder_seed_and_recursion() -> None:
    config = RsiConfig(rsi_window=3, entry_threshold=30, exit_threshold=70)

    rsi = calculate_rsi(close_series([10, 11, 10, 12, 11, 13]), config)

    assert rsi.iloc[:3].isna().all()
    assert rsi.iloc[3] == pytest.approx(75.0)
    assert rsi.iloc[4] == pytest.approx(54.5454545455)
    assert rsi.iloc[5] == pytest.approx(75.0)


def test_rsi_defines_flat_as_50_and_no_losses_as_100() -> None:
    config = RsiConfig(rsi_window=2, entry_threshold=30, exit_threshold=70)

    flat = calculate_rsi(close_series([10, 10, 10]), config)
    rising = calculate_rsi(close_series([10, 11, 12]), config)

    assert flat.iloc[2] == 50.0
    assert rising.iloc[2] == 100.0


def test_rsi_signals_use_strict_threshold_crossings() -> None:
    exit_config = RsiConfig(rsi_window=2, entry_threshold=40, exit_threshold=75)
    entry_config = RsiConfig(rsi_window=2, entry_threshold=40, exit_threshold=75)

    signals = rsi_trade_signals(close_series([10, 9, 8, 9, 10, 11]), exit_config)
    equal_entry = rsi_trade_signals(close_series([10, 12, 9]), entry_config)

    assert signals.tolist() == [0, 0, 1, 0, 0, -1]
    assert equal_entry.tolist() == [0, 0, 0]


def test_rsi_strict_comparison_does_not_add_an_implicit_tolerance(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    crafted_rsi = pd.Series([np.nan, 30.0, 30.0 - 1e-13, 70.0 + 1e-13], index=index)
    monkeypatch.setattr(strategies, "calculate_rsi", lambda close, config: crafted_rsi)
    config = RsiConfig(rsi_window=2, entry_threshold=30, exit_threshold=70)

    signals = rsi_trade_signals(pd.Series(100.0, index=index), config)

    assert signals.tolist() == [0, 0, 1, -1]


def test_calculate_macd_uses_adjust_false_ewm_and_exact_warmup() -> None:
    config = MacdConfig(fast_period=2, slow_period=3, signal_period=2)

    lines = calculate_macd(close_series([1, 2, 3, 2]), config)

    assert lines["macd"].iloc[:2].isna().all()
    assert lines["macd"].iloc[2] == pytest.approx(0.3055555556)
    assert lines["macd"].iloc[3] == pytest.approx(0.0601851852)
    assert lines["signal"].iloc[:3].isna().all()
    assert lines["signal"].iloc[3] == pytest.approx(0.1419753086)


def test_macd_first_valid_point_does_not_trade_and_equal_previous_can_cross() -> None:
    config = MacdConfig(fast_period=2, slow_period=3, signal_period=2)

    signals = macd_trade_signals(close_series([1, 1, 1, 1, 2, 2]), config)

    assert signals.tolist() == [0, 0, 0, 0, 1, 0]


def test_macd_crossover_uses_equality_for_entry_and_exit(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    crafted_lines = pd.DataFrame(
        {
            "macd": [np.nan, 0.0, 1.0, 1.0, 0.0],
            "signal": [np.nan, 0.0, 0.0, 1.0, 1.0],
        },
        index=index,
    )
    monkeypatch.setattr(strategies, "calculate_macd", lambda close, config: crafted_lines)
    config = MacdConfig(fast_period=2, slow_period=3, signal_period=2)

    signals = macd_trade_signals(pd.Series(1.0, index=index), config)

    assert signals.tolist() == [0, 0, 1, 0, -1]


@pytest.mark.parametrize(
    ("config", "direct_function"),
    [
        (MovingAverageConfig(short_window=2, long_window=3), moving_average_trade_signals),
        (RsiConfig(rsi_window=2, entry_threshold=30, exit_threshold=70), rsi_trade_signals),
        (MacdConfig(fast_period=2, slow_period=3, signal_period=2), macd_trade_signals),
    ],
)
def test_dispatcher_preserves_index_and_accumulates_bounded_tranches(
    config: MovingAverageConfig | RsiConfig | MacdConfig,
    direct_function: Callable[[pd.Series, object], pd.Series],
) -> None:
    close = close_series([1, 2, 3, 2, 1, 2])

    signals = generate_trade_signals(close, config)
    targets = accumulate_target_tranches(signals, 2)

    pd.testing.assert_series_equal(signals, direct_function(close, config))
    assert signals.index.equals(close.index)
    assert pd.api.types.is_integer_dtype(signals.dtype)
    assert set(signals.unique()) <= {-1, 0, 1}
    assert targets.index.equals(close.index)
    assert set(targets.unique()) <= {0, 1, 2}
