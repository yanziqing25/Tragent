"""Deterministic long-only indicators, trade signals, and tranche targets."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from trading.quant.models import (
    MacdConfig,
    MovingAverageConfig,
    RsiConfig,
    StrategyConfig,
)

__all__ = [
    "accumulate_target_tranches",
    "calculate_macd",
    "calculate_rsi",
    "generate_trade_signals",
    "macd_trade_signals",
    "moving_average_trade_signals",
    "required_warmup_bars",
    "rsi_trade_signals",
    "validate_training_length",
]


def required_warmup_bars(config: StrategyConfig) -> int:
    """Return the exact close count needed for the first valid indicator value."""
    if isinstance(config, MovingAverageConfig):
        return config.long_window
    if isinstance(config, RsiConfig):
        return config.rsi_window + 1
    if isinstance(config, MacdConfig):
        return config.slow_period + config.signal_period - 1
    raise TypeError("config must be MovingAverageConfig, RsiConfig, or MacdConfig")


def validate_training_length(config: StrategyConfig, training_rows: int) -> None:
    """Require two evaluable bars after a strategy's first valid signal."""
    if isinstance(training_rows, bool) or not isinstance(training_rows, Integral):
        raise ValueError("training_rows must be an integer")

    minimum_rows = required_warmup_bars(config) + 2
    if training_rows < minimum_rows:
        raise ValueError(
            f"training_rows must be at least {minimum_rows} for "
            f"{type(config).__name__}; got {training_rows}"
        )


def calculate_rsi(close: pd.Series, config: RsiConfig) -> pd.Series:
    """Calculate RSI with Wilder's initial average and recursive smoothing."""
    values = _validated_close_for_strategy(close, config)
    window = config.rsi_window
    changes = values.diff().to_numpy(dtype=float)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)

    average_gain = np.full(len(values), np.nan, dtype=float)
    average_loss = np.full(len(values), np.nan, dtype=float)
    average_gain[window] = gains[1 : window + 1].mean()
    average_loss[window] = losses[1 : window + 1].mean()

    for position in range(window + 1, len(values)):
        average_gain[position] = (
            average_gain[position - 1] * (window - 1) + gains[position]
        ) / window
        average_loss[position] = (
            average_loss[position - 1] * (window - 1) + losses[position]
        ) / window

    rsi = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(average_gain) & np.isfinite(average_loss)
    both_zero = valid & (average_gain == 0.0) & (average_loss == 0.0)
    no_losses = valid & (average_loss == 0.0) & (average_gain > 0.0)
    ordinary = valid & (average_loss > 0.0)

    rsi[both_zero] = 50.0
    rsi[no_losses] = 100.0
    relative_strength = np.divide(
        average_gain,
        average_loss,
        out=np.zeros_like(average_gain),
        where=ordinary,
    )
    rsi[ordinary] = 100.0 - 100.0 / (1.0 + relative_strength[ordinary])
    return pd.Series(rsi, index=values.index, name="rsi")


def calculate_macd(close: pd.Series, config: MacdConfig) -> pd.DataFrame:
    """Calculate adjust=False MACD and signal lines with fixed warm-up rules."""
    values = _validated_close_for_strategy(close, config)
    fast_ema = values.ewm(
        span=config.fast_period,
        adjust=False,
        min_periods=config.fast_period,
    ).mean()
    slow_ema = values.ewm(
        span=config.slow_period,
        adjust=False,
        min_periods=config.slow_period,
    ).mean()
    macd = (fast_ema - slow_ema).rename("macd")
    signal = macd.ewm(
        span=config.signal_period,
        adjust=False,
        min_periods=config.signal_period,
    ).mean()
    return pd.DataFrame(
        {"macd": macd.to_numpy(dtype=float), "signal": signal.to_numpy(dtype=float)},
        index=values.index,
    )


def moving_average_trade_signals(
    close: pd.Series,
    config: MovingAverageConfig,
) -> pd.Series:
    """Emit one event when the short SMA crosses the long SMA."""
    values = _validated_close_for_strategy(close, config)
    short_average = values.rolling(
        window=config.short_window,
        min_periods=config.short_window,
    ).mean()
    long_average = values.rolling(
        window=config.long_window,
        min_periods=config.long_window,
    ).mean()
    signals = np.zeros(len(values), dtype=np.int64)
    previous_bullish: bool | None = None
    for offset, (short_value, long_value) in enumerate(
        zip(short_average.to_numpy(dtype=float), long_average.to_numpy(dtype=float), strict=True)
    ):
        if not np.isfinite(short_value) or not np.isfinite(long_value):
            continue
        bullish = short_value > long_value
        if previous_bullish is None:
            if bullish:
                signals[offset] = 1
        elif bullish and not previous_bullish:
            signals[offset] = 1
        elif previous_bullish and not bullish:
            signals[offset] = -1
        previous_bullish = bullish
    return _signal_series(signals, values.index)


def rsi_trade_signals(close: pd.Series, config: RsiConfig) -> pd.Series:
    """Emit independent threshold-crossing Buy and Sell events."""
    rsi = calculate_rsi(close, config)
    signals = np.zeros(len(rsi), dtype=np.int64)
    previous_value: float | None = None

    for offset, value in enumerate(rsi.to_numpy(dtype=float)):
        if not np.isfinite(value):
            continue
        below_entry = value < config.entry_threshold and not _equal_within_roundoff(
            value, config.entry_threshold
        )
        above_exit = value > config.exit_threshold and not _equal_within_roundoff(
            value, config.exit_threshold
        )
        if below_entry and (
            previous_value is None
            or previous_value > config.entry_threshold
            or _equal_within_roundoff(previous_value, config.entry_threshold)
        ):
            signals[offset] = 1
        elif (
            previous_value is not None
            and above_exit
            and (
                previous_value < config.exit_threshold
                or _equal_within_roundoff(previous_value, config.exit_threshold)
            )
        ):
            signals[offset] = -1
        previous_value = value

    return _signal_series(signals, rsi.index)


def macd_trade_signals(close: pd.Series, config: MacdConfig) -> pd.Series:
    """Emit only causal MACD signal-line crossover events."""
    lines = calculate_macd(close, config)
    signals = np.zeros(len(lines), dtype=np.int64)
    previous_macd: float | None = None
    previous_signal: float | None = None

    for offset, (macd, signal) in enumerate(
        lines[["macd", "signal"]].itertuples(index=False, name=None)
    ):
        if not np.isfinite(macd) or not np.isfinite(signal):
            previous_macd = None
            previous_signal = None
            continue

        if previous_macd is not None and previous_signal is not None:
            if previous_macd <= previous_signal and macd > signal:
                signals[offset] = 1
            elif previous_macd >= previous_signal and macd < signal:
                signals[offset] = -1
        previous_macd = macd
        previous_signal = signal

    return _signal_series(signals, lines.index)


def generate_trade_signals(close: pd.Series, config: StrategyConfig) -> pd.Series:
    """Dispatch directly to one deterministic strategy signal function."""
    if isinstance(config, MovingAverageConfig):
        return moving_average_trade_signals(close, config)
    if isinstance(config, RsiConfig):
        return rsi_trade_signals(close, config)
    if isinstance(config, MacdConfig):
        return macd_trade_signals(close, config)
    raise TypeError("config must be MovingAverageConfig, RsiConfig, or MacdConfig")


def accumulate_target_tranches(signals: pd.Series, position_tranches: int) -> pd.Series:
    """Accumulate independent signals into a bounded target-tranche series."""

    if isinstance(position_tranches, bool) or not isinstance(position_tranches, Integral):
        raise ValueError("position_tranches must be an integer")
    if not 1 <= position_tranches <= 20:
        raise ValueError("position_tranches must be between 1 and 20")
    if not isinstance(signals, pd.Series) or signals.empty:
        raise ValueError("signals must be a non-empty Series")
    if not isinstance(signals.index, pd.DatetimeIndex):
        raise ValueError("signals must use a DatetimeIndex")
    if not signals.index.is_monotonic_increasing or not signals.index.is_unique:
        raise ValueError("signals index must be sorted and unique")
    if not signals.isin((-1, 0, 1)).all():
        raise ValueError("signals must contain only -1, 0, and 1")

    targets = np.zeros(len(signals), dtype=np.int64)
    current = 0
    for offset, signal in enumerate(signals.to_numpy(dtype=np.int64)):
        current = min(position_tranches, max(0, current + int(signal)))
        targets[offset] = current
    return pd.Series(targets, index=signals.index, dtype="int64", name="target_tranches")


def _validated_close_for_strategy(
    close: pd.Series,
    config: StrategyConfig,
) -> pd.Series:
    values = _validate_close(close)
    minimum_rows = required_warmup_bars(config)
    if len(values) < minimum_rows:
        raise ValueError(
            f"close must contain at least {minimum_rows} rows for "
            f"{type(config).__name__}; got {len(values)}"
        )
    return values


def _validate_close(close: pd.Series) -> pd.Series:
    if not isinstance(close, pd.Series):
        raise ValueError("close must be a pandas Series")
    if close.empty:
        raise ValueError("close must not be empty")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("close index must be a DatetimeIndex")
    if close.index.hasnans:
        raise ValueError("close index must not contain missing dates")
    if not close.index.is_unique:
        raise ValueError("close index must contain unique dates")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be sorted in ascending order")
    if is_bool_dtype(close.dtype) or not is_numeric_dtype(close.dtype):
        raise ValueError("close values must be numeric")

    values = close.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError("close values must be finite")
    if (values <= 0.0).any():
        raise ValueError("close values must be greater than zero")
    return pd.Series(values, index=close.index, name=close.name, dtype=float)


def _signal_series(values: object, index: pd.DatetimeIndex) -> pd.Series:
    signals = pd.Series(values, index=index, dtype="int64", name="trade_signal")
    if not signals.isin((-1, 0, 1)).all():
        raise ValueError("trade signals must contain only -1, 0, and 1")
    return signals


def _equal_within_roundoff(left: float, right: float) -> bool:
    """Treat only machine-scale representation noise as mathematical equality."""
    scale = max(1.0, abs(left), abs(right))
    tolerance = 4.0 * np.finfo(float).eps * scale
    return abs(left - right) <= tolerance
