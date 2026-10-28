"""Deterministic long-only indicators, trade signals, and tranche targets."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from tragent.quant.models import (
    IntradayRsiConfig,
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
    "intraday_rsi_targets",
    "macd_trade_signals",
    "moving_average_trade_signals",
    "required_warmup_bars",
    "rsi_trade_signals",
    "strategy_targets",
    "validate_training_length",
]

_INTRADAY_RSI = RsiConfig(14, 30, 70)
_INTRADAY_FIRST_SIGNAL = 59
_INTRADAY_ENTRY_CUTOFF = 30
_INTRADAY_COOLDOWN = 5
_INTRADAY_MINIMUM_SESSION = 90


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
    minimum_rows = required_warmup_bars(config) + 2
    if training_rows < minimum_rows:
        raise ValueError(
            f"training_rows must be at least {minimum_rows} for "
            f"{type(config).__name__}; got {training_rows}"
        )


def calculate_rsi(close: pd.Series, config: RsiConfig) -> pd.Series:
    """Calculate RSI with Wilder's initial average and recursive smoothing."""
    window = config.rsi_window
    changes = close.diff().to_numpy(dtype=float)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)

    average_gain = np.full(len(close), np.nan, dtype=float)
    average_loss = np.full(len(close), np.nan, dtype=float)
    average_gain[window] = gains[1 : window + 1].mean()
    average_loss[window] = losses[1 : window + 1].mean()

    for position in range(window + 1, len(close)):
        average_gain[position] = (
            average_gain[position - 1] * (window - 1) + gains[position]
        ) / window
        average_loss[position] = (
            average_loss[position - 1] * (window - 1) + losses[position]
        ) / window

    rsi = np.full(len(close), np.nan, dtype=float)
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
    return pd.Series(rsi, index=close.index, name="rsi")


def calculate_macd(close: pd.Series, config: MacdConfig) -> pd.DataFrame:
    """Calculate adjust=False MACD and signal lines with fixed warm-up rules."""
    fast_ema = close.ewm(
        span=config.fast_period,
        adjust=False,
        min_periods=config.fast_period,
    ).mean()
    slow_ema = close.ewm(
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
        index=close.index,
    )


def moving_average_trade_signals(
    close: pd.Series,
    config: MovingAverageConfig,
) -> pd.Series:
    """Emit one event when the short SMA crosses the long SMA."""
    short_average = close.rolling(
        window=config.short_window,
        min_periods=config.short_window,
    ).mean()
    long_average = close.rolling(
        window=config.long_window,
        min_periods=config.long_window,
    ).mean()

    short_arr = short_average.to_numpy(dtype=float)
    long_arr = long_average.to_numpy(dtype=float)
    valid = np.isfinite(short_arr) & np.isfinite(long_arr)
    signals = np.zeros(len(close), dtype=np.int64)

    if np.any(valid):
        valid_indices = np.flatnonzero(valid)
        bullish = short_arr[valid_indices] > long_arr[valid_indices]

        # First valid observation: enter if already bullish
        if bullish[0]:
            signals[valid_indices[0]] = 1

        # Crossovers between subsequent valid observations
        if len(bullish) > 1:
            transitions = np.diff(bullish.astype(np.int64))
            signals[valid_indices[1:]] = transitions

    return _signal_series(signals, close.index)


def rsi_trade_signals(close: pd.Series, config: RsiConfig) -> pd.Series:
    """Emit independent threshold-crossing Buy and Sell events."""
    rsi = calculate_rsi(close, config)
    signals = np.zeros(len(rsi), dtype=np.int64)

    rsi_arr = rsi.to_numpy(dtype=float)
    valid = np.isfinite(rsi_arr)
    if np.any(valid):
        valid_indices = np.flatnonzero(valid)
        vals = rsi_arr[valid_indices]

        eps_tol = 4.0 * sys.float_info.epsilon
        eq_entry = np.isclose(vals, config.entry_threshold, rtol=eps_tol, atol=eps_tol)
        eq_exit = np.isclose(vals, config.exit_threshold, rtol=eps_tol, atol=eps_tol)

        below_entry = (vals < config.entry_threshold) & (~eq_entry)
        above_exit = (vals > config.exit_threshold) & (~eq_exit)

        # First valid observation: enter if below entry threshold
        if below_entry[0]:
            signals[valid_indices[0]] = 1

        # Subsequent threshold crossings
        if len(vals) > 1:
            prev_vals = vals[:-1]
            prev_eq_entry = eq_entry[:-1]
            prev_eq_exit = eq_exit[:-1]

            cross_entry = below_entry[1:] & ((prev_vals > config.entry_threshold) | prev_eq_entry)
            cross_exit = above_exit[1:] & ((prev_vals < config.exit_threshold) | prev_eq_exit)

            signals[valid_indices[1:][cross_entry]] = 1
            signals[valid_indices[1:][cross_exit]] = -1

    return _signal_series(signals, rsi.index)


def macd_trade_signals(close: pd.Series, config: MacdConfig) -> pd.Series:
    """Emit only causal MACD signal-line crossover events."""
    lines = calculate_macd(close, config)
    signals = np.zeros(len(lines), dtype=np.int64)

    macd_arr = lines["macd"].to_numpy(dtype=float)
    signal_arr = lines["signal"].to_numpy(dtype=float)
    valid = np.isfinite(macd_arr) & np.isfinite(signal_arr)

    if np.count_nonzero(valid) >= 2:
        valid_indices = np.flatnonzero(valid)
        # Only consecutive valid rows can cross (gaps reset causal history)
        is_adjacent = valid_indices[1:] == valid_indices[:-1] + 1
        prev_m = macd_arr[valid_indices[:-1]]
        prev_s = signal_arr[valid_indices[:-1]]
        curr_m = macd_arr[valid_indices[1:]]
        curr_s = signal_arr[valid_indices[1:]]

        cross_up = is_adjacent & (prev_m <= prev_s) & (curr_m > curr_s)
        cross_down = is_adjacent & (prev_m >= prev_s) & (curr_m < curr_s)

        signals[valid_indices[1:][cross_up]] = 1
        signals[valid_indices[1:][cross_down]] = -1

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


def strategy_targets(
    bars: pd.DataFrame,
    config: StrategyConfig,
    position_tranches: int,
    session_labels: tuple[str, ...] | None = None,
) -> pd.Series:
    """Return completed-bar target tranches; the engine fills them at the next open."""
    if isinstance(config, IntradayRsiConfig):
        return intraday_rsi_targets(bars, session_labels, config)
    validate_training_length(config, len(bars))
    signals = generate_trade_signals(bars["close"], config)
    return accumulate_target_tranches(signals, position_tranches)


def intraday_rsi_targets(
    bars: pd.DataFrame, session_labels: tuple[str, ...], config: IntradayRsiConfig
) -> pd.Series:
    """Reset indicators and trade state independently for each complete session."""
    result = pd.Series(0, index=bars.index, dtype="int64", name="target_tranches")
    labels = np.asarray(session_labels)
    for day in dict.fromkeys(session_labels):
        session = bars.loc[labels == day]
        if len(session) < _INTRADAY_MINIMUM_SESSION:
            raise ValueError(
                f"intraday RSI requires at least {_INTRADAY_MINIMUM_SESSION} bars per session"
            )
        result.loc[session.index] = _intraday_rsi_session(session, config)
    return result


def _intraday_rsi_session(bars: pd.DataFrame, config: IntradayRsiConfig) -> np.ndarray:
    """Hold one tranche between an RSI entry and the exit rule locked at that entry."""
    rsi = calculate_rsi(bars["close"], _INTRADAY_RSI)
    lower, upper = _INTRADAY_RSI.entry_threshold, _INTRADAY_RSI.exit_threshold
    entry = ((rsi < lower) & (rsi.shift(1) >= lower)).to_numpy()
    rsi_exit = ((rsi > upper) & (rsi.shift(1) <= upper)).to_numpy()
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    vwap = (typical * bars["volume"]).cumsum() / bars["volume"].cumsum().replace(0, np.nan)
    below = (bars["close"] < vwap).to_numpy()
    at_vwap = (bars["close"] >= vwap).to_numpy()
    last_entry_fill = len(bars) - _INTRADAY_ENTRY_CUTOFF
    targets = np.zeros(len(bars), dtype=np.int64)
    holding, vwap_exit, last_exit = False, False, -_INTRADAY_COOLDOWN
    for signal in range(_INTRADAY_FIRST_SIGNAL, len(bars) - 1):
        fill = signal + 1
        if holding and (at_vwap[signal] if vwap_exit else rsi_exit[signal]):
            holding, last_exit = False, fill
        if (
            not holding
            and entry[signal]
            and (config.policy != "rsi_vwap_exit" or below[signal])
            and fill < last_entry_fill
            and fill - last_exit >= _INTRADAY_COOLDOWN
        ):
            holding = True
            vwap_exit = config.policy != "intraday_rsi" and below[signal]
        targets[signal] = int(holding)
    return targets


def accumulate_target_tranches(signals: pd.Series, position_tranches: int) -> pd.Series:
    """Accumulate independent signals into a bounded target-tranche series."""
    targets = np.zeros(len(signals), dtype=np.int64)
    current = 0
    for offset, signal in enumerate(signals.to_numpy(dtype=np.int64)):
        current = min(position_tranches, max(0, current + int(signal)))
        targets[offset] = current
    return pd.Series(targets, index=signals.index, dtype="int64", name="target_tranches")


def _signal_series(values: object, index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(values, index=index, dtype="int64", name="trade_signal")
