"""Deterministic next-open backtesting for the supported long-only strategies."""

from __future__ import annotations

from math import floor, isfinite, sqrt

import numpy as np
import pandas as pd

from trading.quant.models import (
    TRADE_DISPLAY_INITIAL_CAPITAL,
    AlignedMarketData,
    BacktestCosts,
    DrawdownAnalysis,
    FinalEvaluation,
    PerformanceMetrics,
    SegmentBacktestResult,
    StrategyConfig,
    TradeEvent,
    TrainingEvaluation,
    strategy_name,
)
from trading.quant.strategies import (
    accumulate_target_tranches,
    generate_trade_signals,
    validate_training_length,
)

TRADING_DAYS_PER_YEAR = 252
TRAIN_FRACTION = 0.70


def run_training_evaluation(
    data: AlignedMarketData,
    config: StrategyConfig,
    *,
    costs: BacktestCosts = BacktestCosts(),
    position_tranches: int = 5,
) -> TrainingEvaluation:
    """Evaluate one frozen strategy using only the chronological training prefix."""
    _validate_aligned_data(data)
    if data.bar_interval == "1m":
        return _run_intraday_training_evaluation(
            data,
            config,
            costs=costs,
            position_tranches=position_tranches,
        )
    assert data.benchmark is not None
    split_index = _split_index(len(data.asset.bars))
    validate_training_length(config, split_index)

    asset_train = data.asset.bars.iloc[:split_index].copy()
    benchmark_train = data.benchmark.bars.iloc[:split_index].copy()
    train_signals = generate_trade_signals(asset_train["close"], config)
    train_targets = accumulate_target_tranches(train_signals, position_tranches)
    training = _run_segment(
        asset_train,
        benchmark_train,
        train_targets,
        segment="training",
        previous_close_tranches=0,
        costs=costs,
        position_tranches=position_tranches,
    )

    return TrainingEvaluation(
        symbol=data.asset.symbol,
        benchmark_symbol=data.benchmark.symbol,
        bar_interval="1d",
        strategy_name=strategy_name(config),
        config=config,
        costs=costs,
        data_start=pd.Timestamp(data.asset.bars.index[0]),
        data_end=pd.Timestamp(data.asset.bars.index[-1]),
        split_index=split_index,
        split_date=pd.Timestamp(data.asset.bars.index[split_index]),
        training=training,
        asset_quality=data.asset.quality,
        benchmark_quality=data.benchmark.quality,
        warnings=data.warnings,
    )


def run_final_evaluation(
    data: AlignedMarketData,
    config: StrategyConfig,
    *,
    costs: BacktestCosts = BacktestCosts(),
    position_tranches: int = 5,
) -> FinalEvaluation:
    """Evaluate the test suffix after parameters have been frozen."""
    if data.bar_interval == "1m":
        return _run_intraday_final_evaluation(
            data,
            config,
            costs=costs,
            position_tranches=position_tranches,
        )
    training_evaluation = run_training_evaluation(
        data,
        config,
        costs=costs,
        position_tranches=position_tranches,
    )
    split_index = training_evaluation.split_index
    assert data.benchmark is not None

    full_signals = generate_trade_signals(data.asset.bars["close"], config)
    full_targets = accumulate_target_tranches(full_signals, position_tranches)
    test_targets = full_targets.iloc[split_index:].copy()
    previous_close_tranches = int(full_targets.iloc[split_index - 1])
    test = _run_segment(
        data.asset.bars.iloc[split_index:].copy(),
        data.benchmark.bars.iloc[split_index:].copy(),
        test_targets,
        segment="test",
        previous_close_tranches=previous_close_tranches,
        costs=costs,
        position_tranches=position_tranches,
    )

    return FinalEvaluation(
        symbol=training_evaluation.symbol,
        benchmark_symbol=training_evaluation.benchmark_symbol,
        bar_interval="1d",
        strategy_name=training_evaluation.strategy_name,
        config=config,
        costs=costs,
        data_start=training_evaluation.data_start,
        data_end=training_evaluation.data_end,
        split_index=split_index,
        split_date=training_evaluation.split_date,
        training=training_evaluation.training,
        test=test,
        asset_quality=training_evaluation.asset_quality,
        benchmark_quality=training_evaluation.benchmark_quality,
        warnings=training_evaluation.warnings,
    )


def _run_intraday_training_evaluation(
    data: AlignedMarketData,
    config: StrategyConfig,
    *,
    costs: BacktestCosts,
    position_tranches: int,
) -> TrainingEvaluation:
    split_index = _intraday_split_index(data)
    validate_training_length(config, split_index)
    prices = data.asset.bars.iloc[:split_index].copy()
    labels = data.session_labels[:split_index]
    signals = generate_trade_signals(prices["close"], config)
    targets = accumulate_target_tranches(signals, position_tranches)
    training = _run_intraday_segment(
        prices,
        labels,
        targets,
        segment="training",
        costs=costs,
        position_tranches=position_tranches,
    )
    return TrainingEvaluation(
        symbol=data.asset.symbol,
        benchmark_symbol=None,
        bar_interval="1m",
        strategy_name=strategy_name(config),
        config=config,
        costs=costs,
        data_start=pd.Timestamp(data.asset.bars.index[0]),
        data_end=pd.Timestamp(data.asset.bars.index[-1]),
        split_index=split_index,
        split_date=pd.Timestamp(data.asset.bars.index[split_index]),
        training=training,
        asset_quality=data.asset.quality,
        benchmark_quality=None,
        warnings=data.warnings,
    )


def _run_intraday_final_evaluation(
    data: AlignedMarketData,
    config: StrategyConfig,
    *,
    costs: BacktestCosts,
    position_tranches: int,
) -> FinalEvaluation:
    training = _run_intraday_training_evaluation(
        data,
        config,
        costs=costs,
        position_tranches=position_tranches,
    )
    full_signals = generate_trade_signals(data.asset.bars["close"], config)
    full_targets = accumulate_target_tranches(full_signals, position_tranches)
    split_index = training.split_index
    test = _run_intraday_segment(
        data.asset.bars.iloc[split_index:].copy(),
        data.session_labels[split_index:],
        full_targets.iloc[split_index:].copy(),
        segment="test",
        costs=costs,
        position_tranches=position_tranches,
    )
    return FinalEvaluation(
        symbol=training.symbol,
        benchmark_symbol=None,
        bar_interval="1m",
        strategy_name=training.strategy_name,
        config=config,
        costs=costs,
        data_start=training.data_start,
        data_end=training.data_end,
        split_index=split_index,
        split_date=training.split_date,
        training=training.training,
        test=test,
        asset_quality=training.asset_quality,
        benchmark_quality=None,
        warnings=training.warnings,
    )


def _intraday_split_index(data: AlignedMarketData) -> int:
    if data.training_sessions is None or data.test_sessions is None:
        raise ValueError("minute data requires an explicit session split")
    labels = data.session_labels
    if len(labels) != len(data.asset.bars):
        raise ValueError("minute session labels must match the asset bars")
    unique_sessions = tuple(dict.fromkeys(labels))
    if len(unique_sessions) != data.training_sessions + data.test_sessions:
        raise ValueError("minute session counts do not match the downloaded bars")
    first_test = unique_sessions[data.training_sessions]
    split_index = labels.index(first_test)
    if split_index < 2 or len(labels) - split_index < 2:
        raise ValueError("minute data must contain at least two bars in each segment")
    return split_index


def analyze_drawdown(equity_curve: pd.Series) -> DrawdownAnalysis:
    """Describe the maximum drawdown episode using an initial 1.0 high-water mark."""
    equity = _validated_series(equity_curve, "equity_curve")
    if (equity <= 0).any():
        raise ValueError("equity_curve must contain only positive values")

    high_water = equity.cummax().clip(lower=1.0)
    drawdown = equity / high_water - 1.0
    trough_position = int(np.argmin(drawdown.to_numpy(dtype=float)))
    max_drawdown = float(drawdown.iloc[trough_position])
    first_date = pd.Timestamp(equity.index[0])

    if max_drawdown == 0.0:
        return DrawdownAnalysis(
            max_drawdown=0.0,
            peak_date=first_date,
            trough_date=first_date,
            recovery_date=first_date,
            underwater_bars=0,
        )

    peak_value = float(high_water.iloc[trough_position])
    prefix = equity.iloc[: trough_position + 1].to_numpy(dtype=float)
    peak_candidates = np.flatnonzero(prefix == peak_value)
    peak_position = int(peak_candidates[-1]) if len(peak_candidates) else 0

    recovery_position: int | None = None
    for position in range(trough_position + 1, len(equity)):
        if float(equity.iloc[position]) >= peak_value:
            recovery_position = position
            break

    if recovery_position is None:
        recovery_date = None
        underwater_bars = len(equity) - 1 - peak_position
    else:
        recovery_date = pd.Timestamp(equity.index[recovery_position])
        underwater_bars = recovery_position - peak_position

    return DrawdownAnalysis(
        max_drawdown=max_drawdown,
        peak_date=pd.Timestamp(equity.index[peak_position]),
        trough_date=pd.Timestamp(equity.index[trough_position]),
        recovery_date=recovery_date,
        underwater_bars=underwater_bars,
    )


def _split_index(rows: int) -> int:
    split_index = floor(rows * TRAIN_FRACTION)
    if split_index < 2 or rows - split_index < 2:
        raise ValueError("aligned data must contain at least two training and two test bars")
    return split_index


def _validate_aligned_data(data: AlignedMarketData) -> None:
    asset = data.asset.bars
    if asset.empty:
        raise ValueError("aligned market data must not be empty")
    frames = [("asset", asset)]
    if data.bar_interval == "1d":
        if data.benchmark is None:
            raise ValueError("daily market data requires a benchmark")
        benchmark = data.benchmark.bars
        if len(asset) != len(benchmark) or not asset.index.equals(benchmark.index):
            raise ValueError("asset and benchmark bars must have identical indexes")
        frames.append(("benchmark", benchmark))
    elif data.benchmark is not None:
        raise ValueError("minute market data must not contain a benchmark")
    for label, frame in frames:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{label} bars must use a DatetimeIndex")
        if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError(f"{label} bars index must be sorted and unique")
        if not {"open", "close"}.issubset(frame.columns):
            raise ValueError(f"{label} bars must contain open and close columns")
        values = frame[["open", "close"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{label} open and close values must be finite and positive")


def _run_segment(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    target_tranches: pd.Series,
    *,
    segment: str,
    previous_close_tranches: int,
    costs: BacktestCosts,
    position_tranches: int,
) -> SegmentBacktestResult:
    if segment not in {"training", "test"}:
        raise ValueError("segment must be training or test")
    if isinstance(position_tranches, bool) or not isinstance(position_tranches, int):
        raise ValueError("position_tranches must be an integer")
    if not 1 <= position_tranches <= 20:
        raise ValueError("position_tranches must be between 1 and 20")
    if not 0 <= previous_close_tranches <= position_tranches:
        raise ValueError("previous_close_tranches is outside the configured range")
    if not prices.index.equals(benchmark_prices.index):
        raise ValueError("segment asset and benchmark indexes must match")
    if not target_tranches.index.equals(prices.index):
        raise ValueError("target tranches must match the segment price index")
    if not target_tranches.isin(range(position_tranches + 1)).all():
        raise ValueError("target tranches are outside the configured range")

    executed = target_tranches.shift(1).fillna(0).astype("int64")
    executed.iloc[0] = previous_close_tranches
    strategy = _strategy_account_path(
        prices,
        executed,
        costs=costs,
        position_tranches=position_tranches,
    )
    trades = strategy["trades"]
    assert isinstance(trades, tuple)

    buy_and_hold_positions = pd.Series(1, index=prices.index, dtype="int8", name="position")
    asset_buy_and_hold = _portfolio_path(prices, buy_and_hold_positions, costs)
    benchmark = _portfolio_path(benchmark_prices, buy_and_hold_positions, costs)
    asset_buy_and_hold_return = float(asset_buy_and_hold["equity"].iloc[-1] - 1.0)
    benchmark_return = float(benchmark["equity"].iloc[-1] - 1.0)

    warnings: list[str] = []
    metrics = _performance_metrics(
        strategy["bar_returns"],
        strategy["equity"],
        strategy["drawdown"],
        strategy["turnover"],
        len(trades),
        asset_buy_and_hold_return,
        benchmark_return,
        warnings,
        annualize=True,
    )
    drawdown_analysis = analyze_drawdown(strategy["equity"])
    return SegmentBacktestResult(
        segment=segment,
        bar_returns=strategy["bar_returns"],
        equity_curve=strategy["equity"],
        drawdown_curve=strategy["drawdown"],
        positions=strategy["positions"],
        asset_buy_and_hold_bar_returns=asset_buy_and_hold["bar_returns"],
        asset_buy_and_hold_equity_curve=asset_buy_and_hold["equity"],
        asset_buy_and_hold_drawdown_curve=asset_buy_and_hold["drawdown"],
        benchmark_bar_returns=benchmark["bar_returns"],
        benchmark_equity_curve=benchmark["equity"],
        benchmark_drawdown_curve=benchmark["drawdown"],
        metrics=metrics,
        drawdown=drawdown_analysis,
        trades=trades,
        warnings=tuple(warnings),
    )


def _run_intraday_segment(
    prices: pd.DataFrame,
    session_labels: tuple[str, ...],
    logical_targets: pd.Series,
    *,
    segment: str,
    costs: BacktestCosts,
    position_tranches: int,
) -> SegmentBacktestResult:
    if len(session_labels) != len(prices) or not logical_targets.index.equals(prices.index):
        raise ValueError("minute targets and session labels must match prices")
    executed = logical_targets.shift(1).fillna(0).astype("int64")
    for position in range(len(session_labels)):
        if position == 0 or session_labels[position] != session_labels[position - 1]:
            executed.iloc[position] = 0

    strategy = _strategy_account_path(
        prices,
        executed,
        costs=costs,
        position_tranches=position_tranches,
        session_labels=session_labels,
    )
    buy_and_hold_targets = pd.Series(
        1,
        index=prices.index,
        dtype="int64",
        name="session_buy_and_hold",
    )
    session_buy_and_hold = _strategy_account_path(
        prices,
        buy_and_hold_targets,
        costs=costs,
        position_tranches=1,
        session_labels=session_labels,
    )
    trades = strategy["trades"]
    assert isinstance(trades, tuple)
    asset_buy_and_hold_return = float(session_buy_and_hold["equity"].iloc[-1] - 1.0)
    warnings: list[str] = []
    metrics = _performance_metrics(
        strategy["bar_returns"],
        strategy["equity"],
        strategy["drawdown"],
        strategy["turnover"],
        len(trades),
        asset_buy_and_hold_return,
        None,
        warnings,
        annualize=False,
    )
    empty_benchmark = None
    return SegmentBacktestResult(
        segment=segment,  # type: ignore[arg-type]
        bar_returns=strategy["bar_returns"],
        equity_curve=strategy["equity"],
        drawdown_curve=strategy["drawdown"],
        positions=strategy["positions"],
        asset_buy_and_hold_bar_returns=session_buy_and_hold["bar_returns"],
        asset_buy_and_hold_equity_curve=session_buy_and_hold["equity"],
        asset_buy_and_hold_drawdown_curve=session_buy_and_hold["drawdown"],
        benchmark_bar_returns=empty_benchmark,
        benchmark_equity_curve=empty_benchmark,
        benchmark_drawdown_curve=empty_benchmark,
        metrics=metrics,
        drawdown=analyze_drawdown(strategy["equity"]),
        trades=trades,
        warnings=tuple(warnings),
    )


def _strategy_account_path(
    prices: pd.DataFrame,
    target_tranches: pd.Series,
    *,
    costs: BacktestCosts,
    position_tranches: int,
    session_labels: tuple[str, ...] | None = None,
) -> dict[str, pd.Series | tuple[TradeEvent, ...]]:
    initial_cash = TRADE_DISPLAY_INITIAL_CAPITAL
    tranche_budget = initial_cash / position_tranches
    cash = initial_cash
    shares = 0.0
    cost_basis = 0.0
    open_tranches = 0
    events: list[TradeEvent] = []
    equity_values: list[float] = []
    position_values: list[float] = []

    if session_labels is not None and len(session_labels) != len(prices):
        raise ValueError("session labels must match the account path")

    for position, (date, raw_target) in enumerate(target_tranches.items()):
        target = int(raw_target)
        execution_price = float(prices.loc[date, "open"])
        delta = target - open_tranches
        if delta > 0 and cash > 0:
            cash_outflow = min(tranche_budget * delta, cash)
            trading_cost = cash_outflow * costs.cost_rate
            gross_value = cash_outflow - trading_cost
            if gross_value > 0:
                quantity = gross_value / execution_price
                cash -= cash_outflow
                shares += quantity
                cost_basis += cash_outflow
                open_tranches = target
                average_cost = cost_basis / shares
                cash = max(cash, 0.0)
                events.append(
                    TradeEvent(
                        date=pd.Timestamp(date),
                        action="buy",
                        price=execution_price,
                        quantity=quantity,
                        gross_value=gross_value,
                        trading_cost=trading_cost,
                        net_cash=-cash_outflow,
                        realized_profit=None,
                        average_cost_after=average_cost,
                        position_quantity_after=shares,
                        cash_balance_after=cash,
                    )
                )
        elif delta < 0 and open_tranches > 0 and shares > 0:
            tranches_to_sell = min(-delta, open_tranches)
            average_cost = cost_basis / shares
            quantity = shares * tranches_to_sell / open_tranches
            gross_value = quantity * execution_price
            trading_cost = gross_value * costs.cost_rate
            net_cash = gross_value - trading_cost
            realized_profit = net_cash - quantity * average_cost
            cash += net_cash
            shares -= quantity
            cost_basis -= quantity * average_cost
            open_tranches -= tranches_to_sell
            if open_tranches == 0 or shares <= 1e-12:
                shares = 0.0
                cost_basis = 0.0
                average_cost_after = None
            else:
                average_cost_after = cost_basis / shares
            events.append(
                TradeEvent(
                    date=pd.Timestamp(date),
                    action="sell",
                    price=execution_price,
                    quantity=quantity,
                    gross_value=gross_value,
                    trading_cost=trading_cost,
                    net_cash=net_cash,
                    realized_profit=realized_profit,
                    average_cost_after=average_cost_after,
                    position_quantity_after=shares,
                    cash_balance_after=cash,
                )
            )

        is_session_close = session_labels is not None and (
            position == len(session_labels) - 1
            or session_labels[position] != session_labels[position + 1]
        )
        if is_session_close and shares > 0 and open_tranches > 0:
            close_price = float(prices.loc[date, "close"])
            average_cost = cost_basis / shares
            quantity = shares
            gross_value = quantity * close_price
            trading_cost = gross_value * costs.cost_rate
            net_cash = gross_value - trading_cost
            realized_profit = net_cash - cost_basis
            cash += net_cash
            shares = 0.0
            cost_basis = 0.0
            open_tranches = 0
            events.append(
                TradeEvent(
                    date=pd.Timestamp(date),
                    action="sell",
                    price=close_price,
                    quantity=quantity,
                    gross_value=gross_value,
                    trading_cost=trading_cost,
                    net_cash=net_cash,
                    realized_profit=realized_profit,
                    average_cost_after=None,
                    position_quantity_after=0.0,
                    cash_balance_after=cash,
                )
            )

        equity_values.append(cash + shares * float(prices.loc[date, "close"]))
        position_values.append(open_tranches / position_tranches)

    equity = pd.Series(equity_values, index=prices.index, dtype=float, name="equity") / initial_cash
    bar_returns = equity.pct_change(fill_method=None).rename("bar_return")
    bar_returns.iloc[0] = equity.iloc[0] - 1.0
    high_water = equity.cummax().clip(lower=1.0)
    drawdown = (equity / high_water - 1.0).rename("drawdown")
    positions = pd.Series(position_values, index=prices.index, dtype=float, name="position")
    turnover = positions.diff().fillna(positions.iloc[0]).abs().rename("turnover")
    return {
        "bar_returns": bar_returns,
        "equity": equity,
        "drawdown": drawdown,
        "positions": positions,
        "turnover": turnover,
        "trades": tuple(events),
    }


def _portfolio_path(
    prices: pd.DataFrame,
    new_positions: pd.Series,
    costs: BacktestCosts,
) -> dict[str, pd.Series]:
    old_positions = new_positions.shift(1, fill_value=0).astype("int8")
    turnover = (new_positions - old_positions).abs().astype(float)

    previous_close = prices["close"].shift(1)
    overnight_ratio = (prices["open"] / previous_close).replace([np.inf, -np.inf], np.nan)
    overnight_factor = pd.Series(1.0, index=prices.index)
    held_overnight = old_positions.eq(1)
    overnight_factor.loc[held_overnight] = overnight_ratio.loc[held_overnight]
    overnight_factor.iloc[0] = 1.0

    intraday_factor = pd.Series(1.0, index=prices.index)
    held_intraday = new_positions.eq(1)
    intraday_factor.loc[held_intraday] = (
        prices.loc[held_intraday, "close"] / prices.loc[held_intraday, "open"]
    )

    cost_factor = 1.0 - turnover * costs.cost_rate
    daily_factor = overnight_factor * cost_factor * intraday_factor
    if daily_factor.isna().any() or not np.isfinite(daily_factor.to_numpy()).all():
        raise ValueError("portfolio calculation produced a non-finite daily factor")
    if (daily_factor <= 0).any():
        raise ValueError("portfolio calculation produced a non-positive daily factor")

    bar_returns = (daily_factor - 1.0).rename("bar_return")
    equity = daily_factor.cumprod().rename("equity")
    high_water = equity.cummax().clip(lower=1.0)
    drawdown = (equity / high_water - 1.0).rename("drawdown")

    return {
        "bar_returns": bar_returns,
        "equity": equity,
        "drawdown": drawdown,
        "positions": new_positions.rename("position"),
        "turnover": turnover.rename("turnover"),
    }


def _performance_metrics(
    bar_returns: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series,
    turnover: pd.Series,
    trade_count: int,
    asset_buy_and_hold_return: float,
    benchmark_return: float | None,
    warnings: list[str],
    *,
    annualize: bool,
) -> PerformanceMetrics:
    periods = len(bar_returns)
    if periods < 2:
        raise ValueError("a backtest segment requires at least two daily factors")

    ending_equity = float(equity.iloc[-1])
    total_return = ending_equity - 1.0
    if annualize:
        annualized_return = ending_equity ** (TRADING_DAYS_PER_YEAR / periods) - 1.0
        bar_std = float(bar_returns.std(ddof=1))
        annualized_volatility = bar_std * sqrt(TRADING_DAYS_PER_YEAR)
        if bar_std == 0.0:
            sharpe_ratio = None
            warnings.append("Sharpe ratio is undefined because daily volatility is zero.")
        else:
            sharpe_ratio = float(bar_returns.mean()) / bar_std * sqrt(TRADING_DAYS_PER_YEAR)
    else:
        annualized_return = None
        annualized_volatility = None
        sharpe_ratio = None

    if trade_count < 0:
        raise ValueError("trade_count must be non-negative")
    metric_values = (
        total_return,
        float(drawdown.min()),
        float(turnover.sum()),
        asset_buy_and_hold_return,
    )
    optional_values = (annualized_return, annualized_volatility, benchmark_return)
    if not all(isfinite(value) for value in metric_values):
        raise ValueError("performance metrics must be finite")
    if any(value is not None and not isfinite(value) for value in optional_values):
        raise ValueError("optional performance metrics must be finite or None")
    if sharpe_ratio is not None and not isfinite(sharpe_ratio):
        raise ValueError("Sharpe ratio must be finite or None")

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=float(drawdown.min()),
        trade_count=trade_count,
        turnover=float(turnover.sum()),
        asset_buy_and_hold_return=asset_buy_and_hold_return,
        benchmark_return=benchmark_return,
    )


def _validated_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series) or series.empty:
        raise ValueError(f"{name} must be a non-empty Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must use a DatetimeIndex")
    if not series.index.is_monotonic_increasing or not series.index.is_unique:
        raise ValueError(f"{name} index must be sorted and unique")
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")
    return numeric
