"""Deterministic next-open backtesting for the supported long-only strategies."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from math import floor, sqrt

import numpy as np
import pandas as pd

from tragent.quant.models import (
    TRADE_DISPLAY_INITIAL_CAPITAL,
    AlignedMarketData,
    BacktestCosts,
    DrawdownAnalysis,
    FinalEvaluation,
    MarketData,
    PerformanceMetrics,
    SegmentBacktestResult,
    SegmentName,
    StrategyConfig,
    TradeEvent,
    TrainingEvaluation,
    strategy_name,
)
from tragent.quant.strategies import strategy_targets

TRADING_DAYS_PER_YEAR = 252
TRAIN_FRACTION = 0.70
_DEFAULT_COSTS = BacktestCosts()


def run_training_evaluation(
    data: AlignedMarketData,
    config: StrategyConfig,
    *,
    costs: BacktestCosts = _DEFAULT_COSTS,
    position_tranches: int = 5,
) -> TrainingEvaluation:
    """Evaluate one frozen strategy using only the chronological training prefix."""
    split_index = _split_index(data)
    targets = _targets(data, config, position_tranches, stop=split_index)
    index = data.asset.bars.index
    return TrainingEvaluation(
        symbol=data.asset.symbol,
        benchmark_symbol=None if data.benchmark is None else data.benchmark.symbol,
        bar_interval=data.bar_interval,
        strategy_name=strategy_name(config),
        config=config,
        costs=costs,
        data_start=pd.Timestamp(index[0]),
        data_end=pd.Timestamp(index[-1]),
        split_index=split_index,
        split_date=pd.Timestamp(index[split_index]),
        training=_run_segment(
            data, targets, "training", costs=costs, position_tranches=position_tranches
        ),
        asset_quality=data.asset.quality,
        benchmark_quality=None if data.benchmark is None else data.benchmark.quality,
        warnings=data.warnings,
    )


def run_final_evaluation(
    data: AlignedMarketData,
    config: StrategyConfig,
    *,
    costs: BacktestCosts = _DEFAULT_COSTS,
    position_tranches: int = 5,
) -> FinalEvaluation:
    """Evaluate the test suffix after parameters have been frozen."""
    training = run_training_evaluation(
        data, config, costs=costs, position_tranches=position_tranches
    )
    test = _run_segment(
        data,
        _targets(data, config, position_tranches),
        "test",
        start=training.split_index,
        costs=costs,
        position_tranches=position_tranches,
    )
    return FinalEvaluation(
        **{item.name: getattr(training, item.name) for item in fields(TrainingEvaluation)},
        test=test,
    )


def training_asset(data: AlignedMarketData) -> MarketData:
    """The asset's chronological training prefix, the only bars visible before the lock."""
    split_index = _split_index(data)
    asset = data.asset
    return replace(
        asset,
        bars=asset.bars.iloc[:split_index],
        session_labels=asset.session_labels[:split_index],
    )


def _split_index(data: AlignedMarketData) -> int:
    """Daily data trains on its first 70% of bars, minute data on its training sessions."""
    if data.bar_interval == "1m":
        first_test_session = tuple(dict.fromkeys(data.session_labels))[data.training_sessions]
        return data.session_labels.index(first_test_session)
    rows = len(data.asset.bars)
    split_index = floor(rows * TRAIN_FRACTION)
    if split_index < 2 or rows - split_index < 2:
        raise ValueError("aligned data must contain at least two training and two test bars")
    return split_index


def _targets(
    data: AlignedMarketData,
    config: StrategyConfig,
    position_tranches: int,
    *,
    stop: int | None = None,
) -> pd.Series:
    """Completed-bar targets computed only from the bars before ``stop``."""
    labels = data.session_labels[:stop] if data.bar_interval == "1m" else None
    return strategy_targets(data.asset.bars.iloc[:stop], config, position_tranches, labels)


def analyze_drawdown(equity_curve: pd.Series) -> DrawdownAnalysis:
    """Describe the maximum drawdown episode using an initial 1.0 high-water mark."""
    equity = equity_curve
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


def _run_segment(
    data: AlignedMarketData,
    targets: pd.Series,
    segment: SegmentName,
    *,
    start: int = 0,
    costs: BacktestCosts,
    position_tranches: int,
) -> SegmentBacktestResult:
    """Fill the bars ``start:len(targets)`` at each next open, beside buy-and-hold references."""
    stop = len(targets)
    prices = data.asset.bars.iloc[start:stop]
    # Each bar fills the previous bar's target, so a daily test opens with the last
    # training position; the first bar of a training segment starts from cash.
    executed = targets.shift(1, fill_value=0).astype("int64").iloc[start:]
    if data.bar_interval == "1m":
        labels = np.asarray(data.session_labels[start:stop])
        new_session = labels[1:] != labels[:-1]
        session_end = np.r_[new_session, True]
        # Signals never cross sessions, and every session is flat from its final minute open.
        executed[np.r_[True, new_session] | session_end] = 0
        hold = pd.Series(np.where(session_end, 0, 1), index=prices.index, dtype="int64")
        benchmark = None
    else:
        hold = pd.Series(1, index=prices.index, dtype="int64")
        assert data.benchmark is not None
        benchmark = _account_path(
            data.benchmark.bars.iloc[start:stop], hold, costs=costs, position_tranches=1
        )

    strategy = _account_path(prices, executed, costs=costs, position_tranches=position_tranches)
    buy_and_hold = _account_path(prices, hold, costs=costs, position_tranches=1)
    warnings: list[str] = []
    metrics = _performance_metrics(
        strategy,
        asset_buy_and_hold_return=float(buy_and_hold.equity.iloc[-1] - 1.0),
        benchmark_return=None if benchmark is None else float(benchmark.equity.iloc[-1] - 1.0),
        warnings=warnings,
        annualize=data.bar_interval == "1d",
    )
    return SegmentBacktestResult(
        segment=segment,
        bar_returns=strategy.bar_returns,
        equity_curve=strategy.equity,
        drawdown_curve=strategy.drawdown,
        positions=strategy.positions,
        asset_buy_and_hold_bar_returns=buy_and_hold.bar_returns,
        asset_buy_and_hold_equity_curve=buy_and_hold.equity,
        asset_buy_and_hold_drawdown_curve=buy_and_hold.drawdown,
        benchmark_bar_returns=None if benchmark is None else benchmark.bar_returns,
        benchmark_equity_curve=None if benchmark is None else benchmark.equity,
        benchmark_drawdown_curve=None if benchmark is None else benchmark.drawdown,
        metrics=metrics,
        drawdown=analyze_drawdown(strategy.equity),
        trades=strategy.trades,
        warnings=tuple(warnings),
        mean_exposure=float(strategy.exposure.mean()),
    )


@dataclass(frozen=True, slots=True)
class _AccountPath:
    bar_returns: pd.Series
    equity: pd.Series
    drawdown: pd.Series
    positions: pd.Series
    exposure: pd.Series
    turnover: pd.Series
    trades: tuple[TradeEvent, ...]


def _account_path(
    prices: pd.DataFrame,
    target_tranches: pd.Series,
    *,
    costs: BacktestCosts,
    position_tranches: int,
) -> _AccountPath:
    """Trade equal cash tranches at each open and mark the account at each close."""
    initial_cash = TRADE_DISPLAY_INITIAL_CAPITAL
    tranche_budget = initial_cash / position_tranches
    cash = initial_cash
    shares = 0.0
    cost_basis = 0.0
    open_tranches = 0
    events: list[TradeEvent] = []
    equity_values: list[float] = []
    position_values: list[float] = []
    exposure_values: list[float] = []
    # Positional arrays: per-bar label lookups dominated the cost of every backtest.
    opens = prices["open"].to_numpy(dtype=float)
    closes = prices["close"].to_numpy(dtype=float)
    targets = target_tranches.to_numpy(dtype=np.int64)

    for position, date in enumerate(prices.index):
        target = int(targets[position])
        execution_price = float(opens[position])
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

        market_value = shares * float(closes[position])
        equity_values.append(cash + market_value)
        exposure_values.append(market_value / (cash + market_value))
        position_values.append(open_tranches / position_tranches)

    equity = pd.Series(equity_values, index=prices.index, dtype=float, name="equity") / initial_cash
    bar_returns = equity.pct_change(fill_method=None).rename("bar_return")
    bar_returns.iloc[0] = equity.iloc[0] - 1.0
    high_water = equity.cummax().clip(lower=1.0)
    drawdown = (equity / high_water - 1.0).rename("drawdown")
    positions = pd.Series(position_values, index=prices.index, dtype=float, name="position")
    turnover = positions.diff().fillna(positions.iloc[0]).abs().rename("turnover")
    return _AccountPath(
        bar_returns=bar_returns,
        equity=equity,
        drawdown=drawdown,
        positions=positions,
        exposure=pd.Series(exposure_values, index=prices.index, name="exposure"),
        turnover=turnover,
        trades=tuple(events),
    )


def _performance_metrics(
    path: _AccountPath,
    *,
    asset_buy_and_hold_return: float,
    benchmark_return: float | None,
    warnings: list[str],
    annualize: bool,
) -> PerformanceMetrics:
    bar_returns = path.bar_returns
    ending_equity = float(path.equity.iloc[-1])
    annualized_return = annualized_volatility = sharpe_ratio = None
    if annualize:
        annualized_return = ending_equity ** (TRADING_DAYS_PER_YEAR / len(bar_returns)) - 1.0
        bar_std = float(bar_returns.std(ddof=1))
        annualized_volatility = bar_std * sqrt(TRADING_DAYS_PER_YEAR)
        if bar_std == 0.0:
            warnings.append("Sharpe ratio is undefined because daily volatility is zero.")
        else:
            sharpe_ratio = float(bar_returns.mean()) / bar_std * sqrt(TRADING_DAYS_PER_YEAR)

    return PerformanceMetrics(
        total_return=ending_equity - 1.0,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=float(path.drawdown.min()),
        trade_count=len(path.trades),
        turnover=float(path.turnover.sum()),
        asset_buy_and_hold_return=asset_buy_and_hold_return,
        benchmark_return=benchmark_return,
    )
