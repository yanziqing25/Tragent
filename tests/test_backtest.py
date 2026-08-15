"""Hand-calculated tests for next-open execution and deterministic metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from trading.quant import backtest
from trading.quant.backtest import analyze_drawdown, run_final_evaluation, run_training_evaluation
from trading.quant.models import (
    AlignedMarketData,
    BacktestCosts,
    DataQualityReport,
    MarketData,
    MovingAverageConfig,
)


def market_data(
    symbol: str,
    opens: list[float],
    closes: list[float],
    *,
    start: str = "2024-01-01",
) -> MarketData:
    index = pd.date_range(start, periods=len(opens), freq="D")
    bars = pd.DataFrame(
        {
            "open": opens,
            "high": [max(open_, close) for open_, close in zip(opens, closes, strict=True)],
            "low": [min(open_, close) for open_, close in zip(opens, closes, strict=True)],
            "close": closes,
            "volume": [1_000] * len(opens),
        },
        index=index,
        dtype=float,
    )
    quality = DataQualityReport(
        symbol=symbol,
        valid_rows=len(bars),
        start_date=index[0],
        end_date=index[-1],
    )
    return MarketData(symbol=symbol, bars=bars, quality=quality)


def aligned_data(
    asset_opens: list[float],
    asset_closes: list[float],
    *,
    benchmark_opens: list[float] | None = None,
    benchmark_closes: list[float] | None = None,
) -> AlignedMarketData:
    benchmark_opens = benchmark_opens or [100.0] * len(asset_opens)
    benchmark_closes = benchmark_closes or [100.0] * len(asset_opens)
    return AlignedMarketData(
        asset=market_data("AAPL", asset_opens, asset_closes),
        benchmark=market_data("SPY", benchmark_opens, benchmark_closes),
    )


def intraday_data() -> AlignedMarketData:
    labels = ("2024-01-02",) * 3 + ("2024-01-03",) * 3 + ("2024-01-04",) * 3
    index = pd.DatetimeIndex(
        timestamp
        for day in ("2024-01-02", "2024-01-03", "2024-01-04")
        for timestamp in pd.date_range(f"{day} 14:30:00+00:00", periods=3, freq="1min")
    )
    bars = pd.DataFrame(
        {
            "open": [100.0] * 9,
            "high": [101.0] * 9,
            "low": [99.0] * 9,
            "close": [100.0] * 9,
            "volume": [1_000.0] * 9,
        },
        index=index,
    )
    quality = DataQualityReport(
        symbol="AAPL",
        valid_rows=9,
        start_date=index[0],
        end_date=index[-1],
    )
    return AlignedMarketData(
        asset=MarketData(
            symbol="AAPL",
            bars=bars,
            quality=quality,
            bar_interval="1m",
            session_labels=labels,
        ),
        benchmark=None,
        bar_interval="1m",
        session_labels=labels,
        training_sessions=2,
        test_sessions=1,
    )


def target(values: list[int], index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(values, index=index, dtype="int64", name="target_tranches")


def test_signal_executes_at_the_next_open_with_costs() -> None:
    data = aligned_data(
        [100.0, 100.0, 110.0, 120.0, 130.0, 140.0],
        [100.0, 100.0, 110.0, 120.0, 130.0, 140.0],
    )
    prices = data.asset.bars

    result = backtest._run_segment(
        prices,
        data.benchmark.bars,
        target([1, 2, 3, 2, 1, 1], prices.index),
        segment="training",
        previous_close_tranches=0,
        costs=BacktestCosts(5, 5),
        position_tranches=5,
    )

    assert result.positions.tolist() == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.4, 0.2])
    assert result.bar_returns.iloc[0] == 0.0
    assert result.metrics.trade_count == 5
    assert result.metrics.turnover == pytest.approx(1.0)
    assert result.asset_buy_and_hold_bar_returns.iloc[0] == pytest.approx(-0.001)
    assert result.asset_buy_and_hold_equity_curve.iloc[-1] == pytest.approx(0.999 * 1.40)
    assert result.metrics.asset_buy_and_hold_return == pytest.approx(0.999 * 1.40 - 1.0)

    first_buy, second_buy, third_buy, first_sell, second_sell = result.trades
    assert [trade.action for trade in result.trades] == ["buy", "buy", "buy", "sell", "sell"]
    assert first_buy.price == 100.0
    assert first_buy.quantity == pytest.approx(19.98)
    assert first_buy.gross_value == pytest.approx(1_998.0)
    assert first_buy.trading_cost == pytest.approx(2.0)
    assert first_buy.net_cash == pytest.approx(-2_000.0)
    assert first_buy.cash_balance_after == pytest.approx(8_000.0)
    assert second_buy.cash_balance_after == pytest.approx(6_000.0)
    assert third_buy.cash_balance_after == pytest.approx(4_000.0)
    assert third_buy.average_cost_after == pytest.approx(6_000.0 / 54.7936363636)
    assert first_sell.quantity == pytest.approx(third_buy.position_quantity_after / 3)
    assert first_sell.average_cost_after == pytest.approx(third_buy.average_cost_after)
    assert first_sell.realized_profit == pytest.approx(
        first_sell.net_cash - first_sell.quantity * third_buy.average_cost_after
    )
    assert second_sell.quantity == pytest.approx(first_sell.position_quantity_after / 2)
    assert second_sell.average_cost_after == pytest.approx(third_buy.average_cost_after)
    assert second_sell.position_quantity_after == pytest.approx(first_sell.quantity)
    expected_final_equity = (
        second_sell.cash_balance_after + second_sell.position_quantity_after * 140.0
    ) / 10_000.0
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_final_equity)


def test_asset_and_benchmark_buy_and_hold_use_independent_prices_and_equal_costs() -> None:
    data = aligned_data(
        [100.0, 110.0, 120.0],
        [105.0, 115.0, 126.0],
        benchmark_opens=[200.0, 190.0, 180.0],
        benchmark_closes=[198.0, 189.0, 171.0],
    )
    prices = data.asset.bars

    result = backtest._run_segment(
        prices,
        data.benchmark.bars,
        target([0, 0, 0], prices.index),
        segment="training",
        previous_close_tranches=0,
        costs=BacktestCosts(5, 5),
        position_tranches=1,
    )

    expected_asset_equity = 0.999 * (126.0 / 100.0)
    expected_benchmark_equity = 0.999 * (171.0 / 200.0)
    assert result.asset_buy_and_hold_equity_curve.iloc[-1] == pytest.approx(expected_asset_equity)
    assert result.benchmark_equity_curve.iloc[-1] == pytest.approx(expected_benchmark_equity)
    assert result.metrics.asset_buy_and_hold_return == pytest.approx(expected_asset_equity - 1.0)
    assert result.metrics.benchmark_return == pytest.approx(expected_benchmark_equity - 1.0)
    assert result.asset_buy_and_hold_drawdown_curve.index.equals(prices.index)


def test_entry_skips_prior_overnight_gap_and_exit_keeps_its_gap() -> None:
    data = aligned_data(
        [100.0, 200.0, 110.0],
        [100.0, 220.0, 110.0],
    )
    prices = data.asset.bars

    result = backtest._run_segment(
        prices,
        data.benchmark.bars,
        target([1, 0, 0], prices.index),
        segment="training",
        previous_close_tranches=0,
        costs=BacktestCosts(5, 5),
        position_tranches=1,
    )

    assert result.positions.tolist() == [0, 1, 0]
    assert 1 + result.bar_returns.iloc[1] == pytest.approx(0.999 * (220.0 / 200.0))
    assert 1 + result.bar_returns.iloc[2] == pytest.approx((110.0 / 220.0) * 0.999)


def test_final_test_starts_from_cash_but_uses_last_training_target(monkeypatch) -> None:
    data = aligned_data([100.0] * 10, [100.0] * 10)

    def one_buy_signal(close: pd.Series, _config: MovingAverageConfig) -> pd.Series:
        return pd.Series(
            [1, *([0] * (len(close) - 1))],
            index=close.index,
            dtype="int64",
            name="trade_signal",
        )

    monkeypatch.setattr(backtest, "generate_trade_signals", one_buy_signal)

    evaluation = run_final_evaluation(
        data,
        MovingAverageConfig(short_window=2, long_window=3),
        costs=BacktestCosts(5, 5),
        position_tranches=5,
    )

    assert evaluation.split_index == 7
    assert evaluation.split_date == data.asset.bars.index[7]
    assert evaluation.training.positions.iloc[0] == 0
    assert evaluation.test.positions.iloc[0] == pytest.approx(0.2)
    assert evaluation.test.equity_curve.iloc[0] == pytest.approx(0.9998)
    assert evaluation.test.metrics.trade_count == 1
    assert evaluation.test.metrics.turnover == pytest.approx(0.2)
    assert evaluation.training.asset_buy_and_hold_equity_curve.iloc[0] == pytest.approx(0.999)
    assert evaluation.test.asset_buy_and_hold_equity_curve.iloc[0] == pytest.approx(0.999)
    assert evaluation.test.trades[0].date == data.asset.bars.index[7]
    assert evaluation.test.trades[0].action == "buy"
    assert evaluation.test.trades[0].cash_balance_after == pytest.approx(8_000.0)

    minute = intraday_data()

    def persistent_buy(close: pd.Series, _config: MovingAverageConfig) -> pd.Series:
        return pd.Series(
            [1, *([0] * (len(close) - 1))],
            index=close.index,
            dtype="int64",
            name="trade_signal",
        )

    monkeypatch.setattr(backtest, "generate_trade_signals", persistent_buy)
    intraday = run_final_evaluation(
        minute,
        MovingAverageConfig(short_window=2, long_window=3),
        costs=BacktestCosts(5, 5),
        position_tranches=5,
    )

    assert intraday.split_index == 6
    assert intraday.training.positions.tolist() == pytest.approx([0, 0.2, 0, 0, 0.2, 0])
    assert intraday.test.positions.tolist() == pytest.approx([0, 0.2, 0])
    assert [trade.action for trade in intraday.training.trades] == ["buy", "sell"] * 2
    assert [trade.action for trade in intraday.test.trades] == ["buy", "sell"]
    assert intraday.test.equity_curve.iloc[0] == 1.0
    assert intraday.test.metrics.annualized_return is None
    assert intraday.test.metrics.annualized_volatility is None
    assert intraday.test.metrics.sharpe_ratio is None
    assert intraday.test.metrics.benchmark_return is None


def test_training_evaluation_is_unchanged_when_future_prices_change() -> None:
    first_seven = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    first = aligned_data(first_seven + [107.0, 108.0, 109.0], first_seven + [107, 108, 109])
    second = aligned_data(first_seven + [500.0, 10.0, 800.0], first_seven + [1_000, 5, 900])
    config = MovingAverageConfig(short_window=2, long_window=3)

    first_result = run_training_evaluation(first, config)
    second_result = run_training_evaluation(second, config)

    pd.testing.assert_series_equal(
        first_result.training.bar_returns,
        second_result.training.bar_returns,
    )
    assert first_result.training.metrics == second_result.training.metrics


def test_zero_trade_result_is_valid_and_sharpe_is_none() -> None:
    data = aligned_data([100.0] * 4, [100.0] * 4)
    prices = data.asset.bars

    result = backtest._run_segment(
        prices,
        data.benchmark.bars,
        target([0, 0, 0, 0], prices.index),
        segment="training",
        previous_close_tranches=0,
        costs=BacktestCosts(5, 5),
        position_tranches=5,
    )

    assert result.metrics.total_return == 0.0
    assert result.metrics.annualized_volatility == 0.0
    assert result.metrics.sharpe_ratio is None
    assert result.metrics.trade_count == 0
    assert result.metrics.turnover == 0.0
    assert result.trades == ()
    assert any("Sharpe" in warning for warning in result.warnings)
    assert result.metrics.asset_buy_and_hold_return == pytest.approx(-0.001)
    assert result.metrics.benchmark_return == pytest.approx(-0.001)


def test_drawdown_uses_initial_cash_and_finds_recovery() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    equity = pd.Series([0.99, 0.80, 1.00, 0.90], index=index)

    result = analyze_drawdown(equity)

    assert result.max_drawdown == pytest.approx(-0.20)
    assert result.peak_date == index[0]
    assert result.trough_date == index[1]
    assert result.recovery_date == index[2]
    assert result.underwater_bars == 2


def test_drawdown_uses_latest_equal_peak_and_handles_no_recovery() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    equity = pd.Series([1.10, 1.10, 0.90], index=index)

    result = analyze_drawdown(equity)

    assert result.max_drawdown == pytest.approx(0.90 / 1.10 - 1.0)
    assert result.peak_date == index[1]
    assert result.trough_date == index[2]
    assert result.recovery_date is None
    assert result.underwater_bars == 1


def test_drawdown_preserves_a_tiny_but_real_loss() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    equity = pd.Series([1.0, 1.0 - 5e-16], index=index)

    result = analyze_drawdown(equity)

    assert result.max_drawdown < 0.0
    assert result.trough_date == index[1]


def test_drawdown_rejects_non_finite_or_non_positive_equity() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")

    with pytest.raises(ValueError, match="positive"):
        analyze_drawdown(pd.Series([1.0, 0.0], index=index))
    with pytest.raises(ValueError, match="finite"):
        analyze_drawdown(pd.Series([1.0, float("nan")], index=index))


def test_evaluation_rejects_unaligned_indexes_and_insufficient_test_rows() -> None:
    data = aligned_data([100.0] * 6, [100.0] * 6)
    misaligned_benchmark = MarketData(
        symbol="SPY",
        bars=data.benchmark.bars.set_axis(
            pd.date_range("2025-01-01", periods=6, freq="D"),
            axis="index",
        ),
        quality=data.benchmark.quality,
    )
    misaligned = AlignedMarketData(asset=data.asset, benchmark=misaligned_benchmark)

    with pytest.raises(ValueError, match="identical indexes"):
        run_training_evaluation(misaligned, MovingAverageConfig(2, 3))

    too_short = aligned_data([100.0] * 4, [100.0] * 4)
    with pytest.raises(ValueError, match="at least"):
        run_training_evaluation(too_short, MovingAverageConfig(2, 3))
