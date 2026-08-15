"""Focused tests for the Vue report artifact."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from trading.quant.models import (
    BacktestCosts,
    DataQualityReport,
    DrawdownAnalysis,
    FinalEvaluation,
    MovingAverageConfig,
    PerformanceMetrics,
    SegmentBacktestResult,
    StrategyConfig,
    TradeEvent,
    TrainingEvaluation,
    strategy_name,
)
from trading.reporting import (
    CriticReviewReport,
    ReportInput,
    build_report_view,
    generate_report,
    load_report_view,
)


def segment(
    name: str,
    *,
    sharpe_ratio: float | None = 1.2,
    open_trade: bool = False,
) -> SegmentBacktestResult:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    equity = pd.Series([1.0, 1.1, 1.045], index=index, name="equity")
    drawdown = pd.Series([0.0, 0.0, -0.05], index=index, name="drawdown")
    benchmark_equity = pd.Series([0.999, 1.02, 1.03], index=index, name="equity")
    benchmark_drawdown = pd.Series([-0.001, 0.0, 0.0], index=index, name="drawdown")
    asset_equity = pd.Series([0.999, 1.04, 1.05], index=index, name="equity")
    asset_drawdown = pd.Series([-0.001, 0.0, 0.0], index=index, name="drawdown")
    quantity = 19.98
    buy = TradeEvent(
        date=index[1],
        action="buy",
        price=100.0,
        quantity=quantity,
        gross_value=1_998.0,
        trading_cost=2.0,
        net_cash=-2_000.0,
        realized_profit=None,
        average_cost_after=2_000.0 / quantity,
        position_quantity_after=quantity,
        cash_balance_after=8_000.0,
    )
    sell_gross = quantity * 125.0
    sell_cost = sell_gross * 0.001
    sell = TradeEvent(
        date=index[2],
        action="sell",
        price=125.0,
        quantity=quantity,
        gross_value=sell_gross,
        trading_cost=sell_cost,
        net_cash=sell_gross - sell_cost,
        realized_profit=sell_gross - sell_cost - 2_000.0,
        average_cost_after=None,
        position_quantity_after=0.0,
        cash_balance_after=8_000.0 + sell_gross - sell_cost,
    )
    trades = (buy,) if open_trade else (buy, sell)
    return SegmentBacktestResult(
        segment=name,
        bar_returns=pd.Series([0.0, 0.10, -0.05], index=index, name="bar_return"),
        equity_curve=equity,
        drawdown_curve=drawdown,
        positions=pd.Series([0, 1, 1 if open_trade else 0], index=index, dtype="int8"),
        asset_buy_and_hold_bar_returns=asset_equity.pct_change().fillna(-0.001),
        asset_buy_and_hold_equity_curve=asset_equity,
        asset_buy_and_hold_drawdown_curve=asset_drawdown,
        benchmark_bar_returns=benchmark_equity.pct_change().fillna(-0.001),
        benchmark_equity_curve=benchmark_equity,
        benchmark_drawdown_curve=benchmark_drawdown,
        metrics=PerformanceMetrics(
            total_return=0.045,
            annualized_return=1.5,
            annualized_volatility=0.2,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=-0.05,
            trade_count=len(trades),
            turnover=1.0 if open_trade else 2.0,
            asset_buy_and_hold_return=0.05,
            benchmark_return=0.03,
        ),
        drawdown=DrawdownAnalysis(
            max_drawdown=-0.05,
            peak_date=index[1],
            trough_date=index[2],
            recovery_date=None,
            underwater_bars=1,
        ),
        trades=trades,
        warnings=("Sharpe unavailable." if sharpe_ratio is None else "Synthetic data.",),
    )


def evaluation(
    *,
    sharpe_ratio: float | None = 1.2,
    config: StrategyConfig | None = None,
    open_test_trade: bool = False,
) -> FinalEvaluation:
    config = config or MovingAverageConfig(short_window=2, long_window=3)
    quality = DataQualityReport(
        symbol="AAPL",
        valid_rows=10,
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-01-10"),
        warnings=("One duplicate was removed.",),
    )
    return FinalEvaluation(
        symbol="AAPL",
        benchmark_symbol="SPY",
        strategy_name=strategy_name(config),
        config=config,
        costs=BacktestCosts(5, 5),
        data_start=pd.Timestamp("2024-01-01"),
        data_end=pd.Timestamp("2024-01-10"),
        split_index=7,
        split_date=pd.Timestamp("2024-01-08"),
        training=segment("training", sharpe_ratio=sharpe_ratio),
        test=segment("test", sharpe_ratio=sharpe_ratio, open_trade=open_test_trade),
        asset_quality=quality,
        benchmark_quality=replace(quality, symbol="SPY", warnings=()),
        warnings=("Synthetic evaluation.",),
    )


def report_input(tmp_path: Path, **changes: object) -> ReportInput:
    result = evaluation(sharpe_ratio=None)
    values: dict[str, object] = {
        "research_objective": "Test Vue report",
        "symbol": "AAPL",
        "benchmark_symbol": "SPY",
        "final_evaluations": {"candidate-1": result},
        "selected_fingerprint": "candidate-1",
        "limitations": ("Synthetic and illustrative only.",),
        "runtime_warnings": ("Runtime warning.",),
        "output_dir": tmp_path,
    }
    values.update(changes)
    return ReportInput(**values)


def test_generates_one_parseable_vue_report(tmp_path: Path) -> None:
    first = evaluation()
    selected = evaluation(
        config=MovingAverageConfig(short_window=3, long_window=5),
        open_test_trade=True,
    )
    source = report_input(
        tmp_path,
        final_evaluations={"first": first, "selected": selected},
        selected_fingerprint="selected",
        critic_reviews=(
            CriticReviewReport(
                verdict="accept",
                summary="The conclusion matches the frozen evidence.",
            ),
        ),
        critic_review_status="accepted",
    )

    report_path = generate_report(source)
    report = load_report_view(report_path)

    assert report_path == tmp_path / "report.json"
    assert [path.name for path in tmp_path.iterdir()] == ["report.json"]
    assert report == build_report_view(source)
    assert report.selected_candidate_id == "selected"
    assert len(report.references) == 2
    assert len(report.candidates) == 2
    assert report.critic_review_status == "accepted"
    assert report.candidates[1].segments[1].trades[0].action == "buy"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "figure" not in payload
    assert "chart_config" not in payload


def test_minute_report_uses_session_buy_and_hold_without_a_benchmark(tmp_path: Path) -> None:
    result = evaluation()

    def minute_segment(value: SegmentBacktestResult) -> SegmentBacktestResult:
        return replace(
            value,
            metrics=replace(
                value.metrics,
                annualized_return=None,
                annualized_volatility=None,
                sharpe_ratio=None,
                benchmark_return=None,
            ),
            benchmark_bar_returns=None,
            benchmark_equity_curve=None,
            benchmark_drawdown_curve=None,
        )

    minute = replace(
        result,
        benchmark_symbol=None,
        bar_interval="1m",
        training=minute_segment(result.training),
        test=minute_segment(result.test),
        benchmark_quality=None,
    )
    report = build_report_view(
        ReportInput(
            research_objective="Minute research",
            symbol="AAPL",
            benchmark_symbol=None,
            bar_interval="1m",
            final_evaluations={"minute": minute},
            selected_fingerprint="minute",
            output_dir=tmp_path,
        )
    )

    assert report.benchmark_symbol is None
    assert report.data_source == "yfinance"
    assert len(report.references) == 2
    assert all(reference.benchmark_equity is None for reference in report.references)


def test_no_selection_requires_a_stop_reason(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="stop_reason"):
        generate_report(report_input(tmp_path, selected_fingerprint=None, stop_reason=None))

    source = report_input(
        tmp_path,
        selected_fingerprint=None,
        stop_reason="No strategy met the drawdown constraint.",
    )
    report = build_report_view(source)

    assert report.selected_candidate_id is None
    assert report.candidates[0].id == "candidate-1"


def test_failure_and_training_only_reports_keep_available_evidence(tmp_path: Path) -> None:
    failure = build_report_view(
        report_input(
            tmp_path,
            final_evaluations={},
            selected_fingerprint=None,
            stop_reason="Market data was unavailable.",
            critic_review_status="incomplete",
        )
    )
    assert failure.references == ()
    assert failure.candidates == ()
    assert failure.critic_review_status == "incomplete"

    completed = evaluation()
    training = TrainingEvaluation(
        symbol=completed.symbol,
        benchmark_symbol=completed.benchmark_symbol,
        strategy_name=completed.strategy_name,
        config=completed.config,
        costs=completed.costs,
        data_start=completed.data_start,
        data_end=completed.data_end,
        split_index=completed.split_index,
        split_date=completed.split_date,
        training=completed.training,
        asset_quality=completed.asset_quality,
        benchmark_quality=completed.benchmark_quality,
        warnings=completed.warnings,
    )
    report = build_report_view(
        report_input(
            tmp_path,
            final_evaluations={},
            training_evaluations={"training": training},
            selected_fingerprint=None,
            stop_reason="The Tool call budget was exhausted.",
        )
    )
    assert report.research_incomplete is True
    assert report.references == ()
    assert len(report.candidates) == 1
    assert [segment.segment for segment in report.candidates[0].segments] == ["training"]


def test_selected_fingerprint_must_be_a_final_candidate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selected_fingerprint"):
        generate_report(
            report_input(
                tmp_path,
                selected_fingerprint="unknown",
            )
        )
