"""Materialize deterministic Agent report facts for the Vue application."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pandas import Series, Timestamp
from pydantic import TypeAdapter

from tragent import storage
from tragent.quant.models import (
    TRADE_DISPLAY_INITIAL_CAPITAL,
    FinalEvaluation,
    PerformanceMetrics,
    SegmentBacktestResult,
    StrategyName,
    TradeEvent,
    TrainingEvaluation,
    strategy_parameters,
)
from tragent.research.agent.models import MarketContextEvidence

type ReportEvaluation = FinalEvaluation | TrainingEvaluation


@dataclass(frozen=True, slots=True)
class CriticReviewReport:
    """One bounded Critic Agent review included in the final report."""

    verdict: Literal["accept", "revise", "reject"]
    summary: str
    issues: tuple[str, ...] = ()
    required_revision: str | None = None


@dataclass(frozen=True, slots=True)
class ReportInput:
    """Agent-owned inputs materialized into one semantic report."""

    research_objective: str
    symbol: str
    benchmark_symbol: str | None
    final_evaluations: Mapping[str, FinalEvaluation]
    # Object-store key prefix the report is saved under.
    artifacts: str
    bar_interval: Literal["1d", "1m"] = "1d"
    training_evaluations: Mapping[str, TrainingEvaluation] = field(default_factory=dict)
    selected_fingerprint: str | None = None
    stop_reason: str | None = None
    limitations: tuple[str, ...] = ()
    runtime_warnings: tuple[str, ...] = ()
    critic_reviews: tuple[CriticReviewReport, ...] = ()
    critic_review_status: (
        Literal["not_started", "incomplete", "revision_requested", "accepted", "rejected"] | None
    ) = None
    market_context_summary: str | None = None
    market_context_sources: tuple[MarketContextEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class ReportTradeRow:
    """One deterministic transaction in a report candidate segment."""

    date: str
    action: Literal["buy", "sell"]
    price: float
    quantity: float
    gross_value: float
    trading_cost: float
    net_cash: float
    realized_profit: float | None
    average_cost_after: float | None
    position_quantity_after: float
    cash_balance_after: float


@dataclass(frozen=True, slots=True)
class ReportReferenceSegment:
    """Shared market references for one report segment."""

    segment: Literal["training", "test"]
    timestamps: tuple[str, ...]
    asset_equity: tuple[float, ...]
    asset_drawdown: tuple[float, ...]
    benchmark_equity: tuple[float, ...] | None
    benchmark_drawdown: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class ReportCandidateSegment:
    """One candidate's quantitative evidence for a train or test segment."""

    segment: Literal["training", "test"]
    equity: tuple[float, ...]
    drawdown: tuple[float, ...]
    metrics: PerformanceMetrics
    trades: tuple[ReportTradeRow, ...]
    mean_exposure: float


@dataclass(frozen=True, slots=True)
class ReportCandidate:
    """One Agent-tested strategy identified by its backtest fingerprint."""

    id: str
    strategy_name: StrategyName
    parameters: dict[str, int | float]
    transaction_cost_bps: float
    slippage_bps: float
    segments: tuple[ReportCandidateSegment, ...]


@dataclass(frozen=True, slots=True)
class ResearchReportView:
    """Semantic Agent report returned to the Vue application."""

    research_objective: str
    symbol: str
    benchmark_symbol: str | None
    bar_interval: Literal["1d", "1m"]
    data_start: str | None
    data_end: str | None
    data_source: Literal["yfinance"]
    selected_candidate_id: str | None
    stop_reason: str
    research_incomplete: bool
    references: tuple[ReportReferenceSegment, ...]
    candidates: tuple[ReportCandidate, ...]
    initial_capital: float
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    critic_reviews: tuple[CriticReviewReport, ...]
    critic_review_status: (
        Literal["not_started", "incomplete", "revision_requested", "accepted", "rejected"] | None
    )
    market_context_summary: str | None
    market_context_sources: tuple[MarketContextEvidence, ...]


REPORT_VIEW_FILENAME = "report.json"
_REPORT_VIEW_ADAPTER = TypeAdapter(ResearchReportView)


def generate_report(report_input: ReportInput) -> str:
    """Generate the semantic report consumed by the Vue application, returning its key."""

    return save_report_view(build_report_view(report_input), report_input.artifacts)


def build_report_view(report_input: ReportInput) -> ResearchReportView:
    """Build one semantic report from deterministic Agent research evidence."""

    _validate_report_input(report_input)
    training_only = _training_only_evaluations(report_input)
    data_source = next(
        iter(report_input.final_evaluations.values()),
        next(iter(report_input.training_evaluations.values()), None),
    )
    return ResearchReportView(
        research_objective=report_input.research_objective,
        symbol=report_input.symbol,
        benchmark_symbol=report_input.benchmark_symbol,
        bar_interval=report_input.bar_interval,
        data_start=_timestamp(data_source.data_start) if data_source else None,
        data_end=_timestamp(data_source.data_end) if data_source else None,
        data_source="yfinance",
        selected_candidate_id=report_input.selected_fingerprint,
        stop_reason=(report_input.stop_reason or "").strip(),
        research_incomplete=bool(training_only),
        references=_report_references(report_input),
        candidates=_report_candidates(report_input, training_only),
        initial_capital=TRADE_DISPLAY_INITIAL_CAPITAL,
        warnings=_collect_warnings(report_input),
        limitations=report_input.limitations,
        critic_reviews=report_input.critic_reviews,
        critic_review_status=report_input.critic_review_status,
        market_context_summary=report_input.market_context_summary,
        market_context_sources=report_input.market_context_sources,
    )


def save_report_view(report: ResearchReportView, artifacts: str) -> str:
    """Store the materialized semantic report and return its key."""

    key = f"{artifacts}/{REPORT_VIEW_FILENAME}"
    storage.put(key, _REPORT_VIEW_ADAPTER.dump_json(report), "application/json")
    return key


def load_report_view(key: str) -> ResearchReportView:
    """Load and validate one materialized semantic report."""

    return _REPORT_VIEW_ADAPTER.validate_json(storage.get(key))


def _validate_report_input(report_input: ReportInput) -> None:
    selected = report_input.selected_fingerprint
    if selected is None and not (report_input.stop_reason or "").strip():
        raise ValueError("stop_reason is required when no strategy is selected")
    if selected is not None and selected not in report_input.final_evaluations:
        raise ValueError("selected_fingerprint must identify a final evaluation")


def _report_references(report_input: ReportInput) -> tuple[ReportReferenceSegment, ...]:
    first = next(iter(report_input.final_evaluations.values()), None)
    if first is None:
        return ()
    return tuple(_reference_segment(getattr(first, segment)) for segment in ("training", "test"))


def _reference_segment(segment: SegmentBacktestResult) -> ReportReferenceSegment:
    return ReportReferenceSegment(
        segment=segment.segment,
        timestamps=_timestamps(segment.asset_buy_and_hold_equity_curve),
        asset_equity=_values(segment.asset_buy_and_hold_equity_curve),
        asset_drawdown=_values(segment.asset_buy_and_hold_drawdown_curve),
        benchmark_equity=(
            None
            if segment.benchmark_equity_curve is None
            else _values(segment.benchmark_equity_curve)
        ),
        benchmark_drawdown=(
            None
            if segment.benchmark_drawdown_curve is None
            else _values(segment.benchmark_drawdown_curve)
        ),
    )


def _report_candidates(
    report_input: ReportInput,
    training_only: tuple[tuple[str, TrainingEvaluation], ...],
) -> tuple[ReportCandidate, ...]:
    completed = tuple(
        _report_candidate(
            fingerprint,
            evaluation,
            segments=(evaluation.training, evaluation.test),
        )
        for fingerprint, evaluation in report_input.final_evaluations.items()
    )
    incomplete = tuple(
        _report_candidate(
            fingerprint,
            evaluation,
            segments=(evaluation.training,),
        )
        for fingerprint, evaluation in training_only
    )
    return (*completed, *incomplete)


def _report_candidate(
    fingerprint: str,
    evaluation: ReportEvaluation,
    *,
    segments: tuple[SegmentBacktestResult, ...],
) -> ReportCandidate:
    return ReportCandidate(
        id=fingerprint,
        strategy_name=evaluation.strategy_name,
        parameters=strategy_parameters(evaluation.config),
        transaction_cost_bps=evaluation.costs.transaction_cost_bps,
        slippage_bps=evaluation.costs.slippage_bps,
        segments=tuple(_candidate_segment(segment) for segment in segments),
    )


def _candidate_segment(segment: SegmentBacktestResult) -> ReportCandidateSegment:
    return ReportCandidateSegment(
        segment=segment.segment,
        equity=_values(segment.equity_curve),
        drawdown=_values(segment.drawdown_curve),
        metrics=segment.metrics,
        trades=tuple(_report_trade_row(trade) for trade in segment.trades),
        mean_exposure=segment.mean_exposure,
    )


def _report_trade_row(trade: TradeEvent) -> ReportTradeRow:
    return ReportTradeRow(
        date=_timestamp(trade.date),
        action=trade.action,
        price=trade.price,
        quantity=trade.quantity,
        gross_value=trade.gross_value,
        trading_cost=trade.trading_cost,
        net_cash=trade.net_cash,
        realized_profit=trade.realized_profit,
        average_cost_after=trade.average_cost_after,
        position_quantity_after=trade.position_quantity_after,
        cash_balance_after=trade.cash_balance_after,
    )


def _timestamps(series: Series) -> tuple[str, ...]:
    return tuple(_timestamp(value) for value in series.index)


def _timestamp(value: object) -> str:
    return Timestamp(value).isoformat()


def _values(series: Series) -> tuple[float, ...]:
    return tuple(float(value) for value in series.array)


def _training_only_evaluations(
    report_input: ReportInput,
) -> tuple[tuple[str, TrainingEvaluation], ...]:
    return tuple(
        (fingerprint, evaluation)
        for fingerprint, evaluation in report_input.training_evaluations.items()
        if fingerprint not in report_input.final_evaluations
    )


def _collect_warnings(report_input: ReportInput) -> tuple[str, ...]:
    collected = list(report_input.runtime_warnings)
    for evaluation in report_input.final_evaluations.values():
        collected.extend(evaluation.asset_quality.warnings)
        if evaluation.benchmark_quality is not None:
            collected.extend(evaluation.benchmark_quality.warnings)
        collected.extend(evaluation.warnings)
        collected.extend(evaluation.training.warnings)
        collected.extend(evaluation.test.warnings)
    for _fingerprint, evaluation in _training_only_evaluations(report_input):
        collected.extend(evaluation.asset_quality.warnings)
        if evaluation.benchmark_quality is not None:
            collected.extend(evaluation.benchmark_quality.warnings)
        collected.extend(evaluation.warnings)
        collected.extend(evaluation.training.warnings)
    return tuple(dict.fromkeys(collected))


__all__ = [
    "CriticReviewReport",
    "ReportCandidate",
    "ReportCandidateSegment",
    "ReportInput",
    "ReportReferenceSegment",
    "ReportTradeRow",
    "ResearchReportView",
    "REPORT_VIEW_FILENAME",
    "build_report_view",
    "generate_report",
    "load_report_view",
    "save_report_view",
]
