"""The eight fixed synthetic-evidence cases for the V2 Agent evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, before_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from trading.agent.critic import (
    MAX_CRITIC_AGENT_STEPS,
    MAX_CRITIC_TOOL_CALLS,
    build_critic_messages,
)
from trading.agent.harness import require_single_tool_call
from trading.agent.models import (
    AgentBudget,
    CriticReview,
    ResearchDraft,
    ResearchGoal,
    ToolObservation,
)
from trading.agent.prompt import build_research_messages
from trading.agent.runtime import (
    backtest_fingerprint,
)
from trading.agent.tools import build_critic_tools, build_research_tools
from trading.agent.workspace import CriticRoundRuntime, ResearchWorkspace, RuntimeEvent
from trading.quant import (
    BacktestCosts,
    DataQualityReport,
    DrawdownAnalysis,
    FinalEvaluation,
    PerformanceMetrics,
    RsiConfig,
    SegmentBacktestResult,
    TrainingEvaluation,
)

_RESEARCH_TOOL_NAMES = (
    "get_market_data",
    "calculate_asset_statistics",
    "run_backtest",
    "analyze_drawdown",
    "compare_with_benchmark",
    "submit_research_draft",
)
_CRITIC_TOOL_NAMES = (
    "inspect_selected_drawdown",
    "compare_selected_with_benchmark",
    "submit_critic_review",
)


class CaseContractError(ValueError):
    """The checked-in case contract is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class AgentEvalCase:
    """One fixed decision policy example stored in the LangSmith Dataset."""

    case_id: str
    agent: Literal["research", "critic"]
    scenario: str
    allowed_tools: tuple[str, ...]
    forbidden_tools: tuple[str, ...] = ()
    allowed_verdicts: tuple[str, ...] = ()
    max_decisions: int = 1
    require_training_backtest: bool = False
    forbid_exact_parameters: Mapping[str, int | float] | None = None

    def inputs(self) -> dict[str, object]:
        return {"case_id": self.case_id, "agent": self.agent, "scenario": self.scenario}

    def reference_outputs(self) -> dict[str, object]:
        return {
            "allowed_tools": list(self.allowed_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "allowed_verdicts": list(self.allowed_verdicts),
            "max_decisions": self.max_decisions,
            "require_training_backtest": self.require_training_backtest,
            "forbid_exact_parameters": (
                None if self.forbid_exact_parameters is None else dict(self.forbid_exact_parameters)
            ),
        }


CASES = (
    AgentEvalCase(
        "critic_drawdown_breach",
        "critic",
        "The selected draft breaches the confirmed maximum-drawdown limit.",
        _CRITIC_TOOL_NAMES,
        allowed_verdicts=("revise", "reject"),
        max_decisions=3,
    ),
    AgentEvalCase(
        "critic_sample_divergence",
        "critic",
        "Strong sample-in metrics reverse materially in sample-out evidence.",
        _CRITIC_TOOL_NAMES,
        allowed_verdicts=("revise", "reject"),
        max_decisions=3,
    ),
    AgentEvalCase(
        "critic_low_trade_count",
        "critic",
        "The requested family has one anomalous candidate with only one trade.",
        _CRITIC_TOOL_NAMES,
        allowed_verdicts=("reject",),
        max_decisions=3,
    ),
    AgentEvalCase(
        "critic_conclusion_conflict",
        "critic",
        "The draft claims SPY outperformance while deterministic metrics show less return.",
        _CRITIC_TOOL_NAMES,
        allowed_verdicts=("revise", "reject"),
        max_decisions=3,
    ),
    AgentEvalCase(
        "critic_consistent_accept",
        "critic",
        "Metrics, constraint, benchmark comparison, conclusion, and limitations agree.",
        _CRITIC_TOOL_NAMES,
        allowed_verdicts=("accept",),
        max_decisions=3,
    ),
    AgentEvalCase(
        "critic_second_round_no_revise",
        "critic",
        "A second Critic round still finds a material conclusion conflict.",
        _CRITIC_TOOL_NAMES,
        allowed_verdicts=("accept", "reject"),
        max_decisions=3,
    ),
    AgentEvalCase(
        "research_exact_candidate_repeat",
        "research",
        "An informative RSI candidate already appears in tested_candidates.",
        ("run_backtest", "analyze_drawdown", "compare_with_benchmark"),
        forbidden_tools=(
            "get_market_data",
            "calculate_asset_statistics",
            "submit_research_draft",
        ),
        forbid_exact_parameters={
            "rsi_window": 14,
            "entry_threshold": 30.0,
            "exit_threshold": 70.0,
        },
    ),
    AgentEvalCase(
        "research_required_backtest_missing",
        "research",
        "Market data is ready but no requested RSI candidate has a sample-in backtest.",
        ("run_backtest",),
        forbidden_tools=tuple(name for name in _RESEARCH_TOOL_NAMES if name != "run_backtest"),
        require_training_backtest=True,
    ),
)


def validate_cases() -> dict[str, AgentEvalCase]:
    """Validate and index the intentionally small checked-in case set."""

    by_id = {case.case_id: case for case in CASES}
    if len(CASES) != 8 or len(by_id) != len(CASES):
        raise CaseContractError("The Agent Eval case set must contain eight unique IDs.")
    for case in CASES:
        known = set(_RESEARCH_TOOL_NAMES if case.agent == "research" else _CRITIC_TOOL_NAMES)
        allowed, forbidden = set(case.allowed_tools), set(case.forbidden_tools)
        if not case.case_id or len(case.case_id) > 80:
            raise CaseContractError("An Agent Eval case ID is invalid.")
        if not allowed or allowed & forbidden or not (allowed | forbidden) <= known:
            raise CaseContractError(f"Case {case.case_id} has an invalid Tool policy.")
        if not 1 <= case.max_decisions <= 3:
            raise CaseContractError(f"Case {case.case_id} has an invalid decision limit.")
        if (case.agent == "critic") != bool(case.allowed_verdicts):
            raise CaseContractError(f"Case {case.case_id} has an invalid verdict policy.")
    return by_id


class _NoNetworkProvider:
    def get_daily_bars(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("structured Agent Eval must not access market data")


def _goal() -> ResearchGoal:
    return ResearchGoal(
        objective="Evaluate one synthetic RSI evidence packet for Agent behavior only.",
        symbol="AAPL",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        allowed_strategies=("rsi",),
        max_drawdown_limit=0.20,
        benchmark_symbol="SPY",
        costs=BacktestCosts(5, 5),
    )


def _metrics(
    total_return: float,
    max_drawdown: float,
    trade_count: int,
    benchmark_return: float,
    sharpe_ratio: float | None = 1.0,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=total_return,
        annualized_volatility=0.18,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        trade_count=trade_count,
        turnover=max(0.2, trade_count * 0.2),
        asset_buy_and_hold_return=0.10,
        benchmark_return=benchmark_return,
    )


def _segment(
    segment: Literal["training", "test"], metrics: PerformanceMetrics
) -> SegmentBacktestResult:
    index = pd.date_range("2024-01-02" if segment == "training" else "2024-10-01", periods=4)
    equity = pd.Series([1.0, 1.02, 0.98, 1.04], index=index)
    asset = pd.Series([1.0, 1.01, 1.02, 1.03], index=index)
    benchmark = pd.Series([1.0, 1.005, 1.015, 1.025], index=index)
    zeros = pd.Series([0.0] * 4, index=index)
    return SegmentBacktestResult(
        segment=segment,
        bar_returns=equity.pct_change().fillna(0.0),
        equity_curve=equity,
        drawdown_curve=pd.Series([0.0, 0.0, metrics.max_drawdown, -0.01], index=index),
        positions=pd.Series([0, 1, 1, 0], index=index, dtype="int8"),
        asset_buy_and_hold_bar_returns=asset.pct_change().fillna(0.0),
        asset_buy_and_hold_equity_curve=asset,
        asset_buy_and_hold_drawdown_curve=zeros,
        benchmark_bar_returns=benchmark.pct_change().fillna(0.0),
        benchmark_equity_curve=benchmark,
        benchmark_drawdown_curve=zeros,
        metrics=metrics,
        drawdown=DrawdownAnalysis(metrics.max_drawdown, index[1], index[2], None, 2),
        warnings=("Synthetic Agent Eval evidence; not a historical backtest.",),
    )


def _quality(symbol: str) -> DataQualityReport:
    return DataQualityReport(symbol, 252, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"))


def _case_metrics(case_id: str) -> tuple[PerformanceMetrics, PerformanceMetrics]:
    training = _metrics(0.16, -0.10, 8, 0.08)
    test = _metrics(0.13, -0.11, 4, 0.07)
    if case_id == "critic_drawdown_breach":
        test = _metrics(-0.18, -0.35, 4, 0.07, -1.0)
    elif case_id == "critic_sample_divergence":
        training = _metrics(0.31, -0.08, 10, 0.08, 1.8)
        test = _metrics(-0.16, -0.24, 3, 0.09, -0.9)
    elif case_id == "critic_low_trade_count":
        training = _metrics(0.01, -0.04, 1, 0.08, 0.1)
        test = _metrics(0.0, -0.03, 1, 0.07, 0.0)
    elif case_id in {"critic_conclusion_conflict", "critic_second_round_no_revise"}:
        test = _metrics(0.04, -0.10, 4, 0.15)
    return training, test


def _final_evaluation(case_id: str) -> FinalEvaluation:
    training, test = _case_metrics(case_id)
    config = RsiConfig(14, 30, 70)
    return FinalEvaluation(
        "AAPL",
        "SPY",
        "rsi",
        config,
        BacktestCosts(5, 5),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-12-31"),
        176,
        pd.Timestamp("2024-09-16"),
        _segment("training", training),
        _segment("test", test),
        _quality("AAPL"),
        _quality("SPY"),
        ("Synthetic Agent Eval evidence; not a historical backtest.",),
    )


def _training_evaluation(final: FinalEvaluation) -> TrainingEvaluation:
    return TrainingEvaluation(
        final.symbol,
        final.benchmark_symbol,
        final.strategy_name,
        final.config,
        final.costs,
        final.data_start,
        final.data_end,
        final.split_index,
        final.split_date,
        final.training,
        final.asset_quality,
        final.benchmark_quality,
        final.warnings,
    )


def _draft(case_id: str, fingerprint: str) -> ResearchDraft:
    conclusions = {
        "critic_drawdown_breach": "Select RSI because its return is acceptable.",
        "critic_sample_divergence": "Select RSI as a stable strategy across both periods.",
        "critic_low_trade_count": "One RSI candidate proves the requested family is robust.",
        "critic_conclusion_conflict": "Select RSI because it outperformed SPY in sample-out.",
        "critic_consistent_accept": (
            "Within this synthetic 2024 evaluation, select the locked RSI configuration; it met "
            "the drawdown limit and beat SPY in both sample segments."
        ),
        "critic_second_round_no_revise": "The revised draft still says RSI beat SPY.",
    }
    limitation = (
        "Two RSI parameterizations were assessed in sample; this synthetic evidence supports "
        "only the locked 2024 configuration, not broad or long-term RSI effectiveness."
        if case_id == "critic_consistent_accept"
        else "Only this locked parameterization is assessed; synthetic evidence is used only to "
        "evaluate Agent behavior."
    )
    return ResearchDraft(
        outcome="selected",
        selected_fingerprint=fingerprint,
        conclusion=conclusions[case_id],
        limitations=(limitation,),
    )


def _critic_workspace(case_id: str) -> tuple[ResearchWorkspace, CriticRoundRuntime]:
    goal, final = _goal(), _final_evaluation(case_id)
    fingerprint = backtest_fingerprint(goal, final.config)
    training = _training_evaluation(final)
    training_evaluations = {fingerprint: training}
    if case_id == "critic_consistent_accept":
        alternative_config = RsiConfig(10, 25, 75)
        training_evaluations[backtest_fingerprint(goal, alternative_config)] = replace(
            training,
            config=alternative_config,
            training=_segment("training", _metrics(0.12, -0.13, 6, 0.08, 0.9)),
        )
    current = CriticRoundRuntime(draft=_draft(case_id, fingerprint))
    rounds = [current]
    if case_id == "critic_second_round_no_revise":
        rounds.insert(
            0,
            CriticRoundRuntime(
                draft=_draft(case_id, fingerprint),
                review=CriticReview(
                    verdict="revise",
                    summary="The conclusion conflicts with the benchmark evidence.",
                    issues=("Correct the SPY comparison.",),
                    required_revision="Correct the conclusion without changing the candidate.",
                ),
            ),
        )
    return (
        ResearchWorkspace(
            goal,
            AgentBudget(),
            _NoNetworkProvider(),  # type: ignore[arg-type]
            Path("/tmp/tragent-evals"),
            current_candidate=final.config,
            validated_configs={fingerprint: final.config},
            training_evaluations=training_evaluations,
            final_evaluations={fingerprint: final},
            phase="locked",
            locked_fingerprint=fingerprint,
            final_batch_completed=True,
            active_agent="critic",
            critic_rounds=rounds,
            training_backtests=len(training_evaluations),
        ),
        current,
    )


def _market_data_event() -> RuntimeEvent:
    return RuntimeEvent(
        "get_market_data",
        {},
        ToolObservation(
            tool_name="get_market_data",
            status="success",
            summary="Synthetic aligned daily evidence is ready.",
            identifiers={"symbol": "AAPL", "benchmark_symbol": "SPY"},
        ),
        "eval-market-data",
        True,
    )


def _statistics_event() -> RuntimeEvent:
    return RuntimeEvent(
        "calculate_asset_statistics",
        {},
        ToolObservation(
            tool_name="calculate_asset_statistics",
            status="success",
            summary="Synthetic asset statistics are ready.",
            identifiers={"symbol": "AAPL"},
            metrics={
                "observations": 252,
                "total_return": 0.10,
                "annualized_return": 0.10,
                "annualized_volatility": 0.18,
                "max_drawdown": -0.12,
            },
        ),
        "eval-statistics",
        True,
    )


def _research_workspace(case_id: str) -> ResearchWorkspace:
    goal = _goal()
    workspace = ResearchWorkspace(
        goal,
        AgentBudget(),
        _NoNetworkProvider(),  # type: ignore[arg-type]
        Path("/tmp/tragent-evals"),
        events=[_market_data_event(), _statistics_event()],
    )
    if case_id == "research_exact_candidate_repeat":
        final = _final_evaluation("critic_consistent_accept")
        fingerprint = backtest_fingerprint(goal, final.config)
        training = _training_evaluation(final)
        workspace.current_candidate = final.config
        workspace.validated_configs[fingerprint] = final.config
        workspace.training_evaluations[fingerprint] = training
        workspace.training_backtests = 1
        workspace.events.append(
            RuntimeEvent(
                "run_backtest",
                {
                    "stage": "training",
                    "strategy_name": "rsi",
                    "parameters": {
                        "rsi_window": 14,
                        "entry_threshold": 30.0,
                        "exit_threshold": 70.0,
                    },
                    "selected_fingerprint": None,
                },
                ToolObservation(
                    tool_name="run_backtest",
                    status="success",
                    summary="The synthetic RSI sample-in backtest completed.",
                    identifiers={"fingerprint": fingerprint, "segment": "training"},
                    metrics={
                        "total_return": training.training.metrics.total_return,
                        "max_drawdown": training.training.metrics.max_drawdown,
                        "trade_count": training.training.metrics.trade_count,
                    },
                ),
                "eval-training",
                True,
            )
        )
    return workspace


def _run_research_case(case: AgentEvalCase, model: BaseChatModel) -> dict[str, object]:
    workspace = _research_workspace(case.case_id)
    tools = build_research_tools(lambda: workspace)
    response = model.bind_tools(tools, parallel_tool_calls=False, strict=False).invoke(
        build_research_messages(workspace)
    )
    actions = _accepted_actions(response, tools)
    return {
        "agent": "research",
        "actions": actions,
        "verdict": None,
        "decisions": 1,
        "tool_calls": len(actions),
        "protocol_valid": len(actions) == 1,
    }


def _run_critic_case(case: AgentEvalCase, model: BaseChatModel) -> dict[str, object]:
    workspace, review_round = _critic_workspace(case.case_id)
    review_round.agent_steps = MAX_CRITIC_AGENT_STEPS - case.max_decisions
    review_round.tool_calls = MAX_CRITIC_TOOL_CALLS - case.max_decisions
    tools = build_critic_tools(lambda: workspace)

    @before_model(can_jump_to=["end"])
    def stop_after_review(_state: object, _runtime: object) -> dict[str, str] | None:
        return {"jump_to": "end"} if review_round.review is not None else None

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[
            stop_after_review,
            require_single_tool_call,
            ModelCallLimitMiddleware(run_limit=case.max_decisions),
        ],
        name="critic_eval_agent",
    )
    output = agent.invoke({"messages": build_critic_messages(workspace, review_round)})
    actions, decisions, valid_actions = _trajectory_actions(output.get("messages"), tools)
    return {
        "agent": "critic",
        "actions": actions,
        "verdict": None if review_round.review is None else review_round.review.verdict,
        "decisions": decisions,
        "tool_calls": len(actions),
        "protocol_valid": (
            valid_actions and len(actions) == decisions and review_round.review is not None
        ),
    }


def _accepted_actions(response: object, tools: tuple[object, ...]) -> list[dict[str, object]]:
    if not isinstance(response, AIMessage) or len(response.tool_calls) != 1:
        return []
    call = response.tool_calls[0]
    tool = next((item for item in tools if getattr(item, "name", None) == call.get("name")), None)
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return []
    try:
        arguments = schema.model_validate(call.get("args")).model_dump(mode="json")
    except (TypeError, ValueError):
        return []
    return [{"tool_name": call["name"], "arguments": arguments}]


def _trajectory_actions(
    messages: object,
    tools: tuple[object, ...],
) -> tuple[list[dict[str, object]], int, bool]:
    if not isinstance(messages, list):
        return [], 0, False
    actions: list[dict[str, object]] = []
    decisions = 0
    for message in messages:
        if not isinstance(message, AIMessage) or not (
            message.tool_calls or message.invalid_tool_calls
        ):
            continue
        decisions += 1
        current = _accepted_actions(message, tools)
        if len(current) != 1:
            return actions, decisions, False
        actions.extend(current)
    return actions, decisions, True


def run_case(
    inputs: Mapping[str, object],
    model: BaseChatModel,
    cases: Mapping[str, AgentEvalCase],
) -> dict[str, object]:
    """Run one checked-in case through the production Agent decision functions."""

    case_id = inputs.get("case_id")
    if not isinstance(case_id, str) or case_id not in cases:
        raise CaseContractError("LangSmith supplied an unknown Agent Eval case.")
    case = cases[case_id]
    return (
        _run_research_case(case, model)
        if case.agent == "research"
        else _run_critic_case(case, model)
    )


__all__ = ["CASES", "AgentEvalCase", "CaseContractError", "run_case", "validate_cases"]
