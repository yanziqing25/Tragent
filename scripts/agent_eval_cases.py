"""The eight fixed synthetic-evidence cases for the V2 Agent evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, after_model, before_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from tragent.quant import BacktestCosts, PerformanceMetrics, RsiConfig, strategy_parameters
from tragent.research.agent.critic import (
    MAX_CRITIC_AGENT_STEPS,
    MAX_CRITIC_TOOL_CALLS,
    critic_input_messages,
)
from tragent.research.agent.harness import (
    TradingAgentState,
    critic_prompt,
    require_single_tool_call,
    research_prompt,
)
from tragent.research.agent.models import (
    AgentBudget,
    CriticReview,
    MarketContextEvidence,
    ResearchDraft,
    ResearchGoal,
    ToolObservation,
)
from tragent.research.agent.prompt import research_input_messages
from tragent.research.agent.retrieval import CONTEXT_SOURCES
from tragent.research.agent.runtime import (
    backtest_fingerprint,
)
from tragent.research.agent.tools import build_critic_tools, build_research_tools
from tragent.research.agent.workspace import (
    CandidateRecord,
    CriticRound,
    DrawdownRecord,
    ResearchGraphContext,
    ResearchWorkspace,
    SegmentRecord,
    ToolCallRecord,
)

_RESEARCH_TOOL_NAMES = (
    "get_market_data",
    "calculate_asset_statistics",
    "run_backtest",
    "analyze_drawdown",
    "compare_with_benchmark",
    "search_market_context",
    "submit_research_draft",
)
_CRITIC_TOOL_NAMES = (
    "inspect_selected_drawdown",
    "compare_selected_with_benchmark",
    "inspect_retrieved_context",
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


def _segment(metrics: PerformanceMetrics) -> SegmentRecord:
    return SegmentRecord(
        metrics=metrics,
        drawdown=DrawdownRecord(
            max_drawdown=metrics.max_drawdown,
            peak_date=datetime(2024, 3, 1),
            trough_date=datetime(2024, 4, 1),
            recovery_date=None,
            underwater_bars=22,
        ),
        mean_exposure=0.5,
        warnings=("Synthetic Agent Eval evidence; not a historical backtest.",),
    )


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


def _candidate(
    config: RsiConfig, training: PerformanceMetrics, test: PerformanceMetrics | None = None
) -> CandidateRecord:
    return CandidateRecord(
        fingerprint=backtest_fingerprint(_goal(), config),
        strategy_name="rsi",
        parameters=strategy_parameters(config),
        training=_segment(training),
        test=None if test is None else _segment(test),
    )


def _context_evidence() -> tuple[MarketContextEvidence, ...]:
    return (
        MarketContextEvidence(
            id="eval-sec",
            source_type="official",
            source="sec",
            title="Synthetic SEC evidence",
            excerpt="Synthetic official evidence for Agent policy evaluation.",
            published_at="2024-12-20",
            url="https://www.sec.gov/Archives/eval-sec",
        ),
        MarketContextEvidence(
            id="eval-news",
            source_type="news",
            source="yahoo_finance",
            title="Synthetic news evidence",
            excerpt="Synthetic third-party news context for Agent policy evaluation.",
            published_at="2024-12-21",
            url="https://finance.yahoo.com/eval-news",
        ),
        MarketContextEvidence(
            id="eval-community",
            source_type="community",
            source="stocktwits",
            title="Synthetic community evidence",
            excerpt="Synthetic unverified community sentiment for Agent policy evaluation.",
            published_at="2024-12-22",
            url="https://stocktwits.com/eval-community",
        ),
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
        market_context_summary=(
            "Synthetic official, news, and community evidence is included only to test Critic "
            "source handling; it does not change the deterministic result."
        ),
        cited_evidence_ids=("eval-sec", "eval-news", "eval-community"),
    )


def _critic_workspace(case_id: str) -> ResearchWorkspace:
    training, test = _case_metrics(case_id)
    locked = _candidate(RsiConfig(14, 30, 70), training, test)
    candidates = [locked]
    if case_id == "critic_consistent_accept":
        candidates.append(
            _candidate(RsiConfig(10, 25, 75), _metrics(0.12, -0.13, 6, 0.08, 0.9), test)
        )
    rounds = [CriticRound(draft=_draft(case_id, locked.fingerprint))]
    if case_id == "critic_second_round_no_revise":
        rounds.insert(
            0,
            CriticRound(
                draft=_draft(case_id, locked.fingerprint),
                review=CriticReview(
                    verdict="revise",
                    summary="The conclusion conflicts with the benchmark evidence.",
                    issues=("Correct the SPY comparison.",),
                    required_revision="Correct the conclusion without changing the candidate.",
                ),
            ),
        )
    return ResearchWorkspace(
        goal=_goal(),
        budget=AgentBudget(),
        phase="locked",
        active_agent="critic",
        candidates=candidates,
        locked_fingerprint=locked.fingerprint,
        critic_rounds=rounds,
        journal=[_market_context_record(_context_evidence())],
    )


def _record(tool_name: str, arguments: dict[str, object], observation: ToolObservation):
    return ToolCallRecord(step=1, tool_name=tool_name, arguments=arguments, observation=observation)


def _market_context_record(evidence: tuple[MarketContextEvidence, ...]) -> ToolCallRecord:
    return _record(
        "search_market_context",
        {"queries": dict.fromkeys(CONTEXT_SOURCES, "Synthetic Agent Eval market context")},
        ToolObservation(
            tool_name="search_market_context",
            status="success",
            summary="Synthetic context search covered every configured source.",
            metrics={
                "sec": 1,
                "yahoo_finance": 1,
                "stocktwits": 1,
                "reddit": 0,
            },
            evidence=evidence,
        ),
    )


def _research_workspace(case_id: str) -> ResearchWorkspace:
    journal = [
        _record(
            "get_market_data",
            {},
            ToolObservation(
                tool_name="get_market_data",
                status="success",
                summary="Synthetic aligned daily evidence is ready.",
                identifiers={"symbol": "AAPL", "benchmark_symbol": "SPY"},
            ),
        ),
        _record(
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
        ),
    ]
    workspace = ResearchWorkspace(goal=_goal(), budget=AgentBudget(), journal=journal)
    if case_id == "research_exact_candidate_repeat":
        candidate = _candidate(RsiConfig(14, 30, 70), _case_metrics("critic_consistent_accept")[0])
        metrics = candidate.training.metrics
        workspace.candidates.append(candidate)
        workspace.journal.append(
            _record(
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
                    identifiers={"fingerprint": candidate.fingerprint, "segment": "training"},
                    metrics={
                        "total_return": metrics.total_return,
                        "max_drawdown": metrics.max_drawdown,
                        "trade_count": metrics.trade_count,
                    },
                ),
            )
        )
    return workspace


def _context(model: BaseChatModel) -> ResearchGraphContext:
    return ResearchGraphContext(
        model=model,
        provider=_NoNetworkProvider(),  # type: ignore[arg-type]
        artifacts="evals",
    )


def _conversation(journal: list[ToolCallRecord]) -> list[BaseMessage]:
    """Replay a synthetic journal as the ReAct conversation that produced it."""

    messages: list[BaseMessage] = []
    for index, record in enumerate(journal):
        call_id = f"eval-call-{index}"
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": record.tool_name, "args": record.arguments, "id": call_id}],
            )
        )
        messages.append(ToolMessage(record.observation.to_compact_json(), tool_call_id=call_id))
    return messages


def _run_research_case(case: AgentEvalCase, model: BaseChatModel) -> dict[str, object]:
    workspace = _research_workspace(case.case_id)
    tools = build_research_tools()

    @after_model(can_jump_to=["end"])
    def stop_after_decision(_state: object, _runtime: object) -> dict[str, str]:
        return {"jump_to": "end"}

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[require_single_tool_call, research_prompt, stop_after_decision],
        state_schema=TradingAgentState,
        context_schema=ResearchGraphContext,
        name="research_eval_agent",
    )
    history = [*research_input_messages(workspace), *_conversation(workspace.journal)]
    output = agent.invoke(
        {"messages": history, "workspace": workspace.model_dump(mode="json")},
        context=_context(model),
    )
    actions, decisions, valid_actions = _trajectory_actions(
        output["messages"][len(history) :], tools
    )
    return {
        "agent": "research",
        "actions": actions,
        "verdict": None,
        "decisions": decisions,
        "tool_calls": len(actions),
        "protocol_valid": valid_actions and decisions == 1 and len(actions) == 1,
    }


def _run_critic_case(case: AgentEvalCase, model: BaseChatModel) -> dict[str, object]:
    workspace = _critic_workspace(case.case_id)
    review_round = workspace.critic_rounds[-1]
    review_round.agent_steps = MAX_CRITIC_AGENT_STEPS - case.max_decisions
    review_round.tool_calls = MAX_CRITIC_TOOL_CALLS - case.max_decisions
    tools = build_critic_tools(workspace)

    @before_model(can_jump_to=["end"])
    def stop_after_review(state: TradingAgentState, _runtime: object) -> dict[str, str] | None:
        reviewed = ResearchWorkspace.model_validate(state["workspace"]).critic_rounds[-1]
        return {"jump_to": "end"} if reviewed.review is not None else None

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[
            stop_after_review,
            require_single_tool_call,
            critic_prompt,
            ModelCallLimitMiddleware(run_limit=case.max_decisions),
        ],
        state_schema=TradingAgentState,
        context_schema=ResearchGraphContext,
        name="critic_eval_agent",
    )
    output = agent.invoke(
        {
            "messages": critic_input_messages(workspace),
            "workspace": workspace.model_dump(mode="json"),
        },
        context=_context(model),
    )
    review = ResearchWorkspace.model_validate(output["workspace"]).critic_rounds[-1].review
    actions, decisions, valid_actions = _trajectory_actions(output.get("messages"), tools)
    return {
        "agent": "critic",
        "actions": actions,
        "verdict": None if review is None else review.verdict,
        "decisions": decisions,
        "tool_calls": len(actions),
        "protocol_valid": valid_actions and len(actions) == decisions and review is not None,
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
