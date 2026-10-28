"""User-facing projection of one durable Research Agent task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.types import StateSnapshot

from tragent.quant import StrategyName
from tragent.research.agent.events import CriticBudgetUsage, build_agent_status_snapshot
from tragent.research.agent.models import (
    MAX_MINUTE_SESSIONS,
    BudgetUsage,
    CriticVerdict,
    RequestIssue,
)
from tragent.research.agent.state import GraphState, TaskStatus
from tragent.research.agent.workspace import ToolCallRecord
from tragent.research.task_runs import RunState

_ACTIVE_STATUSES = frozenset({"awaiting_clarification", "awaiting_confirmation", "running"})
_AGENT_NODE_NAMES = frozenset({"research_agent", "critic_agent"})
_RECOVERABLE_TASK_ERROR = "The research task can be resumed."
_FAILED_TASK_ERROR = "The research task failed safely."
_GOAL_PARSING_TASK_ERROR = (
    "Goal Parser could not parse this request. Start a new task to try again."
)


class ResearchGoalView(TypedDict):
    """Stable public goal shape shared by draft and confirmed tasks."""

    objective: str
    symbol: str | None
    bar_interval: Literal["1d", "1m"] | None
    start_date: str | None
    end_date: str | None
    training_sessions: int | None
    test_sessions: int | None
    allowed_strategies: tuple[StrategyName, ...] | None
    max_drawdown_limit: float | None
    position_tranches: int | None
    benchmark_symbol: str | None
    request_issue: RequestIssue | None
    transaction_cost_bps: float | None
    slippage_bps: float | None


@dataclass(frozen=True, slots=True)
class ResearchTaskView:
    """Detached task information returned by the API."""

    thread_id: str
    status: TaskStatus
    # Whether a background worker has this task queued or running right now.
    run_state: RunState | None
    can_resume: bool
    goal: ResearchGoalView | None
    missing_goal_fields: tuple[str, ...]
    clarification_hints: tuple[str, ...]
    clarification_example: str | None
    defaulted_fields: tuple[str, ...]
    budget_used: BudgetUsage
    budget_remaining: BudgetUsage
    active_agent: Literal["research", "critic"] | None
    critic_round: int
    critic_verdict: CriticVerdict | None
    critic_budget_used: CriticBudgetUsage
    critic_budget_remaining: CriticBudgetUsage
    last_agent: Literal["research", "critic"] | None
    last_action: str | None
    last_observation: str | None
    selected_fingerprint: str | None
    stop_reason: str | None
    report_exists: bool
    safe_error: str | None


def build_task_view(
    snapshot: StateSnapshot,
    state: GraphState,
    *,
    report_exists: bool,
    run_state: RunState | None,
) -> ResearchTaskView:
    """Project durable Graph State into the task information shown by the Web UI."""

    status = state.status
    goal = _goal_view(state)
    workspace = state.workspace
    active_agent = workspace.active_agent if workspace else None

    critic_rounds = workspace.critic_rounds if workspace else []
    latest_critic_round = critic_rounds[-1] if critic_rounds else None
    latest_review = latest_critic_round.review if latest_critic_round else None
    agent_status = build_agent_status_snapshot(
        active_agent=active_agent,
        budget=state.budget,
        agent_steps=workspace.agent_steps if workspace else 0,
        tool_calls=workspace.tool_calls if workspace else 0,
        training_backtests=workspace.training_backtests if workspace else 0,
        critic_round=len(critic_rounds),
        critic_verdict=latest_review.verdict if latest_review else None,
        critic_agent_steps=(latest_critic_round.agent_steps if latest_critic_round else 0),
        critic_tool_calls=latest_critic_round.tool_calls if latest_critic_round else 0,
    )

    last_agent, last_entry = _latest_agent_entry(state)
    terminal = workspace.result if workspace else None
    # A queued or running worker owns the task; otherwise checkpointed work can be resumed.
    idle = run_state is None
    has_task_error = idle and _snapshot_has_task_error(snapshot)
    has_pending_node = _snapshot_has_pending_node(snapshot)
    clarification_hints, clarification_example = _clarification_guidance(
        status,
        state.missing_goal_fields,
        goal,
    )

    safe_error: str | None = None
    if has_task_error and status in _ACTIVE_STATUSES:
        safe_error = _RECOVERABLE_TASK_ERROR
    elif status == "failed":
        safe_error = (
            _GOAL_PARSING_TASK_ERROR
            if state.failure_stage == "goal_parsing"
            else _FAILED_TASK_ERROR
        )
    elif state.confirmation_error is not None:
        safe_error = state.confirmation_error

    return ResearchTaskView(
        thread_id=state.thread_id,
        status=status,
        run_state=run_state,
        can_resume=idle and status in _ACTIVE_STATUSES and (has_pending_node or has_task_error),
        goal=goal,
        missing_goal_fields=state.missing_goal_fields,
        clarification_hints=clarification_hints,
        clarification_example=clarification_example,
        defaulted_fields=state.defaulted_fields,
        budget_used=agent_status.budget_used,
        budget_remaining=agent_status.budget_remaining,
        active_agent=active_agent,
        critic_round=len(critic_rounds),
        critic_verdict=latest_review.verdict if latest_review else None,
        critic_budget_used=agent_status.critic_budget_used,
        critic_budget_remaining=agent_status.critic_budget_remaining,
        last_agent=last_agent,
        last_action=last_entry.tool_name if last_entry else None,
        last_observation=last_entry.observation.summary if last_entry else None,
        selected_fingerprint=terminal.selected_fingerprint if terminal else None,
        stop_reason=terminal.stop_reason if terminal else None,
        report_exists=report_exists,
        safe_error=safe_error,
    )


def _goal_view(state: GraphState) -> ResearchGoalView | None:
    if state.workspace is not None:
        goal = state.workspace.goal
        return ResearchGoalView(
            objective=goal.objective,
            symbol=goal.symbol,
            bar_interval=goal.bar_interval,
            start_date=goal.start_date.isoformat() if goal.start_date else None,
            end_date=goal.end_date.isoformat() if goal.end_date else None,
            training_sessions=goal.training_sessions,
            test_sessions=goal.test_sessions,
            allowed_strategies=goal.allowed_strategies,
            max_drawdown_limit=goal.max_drawdown_limit,
            position_tranches=goal.position_tranches,
            benchmark_symbol=goal.benchmark_symbol,
            request_issue=None,
            transaction_cost_bps=goal.costs.transaction_cost_bps,
            slippage_bps=goal.costs.slippage_bps,
        )

    draft_value = state.draft_goal
    if draft_value is None:
        return None
    goal = draft_value
    return ResearchGoalView(
        objective=state.request,
        symbol=goal.symbol,
        bar_interval=goal.bar_interval,
        start_date=goal.start_date,
        end_date=goal.end_date,
        training_sessions=goal.training_sessions,
        test_sessions=goal.test_sessions,
        allowed_strategies=(
            tuple(goal.allowed_strategies) if goal.allowed_strategies is not None else None
        ),
        max_drawdown_limit=(
            float(goal.max_drawdown_limit) if goal.max_drawdown_limit is not None else None
        ),
        position_tranches=goal.position_tranches,
        benchmark_symbol=goal.benchmark_symbol,
        request_issue=goal.request_issue,
        transaction_cost_bps=(
            float(goal.transaction_cost_bps) if goal.transaction_cost_bps is not None else None
        ),
        slippage_bps=float(goal.slippage_bps) if goal.slippage_bps is not None else None,
    )


def _latest_agent_entry(
    state: GraphState,
) -> tuple[Literal["research", "critic"] | None, ToolCallRecord | None]:
    workspace = state.workspace
    if workspace is None:
        return None, None
    research_journal = workspace.journal
    critic_rounds = workspace.critic_rounds
    research_entry = research_journal[-1] if research_journal else None
    if not critic_rounds or not critic_rounds[-1].journal:
        return ("research" if research_entry is not None else None), research_entry

    critic_entry = critic_rounds[-1].journal[-1]
    active_agent = workspace.active_agent
    if active_agent == "critic":
        return "critic", critic_entry
    if active_agent is None:
        latest_review = critic_rounds[-1].review
        if latest_review is None or latest_review.verdict in {"accept", "reject"}:
            return "critic", critic_entry

    if research_entry is not None and research_entry.tool_name != "submit_research_draft":
        return "research", research_entry
    return "critic", critic_entry


def _snapshot_has_task_error(snapshot: StateSnapshot) -> bool:
    return any(task.error is not None for task in snapshot.tasks)


def _snapshot_has_pending_node(snapshot: StateSnapshot) -> bool:
    if snapshot.next:
        return True
    return any(task.name in _AGENT_NODE_NAMES and task.result is None for task in snapshot.tasks)


def _clarification_guidance(
    status: TaskStatus,
    missing_fields: tuple[str, ...],
    goal: Mapping[str, object] | None,
) -> tuple[tuple[str, ...], str | None]:
    if status != "awaiting_clarification" or not missing_fields:
        return (), None

    request_issue = goal.get("request_issue") if goal is not None else None
    hints = tuple(
        _clarification_hint(field, request_issue=request_issue) for field in missing_fields
    )
    return hints, _clarification_example(
        missing_fields,
        request_issue=request_issue,
        goal=goal,
    )


def _clarification_hint(field_name: str, *, request_issue: object) -> str:
    if field_name == "request_details":
        if request_issue == "unsupported_request":
            return "Supported scope / 当前仅支持美股股票的日线或 1 分钟研究，请修改请求"
        if request_issue == "unrelated_request":
            return "Research goal / 请输入美股股票研究目标"
    hints = {
        "symbol": "Stock symbol / 股票代码，例如 NOW",
        "training_sessions": (
            f"Complete in-sample trading days, at most {MAX_MINUTE_SESSIONS} in total / "
            f"样本内完整交易日数量，与样本外合计不超过 {MAX_MINUTE_SESSIONS}，例如 2"
        ),
        "test_sessions": (
            f"Complete out-of-sample trading days, at most {MAX_MINUTE_SESSIONS} in total / "
            f"样本外完整交易日数量，与样本内合计不超过 {MAX_MINUTE_SESSIONS}，例如 1"
        ),
        "bar_interval": "Daily or minute bars / 使用日线还是 1 分钟线",
        "start_date": "Daily start date or range / 日线开始日期，或直接说明完整范围",
        "end_date": "Daily end date or range / 日线结束日期，或直接说明完整范围",
        "benchmark_symbol": "Benchmark / 日线可指定一个美股或 ETF 代码，未指定时默认 SPY",
        "request_details": "Research details / 澄清不支持或有歧义的要求",
    }
    return hints.get(field_name, field_name.replace("_", " "))


def _clarification_example(
    missing_fields: tuple[str, ...],
    *,
    request_issue: object,
    goal: Mapping[str, object] | None,
) -> str:
    if request_issue == "unsupported_request":
        symbol = goal.get("symbol") if goal is not None else None
        return f"改为研究 {symbol or 'AAPL'} 最近三年的日线"
    if request_issue == "unrelated_request":
        return "研究 AAPL 最近三年的日线"
    parts: list[str] = []
    if "symbol" in missing_fields:
        parts.append("标的是 NOW")
    if "training_sessions" in missing_fields:
        parts.append("2个样本内")
    if "test_sessions" in missing_fields:
        parts.append("1个样本外")
    if "bar_interval" in missing_fields:
        parts.append("使用1分钟数据")
    if "start_date" in missing_fields or "end_date" in missing_fields:
        parts.append("日线范围为最近三年")
    if "benchmark_symbol" in missing_fields:
        parts.append("benchmark 使用 QQQ")
    return "，".join(parts) if parts else "请说明你的研究要求"


__all__ = ["ResearchGoalView", "ResearchTaskView", "TaskStatus", "build_task_view"]
