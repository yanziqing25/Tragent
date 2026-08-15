"""User-facing projection of one durable Research Agent task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from trading.agent.events import build_agent_status_snapshot
from trading.agent.models import AgentBudget
from trading.agent.state import GraphState, TaskStatus, copy_plain_json

_ACTIVE_STATUSES = frozenset({"awaiting_clarification", "awaiting_confirmation", "running"})
_AGENT_NODE_NAMES = frozenset({"research_agent", "critic_agent"})
_RECOVERABLE_TASK_ERROR = "The research task can be resumed."
_FAILED_TASK_ERROR = "The research task failed safely."
_GOAL_PARSING_TASK_ERROR = (
    "Goal Parser could not parse this request. Start a new task to try again."
)
_INVALID_CONFIRMATION_ERROR = "Invalid confirmation response."


@dataclass(frozen=True, slots=True)
class ResearchTaskView:
    """Detached task information returned by the API."""

    thread_id: str
    status: TaskStatus
    can_resume: bool
    goal: dict[str, object] | None
    missing_goal_fields: tuple[str, ...]
    clarification_hints: tuple[str, ...]
    clarification_example: str | None
    defaulted_fields: tuple[str, ...]
    budget_used: dict[str, int]
    budget_remaining: dict[str, int]
    active_agent: str | None
    critic_round: int
    critic_verdict: str | None
    critic_budget_used: dict[str, int]
    critic_budget_remaining: dict[str, int]
    last_agent: str | None
    last_action: str | None
    last_observation: str | None
    selected_fingerprint: str | None
    stop_reason: str | None
    report_exists: bool
    safe_error: str | None


def build_task_view(
    snapshot: object,
    state: GraphState,
    *,
    report_exists: bool,
    recoverable: bool = False,
) -> ResearchTaskView:
    """Project durable Graph State into the task information shown by the Web UI."""

    status = cast(TaskStatus, state["status"])
    goal_value = state["confirmed_goal"] or state["draft_goal"]
    goal = None if goal_value is None else cast(dict[str, object], copy_plain_json(goal_value))
    if goal is not None and "objective" not in goal:
        goal["objective"] = state["request"]

    critic_rounds = state["critic_rounds"]
    latest_critic_round = critic_rounds[-1] if critic_rounds else None
    latest_review = None if latest_critic_round is None else latest_critic_round["review"]
    agent_status = build_agent_status_snapshot(
        active_agent=state["active_agent"],
        budget=AgentBudget.model_validate(state["budget"]),
        agent_steps=state["agent_steps"],
        tool_calls=state["tool_calls"],
        training_backtests=state["training_backtests"],
        critic_round=len(critic_rounds),
        critic_verdict=None if latest_review is None else latest_review["verdict"],
        critic_agent_steps=(
            0 if latest_critic_round is None else latest_critic_round["agent_steps"]
        ),
        critic_tool_calls=(0 if latest_critic_round is None else latest_critic_round["tool_calls"]),
    )

    last_agent, last_entry = _latest_agent_entry(state)
    terminal = state["terminal_result"]
    has_task_error = recoverable or _snapshot_has_task_error(snapshot)
    has_pending_node = _snapshot_has_pending_node(snapshot)
    clarification_hints, clarification_example = _clarification_guidance(
        status,
        tuple(state["missing_goal_fields"]),
        goal,
    )

    safe_error: str | None = None
    if has_task_error and status in _ACTIVE_STATUSES:
        safe_error = _RECOVERABLE_TASK_ERROR
    elif status == "failed":
        safe_error = (
            _GOAL_PARSING_TASK_ERROR
            if state["failure_stage"] == "goal_parsing"
            else _FAILED_TASK_ERROR
        )
    elif state["confirmation_error"] is not None:
        safe_error = _INVALID_CONFIRMATION_ERROR

    return ResearchTaskView(
        thread_id=state["thread_id"],
        status=status,
        can_resume=status in _ACTIVE_STATUSES and (has_pending_node or has_task_error),
        goal=goal,
        missing_goal_fields=tuple(state["missing_goal_fields"]),
        clarification_hints=clarification_hints,
        clarification_example=clarification_example,
        defaulted_fields=tuple(state["defaulted_fields"]),
        budget_used=agent_status.budget_used.model_dump(mode="json"),
        budget_remaining=agent_status.budget_remaining.model_dump(mode="json"),
        active_agent=state["active_agent"],
        critic_round=len(critic_rounds),
        critic_verdict=None if latest_review is None else latest_review["verdict"],
        critic_budget_used=agent_status.critic_budget_used.model_dump(mode="json"),
        critic_budget_remaining=agent_status.critic_budget_remaining.model_dump(mode="json"),
        last_agent=last_agent,
        last_action=None if last_entry is None else last_entry["tool_name"],
        last_observation=(
            None if last_entry is None else cast(str, last_entry["observation"]["summary"])
        ),
        selected_fingerprint=None if terminal is None else terminal["selected_fingerprint"],
        stop_reason=None if terminal is None else terminal["stop_reason"],
        report_exists=report_exists,
        safe_error=safe_error,
    )


def _latest_agent_entry(
    state: GraphState,
) -> tuple[str | None, dict[str, object] | None]:
    research_journal = state["message_journal"]
    critic_rounds = state["critic_rounds"]
    research_entry = research_journal[-1] if research_journal else None
    if not critic_rounds or not critic_rounds[-1]["journal"]:
        return ("research" if research_entry is not None else None), research_entry

    critic_entry = critic_rounds[-1]["journal"][-1]
    active_agent = state["active_agent"]
    if active_agent == "critic":
        return "critic", critic_entry
    if active_agent is None:
        latest_review = critic_rounds[-1]["review"]
        if latest_review is None or latest_review["verdict"] in {"accept", "reject"}:
            return "critic", critic_entry

    draft_positions = [
        index
        for index, entry in enumerate(research_journal)
        if entry["tool_name"] == "submit_research_draft"
    ]
    latest_draft_position = draft_positions[len(critic_rounds) - 1]
    if latest_draft_position < len(research_journal) - 1:
        return "research", research_entry
    return "critic", critic_entry


def _snapshot_has_task_error(snapshot: object) -> bool:
    tasks = getattr(snapshot, "tasks", ())
    return isinstance(tasks, tuple) and any(
        getattr(task, "error", None) is not None for task in tasks
    )


def _snapshot_has_pending_node(snapshot: object) -> bool:
    next_nodes = getattr(snapshot, "next", ())
    if isinstance(next_nodes, tuple) and next_nodes:
        return True
    tasks = getattr(snapshot, "tasks", ())
    return isinstance(tasks, tuple) and any(
        getattr(task, "name", None) in _AGENT_NODE_NAMES and getattr(task, "result", None) is None
        for task in tasks
    )


def _clarification_guidance(
    status: TaskStatus,
    missing_fields: tuple[str, ...],
    goal: Mapping[str, object] | None,
) -> tuple[tuple[str, ...], str | None]:
    if status != "awaiting_clarification" or not missing_fields:
        return (), None

    request_issue = goal.get("request_issue") if isinstance(goal, Mapping) else None
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
        "training_sessions": "Complete in-sample trading days / 样本内完整交易日数量，例如 2",
        "test_sessions": "Complete out-of-sample trading days / 样本外完整交易日数量，例如 1",
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
        symbol = goal.get("symbol") if isinstance(goal, Mapping) else None
        safe_symbol = symbol if isinstance(symbol, str) else "AAPL"
        return f"改为研究 {safe_symbol} 最近三年的日线"
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


__all__ = ["ResearchTaskView", "TaskStatus", "build_task_view"]
