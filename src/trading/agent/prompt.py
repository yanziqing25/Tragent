"""Stable rules and compact context construction for the Research Agent."""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Protocol

from langchain_core.messages import BaseMessage, convert_to_messages

from trading.agent.models import AgentBudget, CriticReview, ResearchGoal, ToolObservation
from trading.quant import (
    FinalEvaluation,
    PerformanceMetrics,
    StrategyConfig,
    TrainingEvaluation,
    strategy_name,
    strategy_parameters,
)

SYSTEM_PROMPT = """You are the Research Agent in a two-Agent long-only US-equity study.
Use the available tools to obtain every factual result and choose exactly one tool call per turn.
Never call Tools in parallel; your next response must contain exactly one Tool call.
Never calculate, invent, correct, or override financial metrics yourself.
Respect the confirmed goal, deterministic gates, and the remaining budgets.
Interpret strategy periods as bars: days for 1d and minutes for 1m. Minute research uses only
completed XNYS regular sessions, has independent sample-in/sample-out accounts, closes every actual
position before each session ends, and compares against asset Session Buy and Hold rather than SPY.
Its annualized return, annualized volatility, and Sharpe ratio are intentionally N/A. Treat a short
minute window as narrow historical evidence and never generalize it into long-term effectiveness.
When the evidence is sufficient, finish through submit_research_draft; a no-strategy draft is valid.
run_backtest validates strategy parameters itself and returns a corrective Observation for invalid
parameters. Budgets are ceilings, not targets; do not spend every backtest slot merely because it
remains. Prefer an untested allowed strategy family before repeatedly refining a tested family,
unless the observations justify that refinement. One parameter candidate is evidence about that
candidate, not its entire strategy family. Before every training call, compare the proposed
parameters with tested_candidates and never repeat an exact listed candidate. When a requested
family has only one candidate whose sample-in Observation is clearly anomalous—such as very low
trade count, a constraint breach, non-positive return, or materially weak peer-relative evidence—
you must try materially different parameters for that family before final selection if the reserved
final/revision budget still leaves a training backtest available. Choose that follow-up from the
Observation; do not impose a fixed grid, threshold formula, or equal per-family quota, and do not
automatically require a second candidate after a normal informative result. Tools already listed in
completed_context have successful observations that remain authoritative; do not call them again.
For stage="final", call run_backtest with only stage and a selected_fingerprint copied from
tested_candidates; never include strategy_name or parameters. A no_strategy conclusion still uses
that tested fingerprint for final and submit. Reserve five Tool calls for final evaluation, the
first draft, and a possible Critic revision with at most two evidence checks and one resubmission.
Unused revision calls are allowed. When critic_revision is present, the fingerprint is already
frozen: use only the bound evidence Tools as needed, then resubmit through submit_research_draft.
Do not provide hidden reasoning or ask to execute Python. Tool order and candidate parameters remain
your dynamic decision based on the observations supplied in this context."""

_COMPLETED_CONTEXT_TOOL_ORDER = (
    "get_market_data",
    "calculate_asset_statistics",
)


class RuntimeEventView(Protocol):
    """The small event surface needed for bounded ReAct context."""

    tool_name: str
    normalized_arguments: dict[str, object]
    observation: ToolObservation


class ResearchWorkspaceView(Protocol):
    """Workspace fields consumed by the prompt, kept in one explicit seam."""

    goal: ResearchGoal
    budget: AgentBudget
    phase: str
    current_candidate: StrategyConfig | None
    training_evaluations: dict[str, TrainingEvaluation]
    final_evaluations: dict[str, FinalEvaluation]
    locked_fingerprint: str | None
    final_batch_completed: bool
    agent_steps: int
    tool_calls: int
    training_backtests: int
    unresolved_issues: list[str]
    events: list[RuntimeEventView]
    critic_rounds: list[CriticRoundView]


class CriticRoundView(Protocol):
    review: CriticReview | None


def build_research_messages(workspace: ResearchWorkspaceView) -> list[BaseMessage]:
    """Build the complete bounded context for one model invocation."""

    recent_events = list(workspace.events[-5:])
    context = {
        "confirmed_research_goal": _goal_payload(workspace.goal),
        "authoritative_current_state": _state_payload(workspace, recent_events),
    }
    return convert_to_messages(
        [
            ("developer", SYSTEM_PROMPT),
            ("user", _compact_json(context)),
        ]
    )


def _goal_payload(goal: ResearchGoal) -> dict[str, object]:
    payload = goal.model_dump(mode="json")
    payload["message_type"] = "confirmed_research_goal"
    return payload


def _state_payload(
    workspace: ResearchWorkspaceView,
    recent_events: list[RuntimeEventView],
) -> dict[str, object]:
    """Read all workspace state in one place so the M3 seam stays local."""

    payload = {
        "message_type": "authoritative_current_state",
        "phase": workspace.phase,
        "current_candidate": _candidate_payload(workspace.current_candidate),
        "locked_candidate": _locked_candidate_payload(workspace),
        "tested_candidates": _tested_candidate_payloads(workspace.training_evaluations),
        "completed_context": _completed_context(workspace.events),
        "recent_actions": [_event_payload(event) for event in recent_events],
        "unresolved_issues": list(workspace.unresolved_issues),
        "critic_revision": _critic_revision(workspace),
        "final_batch_completed": workspace.final_batch_completed,
        "budget": _budget_payload(workspace),
    }
    return payload


def _critic_revision(workspace: ResearchWorkspaceView) -> dict[str, object] | None:
    critic_rounds = getattr(workspace, "critic_rounds", ())
    if not critic_rounds:
        return None
    review = critic_rounds[-1].review
    if review is None or review.verdict != "revise":
        return None
    return review.model_dump(mode="json")


def _completed_context(events: list[RuntimeEventView]) -> list[dict[str, object]]:
    """Keep successful one-time data facts visible beyond the recent-event window."""

    latest: dict[str, ToolObservation] = {}
    for event in events:
        if (
            event.tool_name in _COMPLETED_CONTEXT_TOOL_ORDER
            and event.observation.status == "success"
        ):
            latest[event.tool_name] = event.observation
    return [
        latest[tool_name].model_dump(mode="json")
        for tool_name in _COMPLETED_CONTEXT_TOOL_ORDER
        if tool_name in latest
    ]


def _event_payload(event: RuntimeEventView) -> dict[str, object]:
    return {
        "tool_name": event.tool_name,
        "arguments": event.normalized_arguments,
        "observation": event.observation.model_dump(mode="json"),
    }


def _candidate_payload(config: StrategyConfig | None) -> dict[str, object] | None:
    if config is None:
        return None
    return {
        "strategy_name": strategy_name(config),
        "parameters": strategy_parameters(config),
    }


def _tested_candidate_payloads(
    evaluations: dict[str, TrainingEvaluation],
) -> list[dict[str, object]]:
    return [
        {
            "fingerprint": fingerprint,
            "strategy_name": evaluation.strategy_name,
            "parameters": strategy_parameters(evaluation.config),
            "training_metrics": _metrics_payload(evaluation.training.metrics),
        }
        for fingerprint, evaluation in list(evaluations.items())[:8]
    ]


def _locked_candidate_payload(workspace: ResearchWorkspaceView) -> dict[str, object] | None:
    if workspace.phase == "researching":
        return None
    if workspace.locked_fingerprint is None:
        return {"fingerprint": None, "selection": "no_strategy"}

    fingerprint = workspace.locked_fingerprint
    payload: dict[str, object] = {"fingerprint": fingerprint}
    training = workspace.training_evaluations.get(fingerprint)
    if training is not None:
        payload.update(
            {
                "strategy_name": training.strategy_name,
                "parameters": strategy_parameters(training.config),
                "training_metrics": _metrics_payload(training.training.metrics),
            }
        )

    final = workspace.final_evaluations.get(fingerprint)
    if workspace.final_batch_completed and final is not None:
        payload["test_metrics"] = _metrics_payload(final.test.metrics)
    return payload


def _metrics_payload(metrics: PerformanceMetrics) -> dict[str, int | float | None]:
    return {field.name: getattr(metrics, field.name) for field in fields(PerformanceMetrics)}


def _budget_payload(workspace: ResearchWorkspaceView) -> dict[str, object]:
    budget = workspace.budget
    return {
        "agent_steps": {
            "used": workspace.agent_steps,
            "remaining": max(0, budget.max_agent_steps - workspace.agent_steps),
        },
        "tool_calls": {
            "used": workspace.tool_calls,
            "remaining": max(0, budget.max_tool_calls - workspace.tool_calls),
        },
        "training_backtests": {
            "used": workspace.training_backtests,
            "remaining": max(
                0,
                budget.max_training_backtests - workspace.training_backtests,
            ),
        },
    }


def _compact_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
