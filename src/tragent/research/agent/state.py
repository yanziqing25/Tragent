"""The validated Pydantic State checkpointed by the outer Research workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tragent.quant import BacktestCosts
from tragent.research.agent.models import AgentBudget, GoalDraft, LongText
from tragent.research.agent.workspace import ResearchWorkspace

type ConfirmationError = Literal[
    "Invalid confirmation response.",
    "The answer could not be applied; rephrase it and try again.",
]
type TaskStatus = Literal[
    "awaiting_clarification",
    "awaiting_confirmation",
    "running",
    "selected",
    "no_strategy",
    "critic_rejected",
    "failed",
    "budget_exhausted",
    "rejected",
]


class _StrictState(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphState(_StrictState):
    """One task: the request, its HITL goal, and the confirmed research workspace."""

    thread_id: str
    user_id: int = Field(strict=True, ge=1)
    request: LongText
    reference_date: str
    status: TaskStatus
    failure_stage: Literal["goal_parsing"] | None = None
    confirmation_error: ConfirmationError | None = None
    failure_summary: str | None = None
    cost_defaults: BacktestCosts
    position_tranches_default: int = Field(ge=1, le=20)
    budget: AgentBudget
    draft_goal: GoalDraft | None = None
    missing_goal_fields: tuple[str, ...] = ()
    defaulted_fields: tuple[str, ...] = ()
    workspace: ResearchWorkspace | None = None

    def with_workspace(self, workspace: ResearchWorkspace) -> GraphState:
        """Adopt an Agent's workspace; its terminal result decides the task status."""

        status = "running" if workspace.result is None else workspace.result.status
        return self.model_copy(update={"workspace": workspace, "status": status})


def validate_graph_state(value: Mapping[str, object] | GraphState) -> GraphState:
    """Return one detached, validated application State."""

    try:
        if isinstance(value, GraphState):
            return value.model_copy(deep=True)
        return GraphState.model_validate(copy_plain_json(dict(value)))
    except (ValidationError, TypeError, ValueError):
        raise ValueError("Graph State failed validation") from None


def copy_plain_json(value: object) -> object:
    """Return a detached JSON copy for checkpoint updates."""

    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("value must contain finite JSON-compatible values") from None


__all__ = [
    "ConfirmationError",
    "GraphState",
    "TaskStatus",
    "copy_plain_json",
    "validate_graph_state",
]
