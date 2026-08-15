"""Safe business events streamed from the durable Agent workflow."""

from typing import Literal

from langgraph.config import get_stream_writer
from pydantic import BaseModel, ConfigDict, Field

from trading.agent.critic import MAX_CRITIC_AGENT_STEPS, MAX_CRITIC_TOOL_CALLS
from trading.agent.models import AgentBudget, BudgetUsage, CriticVerdict
from trading.agent.workspace import ActiveAgent


class CriticBudgetUsage(BaseModel):
    """Bounded counters for the current Critic review round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Critic model decisions consumed in the active review round.
    agent_steps: int = Field(ge=0)
    # Critic evidence/submission Tool calls consumed in the active review round.
    tool_calls: int = Field(ge=0)


class AgentStatusSnapshot(BaseModel):
    """Small safe runtime snapshot attached to Research/Critic events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Agent currently expected to make the next decision.
    active_agent: Literal["research", "critic"] | None
    # Cumulative Research Agent budget counters.
    budget_used: BudgetUsage
    # Remaining Research Agent allowances.
    budget_remaining: BudgetUsage
    # One-based Critic round count; zero before a draft is submitted.
    critic_round: int = Field(ge=0, le=2)
    # Latest submitted Critic verdict, when one exists.
    critic_verdict: Literal["accept", "revise", "reject"] | None
    # Counters consumed in the active/latest Critic round.
    critic_budget_used: CriticBudgetUsage
    # Remaining allowances in the active/latest Critic round.
    critic_budget_remaining: CriticBudgetUsage


class AgentEvent(BaseModel):
    """One bounded user-facing account of meaningful Agent progress."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Project-defined workflow component that emitted this event.
    source: Literal["goal_parser", "workflow", "research_agent", "critic_agent"]
    # Project-defined semantic category; this is not an SSE protocol field.
    kind: Literal["status", "decision", "tool", "observation"]
    # Bounded user-facing progress sentence displayed by Vue.
    message: str = Field(min_length=1, max_length=200)
    # Tool involved in a tool/observation event, when applicable.
    tool_name: str | None = Field(default=None, min_length=1, max_length=64)
    # Optional counters allowing Vue to refresh progress without fetching full state.
    agent_status: AgentStatusSnapshot | None = None


def build_agent_status_snapshot(
    *,
    active_agent: ActiveAgent,
    budget: AgentBudget,
    agent_steps: int,
    tool_calls: int,
    training_backtests: int,
    critic_round: int,
    critic_agent_steps: int = 0,
    critic_tool_calls: int = 0,
    critic_verdict: CriticVerdict | None = None,
) -> AgentStatusSnapshot:
    """Project live or durable Agent counters into one public status contract."""

    has_critic_round = critic_round > 0
    return AgentStatusSnapshot(
        active_agent=active_agent,
        budget_used=BudgetUsage(
            agent_steps=agent_steps,
            tool_calls=tool_calls,
            training_backtests=training_backtests,
        ),
        budget_remaining=BudgetUsage(
            agent_steps=budget.max_agent_steps - agent_steps,
            tool_calls=budget.max_tool_calls - tool_calls,
            training_backtests=budget.max_training_backtests - training_backtests,
        ),
        critic_round=critic_round,
        critic_verdict=critic_verdict,
        critic_budget_used=CriticBudgetUsage(
            agent_steps=critic_agent_steps,
            tool_calls=critic_tool_calls,
        ),
        critic_budget_remaining=CriticBudgetUsage(
            agent_steps=(MAX_CRITIC_AGENT_STEPS - critic_agent_steps if has_critic_round else 0),
            tool_calls=(MAX_CRITIC_TOOL_CALLS - critic_tool_calls if has_critic_round else 0),
        ),
    )


def emit_agent_event(event: AgentEvent) -> None:
    """Write one event to LangGraph's official custom stream."""

    get_stream_writer()(event.model_dump(mode="json"))


__all__ = [
    "AgentEvent",
    "AgentStatusSnapshot",
    "CriticBudgetUsage",
    "build_agent_status_snapshot",
    "emit_agent_event",
]
