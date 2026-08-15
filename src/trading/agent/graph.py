"""Durable HITL workflow containing Research and Critic ReAct Agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from trading.agent.critic import MAX_CRITIC_AGENT_STEPS
from trading.agent.events import AgentEvent, emit_agent_event
from trading.agent.finalization import finalize_critic_review
from trading.agent.goal import (
    GoalConfirmation,
    GoalSeed,
    apply_goal_answer,
    confirmation_payload,
    materialize_goal,
    parse_goal,
    seed_from_node_state,
    seed_from_validated_state,
)
from trading.agent.harness import RecoverableResearchError, build_agent_harness
from trading.agent.models import GoalDraft
from trading.agent.state import (
    GraphState,
    copy_plain_json,
    restore_workspace_from_state,
    validate_graph_state,
    workspace_to_state,
)
from trading.agent.workspace import ResearchWorkspace
from trading.quant import MarketDataProvider


@dataclass(frozen=True, slots=True)
class ResearchGraphContext:
    """Run-scoped integrations that must never enter checkpoint State."""

    # Stateless LangChain chat model client rebuilt for every start/resume request.
    model: BaseChatModel
    # Runtime market-data service; the durable state stores only its data snapshot.
    provider: MarketDataProvider
    # Task-specific filesystem destination for the deterministic report artifact.
    output_dir: Path


def build_research_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile the outer HITL workflow with two standard ReAct Agent subgraphs."""

    builder = StateGraph(
        # Complete mutable data shared by nodes and persisted by the checkpointer.
        state_schema=GraphState,
        # Non-persisted runtime services available through Runtime.context.
        context_schema=ResearchGraphContext,
        # Smaller payload accepted only when a brand-new graph thread starts.
        input_schema=GoalSeed,
        # Public shape returned when one graph invocation stops or completes.
        output_schema=GraphState,
    )
    builder.add_node("parse_goal", _parse_goal_node)
    builder.add_node(
        "confirm_goal",
        _confirm_goal_node,
        destinations=("parse_goal", "confirm_goal", "research_agent", END),
    )
    builder.add_node(
        "research_agent",
        _research_agent_node,
        destinations=("critic_agent", END),
    )
    builder.add_node(
        "critic_agent",
        _critic_agent_node,
        destinations=("research_agent", END),
    )
    builder.add_edge(START, "parse_goal")
    builder.add_conditional_edges("parse_goal", _route_after_parse)
    return builder.compile(checkpointer=checkpointer, name="durable_research_critic_agents")


def _parse_goal_node(
    state: Mapping[str, object],
    runtime: Runtime[ResearchGraphContext],
) -> GraphState:
    seed = seed_from_node_state(state)
    context = runtime.context
    emit_agent_event(
        AgentEvent(
            source="goal_parser",
            kind="status",
            message="Goal Parser · Parsing the research request...",
        )
    )
    return parse_goal(seed, model=context.model)


def _route_after_parse(state: Mapping[str, object]) -> Literal["confirm_goal", "__end__"]:
    validated = validate_graph_state(state)
    return (
        "confirm_goal"
        if validated["status"] in {"awaiting_clarification", "awaiting_confirmation"}
        else END
    )


def _confirm_goal_node(
    state: Mapping[str, object],
    runtime: Runtime[ResearchGraphContext],
) -> Command:
    validated = validate_graph_state(state)
    if (
        validated["status"] not in {"awaiting_clarification", "awaiting_confirmation"}
        or validated["draft_goal"] is None
    ):
        raise ValueError("only an awaiting goal may enter confirmation")
    context = runtime.context
    emit_agent_event(
        AgentEvent(
            source="workflow",
            kind="status",
            message="Research Workflow · Applying goal confirmation...",
        )
    )
    raw_response = interrupt(confirmation_payload(validated))
    try:
        response = GoalConfirmation.model_validate(raw_response)
    except (ValidationError, TypeError, ValueError):
        invalid = cast(dict[str, object], copy_plain_json(validated))
        invalid["confirmation_error"] = "Invalid confirmation response."
        return Command(
            update=validate_graph_state(invalid),
            goto="confirm_goal",
        )

    if response.answer is not None:
        updated = apply_goal_answer(validated, response.answer, model=context.model)
        return Command(update=updated, goto="confirm_goal")

    if response.action == "approve":
        if validated["status"] != "awaiting_confirmation":
            invalid = cast(dict[str, object], copy_plain_json(validated))
            invalid["confirmation_error"] = "Invalid confirmation response."
            return Command(update=validate_graph_state(invalid), goto="confirm_goal")
        seed = seed_from_validated_state(validated)
        goal = materialize_goal(seed, GoalDraft.model_validate(validated["draft_goal"]))
        approved = cast(dict[str, object], copy_plain_json(validated))
        approved.update(
            {
                "status": "running",
                "failure_stage": None,
                "confirmation_error": None,
                "failure_summary": None,
                "confirmed_goal": goal.model_dump(mode="json"),
                "phase": "researching",
                "active_agent": "research",
            }
        )
        return Command(update=validate_graph_state(approved), goto="research_agent")

    if response.action == "reject":
        rejected = cast(dict[str, object], copy_plain_json(validated))
        rejected.update(
            {
                "status": "rejected",
                "failure_stage": None,
                "confirmation_error": None,
                "failure_summary": None,
                "missing_goal_fields": [],
            }
        )
        return Command(update=validate_graph_state(rejected), goto=END)

    raise ValueError("goal confirmation response is invalid")


def _research_agent_node(
    state: Mapping[str, object],
    runtime: Runtime[ResearchGraphContext],
    config: RunnableConfig,
) -> Command:
    """Run one standard Research ReAct subgraph until draft submission or terminal."""

    validated, context, workspace = _restore_running_workspace(
        state, runtime, active_agent="research"
    )
    agent = build_agent_harness(
        model=context.model,
        kind="research",
        workspace=workspace,
        context_schema=ResearchGraphContext,
    )
    output = agent.invoke(
        {"messages": [], "workspace_state": validated},
        config=_agent_config(config, max_turns=workspace.budget.max_agent_steps),
        context=context,
        durability="sync",
    )
    checkpoint = validate_graph_state(output["workspace_state"])

    if checkpoint["status"] != "running":
        return Command(update=checkpoint, goto=END)
    if checkpoint["active_agent"] != "critic":
        raise RuntimeError("Research Agent ended without submitting a draft")
    return Command(update=checkpoint, goto="critic_agent")


def _critic_agent_node(
    state: Mapping[str, object],
    runtime: Runtime[ResearchGraphContext],
    config: RunnableConfig,
) -> Command:
    """Run one isolated Critic ReAct subgraph and route its submitted verdict."""

    validated, context, workspace = _restore_running_workspace(
        state, runtime, active_agent="critic"
    )
    agent = build_agent_harness(
        model=context.model,
        kind="critic",
        workspace=workspace,
        context_schema=ResearchGraphContext,
    )
    output = agent.invoke(
        {"messages": [], "workspace_state": validated},
        config=_agent_config(config, max_turns=MAX_CRITIC_AGENT_STEPS),
        context=context,
        durability="sync",
    )
    checkpoint = validate_graph_state(output["workspace_state"])

    if checkpoint["status"] != "running":
        return Command(update=checkpoint, goto=END)
    restored, reviewed = _workspace_from_checkpoint(checkpoint, context)
    review = reviewed.critic_rounds[-1].review if reviewed.critic_rounds else None
    if review is None:
        raise RuntimeError("Critic Agent ended without submitting a review")
    if review.verdict == "revise":
        reviewed.active_agent = "research"
        return _workspace_command(
            restored,
            workspace=reviewed,
            goto="research_agent",
        )
    finalize_critic_review(reviewed)
    return _workspace_command(restored, workspace=reviewed, goto=END)


def _restore_running_workspace(
    state: Mapping[str, object],
    runtime: Runtime[ResearchGraphContext],
    *,
    active_agent: Literal["research", "critic"],
) -> tuple[GraphState, ResearchGraphContext, ResearchWorkspace]:
    context = runtime.context
    validated, workspace = restore_workspace_from_state(
        state,
        provider=context.provider,
        output_dir=context.output_dir,
    )
    if validated["status"] != "running" or validated["confirmed_goal"] is None:
        raise ValueError("only a confirmed running task may enter the ReAct loop")
    if validated["active_agent"] != active_agent:
        raise ValueError("the ReAct node does not match the active Agent")
    return validated, context, workspace


def _workspace_command(
    prior_state: GraphState,
    *,
    workspace: ResearchWorkspace,
    goto: str,
) -> Command:
    merged = cast(dict[str, object], copy_plain_json(prior_state))
    merged.update(workspace_to_state(workspace))
    merged.update(
        {
            "status": "running" if workspace.result is None else workspace.result.status,
            "failure_stage": None,
            "confirmation_error": None,
            "failure_summary": None,
        }
    )
    checkpoint = validate_graph_state(merged)
    return Command(update=checkpoint, goto=goto)


def _workspace_from_checkpoint(
    state: Mapping[str, object],
    context: ResearchGraphContext,
) -> tuple[GraphState, ResearchWorkspace]:
    return restore_workspace_from_state(
        state,
        provider=context.provider,
        output_dir=context.output_dir,
    )


def _agent_config(config: RunnableConfig, *, max_turns: int) -> RunnableConfig:
    """Keep the parent checkpoint namespace while sizing standard Agent internals."""

    nested = dict(config)
    nested["recursion_limit"] = max(int(config.get("recursion_limit", 0)), max_turns * 4 + 16)
    return cast(RunnableConfig, nested)


__all__ = [
    "RecoverableResearchError",
    "ResearchGraphContext",
    "build_research_graph",
    "parse_goal",
]
