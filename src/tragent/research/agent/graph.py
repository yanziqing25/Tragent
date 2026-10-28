"""Durable HITL workflow containing Research and Critic ReAct Agents."""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from tragent.research.agent.critic import MAX_CRITIC_AGENT_STEPS, critic_input_messages
from tragent.research.agent.events import AgentEvent, emit_agent_event
from tragent.research.agent.finalization import finalize_critic_review
from tragent.research.agent.goal import (
    GoalConfirmation,
    apply_goal_answer,
    confirmation_payload,
    materialize_goal,
    parse_goal,
)
from tragent.research.agent.harness import RecoverableResearchError, build_agent_harness
from tragent.research.agent.prompt import research_input_messages
from tragent.research.agent.state import GraphState, validate_graph_state
from tragent.research.agent.workspace import ResearchGraphContext, ResearchWorkspace


def build_research_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile the outer HITL workflow with two standard ``create_agent`` subgraphs."""

    builder = StateGraph(
        # Complete validated data shared by nodes and persisted by the checkpointer.
        state_schema=GraphState,
        # Non-persisted runtime services available through Runtime.context.
        context_schema=ResearchGraphContext,
        # Public shape returned when one graph invocation stops or completes.
        output_schema=GraphState,
    )
    builder.add_node("parse_goal", _parse_goal_node)
    builder.add_node(
        "confirm_goal",
        _confirm_goal_node,
        destinations=("confirm_goal", "research_agent", END),
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
    state: GraphState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, object]:
    context = runtime.context
    emit_agent_event(
        runtime.stream_writer,
        AgentEvent(
            source="goal_parser",
            kind="status",
            message="Goal Parser · Parsing the research request...",
        ),
    )
    return parse_goal(state, model=context.model).model_dump(mode="json")


def _route_after_parse(state: GraphState) -> Literal["confirm_goal", "__end__"]:
    return (
        "confirm_goal"
        if state.status in {"awaiting_clarification", "awaiting_confirmation"}
        else END
    )


def _confirm_goal_node(
    state: GraphState,
    runtime: Runtime[ResearchGraphContext],
) -> Command:
    if (
        state.status not in {"awaiting_clarification", "awaiting_confirmation"}
        or state.draft_goal is None
    ):
        raise ValueError("only an awaiting goal may enter confirmation")
    context = runtime.context
    emit_agent_event(
        runtime.stream_writer,
        AgentEvent(
            source="workflow",
            kind="status",
            message="Research Workflow · Applying goal confirmation...",
        ),
    )
    raw_response = interrupt(confirmation_payload(state))
    try:
        response = GoalConfirmation.model_validate(raw_response)
    except (ValidationError, TypeError, ValueError):
        invalid = state.model_dump(mode="json")
        invalid["confirmation_error"] = "Invalid confirmation response."
        return Command(
            update=validate_graph_state(invalid).model_dump(mode="json"),
            goto="confirm_goal",
        )

    if response.answer is not None:
        updated = apply_goal_answer(state, response.answer, model=context.model)
        return Command(update=updated.model_dump(mode="json"), goto="confirm_goal")

    if response.action == "approve":
        if state.status != "awaiting_confirmation":
            invalid = state.model_dump(mode="json")
            invalid["confirmation_error"] = "Invalid confirmation response."
            return Command(
                update=validate_graph_state(invalid).model_dump(mode="json"),
                goto="confirm_goal",
            )
        workspace = ResearchWorkspace(
            goal=materialize_goal(state, state.draft_goal), budget=state.budget
        )
        approved = state.model_copy(
            update={"status": "running", "confirmation_error": None, "workspace": workspace}
        )
        return Command(update=approved.model_dump(mode="json"), goto="research_agent")

    if response.action == "reject":
        rejected = state.model_dump(mode="json")
        rejected.update(
            {
                "status": "rejected",
                "failure_stage": None,
                "confirmation_error": None,
                "failure_summary": None,
                "missing_goal_fields": [],
            }
        )
        return Command(update=validate_graph_state(rejected).model_dump(mode="json"), goto=END)

    raise ValueError("goal confirmation response is invalid")


def _research_agent_node(
    state: GraphState,
    runtime: Runtime[ResearchGraphContext],
    config: RunnableConfig,
) -> Command:
    """Run one standard Research ReAct subgraph until draft submission or terminal."""

    workspace = _active_workspace(state, runtime, active_agent="research")
    researched = _run_agent(
        workspace, "research", runtime, config, max_turns=workspace.budget.max_agent_steps
    )
    update = state.with_workspace(researched).model_dump(mode="json")
    if researched.result is not None:
        return Command(update=update, goto=END)
    if researched.active_agent != "critic":
        raise RuntimeError("Research Agent ended without submitting a draft")
    return Command(update=update, goto="critic_agent")


def _critic_agent_node(
    state: GraphState,
    runtime: Runtime[ResearchGraphContext],
    config: RunnableConfig,
) -> Command:
    """Run one isolated Critic ReAct subgraph and route its submitted verdict."""

    workspace = _active_workspace(state, runtime, active_agent="critic")
    reviewed = _run_agent(workspace, "critic", runtime, config, max_turns=MAX_CRITIC_AGENT_STEPS)
    goto = END
    if reviewed.result is None:
        review = reviewed.critic_rounds[-1].review
        if review is None:
            raise RuntimeError("Critic Agent ended without submitting a review")
        if review.verdict == "revise":
            reviewed.active_agent = "research"
            goto = "research_agent"
        else:
            finalize_critic_review(reviewed)
    return Command(update=state.with_workspace(reviewed).model_dump(mode="json"), goto=goto)


def _active_workspace(
    state: GraphState,
    runtime: Runtime[ResearchGraphContext],
    *,
    active_agent: Literal["research", "critic"],
) -> ResearchWorkspace:
    workspace = state.workspace
    if state.status != "running" or workspace is None:
        raise ValueError("only a confirmed running task may enter the ReAct loop")
    if workspace.active_agent != active_agent:
        raise ValueError("the ReAct node does not match the active Agent")
    return workspace.bind(runtime.context)


def _run_agent(
    workspace: ResearchWorkspace,
    kind: Literal["research", "critic"],
    runtime: Runtime[ResearchGraphContext],
    config: RunnableConfig,
    *,
    max_turns: int,
) -> ResearchWorkspace:
    context = runtime.context
    agent = build_agent_harness(
        model=context.model,
        kind=kind,
        workspace=workspace,
        context_schema=ResearchGraphContext,
    )
    messages = (
        research_input_messages(workspace)
        if kind == "research"
        else critic_input_messages(workspace)
    )
    output = agent.invoke(
        {"messages": messages, "workspace": workspace.model_dump(mode="json")},
        config=_agent_config(config, max_turns=max_turns),
        context=context,
        durability="sync",
    )
    return ResearchWorkspace.model_validate(output["workspace"]).bind(context)


def _agent_config(config: RunnableConfig, *, max_turns: int) -> RunnableConfig:
    """Keep the parent checkpoint namespace while sizing standard Agent internals."""

    nested = dict(config)
    nested["recursion_limit"] = max(int(config.get("recursion_limit", 0)), max_turns * 4 + 16)
    return cast(RunnableConfig, nested)


__all__ = [
    "RecoverableResearchError",
    "build_research_graph",
    "parse_goal",
]
