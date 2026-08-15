"""Standard ``create_agent`` ReAct harness with narrow trading policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    hook_config,
    wrap_model_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from trading.agent.critic import (
    MAX_CRITIC_AGENT_STEPS,
    MAX_CRITIC_TOOL_CALLS,
    build_critic_messages,
    record_critic_event,
)
from trading.agent.events import (
    AgentEvent,
    AgentStatusSnapshot,
    build_agent_status_snapshot,
    emit_agent_event,
)
from trading.agent.finalization import finish_budget_exhausted, finish_failed
from trading.agent.models import ToolObservation
from trading.agent.prompt import build_research_messages
from trading.agent.state import (
    GraphState,
    copy_plain_json,
    restore_workspace_from_state,
    validate_graph_state,
    workspace_to_state,
)
from trading.agent.tools import (
    build_critic_tools,
    build_research_revision_tools,
    build_research_tools,
)
from trading.agent.workspace import (
    ResearchWorkspace,
    consume_agent_step,
    consume_tool_call,
    record_event,
)

type AgentKind = Literal["research", "critic"]


@wrap_model_call
def require_single_tool_call(
    request: ModelRequest[Any],
    handler: Callable[[ModelRequest[Any]], ModelResponse],
) -> ModelResponse:
    """Use the model provider's Tool contract for this Action-only Agent."""

    return handler(
        request.override(
            tool_choice="required",
            model_settings={
                **request.model_settings,
                "parallel_tool_calls": False,
                "strict": False,
            },
        )
    )


class TradingAgentState(AgentState):
    """Private subgraph messages plus the durable application workspace."""

    workspace_state: GraphState


@dataclass(slots=True)
class WorkspaceSession:
    """Run-local pointer refreshed from durable state before every hook and Tool."""

    current: ResearchWorkspace

    def get(self) -> ResearchWorkspace:
        return self.current


class RecoverableResearchError(RuntimeError):
    """Fixed safe error whose latest subgraph checkpoint can be resumed."""

    def __init__(self, *, stage: Literal["model", "tool"]) -> None:
        reasons = {
            "model": "The research model call failed safely.",
            "tool": "The research Tool failed temporarily.",
        }
        self.stage = stage
        super().__init__(reasons[stage])


class TradingLoopMiddleware(AgentMiddleware[TradingAgentState, Any]):
    """Persist the trading workspace and its Agent budgets around standard ReAct hooks."""

    state_schema = TradingAgentState

    def __init__(
        self,
        *,
        kind: AgentKind,
        session: WorkspaceSession,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.session = session

    @property
    def name(self) -> str:
        return f"{self.kind.title()}TradingLoop"

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: TradingAgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        workspace = self._restore(state, runtime)
        review_submitted = (
            self.kind == "critic"
            and bool(workspace.critic_rounds)
            and workspace.critic_rounds[-1].review is not None
        )
        if workspace.result is not None or workspace.active_agent != self.kind or review_submitted:
            return {"workspace_state": self._checkpoint(state), "jump_to": "end"}

        if self.kind == "research":
            if workspace.tool_calls >= workspace.budget.max_tool_calls:
                finish_budget_exhausted(workspace, "The Tool call budget was exhausted.")
                return {"workspace_state": self._checkpoint(state), "jump_to": "end"}
            if not consume_agent_step(workspace):
                finish_budget_exhausted(workspace, "The Agent step budget was exhausted.")
                return {"workspace_state": self._checkpoint(state), "jump_to": "end"}
        else:
            review_round = _active_review_round(workspace)
            if (
                review_round.agent_steps >= MAX_CRITIC_AGENT_STEPS
                or review_round.tool_calls >= MAX_CRITIC_TOOL_CALLS
            ):
                finish_budget_exhausted(workspace, "The Critic review budget was exhausted.")
                return {"workspace_state": self._checkpoint(state), "jump_to": "end"}
            review_round.agent_steps += 1

        emit_agent_event(
            AgentEvent(
                source=("research_agent" if self.kind == "research" else "critic_agent"),
                kind="decision",
                message=(
                    "Research Agent · Deciding the next Action..."
                    if self.kind == "research"
                    else "Critic Agent · Reviewing the research draft..."
                ),
                agent_status=_agent_status_snapshot(workspace),
            )
        )
        return {"workspace_state": self._checkpoint(state)}

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse],
    ) -> ModelResponse:
        workspace = self._restore(cast(TradingAgentState, request.state), request.runtime)
        messages = (
            build_research_messages(workspace)
            if self.kind == "research"
            else build_critic_messages(workspace, _active_review_round(workspace))
        )
        failed = False
        try:
            response = handler(
                request.override(
                    messages=messages,
                    system_message=None,
                )
            )
        except Exception:
            failed = True
        if failed:
            raise RecoverableResearchError(stage="model")
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        state = cast(TradingAgentState, request.state)
        workspace = self._restore(state, request.runtime)
        tool_name = cast(str, request.tool_call["name"])
        call_arguments = cast(dict[str, object], request.tool_call["args"])
        if self.kind == "research":
            if not consume_tool_call(workspace):
                finish_budget_exhausted(workspace, "The Tool call budget was exhausted.")
                return Command(
                    update={"workspace_state": self._checkpoint(state)},
                    goto=END,
                )
        else:
            review_round = _active_review_round(workspace)
            if review_round.tool_calls >= MAX_CRITIC_TOOL_CALLS:
                finish_budget_exhausted(workspace, "The Critic review budget was exhausted.")
                return Command(
                    update={"workspace_state": self._checkpoint(state)},
                    goto=END,
                )
            review_round.tool_calls += 1

        emit_agent_event(
            AgentEvent(
                source=("research_agent" if self.kind == "research" else "critic_agent"),
                kind="tool",
                message=f"{self.kind.title()} Agent · Running {tool_name}...",
                tool_name=tool_name,
                agent_status=_agent_status_snapshot(workspace),
            )
        )
        recoverable_error: RecoverableResearchError | None = None
        try:
            response = handler(request)
        except RecoverableResearchError as error:
            recoverable_error = RecoverableResearchError(stage=error.stage)
        if recoverable_error is not None:
            raise recoverable_error
        response = cast(ToolMessage, response)
        observation = ToolObservation.model_validate_json(cast(str, response.content))

        normalized_arguments = {} if response.status == "error" else call_arguments

        if self.kind == "research":
            record_event(
                workspace,
                tool_name=tool_name,
                normalized_arguments=normalized_arguments,
                observation=observation,
                tool_call_id=request.tool_call.get("id"),
                replayable=True,
            )
        else:
            review_round = _active_review_round(workspace)
            record_critic_event(
                review_round,
                tool_name=tool_name,
                arguments=normalized_arguments,
                observation=observation,
                tool_call_id=request.tool_call.get("id"),
                replayable=True,
            )

        emit_agent_event(
            AgentEvent(
                source=("research_agent" if self.kind == "research" else "critic_agent"),
                kind="observation",
                message=_observation_message(self.kind, tool_name),
                tool_name=tool_name,
                agent_status=_agent_status_snapshot(workspace),
            )
        )

        if workspace.fatal_reason is not None and workspace.result is None:
            finish_failed(workspace, workspace.fatal_reason)
        update = {
            "workspace_state": self._checkpoint(state),
            "messages": [response],
        }
        return Command(update=update)

    def _restore(self, state: TradingAgentState, runtime: Runtime[Any]) -> ResearchWorkspace:
        context = runtime.context
        _validated, workspace = restore_workspace_from_state(
            state["workspace_state"],
            provider=context.provider,
            output_dir=context.output_dir,
        )
        self.session.current = workspace
        return workspace

    def _checkpoint(self, state: TradingAgentState) -> GraphState:
        prior = validate_graph_state(state["workspace_state"])
        workspace = self.session.current
        merged = cast(dict[str, object], copy_plain_json(prior))
        merged.update(workspace_to_state(workspace))
        merged.update(
            {
                "status": "running" if workspace.result is None else workspace.result.status,
                "failure_stage": None,
                "confirmation_error": None,
                "failure_summary": None,
            }
        )
        return validate_graph_state(merged)


def build_agent_harness(
    *,
    model: BaseChatModel,
    kind: AgentKind,
    workspace: ResearchWorkspace,
    context_schema: type[Any],
) -> Any:
    """Build one per-invocation standard ReAct subgraph and its run-local session."""

    session = WorkspaceSession(workspace)
    if kind == "research":
        builder = build_research_revision_tools if workspace.critic_rounds else build_research_tools
        tools = builder(session.get)
    else:
        tools = build_critic_tools(session.get)
    middleware = TradingLoopMiddleware(kind=kind, session=session)
    return create_agent(
        model=model,
        tools=tools,
        middleware=[require_single_tool_call, middleware],
        context_schema=context_schema,
        name=f"{kind}_react_agent",
    )


def _active_review_round(workspace: ResearchWorkspace) -> Any:
    if workspace.active_agent != "critic" or not workspace.critic_rounds:
        raise ValueError("Critic Agent requires an active review round")
    return workspace.critic_rounds[-1]


def _agent_status_snapshot(workspace: ResearchWorkspace) -> AgentStatusSnapshot:
    latest_round = workspace.critic_rounds[-1] if workspace.critic_rounds else None
    return build_agent_status_snapshot(
        active_agent=workspace.active_agent,
        budget=workspace.budget,
        agent_steps=workspace.agent_steps,
        tool_calls=workspace.tool_calls,
        training_backtests=workspace.training_backtests,
        critic_round=len(workspace.critic_rounds),
        critic_verdict=(
            None
            if latest_round is None or latest_round.review is None
            else latest_round.review.verdict
        ),
        critic_agent_steps=0 if latest_round is None else latest_round.agent_steps,
        critic_tool_calls=0 if latest_round is None else latest_round.tool_calls,
    )


def _observation_message(kind: AgentKind, tool_name: str) -> str:
    if tool_name == "submit_research_draft":
        return "Research Agent · Submitted the frozen research draft."
    if tool_name == "submit_critic_review":
        return "Critic Agent · Submitted the review verdict."
    return f"{kind.title()} Agent · Observed {tool_name}."


__all__ = [
    "RecoverableResearchError",
    "TradingAgentState",
    "build_agent_harness",
    "require_single_tool_call",
]
