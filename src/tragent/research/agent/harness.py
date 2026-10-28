"""Standard ``create_agent`` ReAct harness with narrow trading policies."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ModelRetryMiddleware,
    ToolCallRequest,
    dynamic_prompt,
    hook_config,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from tragent.research.agent.critic import (
    MAX_CRITIC_AGENT_STEPS,
    MAX_CRITIC_TOOL_CALLS,
    critic_system_prompt,
)
from tragent.research.agent.events import AgentEvent, emit_agent_event, workspace_status
from tragent.research.agent.finalization import finish_budget_exhausted
from tragent.research.agent.models import ToolObservation
from tragent.research.agent.prompt import research_system_prompt
from tragent.research.agent.tools import (
    build_critic_tools,
    build_research_revision_tools,
    build_research_tools,
)
from tragent.research.agent.workspace import ResearchWorkspace, consume_agent_step
from tragent.research.llm import TRANSIENT_MODEL_ERRORS

type AgentKind = Literal["research", "critic"]
logger = logging.getLogger(__name__)


@wrap_model_call
def require_single_tool_call(
    request: ModelRequest[Any],
    handler: Callable[[ModelRequest[Any]], ModelResponse],
) -> ModelResponse:
    """Request one required Tool call from the model provider."""

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


@wrap_tool_call
def run_first_tool_call_only(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Run one Tool per step even when a provider returns several calls in one response."""

    if request.tool_call["id"] == request.state["messages"][-1].tool_calls[0]["id"]:
        return handler(request)
    # Every call still gets its ToolMessage, and only one Tool updates the workspace.
    name = request.tool_call["name"]
    return ToolMessage(
        ToolObservation(
            tool_name=name,
            status="rejected",
            summary="Only the first Tool call of a step runs; call this Tool again next step.",
        ).to_compact_json(),
        tool_call_id=request.tool_call["id"],
        name=name,
        status="error",
    )


@dynamic_prompt
def research_prompt(request: ModelRequest[Any]) -> str:
    return research_system_prompt(ResearchWorkspace.model_validate(request.state["workspace"]))


@dynamic_prompt
def critic_prompt(request: ModelRequest[Any]) -> str:
    return critic_system_prompt(ResearchWorkspace.model_validate(request.state["workspace"]))


class TradingAgentState(AgentState):
    """The standard ReAct conversation plus the durable research workspace."""

    workspace: dict[str, object]


class RecoverableResearchError(RuntimeError):
    """Fixed safe error after model retries; the latest subgraph checkpoint can be resumed."""

    def __init__(self) -> None:
        super().__init__("The research model call failed safely.")


class TradingLoopMiddleware(AgentMiddleware[TradingAgentState, Any]):
    """Charge the research budgets and end the loop at a handoff, verdict, or exhaustion."""

    state_schema = TradingAgentState

    def __init__(self, *, kind: AgentKind) -> None:
        super().__init__()
        self.kind = kind

    @property
    def name(self) -> str:
        return f"{self.kind.title()}TradingLoop"

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: TradingAgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        workspace = ResearchWorkspace.model_validate(state["workspace"]).bind(runtime.context)
        review_submitted = (
            self.kind == "critic"
            and bool(workspace.critic_rounds)
            and workspace.critic_rounds[-1].review is not None
        )
        if workspace.result is not None or workspace.active_agent != self.kind or review_submitted:
            return {"jump_to": "end"}

        if self.kind == "research":
            if workspace.tool_calls >= workspace.budget.max_tool_calls:
                return _exhausted(workspace, "The Tool call budget was exhausted.")
            if not consume_agent_step(workspace):
                return _exhausted(workspace, "The Agent step budget was exhausted.")
        else:
            review_round = workspace.critic_rounds[-1]
            if (
                review_round.agent_steps >= MAX_CRITIC_AGENT_STEPS
                or review_round.tool_calls >= MAX_CRITIC_TOOL_CALLS
            ):
                return _exhausted(workspace, "The Critic review budget was exhausted.")
            review_round.agent_steps += 1

        emit_agent_event(
            runtime.stream_writer,
            AgentEvent(
                source=("research_agent" if self.kind == "research" else "critic_agent"),
                kind="decision",
                message=(
                    "Research Agent · Deciding the next Action..."
                    if self.kind == "research"
                    else "Critic Agent · Reviewing the research draft..."
                ),
                agent_status=workspace_status(workspace),
            ),
        )
        return {"workspace": workspace.model_dump(mode="json")}

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse],
    ) -> ModelResponse:
        try:
            return handler(request)
        except TRANSIENT_MODEL_ERRORS as error:
            logger.warning(
                "Agent model request interrupted: agent=%s error_type=%s status_code=%s",
                self.kind,
                type(error).__name__,
                getattr(error, "status_code", None),
            )
        # Raised outside the handler so provider details never chain into the checkpoint.
        raise RecoverableResearchError()


def _exhausted(workspace: ResearchWorkspace, reason: str) -> dict[str, Any]:
    finish_budget_exhausted(workspace, reason)
    return {"workspace": workspace.model_dump(mode="json"), "jump_to": "end"}


def build_agent_harness(
    *,
    model: BaseChatModel,
    kind: AgentKind,
    workspace: ResearchWorkspace,
    context_schema: type[Any],
) -> Any:
    """Build one per-invocation standard ReAct subgraph for the active Agent."""

    if kind == "research":
        revising = bool(workspace.critic_rounds)
        tools = build_research_revision_tools() if revising else build_research_tools()
    else:
        tools = build_critic_tools(workspace)
    return create_agent(
        model=model,
        tools=tools,
        middleware=[
            TradingLoopMiddleware(kind=kind),
            # The one retry of a decision, which also covers a stream that drops midway, before
            # the run stops resumable.
            ModelRetryMiddleware(
                max_retries=2, retry_on=TRANSIENT_MODEL_ERRORS, on_failure="error"
            ),
            require_single_tool_call,
            run_first_tool_call_only,
            research_prompt if kind == "research" else critic_prompt,
        ],
        context_schema=context_schema,
        name=f"{kind}_react_agent",
    )


__all__ = [
    "RecoverableResearchError",
    "TradingAgentState",
    "build_agent_harness",
    "critic_prompt",
    "require_single_tool_call",
    "research_prompt",
    "run_first_tool_call_only",
]
