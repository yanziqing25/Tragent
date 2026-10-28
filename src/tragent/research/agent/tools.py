"""LangChain Tools that run deterministic research operations against the durable workspace."""

from collections.abc import Callable
from typing import Annotated, Any, Literal, Self

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    model_validator,
)

from tragent.quant import SegmentName, StrategyName
from tragent.quant.models import INTRADAY_RSI_STRATEGIES
from tragent.research.agent import runtime
from tragent.research.agent.critic import submit_critic_review
from tragent.research.agent.events import AgentEvent, emit_agent_event, workspace_status
from tragent.research.agent.finalization import finish_failed
from tragent.research.agent.models import (
    CriticVerdict,
    Fingerprint,
    LongText,
    Note,
    ResearchOutcome,
    RevisionText,
    ToolObservation,
)
from tragent.research.agent.retrieval import ContextSource
from tragent.research.agent.workspace import ResearchWorkspace, record_tool_call

type AgentKind = Literal["research", "critic"]
type Operation = Callable[..., ToolObservation]

type StrategyParameter = StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)]
type StrategyParameters = Annotated[
    dict[
        Annotated[str, StringConstraints(min_length=1, max_length=64)],
        StrategyParameter,
    ],
    Field(max_length=3),
]
type ReviewSegmentName = Literal["in_sample", "out_of_sample"]
type ContextQuery = Annotated[str, StringConstraints(min_length=2, max_length=300)]


class _ToolArgs(BaseModel):
    """Strict JSON-facing base schema shared by all Research Tools."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Injected by ToolNode; LangChain hides it from the schema the model sees.
    tool_runtime: ToolRuntime = Field(default=None, exclude=True)


class NoArgs(_ToolArgs):
    """Input for Tools whose scope comes entirely from the confirmed goal."""


class RunBacktestArgs(_ToolArgs):
    """Phase-sensitive input for training or irreversible final evaluation."""

    stage: Literal["training", "final"]
    strategy_name: StrategyName | None = None
    parameters: StrategyParameters | None = None
    selected_fingerprint: Fingerprint | None = None

    @model_validator(mode="after")
    def validate_stage_fields(self) -> Self:
        if self.stage == "training":
            if self.strategy_name is None or self.parameters is None:
                raise ValueError("training requires strategy_name and parameters")
            if self.strategy_name in INTRADAY_RSI_STRATEGIES:
                if self.parameters:
                    raise ValueError("fixed intraday policies require parameters={}")
            elif not self.parameters:
                raise ValueError("tunable strategies require non-empty parameters")
            if self.selected_fingerprint is not None:
                raise ValueError("training must not include selected_fingerprint")
        else:
            if self.strategy_name is not None or self.parameters is not None:
                raise ValueError("final must not include strategy_name or parameters")
            if self.selected_fingerprint is None:
                raise ValueError("final requires selected_fingerprint")
        return self


class EvaluationArgs(_ToolArgs):
    """Reference one previously evaluated candidate and an allowed segment."""

    fingerprint: Fingerprint
    segment: SegmentName


class SubmitResearchDraftArgs(_ToolArgs):
    """Submit one frozen Research conclusion for independent review."""

    outcome: ResearchOutcome
    selected_fingerprint: Fingerprint
    conclusion: LongText
    limitations: tuple[Note, ...] = Field(default=(), max_length=10)
    market_context_summary: LongText | None = None


class CriticEvidenceArgs(_ToolArgs):
    """Select a user-facing evidence segment; the fingerprint is injected."""

    segment: ReviewSegmentName


class SearchMarketContextArgs(_ToolArgs):
    """One query per external source after candidate lock; a revision names what was off-target."""

    queries: dict[ContextSource, ContextQuery] = Field(min_length=1)
    revision_reason: Note | None = None


class SubmitCriticReviewArgs(_ToolArgs):
    """Submit one bounded Critic verdict."""

    verdict: CriticVerdict
    summary: LongText
    issues: tuple[Note, ...] = Field(default=(), max_length=8)
    required_revision: (
        Annotated[
            RevisionText, Field(description="Required for revise; omit for accept or reject.")
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_revision_fields(self) -> Self:
        if self.verdict == "revise" and self.required_revision is None:
            raise ValueError("revise requires required_revision")
        if self.verdict != "revise" and self.required_revision is not None:
            raise ValueError("only revise accepts required_revision")
        return self


def build_research_tools() -> tuple[StructuredTool, ...]:
    """Bind the Research Agent's deterministic and context Tools."""

    return (
        _make_tool(
            "research",
            name="get_market_data",
            description="Load the confirmed asset and benchmark data into the workspace.",
            args_schema=NoArgs,
            operation=runtime.get_market_data,
        ),
        _make_tool(
            "research",
            name="calculate_asset_statistics",
            description="Calculate deterministic sample-in statistics for the confirmed asset.",
            args_schema=NoArgs,
            operation=runtime.calculate_asset_statistics,
        ),
        _make_tool(
            "research",
            name="run_backtest",
            description=(
                "For training, supply stage, strategy_name, and parameters. "
                "For final, supply only stage and a selected_fingerprint from tested_candidates; "
                "this irreversibly locks the candidate and evaluates sample-out data. "
                "Use the same fingerprint when submitting, including for no_strategy. "
                "Training parameters: moving_average requires short_window < long_window; "
                "rsi requires rsi_window and 0 < entry_threshold < exit_threshold < 100; "
                "macd requires fast_period < slow_period and signal_period. "
                "1m-only intraday_rsi, rsi_price_exit and rsi_vwap_exit are fixed policies "
                "with parameters={}."
            ),
            args_schema=RunBacktestArgs,
            operation=runtime.run_backtest,
        ),
        _make_tool(
            "research",
            name="analyze_drawdown",
            description="Read deterministic drawdown details for an allowed evaluated segment.",
            args_schema=EvaluationArgs,
            operation=runtime.analyze_drawdown,
        ),
        _make_tool(
            "research",
            name="compare_with_benchmark",
            description=(
                "Compare a strategy with asset and configured benchmark buy-and-hold "
                "for one segment."
            ),
            args_schema=EvaluationArgs,
            operation=runtime.compare_with_benchmark,
        ),
        _make_tool(
            "research",
            name="search_market_context",
            description=(
                "After final evaluation, search external context with one query per source: sec "
                "by business and risk topics, yahoo_finance by company events, stocktwits by the "
                'company name or cashtag, and reddit by the quoted company name, such as "Unity '
                'Software", since unquoted words match posts about anything; strategy terms such '
                "as RSI match none of them. The first search queries every source. If a source's "
                "results are not about the company, search only that source once more with a "
                "revised query and a revision_reason; report sources still without relevant "
                "evidence as gaps."
            ),
            args_schema=SearchMarketContextArgs,
            operation=runtime.search_market_context,
        ),
        _make_tool(
            "research",
            name="submit_research_draft",
            description="Submit the frozen selected or no-strategy draft for Critic review.",
            args_schema=SubmitResearchDraftArgs,
            operation=runtime.submit_research_draft,
        ),
    )


def build_critic_tools(workspace: ResearchWorkspace) -> tuple[StructuredTool, ...]:
    """Bind read-only quantitative and retrieved-context Tools to the review."""

    comparison_description = (
        "Compare the frozen candidate with the current symbol's Session Buy and Hold."
        if workspace.goal.bar_interval == "1m"
        else "Compare the frozen candidate with asset and configured benchmark buy-and-hold."
    )
    return (
        _make_tool(
            "critic",
            name="inspect_selected_drawdown",
            description="Inspect the frozen candidate's deterministic drawdown interval.",
            args_schema=CriticEvidenceArgs,
            operation=_locked_evidence("inspect_selected_drawdown", runtime.analyze_drawdown),
        ),
        _make_tool(
            "critic",
            name="compare_selected_with_benchmark",
            description=comparison_description,
            args_schema=CriticEvidenceArgs,
            operation=_locked_evidence(
                "compare_selected_with_benchmark", runtime.compare_with_benchmark
            ),
        ),
        _make_tool(
            "critic",
            name="inspect_retrieved_context",
            description="Inspect only the frozen external evidence cited by the Research draft.",
            args_schema=NoArgs,
            operation=runtime.inspect_retrieved_context,
        ),
        _make_tool(
            "critic",
            name="submit_critic_review",
            description="Submit the independent accept, revise, or reject verdict.",
            args_schema=SubmitCriticReviewArgs,
            operation=lambda workspace, **arguments: submit_critic_review(
                workspace, workspace.critic_rounds[-1], **arguments
            ),
        ),
    )


def build_research_revision_tools() -> tuple[StructuredTool, ...]:
    """Expose only frozen evidence and resubmission during one Critic revision."""

    return (
        _make_tool(
            "research",
            name="analyze_drawdown",
            description="Inspect the frozen candidate's deterministic drawdown interval.",
            args_schema=CriticEvidenceArgs,
            operation=_locked_evidence("analyze_drawdown", runtime.analyze_drawdown),
        ),
        _make_tool(
            "research",
            name="compare_with_benchmark",
            description=(
                "Compare the frozen candidate with asset and configured benchmark buy-and-hold."
            ),
            args_schema=CriticEvidenceArgs,
            operation=_locked_evidence("compare_with_benchmark", runtime.compare_with_benchmark),
        ),
        _make_tool(
            "research",
            name="inspect_retrieved_context",
            description="Inspect only the frozen external evidence cited by the Research draft.",
            args_schema=NoArgs,
            operation=runtime.inspect_retrieved_context,
        ),
        _make_tool(
            "research",
            name="submit_research_draft",
            description="Submit the revised frozen conclusion for round-two Critic review.",
            args_schema=SubmitResearchDraftArgs,
            operation=runtime.submit_research_draft,
        ),
    )


def _locked_evidence(tool_name: str, operation: Operation) -> Operation:
    """Answer a review question about the frozen candidate only."""

    def inspect(workspace: ResearchWorkspace, *, segment: ReviewSegmentName) -> ToolObservation:
        fingerprint = workspace.locked_fingerprint
        if fingerprint is None:
            return ToolObservation(
                tool_name=tool_name,
                status="rejected",
                summary="Review evidence requires the frozen fingerprint.",
            )
        observation = operation(
            workspace,
            fingerprint=fingerprint,
            segment="training" if segment == "in_sample" else "test",
        )
        return observation.model_copy(update={"tool_name": tool_name})

    return inspect


def _make_tool(
    kind: AgentKind,
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    operation: Operation,
) -> StructuredTool:
    """Create one Tool that counts, journals, and checkpoints its own Action."""

    def invalid_arguments(_: object) -> str:
        return ToolObservation(
            tool_name=name,
            status="rejected",
            summary="Arguments did not match the declared Tool schema.",
        ).to_compact_json()

    def invoke_operation(tool_runtime: ToolRuntime, **arguments: Any) -> Command:
        workspace = ResearchWorkspace.model_validate(tool_runtime.state["workspace"]).bind(
            tool_runtime.context
        )
        if kind == "research":
            workspace.tool_calls += 1
            journal, step = workspace.journal, workspace.agent_steps
        else:
            review_round = workspace.critic_rounds[-1]
            review_round.tool_calls += 1
            journal, step = review_round.journal, review_round.agent_steps
        source = "research_agent" if kind == "research" else "critic_agent"
        emit_agent_event(
            tool_runtime.stream_writer,
            AgentEvent(
                source=source,
                kind="tool",
                message=f"{kind.title()} Agent · Running {name}...",
                tool_name=name,
                agent_status=workspace_status(workspace),
            ),
        )
        observation = operation(workspace, **arguments)
        record_tool_call(
            journal,
            step=step,
            tool_name=name,
            arguments=arguments,
            observation=observation,
        )
        emit_agent_event(
            tool_runtime.stream_writer,
            AgentEvent(
                source=source,
                kind="observation",
                message=_observation_message(kind, name),
                tool_name=name,
                agent_status=workspace_status(workspace),
            ),
        )
        if workspace.fatal_reason is not None and workspace.result is None:
            finish_failed(workspace, workspace.fatal_reason)
        return Command(
            update={
                "workspace": workspace.model_dump(mode="json"),
                "messages": [
                    ToolMessage(
                        observation.to_compact_json(), tool_call_id=tool_runtime.tool_call_id
                    )
                ],
            }
        )

    return StructuredTool.from_function(
        func=invoke_operation,
        name=name,
        description=description,
        args_schema=args_schema,
        infer_schema=False,
        handle_validation_error=invalid_arguments,
    )


def _observation_message(kind: AgentKind, tool_name: str) -> str:
    if tool_name == "submit_research_draft":
        return "Research Agent · Submitted the frozen research draft."
    if tool_name == "submit_critic_review":
        return "Critic Agent · Submitted the review verdict."
    return f"{kind.title()} Agent · Observed {tool_name}."


__all__ = [
    "CriticEvidenceArgs",
    "EvaluationArgs",
    "NoArgs",
    "RunBacktestArgs",
    "SubmitCriticReviewArgs",
    "SubmitResearchDraftArgs",
    "build_critic_tools",
    "build_research_revision_tools",
    "build_research_tools",
]
