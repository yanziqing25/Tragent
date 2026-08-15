"""Six thin LangChain Tool wrappers for the deterministic research runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from trading.agent.models import CriticVerdict, Fingerprint, ResearchOutcome, ToolObservation
from trading.quant import SegmentName, StrategyName

if TYPE_CHECKING:
    from trading.agent.workspace import CriticRoundRuntime, ResearchWorkspace

type StrategyParameter = StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)]
type StrategyParameters = Annotated[
    dict[
        Annotated[StrictStr, StringConstraints(min_length=1, max_length=64)],
        StrategyParameter,
    ],
    Field(min_length=1, max_length=3),
]
type ReviewSegmentName = Literal["in_sample", "out_of_sample"]

_Conclusion = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000),
]
_Limitation = Annotated[StrictStr, StringConstraints(max_length=500)]


class _ToolArgs(BaseModel):
    """Strict JSON-facing base schema shared by all Research Tools."""

    model_config = ConfigDict(extra="forbid")


class NoArgs(_ToolArgs):
    """Input for Tools whose scope comes entirely from the confirmed goal."""


class RunBacktestArgs(_ToolArgs):
    """Phase-sensitive input for training or irreversible final evaluation."""

    stage: Literal["training", "final"]
    strategy_name: StrategyName | None = None
    parameters: StrategyParameters | None = None
    selected_fingerprint: Fingerprint | None = None

    @model_validator(mode="after")
    def validate_stage_fields(self) -> RunBacktestArgs:
        if self.stage == "training":
            if self.strategy_name is None or self.parameters is None:
                raise ValueError("training requires strategy_name and parameters")
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
    conclusion: _Conclusion
    limitations: tuple[_Limitation, ...] = Field(default=(), max_length=10)

    @field_validator("limitations")
    @classmethod
    def limitations_must_not_be_blank(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("limitations must not contain blank entries")
        return values


class CriticEvidenceArgs(_ToolArgs):
    """Select a user-facing evidence segment; the fingerprint is injected."""

    segment: ReviewSegmentName


class SubmitCriticReviewArgs(_ToolArgs):
    """Submit one bounded Critic verdict."""

    verdict: CriticVerdict
    summary: _Conclusion
    issues: tuple[_Limitation, ...] = Field(default=(), max_length=8)
    required_revision: (
        Annotated[
            StrictStr,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000),
            Field(description="Required for revise; omit for accept or reject."),
        ]
        | None
    ) = None

    @field_validator("issues")
    @classmethod
    def issues_must_not_be_blank(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("issues must not contain blank entries")
        return values

    @model_validator(mode="after")
    def validate_revision_fields(self) -> SubmitCriticReviewArgs:
        if self.verdict == "revise" and self.required_revision is None:
            raise ValueError("revise requires required_revision")
        if self.verdict != "revise" and self.required_revision is not None:
            raise ValueError("only revise accepts required_revision")
        return self


def build_research_tools(
    get_workspace: Callable[[], ResearchWorkspace],
) -> tuple[StructuredTool, ...]:
    """Bind the fixed six Tool schemas to the current durable workspace."""

    # Imported lazily so schema-only tests and imports do not create runtime state.
    from trading.agent import runtime

    return (
        _make_tool(
            name="get_market_data",
            description="Load the confirmed asset and SPY daily data into the workspace.",
            args_schema=NoArgs,
            operation=lambda: runtime.get_market_data(get_workspace()),
        ),
        _make_tool(
            name="calculate_asset_statistics",
            description="Calculate deterministic statistics for the confirmed asset.",
            args_schema=NoArgs,
            operation=lambda: runtime.calculate_asset_statistics(get_workspace()),
        ),
        _make_tool(
            name="run_backtest",
            description=(
                "Run a training evaluation or irreversibly lock and run final evaluation. "
                "Training parameters: moving_average requires short_window < long_window; "
                "rsi requires rsi_window and 0 < entry_threshold < exit_threshold < 100; "
                "macd requires fast_period < slow_period and signal_period."
            ),
            args_schema=RunBacktestArgs,
            operation=lambda stage, strategy_name=None, parameters=None, selected_fingerprint=None: (
                runtime.run_backtest(
                    get_workspace(),
                    stage=stage,
                    strategy_name=strategy_name,
                    parameters=parameters,
                    selected_fingerprint=selected_fingerprint,
                )
            ),
        ),
        _make_tool(
            name="analyze_drawdown",
            description="Read deterministic drawdown details for an allowed evaluated segment.",
            args_schema=EvaluationArgs,
            operation=lambda fingerprint, segment: runtime.analyze_drawdown(
                get_workspace(),
                fingerprint=fingerprint,
                segment=segment,
            ),
        ),
        _make_tool(
            name="compare_with_benchmark",
            description="Compare a strategy with asset and SPY buy-and-hold for one segment.",
            args_schema=EvaluationArgs,
            operation=lambda fingerprint, segment: runtime.compare_with_benchmark(
                get_workspace(),
                fingerprint=fingerprint,
                segment=segment,
            ),
        ),
        _make_tool(
            name="submit_research_draft",
            description="Submit the frozen selected or no-strategy draft for Critic review.",
            args_schema=SubmitResearchDraftArgs,
            operation=lambda outcome, selected_fingerprint, conclusion, limitations=(): (
                runtime.submit_research_draft(
                    get_workspace(),
                    outcome=outcome,
                    selected_fingerprint=selected_fingerprint,
                    conclusion=conclusion,
                    limitations=limitations,
                )
            ),
        ),
    )


def build_critic_tools(
    get_workspace: Callable[[], ResearchWorkspace],
) -> tuple[StructuredTool, ...]:
    """Bind three permission-restricted Tools to the active frozen review."""

    from trading.agent import runtime
    from trading.agent.critic import submit_critic_review

    comparison_description = (
        "Compare the frozen candidate with the current symbol's Session Buy and Hold."
        if get_workspace().goal.bar_interval == "1m"
        else "Compare the frozen candidate with asset and SPY buy-and-hold."
    )
    return (
        _make_tool(
            name="inspect_selected_drawdown",
            description="Inspect the frozen candidate's deterministic drawdown interval.",
            args_schema=CriticEvidenceArgs,
            operation=lambda segment: _locked_evidence(
                get_workspace(),
                "inspect_selected_drawdown",
                runtime.analyze_drawdown,
                segment,
            ),
        ),
        _make_tool(
            name="compare_selected_with_benchmark",
            description=comparison_description,
            args_schema=CriticEvidenceArgs,
            operation=lambda segment: _locked_evidence(
                get_workspace(),
                "compare_selected_with_benchmark",
                runtime.compare_with_benchmark,
                segment,
            ),
        ),
        _make_tool(
            name="submit_critic_review",
            description="Submit the independent accept, revise, or reject verdict.",
            args_schema=SubmitCriticReviewArgs,
            operation=lambda verdict, summary, issues=(), required_revision=None: (
                submit_critic_review(
                    get_workspace(),
                    _active_review_round(get_workspace()),
                    verdict=verdict,
                    summary=summary,
                    issues=issues,
                    required_revision=required_revision,
                )
            ),
        ),
    )


def build_research_revision_tools(
    get_workspace: Callable[[], ResearchWorkspace],
) -> tuple[StructuredTool, ...]:
    """Expose only frozen evidence and resubmission during one Critic revision."""

    from trading.agent import runtime

    return (
        _make_tool(
            name="analyze_drawdown",
            description="Inspect the frozen candidate's deterministic drawdown interval.",
            args_schema=CriticEvidenceArgs,
            operation=lambda segment: _locked_evidence(
                get_workspace(),
                "analyze_drawdown",
                runtime.analyze_drawdown,
                segment,
            ),
        ),
        _make_tool(
            name="compare_with_benchmark",
            description="Compare the frozen candidate with asset and SPY buy-and-hold.",
            args_schema=CriticEvidenceArgs,
            operation=lambda segment: _locked_evidence(
                get_workspace(),
                "compare_with_benchmark",
                runtime.compare_with_benchmark,
                segment,
            ),
        ),
        _make_tool(
            name="submit_research_draft",
            description="Submit the revised frozen conclusion for round-two Critic review.",
            args_schema=SubmitResearchDraftArgs,
            operation=lambda outcome, selected_fingerprint, conclusion, limitations=(): (
                runtime.submit_research_draft(
                    get_workspace(),
                    outcome=outcome,
                    selected_fingerprint=selected_fingerprint,
                    conclusion=conclusion,
                    limitations=limitations,
                )
            ),
        ),
    )


def _active_review_round(workspace: ResearchWorkspace) -> CriticRoundRuntime:
    if not workspace.critic_rounds:
        raise ValueError("Critic Tools require an active review round")
    return workspace.critic_rounds[-1]


def _locked_evidence(
    workspace: ResearchWorkspace,
    tool_name: str,
    operation: Callable[..., ToolObservation],
    segment: ReviewSegmentName,
) -> ToolObservation:
    fingerprint = workspace.locked_fingerprint
    if fingerprint is None:
        return ToolObservation(
            tool_name=tool_name,
            status="rejected",
            summary="Review evidence requires the frozen fingerprint.",
        )
    internal_segment = "training" if segment == "in_sample" else "test"
    observation = operation(
        workspace,
        fingerprint=fingerprint,
        segment=internal_segment,
    )
    return observation.model_copy(update={"tool_name": tool_name})


def _make_tool(
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    operation: Callable[..., ToolObservation],
) -> StructuredTool:
    """Create one explicit Tool without adding a registry or runtime abstraction."""

    def invalid_arguments(_: object) -> str:
        return ToolObservation(
            tool_name=name,
            status="rejected",
            summary="Arguments did not match the declared Tool schema.",
        ).to_compact_json()

    def invoke_operation(**arguments: Any) -> str:
        return operation(**arguments).to_compact_json()

    return StructuredTool.from_function(
        func=invoke_operation,
        name=name,
        description=description,
        args_schema=args_schema,
        infer_schema=False,
        handle_validation_error=invalid_arguments,
    )


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
