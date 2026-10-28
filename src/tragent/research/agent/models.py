"""Compact JSON-facing contracts for the Milestone 2 Research Agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from tragent.quant import (
    BacktestCosts,
    FinalEvaluation,
    PerformanceMetrics,
    StrategyName,
    TrainingEvaluation,
)
from tragent.quant.models import (
    INTRADAY_RSI_BASELINE,
    INTRADAY_RSI_STRATEGIES,
    INTRADAY_RSI_VARIANTS,
)

type BarInterval = Literal["1d", "1m"]
type RequestIssue = Literal[
    "ambiguous_fields",
    "unsupported_request",
    "unrelated_request",
]
type ObservationStatus = Literal["success", "rejected", "error"]
type ContextSourceType = Literal["official", "news", "community"]
type ResearchStatus = Literal[
    "selected",
    "no_strategy",
    "critic_rejected",
    "failed",
    "budget_exhausted",
]
type ResearchOutcome = Literal["selected", "no_strategy"]
type CriticVerdict = Literal["accept", "revise", "reject"]
type Fingerprint = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
type MetricName = Annotated[str, StringConstraints(min_length=1, max_length=64)]
type MetricValue = StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)] | None
type Metrics = Annotated[dict[MetricName, MetricValue], Field(max_length=32)]
type Symbol = Annotated[
    str, StringConstraints(strip_whitespace=True, to_upper=True, min_length=1, max_length=32)
]
# Model-written prose: conclusions and summaries, revision requests, and list entries.
type LongText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)
]
type RevisionText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)
]
type Note = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
type WarningText = Annotated[str, StringConstraints(max_length=300)]
type EvidenceId = Annotated[str, StringConstraints(min_length=1, max_length=128)]
# Half the 2,000-character request, so an answer always fits beside the original request.
type ClarificationAnswer = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)
]

# Yahoo serves 1m bars for about the last 30 calendar days, roughly 20 XNYS sessions.
MAX_MINUTE_SESSIONS = 20
_MAX_AGENT_STEPS = 40
_MAX_TOOL_CALLS = 30
_MAX_TRAINING_BACKTESTS = 11
_MAX_OBSERVATION_JSON = 6_000
_TRUNCATION_WARNING = "Observation text was truncated to safety limits."
_PERFORMANCE_METRIC_NAMES = frozenset(
    {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "trade_count",
        "turnover",
        "asset_buy_and_hold_return",
        "benchmark_return",
    }
)


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GoalDraft(BaseModel):
    """Bounded partial goal used only by parsing and HITL clarification."""

    model_config = ConfigDict(extra="forbid")

    symbol: str | None = None
    bar_interval: BarInterval | None = None
    start_date: str | None = None
    end_date: str | None = None
    training_sessions: int | None = Field(default=None, ge=1)
    test_sessions: int | None = Field(default=None, ge=1)
    allowed_strategies: list[StrategyName] | None = Field(default=None, max_length=6)
    max_drawdown_limit: FiniteFloat | None = None
    position_tranches: int | None = Field(default=None, ge=1, le=20)
    transaction_cost_bps: FiniteFloat | None = None
    slippage_bps: FiniteFloat | None = None
    benchmark_symbol: str | None = None
    request_issue: RequestIssue | None = None


def with_matched_baseline(strategies: Sequence[object]) -> tuple[object, ...]:
    """Deduplicate a strategy scope; fixed RSI exit variants carry their intraday_rsi baseline."""
    scope: list[object] = []
    for name in strategies:
        if name not in scope:
            scope.append(name)
    if INTRADAY_RSI_BASELINE not in scope and any(name in INTRADAY_RSI_VARIANTS for name in scope):
        scope.append(INTRADAY_RSI_BASELINE)
    return tuple(scope)


class ResearchGoal(_FrozenContract):
    """A structured research goal that has already been confirmed by the user."""

    objective: LongText
    symbol: Symbol
    bar_interval: BarInterval = "1d"
    start_date: date | None = None
    end_date: date | None = None
    training_sessions: int | None = Field(default=None, ge=1, strict=True)
    test_sessions: int | None = Field(default=None, ge=1, strict=True)
    benchmark_symbol: Symbol | None = "SPY"
    allowed_strategies: tuple[StrategyName, ...] = Field(min_length=1)
    max_drawdown_limit: Annotated[FiniteFloat, Field(ge=0, lt=1)] | None = None
    costs: BacktestCosts = Field(default_factory=BacktestCosts)
    position_tranches: int = Field(default=5, ge=1, le=20, strict=True)

    @field_validator("allowed_strategies")
    @classmethod
    def add_matched_baseline(cls, value: tuple[StrategyName, ...]) -> tuple[StrategyName, ...]:
        return with_matched_baseline(value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_timeframe_fields(self) -> ResearchGoal:
        if self.bar_interval == "1d":
            if any(name in INTRADAY_RSI_STRATEGIES for name in self.allowed_strategies):
                raise ValueError("intraday RSI policies require 1m bars")
            if self.start_date is None or self.end_date is None:
                raise ValueError("daily goals require start_date and end_date")
            if self.start_date > self.end_date:
                raise ValueError("start_date must be on or before end_date")
            if self.training_sessions is not None or self.test_sessions is not None:
                raise ValueError("daily goals must not contain session counts")
            if self.benchmark_symbol is None:
                raise ValueError("daily goals require a benchmark symbol")
            return self

        if self.start_date is not None:
            raise ValueError("minute goals must not contain start_date")
        if self.end_date is None:
            raise ValueError("minute goals require a historical cutoff date")
        if self.training_sessions is None or self.test_sessions is None:
            raise ValueError("minute goals require training and test session counts")
        if self.training_sessions + self.test_sessions > MAX_MINUTE_SESSIONS:
            raise ValueError(f"minute goals cover at most {MAX_MINUTE_SESSIONS} sessions")
        if self.benchmark_symbol is not None:
            raise ValueError("minute goals do not use a benchmark symbol")
        return self


class AgentBudget(_FrozenContract):
    """Hard-bounded Agent budgets; callers may only lower the defaults."""

    max_agent_steps: int = Field(default=_MAX_AGENT_STEPS, ge=1, le=_MAX_AGENT_STEPS, strict=True)
    max_tool_calls: int = Field(default=_MAX_TOOL_CALLS, ge=1, le=_MAX_TOOL_CALLS, strict=True)
    max_training_backtests: int = Field(
        default=_MAX_TRAINING_BACKTESTS,
        ge=1,
        le=_MAX_TRAINING_BACKTESTS,
        strict=True,
    )

    @model_validator(mode="after")
    def validate_budget_relationships(self) -> AgentBudget:
        if self.max_tool_calls > self.max_agent_steps:
            raise ValueError("max_tool_calls must not exceed max_agent_steps")
        return self


class MarketContextEvidence(_FrozenContract):
    """One bounded external source excerpt returned by the context Tool."""

    id: str = Field(min_length=1, max_length=128)
    source_type: ContextSourceType
    source: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=250)
    excerpt: str = Field(min_length=1, max_length=600)
    published_at: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=1, max_length=500)


class ToolObservation(_FrozenContract):
    """A bounded Tool result safe to return to the model."""

    tool_name: str = Field(min_length=1, max_length=64)
    status: ObservationStatus
    summary: str = ""
    identifiers: dict[
        Annotated[str, StringConstraints(min_length=1, max_length=64)],
        Annotated[str, StringConstraints(max_length=128)],
    ] = Field(default_factory=dict, max_length=4)
    metrics: Metrics = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    evidence: tuple[MarketContextEvidence, ...] = Field(default=(), max_length=8)

    @model_validator(mode="before")
    @classmethod
    def trim_optional_text(cls, value: Mapping[str, object]) -> dict[str, object]:
        prepared = dict(value)
        summary = prepared.get("summary", "")
        warnings = list(prepared.get("warnings", ()))
        truncated = len(summary) > 800 or len(warnings) > 10 or any(len(w) > 300 for w in warnings)
        prepared["summary"] = _truncate(summary, 800)
        warnings = [_truncate(warning, 300) for warning in warnings[:10]]
        if truncated:
            warnings = _append_truncation_warning(warnings)
        prepared["warnings"] = tuple(warnings)
        return prepared

    @model_validator(mode="after")
    def enforce_total_json_limit(self) -> ToolObservation:
        if len(self.model_dump_json()) > _MAX_OBSERVATION_JSON:
            raise ValueError(
                f"ToolObservation JSON must not exceed {_MAX_OBSERVATION_JSON} characters"
            )
        return self

    def to_compact_json(self) -> str:
        """Return the bounded JSON representation consumed by ToolMessage."""

        return self.model_dump_json()


class BudgetUsage(_FrozenContract):
    """Counters copied into the final result."""

    agent_steps: int = Field(default=0, ge=0, le=_MAX_AGENT_STEPS, strict=True)
    tool_calls: int = Field(default=0, ge=0, le=_MAX_TOOL_CALLS, strict=True)
    training_backtests: int = Field(default=0, ge=0, le=_MAX_TRAINING_BACKTESTS, strict=True)


def performance_metrics_dict(metrics: PerformanceMetrics) -> dict[str, int | float | None]:
    """Copy trusted quantitative metrics into an Agent-facing mapping."""

    return asdict(metrics)


def evaluation_warnings(
    evaluation: TrainingEvaluation | FinalEvaluation,
    *,
    include_test: bool,
) -> tuple[str, ...]:
    """Collect bounded warnings exposed by an Agent observation or result."""

    collected = [
        *evaluation.asset_quality.warnings,
        *evaluation.warnings,
        *evaluation.training.warnings,
    ]
    if evaluation.benchmark_quality is not None:
        collected.extend(evaluation.benchmark_quality.warnings)
    if include_test and isinstance(evaluation, FinalEvaluation):
        collected.extend(evaluation.test.warnings)
    return safe_warnings(collected)


def safe_warnings(values: Sequence[str]) -> tuple[str, ...]:
    """Bound and deduplicate warnings before constructing strict Agent contracts."""

    result: list[str] = []
    for value in values:
        safe = _truncate(value, 300)
        if safe not in result:
            result.append(safe)
        if len(result) == 10:
            break
    return tuple(result)


class ResearchDraft(_FrozenContract):
    """A frozen Research submission awaiting independent Critic review."""

    outcome: ResearchOutcome
    selected_fingerprint: Fingerprint
    conclusion: LongText
    limitations: tuple[Note, ...] = Field(default=(), max_length=10)
    market_context_summary: LongText | None = None
    cited_evidence_ids: tuple[EvidenceId, ...] = Field(default=(), max_length=8)


class CriticReview(_FrozenContract):
    """One bounded, evidence-grounded Critic verdict."""

    verdict: CriticVerdict
    summary: LongText
    issues: tuple[Note, ...] = Field(default=(), max_length=8)
    required_revision: RevisionText | None = None

    @model_validator(mode="after")
    def validate_revision_relationship(self) -> CriticReview:
        if self.verdict == "revise" and self.required_revision is None:
            raise ValueError("revise verdict requires required_revision")
        if self.verdict != "revise" and self.required_revision is not None:
            raise ValueError("only revise verdict accepts required_revision")
        return self


class ResearchResult(_FrozenContract):
    """The structured terminal result produced by the Research Agent."""

    status: ResearchStatus
    selected_fingerprint: Fingerprint | None = None
    evaluated_fingerprint: Fingerprint | None = None
    training_metrics: Metrics = Field(default_factory=dict)
    test_metrics: Metrics = Field(default_factory=dict)
    conclusion: LongText
    stop_reason: LongText | None = None
    warnings: tuple[WarningText, ...] = Field(default=(), max_length=10)
    limitations: tuple[Note, ...] = Field(default=(), max_length=10)
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)

    @field_validator("training_metrics", "test_metrics")
    @classmethod
    def metrics_are_complete(cls, value: dict[str, MetricValue]) -> dict[str, MetricValue]:
        if value and frozenset(value) != _PERFORMANCE_METRIC_NAMES:
            raise ValueError("result metrics must contain the complete performance metric set")
        return value

    @model_validator(mode="after")
    def validate_status_contract(self) -> ResearchResult:
        if self.status == "selected":
            if self.selected_fingerprint is None:
                raise ValueError("selected result requires selected_fingerprint")
            if self.evaluated_fingerprint != self.selected_fingerprint:
                raise ValueError("selected and evaluated fingerprints must match")
            if not self.training_metrics or not self.test_metrics:
                raise ValueError("selected result requires training and test metrics")
            if self.stop_reason is not None:
                raise ValueError("selected result must not have a stop_reason")
        else:
            if self.selected_fingerprint is not None:
                raise ValueError("non-selected result must not select a fingerprint")
            if self.stop_reason is None:
                raise ValueError("non-selected result requires a stop_reason")

        if self.status == "no_strategy":
            if self.evaluated_fingerprint is None and (self.training_metrics or self.test_metrics):
                raise ValueError(
                    "no_strategy result without an evaluated fingerprint must not contain metrics"
                )
            if self.evaluated_fingerprint is not None and (
                not self.training_metrics or not self.test_metrics
            ):
                raise ValueError("evaluated no_strategy result requires trusted metrics")
        elif self.status == "critic_rejected":
            if self.evaluated_fingerprint is None:
                raise ValueError("critic_rejected result requires the reviewed fingerprint")
            if not self.training_metrics or not self.test_metrics:
                raise ValueError("critic_rejected result requires trusted metrics")
        return self


def _truncate(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    if maximum <= 3:
        return value[:maximum]
    return value[: maximum - 3] + "..."


def _append_truncation_warning(warnings: list[object]) -> list[object]:
    if _TRUNCATION_WARNING in warnings:
        return warnings[:10]
    if len(warnings) >= 10:
        return [*warnings[:9], _TRUNCATION_WARNING]
    return [*warnings, _TRUNCATION_WARNING]
