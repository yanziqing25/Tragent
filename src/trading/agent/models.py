"""Compact JSON-facing contracts for the Milestone 2 Research Agent."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from math import isfinite
from numbers import Integral, Real
from typing import Annotated, Literal

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

from trading.quant import (
    BacktestCosts,
    FinalEvaluation,
    PerformanceMetrics,
    StrategyName,
    TrainingEvaluation,
)

type BarInterval = Literal["1d", "1m"]
type RequestIssue = Literal[
    "ambiguous_fields",
    "unsupported_request",
    "unrelated_request",
]
type ObservationStatus = Literal["success", "rejected", "error"]
type ResearchStatus = Literal[
    "selected",
    "no_strategy",
    "critic_rejected",
    "failed",
    "budget_exhausted",
]
type ResearchOutcome = Literal["selected", "no_strategy"]
type CriticVerdict = Literal["accept", "revise", "reject"]
type Fingerprint = Annotated[StrictStr, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type MetricValue = int | float | None

_MAX_AGENT_STEPS = 40
_MAX_TOOL_CALLS = 30
_MAX_TRAINING_BACKTESTS = 8
_MAX_OBSERVATION_JSON = 6_000
_MAX_TRACE_ARGUMENTS_JSON = 8_000
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

    symbol: StrictStr | None = None
    bar_interval: BarInterval | None = None
    start_date: StrictStr | None = None
    end_date: StrictStr | None = None
    training_sessions: StrictInt | None = Field(default=None, ge=1, le=60)
    test_sessions: StrictInt | None = Field(default=None, ge=1, le=60)
    allowed_strategies: list[StrategyName] | None = Field(default=None, max_length=3)
    max_drawdown_limit: StrictInt | StrictFloat | None = None
    position_tranches: StrictInt | None = Field(default=None, ge=1, le=20)
    transaction_cost_bps: StrictInt | StrictFloat | None = None
    slippage_bps: StrictInt | StrictFloat | None = None
    benchmark_symbol: StrictStr | None = None
    request_issue: RequestIssue | None = None


class ResearchGoal(_FrozenContract):
    """A structured research goal that has already been confirmed by the user."""

    objective: StrictStr
    symbol: StrictStr
    bar_interval: BarInterval = "1d"
    start_date: date | None = None
    end_date: date | None = None
    training_sessions: StrictInt | None = Field(default=None, ge=1, le=60)
    test_sessions: StrictInt | None = Field(default=None, ge=1, le=60)
    benchmark_symbol: StrictStr | None = "SPY"
    allowed_strategies: tuple[StrategyName, ...]
    max_drawdown_limit: float | None = None
    costs: BacktestCosts = Field(default_factory=BacktestCosts)
    position_tranches: StrictInt = Field(default=5, ge=1, le=20)

    @field_validator("objective")
    @classmethod
    def normalize_objective(cls, value: str) -> str:
        normalized = value.strip()
        if not 1 <= len(normalized) <= 2_000:
            raise ValueError("objective must contain between 1 and 2000 characters")
        return normalized

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not 1 <= len(normalized) <= 32:
            raise ValueError("symbol must contain between 1 and 32 characters")
        return normalized

    @field_validator("benchmark_symbol")
    @classmethod
    def normalize_benchmark_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not 1 <= len(normalized) <= 32:
            raise ValueError("benchmark_symbol must contain between 1 and 32 characters")
        return normalized

    @field_validator("allowed_strategies", mode="before")
    @classmethod
    def deduplicate_strategies(cls, value: object) -> object:
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("allowed_strategies must be a sequence")
        deduplicated: list[object] = []
        for item in value:
            if item not in deduplicated:
                deduplicated.append(item)
        if not deduplicated:
            raise ValueError("allowed_strategies must not be empty")
        return tuple(deduplicated)

    @field_validator("max_drawdown_limit", mode="before")
    @classmethod
    def validate_drawdown_limit(cls, value: object) -> float | None:
        if value is None:
            return None
        normalized = _finite_number("max_drawdown_limit", value)
        if not 0.0 <= normalized < 1.0:
            raise ValueError("max_drawdown_limit must satisfy 0 <= value < 1")
        return normalized

    @field_validator("costs", mode="before")
    @classmethod
    def preserve_strict_cost_validation(cls, value: object) -> object:
        if isinstance(value, BacktestCosts):
            return value
        if not isinstance(value, Mapping):
            return value
        allowed = {"transaction_cost_bps", "slippage_bps"}
        if not all(isinstance(key, str) for key in value) or set(value) - allowed:
            raise ValueError("costs contains unsupported fields")
        defaults = BacktestCosts()
        try:
            return BacktestCosts(
                transaction_cost_bps=value.get(
                    "transaction_cost_bps",
                    defaults.transaction_cost_bps,
                ),
                slippage_bps=value.get("slippage_bps", defaults.slippage_bps),
            )
        except ValueError:
            raise ValueError("costs must contain valid finite numeric values") from None

    @model_validator(mode="after")
    def validate_timeframe_fields(self) -> ResearchGoal:
        if self.bar_interval == "1d":
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
        if self.benchmark_symbol is not None:
            raise ValueError("minute goals do not use a benchmark symbol")
        return self


class AgentBudget(_FrozenContract):
    """Hard-bounded Agent budgets; callers may only lower the defaults."""

    max_agent_steps: StrictInt = Field(default=_MAX_AGENT_STEPS, ge=1, le=_MAX_AGENT_STEPS)
    max_tool_calls: StrictInt = Field(default=_MAX_TOOL_CALLS, ge=1, le=_MAX_TOOL_CALLS)
    max_training_backtests: StrictInt = Field(
        default=_MAX_TRAINING_BACKTESTS,
        ge=1,
        le=_MAX_TRAINING_BACKTESTS,
    )

    @model_validator(mode="after")
    def validate_budget_relationships(self) -> AgentBudget:
        if self.max_tool_calls > self.max_agent_steps:
            raise ValueError("max_tool_calls must not exceed max_agent_steps")
        return self


class ToolObservation(_FrozenContract):
    """A bounded Tool result safe to return to the model."""

    tool_name: StrictStr = Field(min_length=1, max_length=64)
    status: ObservationStatus
    summary: StrictStr = ""
    identifiers: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    metrics: dict[StrictStr, MetricValue] = Field(default_factory=dict)
    warnings: tuple[StrictStr, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def trim_optional_text(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        prepared = dict(value)
        summary = prepared.get("summary", "")
        if not isinstance(summary, str):
            return prepared

        truncated = len(summary) > 800
        prepared["summary"] = _truncate(summary, 800)
        raw_warnings = prepared.get("warnings", ())
        if isinstance(raw_warnings, str) or not isinstance(raw_warnings, Sequence):
            return prepared

        normalized_warnings: list[object] = []
        for warning in raw_warnings:
            if not isinstance(warning, str):
                normalized_warnings.append(warning)
                continue
            if len(warning) > 300:
                truncated = True
            normalized_warnings.append(_truncate(warning, 300))
        if len(normalized_warnings) > 10:
            truncated = True
            normalized_warnings = normalized_warnings[:10]
        if truncated:
            normalized_warnings = _append_truncation_warning(normalized_warnings)
        prepared["warnings"] = tuple(normalized_warnings)
        return prepared

    @field_validator("identifiers", mode="before")
    @classmethod
    def validate_identifiers(cls, value: object) -> dict[str, str]:
        if not isinstance(value, Mapping) or len(value) > 4:
            raise ValueError("identifiers must be a mapping with at most four entries")
        normalized: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise ValueError("identifier keys and values must be strings")
            if not 1 <= len(key) <= 64 or len(item) > 128:
                raise ValueError("identifier keys or values exceed their size limit")
            normalized[key] = item
        return normalized

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(cls, value: object) -> dict[str, MetricValue]:
        return _validated_metrics(value)

    @model_validator(mode="after")
    def enforce_total_json_limit(self) -> ToolObservation:
        if len(self.model_dump_json()) <= _MAX_OBSERVATION_JSON:
            return self

        object.__setattr__(self, "summary", "")
        object.__setattr__(self, "warnings", (_TRUNCATION_WARNING,))
        if len(self.model_dump_json()) > _MAX_OBSERVATION_JSON:
            raise ValueError("ToolObservation JSON must not exceed 6000 characters")
        return self

    def to_compact_json(self) -> str:
        """Return the bounded JSON representation consumed by ToolMessage."""

        return self.model_dump_json()


class BudgetUsage(_FrozenContract):
    """Counters copied into the final result."""

    agent_steps: StrictInt = Field(default=0, ge=0, le=_MAX_AGENT_STEPS)
    tool_calls: StrictInt = Field(default=0, ge=0, le=_MAX_TOOL_CALLS)
    training_backtests: StrictInt = Field(default=0, ge=0, le=_MAX_TRAINING_BACKTESTS)


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
        if not isinstance(value, str):
            continue
        safe = _truncate(value, 300)
        if safe not in result:
            result.append(safe)
        if len(result) == 10:
            break
    return tuple(result)


class ResearchTraceRecord(_FrozenContract):
    """One compact, observable Action/Observation record without model reasoning."""

    step: StrictInt = Field(ge=1, le=_MAX_AGENT_STEPS)
    tool_name: StrictStr = Field(min_length=1, max_length=64)
    arguments: dict[StrictStr, object] = Field(default_factory=dict)
    observation: ToolObservation

    @field_validator("arguments", mode="before")
    @classmethod
    def validate_arguments(cls, value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError("arguments must be a mapping")
        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            normalized = json.loads(encoded)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("arguments must contain only finite JSON values") from None
        if not isinstance(normalized, dict) or not all(isinstance(key, str) for key in normalized):
            raise ValueError("arguments must use string keys")
        compact = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(compact) > _MAX_TRACE_ARGUMENTS_JSON:
            raise ValueError("arguments JSON must not exceed 8000 characters")
        return normalized


class ResearchDraft(_FrozenContract):
    """A frozen Research submission awaiting independent Critic review."""

    outcome: ResearchOutcome
    selected_fingerprint: Fingerprint
    conclusion: StrictStr = Field(min_length=1, max_length=2_000)
    limitations: tuple[StrictStr, ...] = ()

    @field_validator("conclusion")
    @classmethod
    def normalize_conclusion(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("conclusion must not be blank")
        return normalized

    @field_validator("limitations", mode="before")
    @classmethod
    def validate_limitations(cls, value: object) -> tuple[str, ...]:
        normalized = _bounded_strings(
            value,
            maximum_items=10,
            maximum_length=500,
            field="limitations",
        )
        if any(not item.strip() for item in normalized):
            raise ValueError("limitations must not contain blank entries")
        return normalized


class CriticReview(_FrozenContract):
    """One bounded, evidence-grounded Critic verdict."""

    verdict: CriticVerdict
    summary: StrictStr = Field(min_length=1, max_length=2_000)
    issues: tuple[StrictStr, ...] = ()
    required_revision: StrictStr | None = Field(default=None, min_length=1, max_length=1_000)

    @field_validator("summary")
    @classmethod
    def normalize_summary(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("summary must not be blank")
        return normalized

    @field_validator("issues", mode="before")
    @classmethod
    def validate_issues(cls, value: object) -> tuple[str, ...]:
        normalized = _bounded_strings(
            value,
            maximum_items=8,
            maximum_length=500,
            field="issues",
        )
        if any(not item.strip() for item in normalized):
            raise ValueError("issues must not contain blank entries")
        return normalized

    @field_validator("required_revision")
    @classmethod
    def normalize_required_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("required_revision must not be blank")
        return normalized

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
    goal: ResearchGoal
    selected_fingerprint: Fingerprint | None = None
    evaluated_fingerprint: Fingerprint | None = None
    training_metrics: dict[StrictStr, MetricValue] = Field(default_factory=dict)
    test_metrics: dict[StrictStr, MetricValue] = Field(default_factory=dict)
    conclusion: StrictStr = Field(min_length=1, max_length=2_000)
    stop_reason: StrictStr | None = Field(default=None, min_length=1, max_length=2_000)
    warnings: tuple[StrictStr, ...] = ()
    limitations: tuple[StrictStr, ...] = ()
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)
    trace: tuple[ResearchTraceRecord, ...] = ()

    @field_validator("training_metrics", "test_metrics", mode="before")
    @classmethod
    def validate_result_metrics(cls, value: object) -> dict[str, MetricValue]:
        normalized = _validated_metrics(value)
        if normalized and frozenset(normalized) != _PERFORMANCE_METRIC_NAMES:
            raise ValueError("result metrics must contain the complete performance metric set")
        return normalized

    @field_validator("warnings", mode="before")
    @classmethod
    def validate_warnings(cls, value: object) -> tuple[str, ...]:
        return _bounded_strings(value, maximum_items=10, maximum_length=300, field="warnings")

    @field_validator("limitations", mode="before")
    @classmethod
    def validate_limitations(cls, value: object) -> tuple[str, ...]:
        return _bounded_strings(
            value,
            maximum_items=10,
            maximum_length=500,
            field="limitations",
        )

    @field_validator("trace", mode="before")
    @classmethod
    def validate_trace_length(cls, value: object) -> object:
        if isinstance(value, Sequence) and not isinstance(value, str) and len(value) <= 40:
            return value
        raise ValueError("trace must contain at most 40 records")

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
            if self.stop_reason is None or not self.stop_reason.strip():
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


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _validated_metrics(value: object) -> dict[str, MetricValue]:
    if not isinstance(value, Mapping) or len(value) > 32:
        raise ValueError("metrics must be a mapping with at most 32 entries")
    normalized: dict[str, MetricValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 64:
            raise ValueError("metric keys must contain between 1 and 64 characters")
        if item is None:
            normalized[key] = None
        elif isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError("metric values must be finite numbers, integers, or None")
        elif isinstance(item, Integral):
            normalized[key] = int(item)
        else:
            number = float(item)
            if not isfinite(number):
                raise ValueError("metric values must be finite")
            normalized[key] = number
    return normalized


def _bounded_strings(
    value: object,
    *,
    maximum_items: int,
    maximum_length: int,
    field: str,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) > maximum_items:
        raise ValueError(f"{field} exceeds its item limit")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or len(item) > maximum_length:
            raise ValueError(f"{field} entries exceed their size limit")
        normalized.append(item)
    return tuple(normalized)


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
