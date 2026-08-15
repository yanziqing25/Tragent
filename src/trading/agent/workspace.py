"""Mutable working memory for one Research and Critic Agent run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from trading.agent.models import (
    AgentBudget,
    CriticReview,
    ResearchDraft,
    ResearchGoal,
    ResearchResult,
    ResearchTraceRecord,
    ToolObservation,
)
from trading.quant import (
    AlignedMarketData,
    AssetStatistics,
    FinalEvaluation,
    MarketDataProvider,
    StrategyConfig,
    TrainingEvaluation,
)

type WorkspacePhase = Literal["researching", "locked", "completed"]
type ActiveAgent = Literal["research", "critic"] | None


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """One compact event; only complete valid Tool interactions are replayable."""

    tool_name: str
    normalized_arguments: dict[str, object]
    observation: ToolObservation
    tool_call_id: str | None = None
    replayable: bool = False


@dataclass(slots=True)
class CriticRoundRuntime:
    """Mutable state for one bounded Critic ReAct review."""

    draft: ResearchDraft
    agent_steps: int = 0
    tool_calls: int = 0
    events: list[RuntimeEvent] = field(default_factory=list)
    event_steps: list[int] = field(default_factory=list)
    review: CriticReview | None = None


@dataclass(slots=True)
class ResearchWorkspace:
    """Working memory reconstructed from Graph State for the active Agent turn."""

    goal: ResearchGoal
    budget: AgentBudget
    provider: MarketDataProvider
    output_dir: Path
    aligned_data: AlignedMarketData | None = None
    asset_statistics: AssetStatistics | None = None
    current_candidate: StrategyConfig | None = None
    validated_configs: dict[str, StrategyConfig] = field(default_factory=dict)
    fingerprint_payloads: dict[str, dict[str, object]] = field(default_factory=dict)
    training_evaluations: dict[str, TrainingEvaluation] = field(default_factory=dict)
    final_evaluations: dict[str, FinalEvaluation] = field(default_factory=dict)
    phase: WorkspacePhase = "researching"
    locked_fingerprint: str | None = None
    final_batch_completed: bool = False
    active_agent: ActiveAgent = "research"
    critic_rounds: list[CriticRoundRuntime] = field(default_factory=list)
    agent_steps: int = 0
    tool_calls: int = 0
    training_backtests: int = 0
    cached_observations: dict[str, ToolObservation] = field(default_factory=dict)
    download_failures: int = 0
    trace: list[ResearchTraceRecord] = field(default_factory=list)
    events: list[RuntimeEvent] = field(default_factory=list)
    unresolved_issues: list[str] = field(default_factory=list)
    runtime_warnings: list[str] = field(default_factory=list)
    fatal_reason: str | None = None
    result: ResearchResult | None = None


def consume_agent_step(workspace: ResearchWorkspace) -> bool:
    """Consume one model step, returning False without mutation at the limit."""

    if workspace.agent_steps >= workspace.budget.max_agent_steps:
        return False
    workspace.agent_steps += 1
    return True


def consume_tool_call(workspace: ResearchWorkspace) -> bool:
    """Consume one known single-Action attempt before standard ToolNode execution."""

    if workspace.tool_calls >= workspace.budget.max_tool_calls:
        return False
    workspace.tool_calls += 1
    return True


def lock_selection(workspace: ResearchWorkspace, fingerprint: str | None) -> bool:
    """Irreversibly lock one trained candidate or an explicit no-selection choice."""

    if workspace.phase != "researching":
        return False
    if fingerprint is not None and fingerprint not in workspace.training_evaluations:
        return False
    workspace.locked_fingerprint = fingerprint
    workspace.phase = "locked"
    return True


def set_fatal(workspace: ResearchWorkspace, reason: str) -> None:
    """Record the first fatal Agent runtime failure."""

    if workspace.fatal_reason is None:
        workspace.fatal_reason = reason[:2_000]
    if workspace.fatal_reason not in workspace.unresolved_issues:
        workspace.unresolved_issues.append(workspace.fatal_reason)


def add_runtime_warning(workspace: ResearchWorkspace, warning: str) -> None:
    """Append one unique warning to the Agent workspace."""

    if warning not in workspace.runtime_warnings:
        workspace.runtime_warnings.append(warning)


def record_event(
    workspace: ResearchWorkspace,
    *,
    tool_name: str,
    normalized_arguments: Mapping[str, object],
    observation: ToolObservation,
    tool_call_id: str | None,
    replayable: bool,
) -> RuntimeEvent:
    """Append one normalized Action/Observation pair to the Agent journal."""

    normalized = json.loads(canonical_json(dict(normalized_arguments)))
    assert isinstance(normalized, dict)
    event = RuntimeEvent(
        tool_name=tool_name,
        normalized_arguments=normalized,
        observation=observation,
        tool_call_id=tool_call_id,
        replayable=bool(replayable and tool_call_id),
    )
    workspace.events.append(event)
    record = ResearchTraceRecord(
        step=max(1, workspace.agent_steps),
        tool_name=tool_name,
        arguments=normalized,
        observation=observation,
    )
    workspace.trace.append(record)
    if workspace.result is not None:
        workspace.result = workspace.result.model_copy(update={"trace": tuple(workspace.trace)})
    return event


def canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError):
        raise ValueError("payload must contain only finite JSON values") from None


__all__ = [
    "ActiveAgent",
    "CriticRoundRuntime",
    "ResearchWorkspace",
    "RuntimeEvent",
    "canonical_json",
    "add_runtime_warning",
    "consume_agent_step",
    "consume_tool_call",
    "lock_selection",
    "record_event",
    "set_fatal",
]
