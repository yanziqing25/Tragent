"""The durable research workspace shared by the Research and Critic Agents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal, Self

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, PrivateAttr

from tragent.quant import (
    AlignedMarketData,
    MarketDataProvider,
    PerformanceMetrics,
    SegmentBacktestResult,
    StrategyConfig,
    StrategyName,
    build_strategy_config,
)
from tragent.research.agent.market_snapshot import load_market_data, save_market_data
from tragent.research.agent.models import (
    AgentBudget,
    CriticReview,
    Fingerprint,
    MarketContextEvidence,
    ResearchDraft,
    ResearchGoal,
    ResearchResult,
    ToolObservation,
)
from tragent.research.agent.retrieval import MarketContextRetriever

type WorkspacePhase = Literal["researching", "locked", "completed"]
# Which ReAct Agent owns the next decision; None once the run is terminal.
type ActiveAgent = Literal["research", "critic"] | None


@dataclass(frozen=True, slots=True)
class ResearchGraphContext:
    """Run-scoped integrations that must never enter checkpoint State."""

    # Stateless LangChain chat model client rebuilt for every start/resume request.
    model: BaseChatModel
    # Market-data service used once; later turns read the frozen snapshot among the artifacts.
    provider: MarketDataProvider
    # Object-store key prefix of the task's market-data snapshot and report.
    artifacts: str
    # PostgreSQL semantic context service; runtime-only like the market provider.
    retriever: MarketContextRetriever | None = None


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCallRecord(_Record):
    """One Tool call and its bounded Observation, in a Research or Critic journal."""

    step: int
    tool_name: str
    arguments: dict[str, object]
    observation: ToolObservation


class DrawdownRecord(_Record):
    max_drawdown: float
    peak_date: datetime
    trough_date: datetime
    recovery_date: datetime | None
    underwater_bars: int


class SegmentRecord(_Record):
    """The deterministic evidence Agents may read about one evaluated segment."""

    metrics: PerformanceMetrics
    drawdown: DrawdownRecord
    mean_exposure: float
    warnings: tuple[str, ...]

    @classmethod
    def from_result(cls, segment: SegmentBacktestResult, warnings: tuple[str, ...]) -> Self:
        return cls(
            metrics=segment.metrics,
            drawdown=DrawdownRecord(**asdict(segment.drawdown)),
            mean_exposure=segment.mean_exposure,
            warnings=warnings,
        )


class CandidateRecord(_Record):
    """One trained strategy; the frozen final batch adds its sample-out evidence."""

    fingerprint: Fingerprint
    strategy_name: StrategyName
    parameters: dict[str, int | float]
    training: SegmentRecord
    test: SegmentRecord | None = None

    @property
    def config(self) -> StrategyConfig:
        return build_strategy_config(self.strategy_name, self.parameters)


class CriticRound(_Record):
    """One frozen Research draft and the Critic work performed against it."""

    draft: ResearchDraft
    agent_steps: int = 0
    tool_calls: int = 0
    journal: list[ToolCallRecord] = []
    review: CriticReview | None = None


class ResearchWorkspace(_Record):
    """Everything a confirmed research run decides and observes, checkpointed after each action.

    Runtime services are bound privately for one Agent turn and never enter the checkpoint.
    The immutable market-data snapshot is stored once in the task directory.
    """

    goal: ResearchGoal
    budget: AgentBudget
    phase: WorkspacePhase = "researching"
    active_agent: ActiveAgent = "research"
    agent_steps: int = 0
    tool_calls: int = 0
    download_failures: int = 0
    journal: list[ToolCallRecord] = []
    candidates: list[CandidateRecord] = []
    locked_fingerprint: Fingerprint | None = None
    critic_rounds: list[CriticRound] = []
    runtime_warnings: list[str] = []
    fatal_reason: str | None = None
    result: ResearchResult | None = None

    _context: ResearchGraphContext | None = PrivateAttr(default=None)
    _aligned_data: AlignedMarketData | None = PrivateAttr(default=None)

    def bind(self, context: ResearchGraphContext) -> Self:
        self._context = context
        return self

    @property
    def context(self) -> ResearchGraphContext:
        assert self._context is not None, "the workspace is not bound to a run context"
        return self._context

    @property
    def provider(self) -> MarketDataProvider:
        return self.context.provider

    @property
    def retriever(self) -> MarketContextRetriever | None:
        return self.context.retriever

    @property
    def artifacts(self) -> str:
        return self.context.artifacts

    @property
    def aligned_data(self) -> AlignedMarketData | None:
        if self._aligned_data is None and self.latest_success("get_market_data") is not None:
            self._aligned_data = load_market_data(self.artifacts)
        return self._aligned_data

    def store_market_data(self, data: AlignedMarketData) -> None:
        save_market_data(self.artifacts, data)
        self._aligned_data = data

    @property
    def training_backtests(self) -> int:
        return len(self.candidates)

    def candidate(self, fingerprint: str | None) -> CandidateRecord | None:
        return next((item for item in self.candidates if item.fingerprint == fingerprint), None)

    @property
    def final_batch_completed(self) -> bool:
        return self.phase != "researching" and any(item.test for item in self.candidates)

    def latest_success(self, tool_name: str) -> ToolObservation | None:
        return next(
            (
                record.observation
                for record in reversed(self.journal)
                if record.tool_name == tool_name and record.observation.status == "success"
            ),
            None,
        )

    @property
    def retrieved_evidence(self) -> dict[str, MarketContextEvidence]:
        return {
            evidence.id: evidence
            for record in self.journal
            for evidence in record.observation.evidence
        }


def consume_agent_step(workspace: ResearchWorkspace) -> bool:
    """Consume one model step, returning False without mutation at the limit."""

    if workspace.agent_steps >= workspace.budget.max_agent_steps:
        return False
    workspace.agent_steps += 1
    return True


def set_fatal(workspace: ResearchWorkspace, reason: str) -> None:
    """Record the first fatal Agent runtime failure."""

    if workspace.fatal_reason is None:
        workspace.fatal_reason = reason[:2_000]


def add_runtime_warning(workspace: ResearchWorkspace, warning: str) -> None:
    """Append one unique warning to the Agent workspace."""

    if warning not in workspace.runtime_warnings:
        workspace.runtime_warnings.append(warning)


def record_tool_call(
    journal: list[ToolCallRecord],
    *,
    step: int,
    tool_name: str,
    arguments: Mapping[str, object],
    observation: ToolObservation,
) -> None:
    """Append one Action/Observation pair to a Research or Critic journal."""

    journal.append(
        ToolCallRecord(
            step=max(1, step),
            tool_name=tool_name,
            arguments=dict(arguments),
            observation=observation,
        )
    )


def canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError):
        raise ValueError("payload must contain only finite JSON values") from None


__all__ = [
    "ActiveAgent",
    "CandidateRecord",
    "CriticRound",
    "DrawdownRecord",
    "ResearchGraphContext",
    "ResearchWorkspace",
    "SegmentRecord",
    "ToolCallRecord",
    "WorkspacePhase",
    "add_runtime_warning",
    "canonical_json",
    "consume_agent_step",
    "record_tool_call",
    "set_fatal",
]
