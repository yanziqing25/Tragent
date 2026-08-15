"""Validated plain-JSON State and adapters for the durable Research workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)

from trading.agent.market_snapshot import (
    DataSnapshotDTO,
    aligned_data_from_snapshot,
    aligned_data_to_snapshot,
)
from trading.agent.models import (
    AgentBudget,
    BudgetUsage,
    CriticReview,
    Fingerprint,
    GoalDraft,
    ResearchDraft,
    ResearchGoal,
    ResearchResult,
    ResearchStatus,
    ResearchTraceRecord,
    ToolObservation,
)
from trading.agent.workspace import (
    CriticRoundRuntime,
    ResearchWorkspace,
    RuntimeEvent,
)
from trading.quant import (
    FinalEvaluation,
    MarketDataProvider,
    StrategyConfig,
    StrategyName,
    TrainingEvaluation,
    build_strategy_config,
    strategy_name,
    strategy_parameters,
)
from trading.quant import run_final_evaluation as quant_run_final_evaluation
from trading.quant import run_training_evaluation as quant_run_training_evaluation

type TaskStatus = Literal[
    "awaiting_clarification",
    "awaiting_confirmation",
    "running",
    "selected",
    "no_strategy",
    "critic_rejected",
    "failed",
    "budget_exhausted",
    "rejected",
]
# Which ReAct Agent currently owns the next decision; None outside an Agent loop.
type ActiveAgentState = Literal["research", "critic"] | None
# Coarse deterministic phase inside the confirmed research workspace.
type WorkspacePhaseState = Literal["researching", "locked", "completed"]
type PlainObject = dict[str, object]


class JournalEntryState(TypedDict):
    """One durable Action/Observation record from a Tool call."""

    # One-based Agent step associated with this Tool call.
    step: int
    # Stable LangChain Tool name selected by the model.
    tool_name: str
    # Validated JSON arguments supplied by the model.
    arguments: dict[str, object]
    # Bounded deterministic result returned to the model.
    observation: dict[str, object]
    # Provider-generated identifier linking the Tool result to its Tool call.
    tool_call_id: str | None
    # Whether this recorded result may be reused while reconstructing Agent messages.
    replayable: bool


class TerminalResultState(TypedDict):
    """Final JSON-safe outcome stored after the research workflow terminates."""

    # Business outcome, such as selected, no_strategy, failed, or budget_exhausted.
    status: ResearchStatus
    # Candidate selected before irreversible out-of-sample evaluation, when applicable.
    selected_fingerprint: str | None
    # Candidate whose final evaluation produced the terminal metrics.
    evaluated_fingerprint: str | None
    # Metrics from the in-sample research segment.
    training_metrics: dict[str, int | float | None]
    # Metrics from the held-out final segment.
    test_metrics: dict[str, int | float | None]
    # Human-readable conclusion produced from validated evidence.
    conclusion: str
    # Optional bounded reason explaining why execution stopped.
    stop_reason: str | None
    # Non-fatal caveats discovered during execution.
    warnings: list[str]
    # Known limits of the final conclusion.
    limitations: list[str]
    # Final counters describing how much of the Agent budget was consumed.
    budget_usage: dict[str, int]


class CriticRoundState(TypedDict):
    """One frozen Research draft and the Critic work performed against it."""

    # Research draft frozen before this review round begins.
    draft: dict[str, object]
    # Number of Critic model decisions consumed in this round.
    agent_steps: int
    # Number of Critic Tool calls consumed in this round.
    tool_calls: int
    # Critic-only evidence inspection journal.
    journal: list[JournalEntryState]
    # Submitted accept/revise/reject decision, or None while review is active.
    review: dict[str, object] | None


class GraphState(TypedDict):
    """Complete JSON-safe task state checkpointed by the outer LangGraph workflow."""

    # --- Identity and immutable request inputs ---
    # Canonical UUID4 used as the LangGraph thread and public task identifier.
    thread_id: str
    # Authenticated task owner; every read/resume operation verifies this value.
    user_id: int
    # Normalized natural-language request retained as the task's source instruction.
    request: str
    # Stable ISO date used for deterministic relative-date interpretation.
    reference_date: str

    # --- Top-level workflow status and safe failures ---
    # Public lifecycle status used for routing, resume rules, and API views.
    status: TaskStatus
    # Deterministic stage that failed before an Agent workspace existed.
    failure_stage: Literal["goal_parsing"] | None
    # Safe validation error from an invalid human confirmation payload.
    confirmation_error: Literal["Invalid confirmation response."] | None
    # Bounded user-facing failure description that never contains raw secrets.
    failure_summary: str | None

    # --- Server-controlled defaults copied into durable state ---
    # Transaction cost and slippage defaults used when the user omits them.
    cost_defaults: dict[str, float]
    # Default number of equal position tranches used when the user omits it.
    position_tranches_default: int

    # --- Goal parsing and human confirmation ---
    # Partial structured goal produced by the one-shot Goal Parser model.
    draft_goal: dict[str, object] | None
    # Fully validated goal frozen after explicit user approval.
    confirmed_goal: dict[str, object] | None
    # Goal fields that require an additional user answer before approval.
    missing_goal_fields: list[str]
    # Goal fields filled from deterministic server defaults rather than user text.
    defaulted_fields: list[str]

    # --- Agent orchestration and budgets ---
    # Immutable maximum Agent, Tool, and training-backtest allowances.
    budget: dict[str, int]
    # Current deterministic workspace phase: researching, locked, or completed.
    phase: WorkspacePhaseState | None
    # Agent expected to make the next decision, or None outside the ReAct phase.
    active_agent: ActiveAgentState
    # Frozen Critic review rounds, including at most one requested revision.
    critic_rounds: list[CriticRoundState]
    # Research Agent model decisions consumed so far.
    agent_steps: int
    # Research Agent Tool calls consumed so far.
    tool_calls: int
    # Unique in-sample backtests consumed so far.
    training_backtests: int
    # Market-data download failures recorded for bounded retry behavior.
    download_failures: int

    # --- Durable Research Agent memory ---
    # Ordered Tool Action/Observation journal used to reconstruct bounded prompts.
    message_journal: list[JournalEntryState]
    # Evidence or validation issues the Research Agent still needs to resolve.
    unresolved_issues: list[str]
    # Non-fatal runtime warnings accumulated across Tools.
    runtime_warnings: list[str]
    # Fatal deterministic reason waiting to be converted into a terminal result.
    fatal_reason: str | None

    # --- Market data, candidates, and evaluation bookkeeping ---
    # JSON snapshot of aligned asset/benchmark bars; never a live DataFrame.
    data_snapshot: dict[str, object] | None
    # Strategy configuration currently being considered by the Research Agent.
    current_candidate: dict[str, object] | None
    # Validated strategy configurations keyed by their deterministic fingerprint.
    validated_configs: dict[str, dict[str, object]]
    # Fingerprints that already received an in-sample evaluation.
    training_evaluations: list[str]
    # Fingerprints that already received an irreversible held-out evaluation.
    final_evaluations: list[str]
    # Frozen candidate selected before held-out evaluation; cannot be changed later.
    locked_fingerprint: str | None
    # Whether the one allowed held-out evaluation batch has completed.
    final_batch_completed: bool

    # --- Terminal output ---
    # Final result payload; None until the workflow reaches a terminal outcome.
    terminal_result: TerminalResultState | None


class _StrictDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _CostDefaultsDTO(_StrictDTO):
    transaction_cost_bps: float
    slippage_bps: float


class _CandidateDTO(_StrictDTO):
    strategy_name: StrategyName
    parameters: dict[str, int | float]


class _JournalEntryDTO(_StrictDTO):
    step: int
    tool_name: str
    arguments: dict[str, object]
    observation: ToolObservation
    tool_call_id: str | None = None
    replayable: bool


class _CriticRoundDTO(_StrictDTO):
    draft: ResearchDraft
    agent_steps: int
    tool_calls: int
    journal: tuple[_JournalEntryDTO, ...]
    review: CriticReview | None = None


class _TerminalResultDTO(_StrictDTO):
    status: ResearchStatus
    selected_fingerprint: Fingerprint | None = None
    evaluated_fingerprint: Fingerprint | None = None
    training_metrics: dict[str, int | float | None] = Field(default_factory=dict)
    test_metrics: dict[str, int | float | None] = Field(default_factory=dict)
    conclusion: str
    stop_reason: str | None = None
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    budget_usage: BudgetUsage


class _GraphStateDTO(_StrictDTO):
    thread_id: str
    user_id: int = Field(strict=True, ge=1)
    request: str
    reference_date: str
    status: TaskStatus
    failure_stage: Literal["goal_parsing"] | None
    confirmation_error: Literal["Invalid confirmation response."] | None
    failure_summary: str | None
    cost_defaults: _CostDefaultsDTO
    position_tranches_default: int
    draft_goal: GoalDraft | None
    confirmed_goal: ResearchGoal | None
    missing_goal_fields: tuple[str, ...]
    defaulted_fields: tuple[str, ...]
    budget: AgentBudget
    phase: WorkspacePhaseState | None
    active_agent: ActiveAgentState
    critic_rounds: tuple[_CriticRoundDTO, ...]
    agent_steps: int
    tool_calls: int
    training_backtests: int
    download_failures: int
    message_journal: tuple[_JournalEntryDTO, ...]
    unresolved_issues: tuple[str, ...]
    runtime_warnings: tuple[str, ...]
    fatal_reason: str | None
    data_snapshot: DataSnapshotDTO | None
    current_candidate: _CandidateDTO | None
    validated_configs: dict[Fingerprint, _CandidateDTO]
    training_evaluations: tuple[Fingerprint, ...]
    final_evaluations: tuple[Fingerprint, ...]
    locked_fingerprint: Fingerprint | None
    final_batch_completed: bool
    terminal_result: _TerminalResultDTO | None


_JOURNAL_ADAPTER = TypeAdapter(list[_JournalEntryDTO])
_CRITIC_ROUNDS_ADAPTER = TypeAdapter(list[_CriticRoundDTO])


def validate_graph_state(value: Mapping[str, object]) -> GraphState:
    """Return one detached, validated plain-JSON application State."""

    _dto, normalized = _parse_graph_state(value)
    return normalized


def workspace_to_state(
    workspace: ResearchWorkspace,
) -> dict[str, object]:
    """Serialize the durable runtime and quant fields owned by one workspace."""

    snapshot = (
        None if workspace.aligned_data is None else aligned_data_to_snapshot(workspace.aligned_data)
    )
    payload: dict[str, object] = {
        "phase": workspace.phase,
        "active_agent": workspace.active_agent if workspace.result is None else None,
        "critic_rounds": _critic_rounds_from_workspace(workspace),
        "agent_steps": workspace.agent_steps,
        "tool_calls": workspace.tool_calls,
        "training_backtests": workspace.training_backtests,
        "download_failures": workspace.download_failures,
        "message_journal": _journal_from_workspace(workspace),
        "unresolved_issues": list(workspace.unresolved_issues),
        "runtime_warnings": list(workspace.runtime_warnings),
        "fatal_reason": workspace.fatal_reason,
        "data_snapshot": snapshot,
        "current_candidate": _candidate_to_state(workspace.current_candidate),
        "validated_configs": {
            fingerprint: _candidate_to_state(config)
            for fingerprint, config in workspace.validated_configs.items()
        },
        "training_evaluations": list(workspace.training_evaluations),
        "final_evaluations": list(workspace.final_evaluations),
        "locked_fingerprint": workspace.locked_fingerprint,
        "final_batch_completed": workspace.final_batch_completed,
        "terminal_result": (
            None if workspace.result is None else _terminal_result_to_state(workspace.result)
        ),
    }
    return cast(dict[str, object], copy_plain_json(payload))


def restore_workspace_from_state(
    state: Mapping[str, object],
    *,
    provider: MarketDataProvider,
    output_dir: Path,
) -> tuple[GraphState, ResearchWorkspace]:
    """Validate State once and return its normalized mapping and restored workspace."""

    dto, _normalized = _parse_graph_state(state)
    root = Path(output_dir)
    try:
        goal = dto.confirmed_goal
        if goal is None or dto.phase is None:
            raise ValueError("checkpoint does not contain a confirmed research workspace")

        aligned_data = (
            None if dto.data_snapshot is None else aligned_data_from_snapshot(dto.data_snapshot)
        )
        configs = {
            fingerprint: _config_from_candidate(candidate)
            for fingerprint, candidate in dto.validated_configs.items()
        }
        current_candidate = (
            None if dto.current_candidate is None else _config_from_candidate(dto.current_candidate)
        )

        training_evaluations: dict[str, TrainingEvaluation] = {}
        for fingerprint in dto.training_evaluations:
            if aligned_data is None:
                raise ValueError("training State has no market-data snapshot")
            evaluation = quant_run_training_evaluation(
                aligned_data,
                configs[fingerprint],
                costs=goal.costs,
                position_tranches=goal.position_tranches,
            )
            training_evaluations[fingerprint] = evaluation

        final_evaluations: dict[str, FinalEvaluation] = {}
        for fingerprint in dto.final_evaluations:
            if aligned_data is None:
                raise ValueError("final State has no market-data snapshot")
            evaluation = quant_run_final_evaluation(
                aligned_data,
                configs[fingerprint],
                costs=goal.costs,
                position_tranches=goal.position_tranches,
            )
            final_evaluations[fingerprint] = evaluation

        events, trace = _history_from_entries(dto.message_journal)
        cached_observations = _cached_observations_from_history(events)
        result = (
            None
            if dto.terminal_result is None
            else _result_from_dto(
                dto.terminal_result,
                goal=goal,
                trace=trace,
            )
        )
        critic_rounds = _critic_rounds_from_dto(dto.critic_rounds)
        workspace = ResearchWorkspace(
            goal=goal,
            budget=dto.budget,
            provider=provider,
            output_dir=root,
            aligned_data=aligned_data,
            current_candidate=current_candidate,
            validated_configs=configs,
            training_evaluations=training_evaluations,
            final_evaluations=final_evaluations,
            phase=dto.phase,
            locked_fingerprint=dto.locked_fingerprint,
            final_batch_completed=dto.final_batch_completed,
            active_agent=dto.active_agent,
            critic_rounds=critic_rounds,
            agent_steps=dto.agent_steps,
            tool_calls=dto.tool_calls,
            training_backtests=dto.training_backtests,
            cached_observations=cached_observations,
            download_failures=dto.download_failures,
            trace=trace,
            events=events,
            unresolved_issues=list(dto.unresolved_issues),
            runtime_warnings=list(dto.runtime_warnings),
            fatal_reason=dto.fatal_reason,
            result=result,
        )
        return _normalized, workspace
    except (ValidationError, TypeError, ValueError, KeyError, IndexError):
        raise ValueError("checkpoint workspace could not be rebuilt") from None


def _journal_from_workspace(workspace: ResearchWorkspace) -> list[JournalEntryState]:
    """Create the single checkpoint journal from matching runtime events and trace records."""

    if len(workspace.events) != len(workspace.trace):
        raise ValueError("workspace event and trace histories do not match")
    entries: list[dict[str, object]] = []
    for event, trace in zip(workspace.events, workspace.trace, strict=True):
        trace_payload = trace.model_dump(mode="json")
        event_observation = event.observation.model_dump(mode="json")
        if (
            trace.tool_name != event.tool_name
            or trace_payload["arguments"] != event.normalized_arguments
            or trace_payload["observation"] != event_observation
        ):
            raise ValueError("workspace event and trace histories do not match")
        entries.append(
            {
                "step": trace.step,
                "tool_name": event.tool_name,
                "arguments": copy_plain_json(event.normalized_arguments),
                "observation": event.observation.model_dump(mode="json"),
                "tool_call_id": event.tool_call_id,
                "replayable": event.replayable,
            }
        )
    try:
        validated = _JOURNAL_ADAPTER.validate_python(entries)
    except (ValidationError, TypeError, ValueError):
        raise ValueError("workspace journal failed validation") from None
    return cast(list[JournalEntryState], _JOURNAL_ADAPTER.dump_python(validated, mode="json"))


def _critic_rounds_from_workspace(
    workspace: ResearchWorkspace,
) -> list[CriticRoundState]:
    payload: list[dict[str, object]] = []
    for review_round in workspace.critic_rounds:
        if len(review_round.events) != len(review_round.event_steps):
            raise ValueError("Critic event and step histories do not match")
        journal = [
            {
                "step": step,
                "tool_name": event.tool_name,
                "arguments": copy_plain_json(event.normalized_arguments),
                "observation": event.observation.model_dump(mode="json"),
                "tool_call_id": event.tool_call_id,
                "replayable": event.replayable,
            }
            for event, step in zip(
                review_round.events,
                review_round.event_steps,
                strict=True,
            )
        ]
        payload.append(
            {
                "draft": review_round.draft.model_dump(mode="json"),
                "agent_steps": review_round.agent_steps,
                "tool_calls": review_round.tool_calls,
                "journal": journal,
                "review": (
                    None
                    if review_round.review is None
                    else review_round.review.model_dump(mode="json")
                ),
            }
        )
    try:
        validated = _CRITIC_ROUNDS_ADAPTER.validate_python(payload)
    except (ValidationError, TypeError, ValueError):
        raise ValueError("Critic round State failed validation") from None
    return cast(
        list[CriticRoundState],
        _CRITIC_ROUNDS_ADAPTER.dump_python(validated, mode="json"),
    )


def _critic_rounds_from_dto(
    rounds: Sequence[_CriticRoundDTO],
) -> list[CriticRoundRuntime]:
    restored: list[CriticRoundRuntime] = []
    for review_round in rounds:
        events, _trace = _history_from_entries(review_round.journal)
        restored.append(
            CriticRoundRuntime(
                draft=review_round.draft,
                agent_steps=review_round.agent_steps,
                tool_calls=review_round.tool_calls,
                events=events,
                event_steps=[entry.step for entry in review_round.journal],
                review=review_round.review,
            )
        )
    return restored


def _history_from_entries(
    entries: Sequence[_JournalEntryDTO],
) -> tuple[list[RuntimeEvent], list[ResearchTraceRecord]]:
    events: list[RuntimeEvent] = []
    trace: list[ResearchTraceRecord] = []
    for entry in entries:
        observation = entry.observation
        arguments = cast(dict[str, object], copy_plain_json(entry.arguments))
        events.append(
            RuntimeEvent(
                tool_name=entry.tool_name,
                normalized_arguments=arguments,
                observation=observation,
                tool_call_id=entry.tool_call_id,
                replayable=entry.replayable,
            )
        )
        trace.append(
            ResearchTraceRecord(
                step=entry.step,
                tool_name=entry.tool_name,
                arguments=arguments,
                observation=observation,
            )
        )
    return events, trace


def _terminal_result_to_state(result: ResearchResult) -> TerminalResultState:
    """Encode a terminal result without duplicating its confirmed goal or trace."""

    payload = result.model_dump(mode="json", exclude={"goal", "trace"})
    try:
        dto = _TerminalResultDTO.model_validate(payload)
    except (ValidationError, TypeError, ValueError):
        raise ValueError("terminal result snapshot failed validation") from None
    return cast(TerminalResultState, dto.model_dump(mode="json"))


def _result_from_dto(
    dto: _TerminalResultDTO,
    *,
    goal: ResearchGoal,
    trace: Sequence[ResearchTraceRecord],
) -> ResearchResult:
    payload = dto.model_dump(mode="python")
    try:
        return ResearchResult(goal=goal, trace=tuple(trace), **payload)
    except (ValidationError, TypeError, ValueError):
        raise ValueError("terminal result snapshot failed validation") from None


def copy_plain_json(value: object) -> object:
    """Return a detached JSON copy for checkpoint updates."""

    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("value must contain finite JSON-compatible values") from None


def _parse_graph_state(
    value: Mapping[str, object],
) -> tuple[_GraphStateDTO, GraphState]:
    if not isinstance(value, Mapping):
        raise ValueError("Graph State must be a mapping")
    try:
        dto = _GraphStateDTO.model_validate(value)
    except (ValidationError, TypeError, ValueError):
        raise ValueError("Graph State failed validation") from None
    normalized = cast(GraphState, dto.model_dump(mode="json"))
    return dto, normalized


def _candidate_to_state(config: StrategyConfig | None) -> dict[str, object] | None:
    if config is None:
        return None
    try:
        dto = _CandidateDTO(
            strategy_name=strategy_name(config),
            parameters=strategy_parameters(config),
        )
    except (ValidationError, TypeError, ValueError):
        raise ValueError("strategy candidate failed State validation") from None
    return cast(dict[str, object], dto.model_dump(mode="json"))


def _config_from_candidate(candidate: _CandidateDTO) -> StrategyConfig:
    return build_strategy_config(candidate.strategy_name, candidate.parameters)


def _cached_observations_from_history(
    events: Sequence[RuntimeEvent],
) -> dict[str, ToolObservation]:
    return {
        event.tool_name: event.observation
        for event in events
        if event.tool_name in {"get_market_data", "calculate_asset_statistics"}
        and event.observation.status == "success"
        and event.replayable
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "CriticRoundState",
    "GraphState",
    "JournalEntryState",
    "TaskStatus",
    "TerminalResultState",
    "aligned_data_to_snapshot",
    "copy_plain_json",
    "restore_workspace_from_state",
    "validate_graph_state",
    "workspace_to_state",
]
