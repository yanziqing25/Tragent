"""Thin application commands for one durable Research Agent task."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from langgraph.types import Command
from langsmith import Client, tracing_context
from pydantic import BaseModel, ConfigDict, Field

from trading.agent.events import AgentEvent
from trading.agent.graph import (
    RecoverableResearchError,
    ResearchGraphContext,
    build_research_graph,
)
from trading.agent.models import AgentBudget
from trading.agent.state import (
    GraphState,
    validate_graph_state,
)
from trading.checkpoint import DatabasePool, open_checkpointer, open_store
from trading.config import PROJECT_ROOT, Settings
from trading.llm import build_llm
from trading.quant import BacktestCosts, YFinanceProvider
from trading.reporting import REPORT_VIEW_FILENAME, ResearchReportView, load_report_view
from trading.task_view import ResearchTaskView, TaskStatus, build_task_view

_REPORTABLE_STATUSES = frozenset(
    {"selected", "no_strategy", "critic_rejected", "failed", "budget_exhausted"}
)
_TERMINAL_STATUSES = _REPORTABLE_STATUSES | {"rejected"}
_REPORT_READ_ERROR = "research report could not be read"
_TASK_SUMMARY_NAMESPACE = "research_tasks"
_AGENT_NODE_NAMES = frozenset({"research_agent", "critic_agent"})


class _ResearchApplicationError(ValueError):
    """A user-facing research command error."""


class ResearchTaskSummary(BaseModel):
    """Safe history row stored independently from the durable Graph State."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Durable task identifier used to open the task detail page.
    thread_id: str = Field(min_length=36, max_length=36)
    # Bounded one-line preview of the original user request.
    request_summary: str = Field(min_length=1, max_length=160)
    # Current public lifecycle status copied from the durable Graph State.
    status: TaskStatus
    # Agent that is active now or most recently performed an Action.
    agent: Literal["research", "critic"] | None = None
    # Most recent Tool name recorded in the durable journal.
    last_action: str | None = Field(default=None, min_length=1, max_length=64)
    # Latest Critic decision, when a review has been submitted.
    critic_verdict: Literal["accept", "revise", "reject"] | None = None
    # PostgreSQL Store timestamp used for history ordering and display.
    updated_at: datetime
    # Whether the API currently permits a resume command for this task.
    can_resume: bool
    # Whether the deterministic report JSON artifact is available.
    report_exists: bool


class ResearchTaskStarted(BaseModel):
    """The stable task identity emitted before user-facing Agent progress."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # New LangGraph thread ID sent in the project-defined SSE "thread" event.
    thread_id: str = Field(min_length=36, max_length=36)


# Internal business-event union converted by the FastAPI layer into named SSE events.
type ResearchStreamItem = AgentEvent | ResearchTaskStarted | ResearchTaskView


def stream_start_research(
    request: str,
    *,
    user_id: int,
    settings: Settings,
    pool: DatabasePool,
    reference_date: date | None = None,
) -> Iterator[ResearchStreamItem]:
    """Yield one task identity, its Agent events, and the durable task view."""

    normalized_request = _normalize_request(request)
    normalized_reference_date = _normalize_reference_date(reference_date)
    thread_id = str(uuid4())
    costs = BacktestCosts(
        transaction_cost_bps=settings.default_transaction_cost_bps,
        slippage_bps=settings.default_slippage_bps,
    )
    seed = {
        "thread_id": thread_id,
        "user_id": user_id,
        "request": normalized_request,
        "reference_date": normalized_reference_date.isoformat(),
        "cost_defaults": {
            "transaction_cost_bps": costs.transaction_cost_bps,
            "slippage_bps": costs.slippage_bps,
        },
        "position_tranches_default": settings.default_position_tranches,
        "budget": AgentBudget().model_dump(mode="json"),
    }
    config = _graph_config(thread_id)

    with _research_tracing(
        settings,
        metadata=_trace_metadata(thread_id),
    ):
        with open_checkpointer(pool) as checkpointer:
            graph = build_research_graph(checkpointer=checkpointer)
            context = _build_graph_context(
                thread_id,
                settings=settings,
            )
            with closing(
                _stream_graph(
                    graph,
                    seed,
                    config,
                    context=context,
                )
            ) as events:
                first_event = next(events, None)
                yield ResearchTaskStarted(thread_id=thread_id)
                if first_event is not None:
                    yield first_event
                yield from events
            snapshot = graph.get_state(config)
            state = _state_from_snapshot(
                snapshot,
                expected_thread_id=thread_id,
                expected_user_id=user_id,
            )
            view = _task_view(snapshot, state=state)
    _save_task_summary(pool, state, view)
    yield view


def stream_resume_research(
    thread_id: str,
    *,
    user_id: int,
    settings: Settings,
    pool: DatabasePool,
    action: str | None = None,
    answer: str | None = None,
) -> Iterator[ResearchStreamItem]:
    """Yield Agent events followed by the resumed durable task view."""

    normalized_thread_id = _canonical_thread_id(thread_id)
    config = _graph_config(normalized_thread_id)

    with open_checkpointer(pool) as checkpointer:
        graph = build_research_graph(checkpointer=checkpointer)
        snapshot = graph.get_state(config)
        state = _state_from_snapshot(
            snapshot,
            expected_thread_id=normalized_thread_id,
            expected_user_id=user_id,
        )
        invoke_input = _resume_input(state, action=action, answer=answer)
        if state["status"] in _TERMINAL_STATUSES:
            view = _task_view(snapshot, state=state)
        else:
            with _research_tracing(
                settings,
                metadata=_trace_metadata(normalized_thread_id, state=state),
            ):
                context = _build_graph_context(
                    normalized_thread_id,
                    settings=settings,
                )
                recoverable = False
                try:
                    yield from _stream_graph(
                        graph,
                        invoke_input,
                        config,
                        context=context,
                    )
                except RecoverableResearchError:
                    recoverable = True
                snapshot = graph.get_state(config)
                state = (
                    _active_agent_state(
                        snapshot,
                        checkpointer=checkpointer,
                        expected_thread_id=normalized_thread_id,
                        expected_user_id=user_id,
                    )
                    if recoverable
                    else _state_from_snapshot(
                        snapshot,
                        expected_thread_id=normalized_thread_id,
                        expected_user_id=user_id,
                    )
                )
                view = _task_view(
                    snapshot,
                    recoverable=recoverable,
                    state=state,
                )
    _save_task_summary(pool, state, view)
    yield view


def get_research_status(
    thread_id: str,
    *,
    user_id: int,
    pool: DatabasePool,
) -> ResearchTaskView:
    """Read one checkpoint as a detached safe view without runtime integrations."""

    normalized_thread_id = _canonical_thread_id(thread_id)
    config = _graph_config(normalized_thread_id)
    with tracing_context(enabled=False):
        with open_checkpointer(pool) as checkpointer:
            graph = build_research_graph(checkpointer=checkpointer)
            snapshot = graph.get_state(config)
            state = _active_agent_state(
                snapshot,
                checkpointer=checkpointer,
                expected_thread_id=normalized_thread_id,
                expected_user_id=user_id,
            )
            return _task_view(
                snapshot,
                state=state,
            )


def list_research_tasks(
    *,
    user_id: int,
    pool: DatabasePool,
    limit: int = 20,
    offset: int = 0,
) -> tuple[ResearchTaskSummary, ...]:
    """List safe task summaries without reading checkpoint internals."""
    if type(limit) is not int or not 1 <= limit <= 100:
        raise _ResearchApplicationError("research task list limit must be between 1 and 100")
    if type(offset) is not int or offset < 0:
        raise _ResearchApplicationError("research task list offset must be non-negative")

    with open_store(pool) as store:
        items = store.search(
            (_TASK_SUMMARY_NAMESPACE, str(user_id)),
            limit=limit,
            offset=offset,
        )
    summaries: list[ResearchTaskSummary] = []
    for item in items:
        value = item.value
        if not isinstance(value, Mapping):
            raise _ResearchApplicationError("research task summary is invalid")
        summary = ResearchTaskSummary.model_validate({**value, "updated_at": item.updated_at})
        if summary.thread_id != item.key or not _is_canonical_uuid4(summary.thread_id):
            raise _ResearchApplicationError("research task summary is invalid")
        summaries.append(summary)
    return tuple(summaries)


def get_research_report_view(
    thread_id: str,
    *,
    user_id: int,
    pool: DatabasePool,
) -> ResearchReportView:
    """Return deterministic structured report data for the Vue renderer."""
    report_view_path = _verified_report_path(
        thread_id,
        user_id=user_id,
        pool=pool,
    )
    try:
        return load_report_view(report_view_path)
    except (OSError, UnicodeError, ValueError):
        raise _ResearchApplicationError(_REPORT_READ_ERROR) from None


def _verified_report_path(
    thread_id: str,
    *,
    user_id: int,
    pool: DatabasePool,
) -> Path:

    normalized_thread_id = _canonical_thread_id(thread_id)
    config = _graph_config(normalized_thread_id)
    with tracing_context(enabled=False):
        with open_checkpointer(pool) as checkpointer:
            graph = build_research_graph(checkpointer=checkpointer)
            snapshot = graph.get_state(config)
            state = _state_from_snapshot(
                snapshot,
                expected_thread_id=normalized_thread_id,
                expected_user_id=user_id,
            )

    if state["status"] not in _REPORTABLE_STATUSES:
        raise _ResearchApplicationError("research report requires a reportable terminal task")
    report_path = _report_directory(normalized_thread_id) / REPORT_VIEW_FILENAME
    if not report_path.is_file():
        raise _ResearchApplicationError("research report file was not found")
    return report_path


@contextmanager
def _research_tracing(
    settings: Settings,
    *,
    metadata: Mapping[str, object],
) -> Iterator[None]:
    client = Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        omit_traced_runtime_info=True,
    )
    try:
        with tracing_context(
            enabled=True,
            client=client,
            project_name=settings.langsmith_project,
            metadata=dict(metadata),
            parent=False,
        ):
            yield
    finally:
        try:
            client.flush(timeout=5.0)
        finally:
            client.close(timeout=0.0)


def _trace_metadata(
    thread_id: str,
    *,
    state: GraphState | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {"thread_id": thread_id}
    if state is None:
        return metadata
    metadata["status"] = state["status"]
    if state["phase"] is not None:
        metadata["phase"] = state["phase"]
    terminal = state["terminal_result"]
    if terminal is not None:
        for field_name in ("selected_fingerprint", "evaluated_fingerprint"):
            value = terminal[field_name]
            if value is not None:
                metadata[field_name] = value
    return metadata


def _build_graph_context(
    thread_id: str,
    *,
    settings: Settings,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        model=build_llm(settings),
        provider=YFinanceProvider(),
        output_dir=_report_directory(thread_id),
    )


def _stream_graph(
    graph: Any,
    value: object,
    config: Mapping[str, object],
    *,
    context: ResearchGraphContext,
) -> Iterator[AgentEvent]:
    parts = graph.stream(
        value,
        config,
        context=context,
        durability="sync",
        stream_mode="custom",
        version="v2",
        subgraphs=True,
    )
    try:
        for part in parts:
            yield AgentEvent.model_validate(part["data"])
    finally:
        close = getattr(parts, "close", None)
        if callable(close):
            close()


def _resume_input(
    state: GraphState,
    *,
    action: str | None,
    answer: str | None,
) -> Command | None:
    status = state["status"]
    if status in {"awaiting_clarification", "awaiting_confirmation"}:
        if answer is not None:
            if action is not None:
                raise _ResearchApplicationError(
                    "a goal answer cannot be combined with a confirmation action"
                )
            return Command(resume={"answer": _normalize_request(answer)})
        allowed_actions = {"reject"}
        if status == "awaiting_confirmation":
            allowed_actions.add("approve")
        if action not in allowed_actions:
            raise _ResearchApplicationError(
                "an awaiting research task requires an answer or valid confirmation action"
            )
        return Command(resume={"action": action})

    if status == "running":
        if action is not None or answer is not None:
            raise _ResearchApplicationError(
                "a running research task resumes without an action or answer"
            )
        return None

    if status in _TERMINAL_STATUSES:
        if action is not None or answer is not None:
            raise _ResearchApplicationError(
                "a terminal research task does not accept an action or answer"
            )
        return None
    raise _ResearchApplicationError("research task status is invalid")


def _task_view(
    snapshot: object,
    *,
    recoverable: bool = False,
    state: GraphState,
) -> ResearchTaskView:
    return build_task_view(
        snapshot,
        state,
        report_exists=_report_exists(state),
        recoverable=recoverable,
    )


def _save_task_summary(
    pool: DatabasePool,
    state: GraphState,
    view: ResearchTaskView,
) -> None:
    with open_store(pool) as store:
        store.put(
            (_TASK_SUMMARY_NAMESPACE, str(state["user_id"])),
            view.thread_id,
            {
                "thread_id": view.thread_id,
                "request_summary": " ".join(state["request"].split())[:160],
                "status": view.status,
                "agent": view.active_agent or view.last_agent,
                "last_action": view.last_action,
                "critic_verdict": view.critic_verdict,
                "can_resume": view.can_resume,
                "report_exists": view.report_exists,
            },
            index=False,
        )


def _state_from_snapshot(
    snapshot: object,
    *,
    expected_thread_id: str,
    expected_user_id: int,
) -> GraphState:
    values = getattr(snapshot, "values", None)
    if not isinstance(values, Mapping) or not values:
        raise _ResearchApplicationError("research task was not found")
    try:
        state = validate_graph_state(values)
    except (TypeError, ValueError):
        raise _ResearchApplicationError("research task checkpoint is invalid") from None
    if state["thread_id"] != expected_thread_id:
        raise _ResearchApplicationError("research task checkpoint is invalid")
    if state["user_id"] != expected_user_id:
        raise _ResearchApplicationError("research task was not found")
    return state


def _active_agent_state(
    snapshot: object,
    *,
    checkpointer: object,
    expected_thread_id: str,
    expected_user_id: int,
) -> GraphState:
    """Read the active Agent's durable workspace when the parent state is stale."""

    parent = _state_from_snapshot(
        snapshot,
        expected_thread_id=expected_thread_id,
        expected_user_id=expected_user_id,
    )
    tasks = getattr(snapshot, "tasks", ())
    if not isinstance(tasks, tuple):
        return parent

    for task in reversed(tasks):
        task_name = getattr(task, "name", None)
        task_id = getattr(task, "id", None)
        if task_name not in _AGENT_NODE_NAMES or not isinstance(task_id, str):
            continue
        checkpoint_tuple = checkpointer.get_tuple(
            {
                "configurable": {
                    "thread_id": expected_thread_id,
                    "checkpoint_ns": f"{task_name}:{task_id}",
                }
            }
        )
        if checkpoint_tuple is None:
            continue
        channel_values = checkpoint_tuple.checkpoint["channel_values"]
        workspace_state = channel_values.get("workspace_state")
        if workspace_state is None:
            continue
        child = validate_graph_state(workspace_state)
        if child["thread_id"] != expected_thread_id:
            raise _ResearchApplicationError("research task checkpoint is invalid")
        if child["user_id"] != expected_user_id:
            raise _ResearchApplicationError("research task was not found")
        return child
    return parent


def _report_exists(state: GraphState) -> bool:
    if state["status"] not in _REPORTABLE_STATUSES:
        return False
    return (_report_directory(state["thread_id"]) / REPORT_VIEW_FILENAME).is_file()


def _is_canonical_uuid4(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


def _normalize_request(value: object) -> str:
    if type(value) is not str:
        raise _ResearchApplicationError("research request must be a string")
    normalized = value.strip()
    if not 1 <= len(normalized) <= 2_000:
        raise _ResearchApplicationError(
            "research request must contain between 1 and 2000 characters"
        )
    return normalized


def _normalize_reference_date(value: date | None) -> date:
    if value is None:
        return date.today()
    if type(value) is not date:
        raise _ResearchApplicationError("reference_date must be a date")
    return value


def _canonical_thread_id(value: object) -> str:
    if not _is_canonical_uuid4(value):
        raise _ResearchApplicationError("research thread ID must be a canonical UUID4")
    return cast(str, value)


def _graph_config(thread_id: str) -> dict[str, object]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 96,
        "run_name": "research_workflow",
    }


def _report_root() -> Path:
    return Path(PROJECT_ROOT) / "artifacts" / "reports"


def _report_directory(thread_id: str) -> Path:
    return _report_root() / thread_id


__all__ = [
    "ResearchStreamItem",
    "ResearchTaskSummary",
    "ResearchTaskStarted",
    "ResearchTaskView",
    "get_research_report_view",
    "get_research_status",
    "list_research_tasks",
    "stream_resume_research",
    "stream_start_research",
]
