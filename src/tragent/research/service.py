"""Application commands for durable Research Agent tasks run by background workers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import date, datetime
from typing import Any, Literal
from uuid import uuid4

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.types import Command, StateSnapshot
from langsmith import Client, tracing_context
from pydantic import BaseModel, ConfigDict, Field
from redis import Redis

from tragent import storage
from tragent.config import Settings
from tragent.db import DatabasePool
from tragent.quant import BacktestCosts, YFinanceProvider, latest_completed_us_market_session
from tragent.research import task_runs
from tragent.research.agent.events import AgentEvent
from tragent.research.agent.goal import initial_graph_state
from tragent.research.agent.graph import RecoverableResearchError, build_research_graph
from tragent.research.agent.models import AgentBudget
from tragent.research.agent.retrieval import MarketContextRetriever
from tragent.research.agent.state import GraphState, validate_graph_state
from tragent.research.agent.workspace import ResearchGraphContext, ResearchWorkspace
from tragent.research.llm import build_llm
from tragent.research.reporting import REPORT_VIEW_FILENAME, ResearchReportView, load_report_view
from tragent.research.task_view import ResearchTaskView, TaskStatus, build_task_view

# Hands one run to the background queue: thread ID, owner, and an optional HITL resume value.
type Enqueue = Callable[[str, int, dict[str, str] | None], None]
logger = logging.getLogger(__name__)

_REPORTABLE_STATUSES = frozenset(
    {"selected", "no_strategy", "critic_rejected", "failed", "budget_exhausted"}
)
_TERMINAL_STATUSES = _REPORTABLE_STATUSES | {"rejected"}
_REPORT_READ_ERROR = "research report could not be read"
_TASK_SUMMARY_NAMESPACE = "research_tasks"
_AGENT_NODE_NAMES = frozenset({"research_agent", "critic_agent"})
_TRY_TASK_LOCK = "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired"
_UNLOCK_TASK = "SELECT pg_advisory_unlock(hashtextextended(%s, 0))"


class ResearchApplicationError(ValueError):
    """A user-facing research command error."""


class TaskNotFoundError(ResearchApplicationError):
    """Raised when a research task is not found."""


class ReportNotFoundError(ResearchApplicationError):
    """Raised when a research report file is not found."""


class TaskBusyError(ResearchApplicationError):
    """Raised when another request is already advancing the same task."""


class TaskLimitError(ResearchApplicationError):
    """Raised when the user already has as many queued or running tasks as allowed."""


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


def start_research(
    request: str,
    *,
    user_id: int,
    settings: Settings,
    pool: DatabasePool,
    redis: Redis,
    enqueue: Enqueue,
    reference_date: date | None = None,
) -> ResearchTaskView:
    """Checkpoint a new task and queue its Goal Parser run."""

    reference_date = reference_date or latest_completed_us_market_session()
    thread_id = str(uuid4())
    initial_state = initial_graph_state(
        {
            "thread_id": thread_id,
            "user_id": user_id,
            "request": request,
            "reference_date": reference_date.isoformat(),
            "cost_defaults": BacktestCosts(
                transaction_cost_bps=settings.default_transaction_cost_bps,
                slippage_bps=settings.default_slippage_bps,
            ),
            "position_tranches_default": settings.default_position_tranches,
            "budget": AgentBudget(),
        }
    )
    # Take the user's slot first, so a full quota creates nothing.
    _claim(redis, thread_id, user_id=user_id, settings=settings)
    graph = build_research_graph(checkpointer=PostgresSaver(pool))
    try:
        # Persist the task before any model call so it is visible while it waits in the queue.
        graph.invoke(
            initial_state.model_dump(mode="json"),
            _graph_config(thread_id),
            interrupt_before=["parse_goal"],
            durability="sync",
        )
    except Exception:
        task_runs.finish_run(redis, thread_id, user_id=user_id)
        raise
    _queue(enqueue, redis, thread_id, user_id, None)
    state, view = _read_task(thread_id, user_id=user_id, pool=pool, redis=redis)
    _save_task_summary(pool, state, view)
    return view


def resume_research(
    thread_id: str,
    *,
    user_id: int,
    settings: Settings,
    pool: DatabasePool,
    redis: Redis,
    enqueue: Enqueue,
    action: str | None = None,
    answer: str | None = None,
) -> ResearchTaskView:
    """Validate a HITL decision or continuation and queue the run that applies it."""

    state, view = _read_task(thread_id, user_id=user_id, pool=pool, redis=redis)
    resume = _resume_value(state, action=action, answer=answer)
    if state.status in _TERMINAL_STATUSES:
        return view
    _claim(redis, thread_id, user_id=user_id, settings=settings)
    _queue(enqueue, redis, thread_id, user_id, resume)
    return _read_task(thread_id, user_id=user_id, pool=pool, redis=redis)[1]


def _claim(redis: Redis, thread_id: str, *, user_id: int, settings: Settings) -> None:
    limit = settings.max_active_runs_per_user
    claim = task_runs.claim_run(redis, thread_id, user_id=user_id, limit=limit)
    if claim == "busy":
        raise TaskBusyError("research task is already queued or running")
    if claim == "full":
        raise TaskLimitError(
            f"you already have {limit} research tasks queued or running; wait for one to finish"
        )


def _queue(
    enqueue: Enqueue,
    redis: Redis,
    thread_id: str,
    user_id: int,
    resume: dict[str, str] | None,
) -> None:
    try:
        enqueue(thread_id, user_id, resume)
    except Exception:
        # The run never reached the broker, so its claim must not hold the user's slot.
        task_runs.finish_run(redis, thread_id, user_id=user_id)
        raise


def run_research(
    thread_id: str,
    *,
    user_id: int,
    resume: dict[str, str] | None,
    settings: Settings,
    pool: DatabasePool,
    redis: Redis,
) -> None:
    """Advance one task to its next pause, publishing live Agent events for subscribers."""

    try:
        with _task_lock(pool, thread_id):
            try:
                _advance(
                    thread_id,
                    user_id=user_id,
                    resume=resume,
                    settings=settings,
                    pool=pool,
                    redis=redis,
                )
            finally:
                task_runs.finish_run(redis, thread_id, user_id=user_id)
                _save_task_summary(
                    pool, *_read_task(thread_id, user_id=user_id, pool=pool, redis=redis)
                )
    except ResearchApplicationError as error:
        # A redelivered or duplicated run whose task moved on; the current owner reports it.
        logger.warning("Skipped research run: thread_id=%s reason=%s", thread_id, error)


def _advance(
    thread_id: str,
    *,
    user_id: int,
    resume: dict[str, str] | None,
    settings: Settings,
    pool: DatabasePool,
    redis: Redis,
) -> None:
    task_runs.mark_running(redis, thread_id, user_id=user_id)
    config = _graph_config(thread_id)
    graph = build_research_graph(checkpointer=PostgresSaver(pool))
    state = _state_from_snapshot(
        graph.get_state(config), expected_thread_id=thread_id, expected_user_id=user_id
    )
    resume = _resume_value(state, **(resume or {}))
    if state.status in _TERMINAL_STATUSES:
        return
    with _research_tracing(settings, metadata=_trace_metadata(thread_id, state=state)):
        context = _build_graph_context(thread_id, settings=settings, pool=pool)
        try:
            for event in _stream_graph(
                graph,
                None if resume is None else Command(resume=resume),
                config,
                context=context,
            ):
                task_runs.publish(
                    redis, thread_id, "agent", event.model_dump_json(), user_id=user_id
                )
        except RecoverableResearchError:
            # The failed decision stays checkpointed; the task view offers a manual resume.
            pass


def get_research_status(
    thread_id: str,
    *,
    user_id: int,
    pool: DatabasePool,
    redis: Redis,
) -> ResearchTaskView:
    """Read one checkpoint and its run marker as a detached safe view."""

    return _read_task(thread_id, user_id=user_id, pool=pool, redis=redis)[1]


def _read_task(
    thread_id: str,
    *,
    user_id: int,
    pool: DatabasePool,
    redis: Redis,
) -> tuple[GraphState, ResearchTaskView]:
    checkpointer = PostgresSaver(pool)
    with tracing_context(enabled=False):
        snapshot = build_research_graph(checkpointer=checkpointer).get_state(
            _graph_config(thread_id)
        )
    state = _active_agent_state(
        snapshot,
        checkpointer,
        expected_thread_id=thread_id,
        expected_user_id=user_id,
    )
    view = build_task_view(
        snapshot,
        state,
        report_exists=_report_exists(state),
        run_state=task_runs.run_state(redis, thread_id),
    )
    return state, view


def list_research_tasks(
    *,
    user_id: int,
    pool: DatabasePool,
    limit: int = 20,
    offset: int = 0,
) -> tuple[ResearchTaskSummary, ...]:
    """Return ordered, pageable task cards from the application store."""

    store = PostgresStore(pool)
    items = store.search(
        (_TASK_SUMMARY_NAMESPACE, str(user_id)),
        limit=limit,
        offset=offset,
    )
    return tuple(
        ResearchTaskSummary.model_validate({**item.value, "updated_at": item.updated_at})
        for item in items
    )


def get_research_report_view(
    thread_id: str,
    *,
    user_id: int,
    pool: DatabasePool,
) -> ResearchReportView:
    """Return the validated semantic report artifact for one completed task."""

    key = _verified_report_key(thread_id, user_id=user_id, pool=pool)
    try:
        return load_report_view(key)
    except ValueError:
        raise ResearchApplicationError(_REPORT_READ_ERROR) from None


def _verified_report_key(
    thread_id: str,
    *,
    user_id: int,
    pool: DatabasePool,
) -> str:

    config = _graph_config(thread_id)
    with tracing_context(enabled=False):
        checkpointer = PostgresSaver(pool)
        graph = build_research_graph(checkpointer=checkpointer)
        snapshot = graph.get_state(config)
        state = _state_from_snapshot(
            snapshot,
            expected_thread_id=thread_id,
            expected_user_id=user_id,
        )

    if state.status not in _REPORTABLE_STATUSES:
        raise ResearchApplicationError("research report requires a reportable terminal task")
    key = f"{_artifacts(thread_id)}/{REPORT_VIEW_FILENAME}"
    if not storage.exists(key):
        raise ReportNotFoundError("research report file was not found")
    return key


@contextmanager
def _task_lock(pool: DatabasePool, thread_id: str) -> Iterator[None]:
    """Hold a PostgreSQL session lock so only one worker advances a task at a time."""

    # The lock lives as long as this connection, so a crashed process releases it too.
    with pool.connection() as connection:
        row = connection.execute(_TRY_TASK_LOCK, (thread_id,)).fetchone()
        if not row or not row["acquired"]:
            raise TaskBusyError("research task is already running")
        try:
            yield
        finally:
            connection.execute(_UNLOCK_TASK, (thread_id,))


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
    metadata["status"] = state.status
    workspace = state.workspace
    if workspace is None:
        return metadata
    metadata["phase"] = workspace.phase
    terminal = workspace.result
    if terminal is not None:
        for field_name in ("selected_fingerprint", "evaluated_fingerprint"):
            value = getattr(terminal, field_name)
            if value is not None:
                metadata[field_name] = value
    return metadata


def _build_graph_context(
    thread_id: str,
    *,
    settings: Settings,
    pool: DatabasePool,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        model=build_llm(settings),
        provider=YFinanceProvider(),
        artifacts=_artifacts(thread_id),
        retriever=MarketContextRetriever(pool, settings),
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
    with closing(parts):
        for part in parts:
            yield AgentEvent.model_validate(part["data"])


def _resume_value(
    state: GraphState,
    *,
    action: str | None = None,
    answer: str | None = None,
) -> dict[str, str] | None:
    """The HITL value a run resumes with; None continues a running task from its checkpoint."""

    status = state.status
    if status in {"awaiting_clarification", "awaiting_confirmation"}:
        if answer is not None:
            return {"answer": answer}
        allowed_actions = {"reject"}
        if status == "awaiting_confirmation":
            allowed_actions.add("approve")
        if action not in allowed_actions:
            raise ResearchApplicationError(
                "an awaiting research task requires an answer or valid confirmation action"
            )
        return {"action": action}

    if status == "running":
        if action is not None or answer is not None:
            raise ResearchApplicationError(
                "a running research task resumes without an action or answer"
            )
        return None

    if status in _TERMINAL_STATUSES:
        if action is not None or answer is not None:
            raise ResearchApplicationError(
                "a terminal research task does not accept an action or answer"
            )
        return None
    raise ResearchApplicationError("research task status is invalid")


def _save_task_summary(
    pool: DatabasePool,
    state: GraphState,
    view: ResearchTaskView,
) -> None:
    store = PostgresStore(pool)
    store.put(
        (_TASK_SUMMARY_NAMESPACE, str(state.user_id)),
        view.thread_id,
        {
            "thread_id": view.thread_id,
            "request_summary": " ".join(state.request.split())[:160],
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
    snapshot: StateSnapshot,
    *,
    expected_thread_id: str,
    expected_user_id: int,
) -> GraphState:
    values = snapshot.values
    # A thread that was never started has an empty checkpoint.
    if not values:
        raise TaskNotFoundError("research task was not found")
    try:
        state = validate_graph_state(values)
    except (TypeError, ValueError):
        raise ResearchApplicationError("research task checkpoint is invalid") from None
    if state.thread_id != expected_thread_id:
        raise ResearchApplicationError("research task checkpoint is invalid")
    if state.user_id != expected_user_id:
        raise TaskNotFoundError("research task was not found")
    return state


def _active_agent_state(
    snapshot: StateSnapshot,
    checkpointer: PostgresSaver,
    *,
    expected_thread_id: str,
    expected_user_id: int,
) -> GraphState:
    """Adopt the unfinished Agent node's own workspace, which is newer than the parent's.

    The ReAct harness is built inside its node, so ``get_state(subgraphs=True)`` cannot find
    it; its checkpoints are read directly from the node task's checkpoint namespace.
    """

    parent = _state_from_snapshot(
        snapshot,
        expected_thread_id=expected_thread_id,
        expected_user_id=expected_user_id,
    )
    for task in reversed(snapshot.tasks):
        if task.name not in _AGENT_NODE_NAMES:
            continue
        saved = checkpointer.get_tuple(
            {
                "configurable": {
                    "thread_id": expected_thread_id,
                    "checkpoint_ns": f"{task.name}:{task.id}",
                }
            }
        )
        workspace = None if saved is None else saved.checkpoint["channel_values"].get("workspace")
        if workspace is not None:
            return parent.with_workspace(ResearchWorkspace.model_validate(workspace))
    return parent


def _report_exists(state: GraphState) -> bool:
    if state.status not in _REPORTABLE_STATUSES:
        return False
    return storage.exists(f"{_artifacts(state.thread_id)}/{REPORT_VIEW_FILENAME}")


def _graph_config(thread_id: str) -> dict[str, object]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 96,
        "run_name": "research_workflow",
    }


def _artifacts(thread_id: str) -> str:
    return f"research/{thread_id}"


__all__ = [
    "Enqueue",
    "ReportNotFoundError",
    "ResearchApplicationError",
    "ResearchTaskSummary",
    "ResearchTaskView",
    "TaskBusyError",
    "TaskLimitError",
    "TaskNotFoundError",
    "get_research_report_view",
    "get_research_status",
    "list_research_tasks",
    "resume_research",
    "run_research",
    "start_research",
]
