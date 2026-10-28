"""Application-boundary tests for durable research task commands."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import fakeredis
import langsmith
import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command, PregelTask, StateSnapshot
from pydantic import SecretStr

import tragent.research.agent.finalization as finalization_module
import tragent.research.service as research_module
import tragent.research.task_view as task_view_module
from tragent.config import Settings
from tragent.db import DatabasePool
from tragent.quant import BacktestCosts
from tragent.research import task_runs
from tragent.research.agent.events import AgentEvent
from tragent.research.agent.graph import RecoverableResearchError
from tragent.research.agent.models import AgentBudget, GoalDraft, ResearchGoal
from tragent.research.agent.state import validate_graph_state
from tragent.research.service import (
    ResearchTaskSummary,
    ResearchTaskView,
    TaskBusyError,
    TaskLimitError,
    get_research_report_view,
    get_research_status,
    list_research_tasks,
    resume_research,
    run_research,
    start_research,
)
from tragent.research.task_view import build_task_view

TEST_POOL = cast(DatabasePool, object())


@pytest.fixture
def redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


class _Queue:
    """Records what the API hands to the Celery queue."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, int, dict[str, str] | None]] = []

    def __call__(self, thread_id: str, user_id: int, resume: dict[str, str] | None) -> None:
        self.runs.append((thread_id, user_id, resume))


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_api_key=SecretStr("test-llm-key"),
        llm_model="test-model",
        llm_base_url="https://llm.example/v1",
        embedding_api_key=SecretStr("test-embedding-key"),
        embedding_model="test-embedding-model",
        embedding_base_url="https://embeddings.example/v1",
        embedding_dimensions=1536,
        sec_user_agent="Test User test@example.test",
        langsmith_api_key=SecretStr("test-langsmith-key"),
        langsmith_project="test-project",
        postgres_host="localhost",
        postgres_port=5432,
        postgres_user="test",
        postgres_password=SecretStr("test"),
        postgres_database="test",
        redis_host="localhost",
        redis_port=6379,
        jwt_secret=SecretStr("test-jwt-secret-with-at-least-32-characters"),
        s3_access_key="test-s3-key",
        s3_secret_key=SecretStr("test-s3-secret"),
        default_transaction_cost_bps=7.5,
        default_slippage_bps=2.5,
    )


def _state_snapshot(
    values: Mapping[str, object],
    *,
    tasks: tuple[PregelTask, ...] = (),
    next_nodes: tuple[str, ...] = (),
) -> StateSnapshot:
    return StateSnapshot(
        values=values,
        next=next_nodes,
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=tasks,
        interrupts=(),
    )


@dataclass(slots=True)
class _StreamCall:
    value: object
    config: Mapping[str, object]
    context: object
    durability: str | None
    stream_mode: str | None
    version: str | None
    subgraphs: bool


class _CheckpointerSpy:
    def __init__(self) -> None:
        self.create_calls = 0
        self.setup_calls = 0
        # Agent subgraph checkpoints by namespace, as the ReAct harness writes them.
        self.agent_checkpoints: dict[str, dict[str, object]] = {}

    def get_tuple(self, config: Mapping[str, Any]) -> SimpleNamespace | None:
        values = self.agent_checkpoints.get(config["configurable"]["checkpoint_ns"])
        return None if values is None else SimpleNamespace(checkpoint={"channel_values": values})

    def setup(self) -> None:
        self.setup_calls += 1


@dataclass(slots=True)
class _StorePut:
    namespace: tuple[str, ...]
    key: str
    value: dict[str, object]
    index: bool | None


class _StoreSpy:
    def __init__(self) -> None:
        self.create_calls = 0
        self.put_calls: list[_StorePut] = []
        self.search_calls: list[dict[str, object]] = []
        self.items: list[object] = []

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, object],
        index: bool | None = None,
    ) -> None:
        self.put_calls.append(_StorePut(namespace, key, value, index))

    def search(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int,
        offset: int,
    ) -> list[object]:
        self.search_calls.append({"namespace": namespace, "limit": limit, "offset": offset})
        return self.items


class _FakeLangSmithClient:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def flush(self, timeout: float | None = None) -> None:
        del timeout

    def close(self, timeout: float | None = None) -> None:
        del timeout


@contextmanager
def _no_network_tracing_context(**_kwargs: object) -> Iterator[None]:
    yield


@pytest.fixture(autouse=True)
def no_network_langsmith(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    monkeypatch.setattr(langsmith, "Client", _FakeLangSmithClient)
    monkeypatch.setattr(langsmith, "tracing_context", _no_network_tracing_context)
    monkeypatch.setattr(research_module, "Client", _FakeLangSmithClient, raising=False)
    monkeypatch.setattr(research_module, "tracing_context", _no_network_tracing_context)
    yield


class _GraphSpy:
    """Only the compile/stream/get_state seam used by the application module."""

    def __init__(
        self,
        checkpointer: _CheckpointerSpy,
        snapshot: StateSnapshot,
        *,
        after_stream: Callable[[object], StateSnapshot] | None = None,
        events: tuple[AgentEvent, ...] = (),
    ) -> None:
        self.checkpointer = checkpointer
        self.snapshot = snapshot
        self.after_stream = after_stream
        self.events = events
        self.stream_calls: list[_StreamCall] = []
        self.invoke_calls: list[dict[str, object]] = []
        self.get_state_calls: list[Mapping[str, object]] = []

    def invoke(
        self,
        value: Mapping[str, object],
        config: Mapping[str, object],
        **kwargs: object,
    ) -> None:
        self.invoke_calls.append({"value": value, "config": config, **kwargs})
        self.snapshot = _state_snapshot(value, next_nodes=("parse_goal",))

    def stream(
        self,
        value: object,
        config: Mapping[str, object] | None = None,
        *,
        context: object = None,
        durability: str | None = None,
        stream_mode: str | None = None,
        version: str | None = None,
        subgraphs: bool = False,
        **_kwargs: object,
    ) -> Iterator[dict[str, object]]:
        assert config is not None
        self.stream_calls.append(
            _StreamCall(
                value=value,
                config=config,
                context=context,
                durability=durability,
                stream_mode=stream_mode,
                version=version,
                subgraphs=subgraphs,
            )
        )
        if self.after_stream is not None:
            self.snapshot = self.after_stream(value)
        for event in self.events:
            yield {
                "type": "custom",
                "ns": (),
                "data": event.model_dump(mode="json"),
            }

    def get_state(self, config: Mapping[str, object]) -> StateSnapshot:
        self.get_state_calls.append(config)
        return self.snapshot


class _NeverProvider:
    def get_daily_bars(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("the application seam must not download market data")


def _goal(
    request: str = "Research AAPL with the supported long-only strategies.",
) -> dict[str, object]:
    return ResearchGoal(
        objective=request,
        symbol="AAPL",
        start_date=date(2021, 7, 13),
        end_date=date(2026, 7, 13),
        allowed_strategies=("moving_average", "rsi", "macd"),
        max_drawdown_limit=0.2,
        benchmark_symbol="SPY",
        costs=BacktestCosts(transaction_cost_bps=7.5, slippage_bps=2.5),
    ).model_dump(mode="json")


def _goal_draft(request: str) -> dict[str, object]:
    goal = _goal(request)
    costs = goal["costs"]
    assert isinstance(costs, dict)
    return GoalDraft(
        symbol=goal["symbol"],
        bar_interval=goal["bar_interval"],
        start_date=goal["start_date"],
        end_date=goal["end_date"],
        allowed_strategies=goal["allowed_strategies"],
        max_drawdown_limit=goal["max_drawdown_limit"],
        position_tranches=goal["position_tranches"],
        transaction_cost_bps=costs["transaction_cost_bps"],
        slippage_bps=costs["slippage_bps"],
        benchmark_symbol=goal["benchmark_symbol"],
    ).model_dump(mode="json")


def _state(
    thread_id: str,
    *,
    status: str = "awaiting_confirmation",
    request: str = "Research AAPL with the supported long-only strategies.",
) -> dict[str, object]:
    researching = status == "running"
    terminal = status in {"selected", "no_strategy", "failed", "budget_exhausted"}
    workspace: dict[str, object] | None = None
    if researching or terminal:
        workspace = {
            "goal": _goal(request),
            "budget": AgentBudget().model_dump(mode="json"),
            "phase": "researching" if researching else "completed",
            "active_agent": "research" if researching else None,
        }
    if terminal:
        workspace["result"] = {
            "status": status,
            "conclusion": "The research run stopped safely without selecting a strategy.",
            "stop_reason": "The research run stopped safely.",
        }
    state: dict[str, object] = {
        "thread_id": thread_id,
        "user_id": 1,
        "request": request,
        "reference_date": "2026-07-13",
        "status": status,
        "cost_defaults": {
            "transaction_cost_bps": 7.5,
            "slippage_bps": 2.5,
        },
        "position_tranches_default": 5,
        "draft_goal": _goal_draft(request),
        "defaulted_fields": [
            "start_date",
            "end_date",
            "allowed_strategies",
            "max_drawdown_limit",
            "transaction_cost_bps",
            "slippage_bps",
            "benchmark_symbol",
        ],
        "budget": AgentBudget().model_dump(mode="json"),
        "workspace": workspace,
    }
    return validate_graph_state(state).model_dump(mode="json")


def _snapshot(
    state: Mapping[str, object],
    *,
    recoverable_error: Exception | None = None,
) -> StateSnapshot:
    tasks: tuple[PregelTask, ...] = ()
    next_nodes: tuple[str, ...] = ()
    if recoverable_error is not None:
        tasks = (
            PregelTask(
                id="recoverable-task",
                name="research_agent",
                path=(),
                error=recoverable_error,
            ),
        )
        next_nodes = ("research_agent",)
    return _state_snapshot(state, tasks=tasks, next_nodes=next_nodes)


def _install_storage(
    monkeypatch: pytest.MonkeyPatch,
    checkpointer: _CheckpointerSpy,
    graph: _GraphSpy,
    *,
    store: _StoreSpy | None = None,
) -> _StoreSpy:
    store = store or _StoreSpy()

    def create_checkpointer(pool: DatabasePool) -> _CheckpointerSpy:
        assert pool is TEST_POOL
        checkpointer.create_calls += 1
        return checkpointer

    def create_store(pool: DatabasePool) -> _StoreSpy:
        assert pool is TEST_POOL
        store.create_calls += 1
        return store

    def build_fake(*, checkpointer: object = None) -> _GraphSpy:
        assert checkpointer is graph.checkpointer
        return graph

    @contextmanager
    def unlocked(pool: DatabasePool, _thread_id: str) -> Iterator[None]:
        assert pool is TEST_POOL
        yield

    monkeypatch.setattr(research_module, "PostgresSaver", create_checkpointer)
    monkeypatch.setattr(research_module, "PostgresStore", create_store)
    monkeypatch.setattr(research_module, "build_research_graph", build_fake)
    monkeypatch.setattr(research_module, "_task_lock", unlocked)
    return store


def _install_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[BaseChatModel, _NeverProvider, list[Settings], list[None]]:
    model = MagicMock(spec=BaseChatModel)
    provider = _NeverProvider()
    model_calls: list[Settings] = []
    provider_calls: list[None] = []

    def build_model(settings: Settings) -> BaseChatModel:
        model_calls.append(settings)
        return model

    def build_provider() -> _NeverProvider:
        provider_calls.append(None)
        return provider

    monkeypatch.setattr(research_module, "build_llm", build_model)
    monkeypatch.setattr(research_module, "YFinanceProvider", build_provider)
    return model, provider, model_calls, provider_calls


def _install_tracing(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    @contextmanager
    def tracing_spy(**kwargs: object) -> Iterator[None]:
        calls.append(dict(kwargs))
        yield

    monkeypatch.setattr(research_module, "tracing_context", tracing_spy)
    return calls


def _thread_from_config(config: Mapping[str, object]) -> str:
    configurable = config.get("configurable")
    assert isinstance(configurable, Mapping)
    thread_id = configurable.get("thread_id")
    assert isinstance(thread_id, str)
    return thread_id


def _context_item(context: object, name: str) -> object:
    if isinstance(context, Mapping):
        return context[name]
    return getattr(context, name)


def test_start_checkpoints_the_task_and_queues_goal_parsing_without_model_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
    redis: fakeredis.FakeRedis,
) -> None:
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _state_snapshot({}))
    store = _install_storage(monkeypatch, checkpointer, graph)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("starting a task must leave model work to the worker")

    monkeypatch.setattr(research_module, "build_llm", forbidden)
    monkeypatch.setattr(research_module, "YFinanceProvider", forbidden)
    queue = _Queue()

    view = start_research(
        "Research AAPL daily strategies.",
        user_id=1,
        settings=settings,
        pool=TEST_POOL,
        redis=redis,
        enqueue=queue,
        reference_date=date(2026, 7, 13),
    )

    assert UUID(view.thread_id).version == 4
    assert view.status == "running"
    assert view.run_state == "queued"
    assert view.can_resume is False
    assert queue.runs == [(view.thread_id, 1, None)]
    (call,) = graph.invoke_calls
    assert call["interrupt_before"] == ["parse_goal"]
    assert call["durability"] == "sync"
    assert _thread_from_config(call["config"]) == view.thread_id
    assert call["value"]["cost_defaults"] == {"transaction_cost_bps": 7.5, "slippage_bps": 2.5}
    assert graph.stream_calls == []
    assert store.put_calls[0].value["status"] == "running"


def test_start_with_full_slots_creates_nothing(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    redis: fakeredis.FakeRedis,
) -> None:
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _state_snapshot({}))
    store = _install_storage(monkeypatch, checkpointer, graph)
    for index in range(settings.max_active_runs_per_user):
        task_runs.claim_run(redis, f"task-{index}", user_id=1, limit=2)
    queue = _Queue()

    with pytest.raises(TaskLimitError, match="already have 2 research tasks"):
        start_research(
            "Research AAPL daily strategies.",
            user_id=1,
            settings=settings,
            pool=TEST_POOL,
            redis=redis,
            enqueue=queue,
            reference_date=date(2026, 7, 13),
        )

    assert graph.invoke_calls == []
    assert queue.runs == []
    assert store.put_calls == []


def test_a_failed_enqueue_releases_the_claim(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    redis: fakeredis.FakeRedis,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _snapshot(_state(thread_id, status="running")))
    _install_storage(monkeypatch, checkpointer, graph)

    def broker_down(*_args: object) -> None:
        raise ConnectionError("broker unavailable")

    with pytest.raises(ConnectionError):
        resume_research(
            thread_id,
            user_id=1,
            settings=settings,
            pool=TEST_POOL,
            redis=redis,
            enqueue=broker_down,
        )

    assert task_runs.run_state(redis, thread_id) is None


@pytest.mark.parametrize(
    ("action", "answer", "resume_payload"),
    [
        ("approve", None, {"action": "approve"}),
        (
            None,
            "Research MSFT with the supported long-only strategies.",
            {"answer": "Research MSFT with the supported long-only strategies."},
        ),
        ("reject", None, {"action": "reject"}),
    ],
)
def test_resume_queues_only_valid_confirmation_actions_for_awaiting_task(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    tmp_path: Path,
    redis: fakeredis.FakeRedis,
    action: str | None,
    answer: str | None,
    resume_payload: dict[str, str],
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _snapshot(_state(thread_id)))
    _install_storage(monkeypatch, checkpointer, graph)
    queue = _Queue()

    view = resume_research(
        thread_id,
        user_id=1,
        settings=settings,
        pool=TEST_POOL,
        redis=redis,
        enqueue=queue,
        action=action,
        answer=answer,
    )

    assert queue.runs == [(thread_id, 1, resume_payload)]
    assert view.status == "awaiting_confirmation"
    assert view.run_state == "queued"
    assert graph.stream_calls == []
    with pytest.raises(TaskBusyError, match="already queued or running"):
        resume_research(
            thread_id,
            user_id=1,
            settings=settings,
            pool=TEST_POOL,
            redis=redis,
            enqueue=queue,
            action=action,
            answer=answer,
        )
    assert len(queue.runs) == 1


@pytest.mark.parametrize(
    ("status", "action", "answer"),
    [
        ("awaiting_confirmation", None, None),
        ("awaiting_confirmation", "invalid", None),
        ("running", "approve", None),
        ("running", None, "Research MSFT."),
    ],
)
def test_resume_rejects_invalid_action_request_and_phase_before_queueing(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    redis: fakeredis.FakeRedis,
    status: str,
    action: str | None,
    answer: str | None,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _snapshot(_state(thread_id, status=status)))
    _install_storage(monkeypatch, checkpointer, graph)
    queue = _Queue()

    with pytest.raises(ValueError, match="research task|action|answer|confirmation"):
        resume_research(
            thread_id,
            user_id=1,
            settings=settings,
            pool=TEST_POOL,
            redis=redis,
            enqueue=queue,
            action=action,
            answer=answer,
        )

    assert queue.runs == []
    assert task_runs.run_state(redis, thread_id) is None


class _LockConnection:
    def __init__(self, acquired: bool) -> None:
        self.acquired = acquired
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, sql: str, params: tuple[object, ...]) -> SimpleNamespace:
        self.statements.append((sql, params))
        return SimpleNamespace(fetchone=lambda: {"acquired": self.acquired})


class _LockPool:
    def __init__(self, acquired: bool) -> None:
        self.connection_spy = _LockConnection(acquired)

    @contextmanager
    def connection(self) -> Iterator[_LockConnection]:
        yield self.connection_spy


def test_task_lock_releases_after_failures_and_rejects_a_busy_task() -> None:
    thread_id = str(uuid4())
    pool = _LockPool(acquired=True)
    with pytest.raises(RuntimeError, match="interrupted"):
        with research_module._task_lock(cast(DatabasePool, pool), thread_id):
            raise RuntimeError("interrupted")
    statements = pool.connection_spy.statements
    assert [sql.split("(")[0] for sql, _ in statements] == [
        "SELECT pg_try_advisory_lock",
        "SELECT pg_advisory_unlock",
    ]
    assert all(params == (thread_id,) for _, params in statements)

    busy = _LockPool(acquired=False)
    with pytest.raises(TaskBusyError):
        with research_module._task_lock(cast(DatabasePool, busy), thread_id):
            raise AssertionError("a busy task must not run")
    assert len(busy.connection_spy.statements) == 1


def test_worker_streams_live_events_then_clears_the_run_and_refreshes_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
    redis: fakeredis.FakeRedis,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    event = AgentEvent(
        source="research_agent", kind="decision", message="Research Agent · Deciding..."
    )
    graph = _GraphSpy(
        checkpointer,
        _snapshot(_state(thread_id)),
        after_stream=lambda _value: _snapshot(_state(thread_id, status="rejected")),
        events=(event,),
    )
    store = _install_storage(monkeypatch, checkpointer, graph)
    _install_runtime_dependencies(monkeypatch)
    trace_calls = _install_tracing(monkeypatch)
    assert task_runs.claim_run(redis, thread_id, user_id=1, limit=2) == "claimed"
    subscriber = redis.pubsub(ignore_subscribe_messages=True)
    subscriber.subscribe(task_runs.events_channel(thread_id))

    run_research(
        thread_id,
        user_id=1,
        resume={"action": "reject"},
        settings=settings,
        pool=TEST_POOL,
        redis=redis,
    )

    (call,) = graph.stream_calls
    assert isinstance(call.value, Command)
    assert call.value.resume == {"action": "reject"}
    assert call.durability == "sync"
    assert call.config["recursion_limit"] == 96
    # One traced run, then an untraced read of the final checkpoint for the summary.
    assert [call["enabled"] for call in trace_calls] == [True, False]
    messages = (subscriber.get_message() for _ in range(3))
    published = [message["data"] for message in messages if message is not None]
    assert published == [f"agent\n{event.model_dump_json()}", "done\n{}"]
    assert task_runs.run_state(redis, thread_id) is None
    assert store.put_calls[-1].value["status"] == "rejected"


def test_running_task_continues_from_its_checkpoint_without_a_hitl_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
    redis: fakeredis.FakeRedis,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(
        checkpointer,
        _snapshot(_state(thread_id, status="running")),
        after_stream=lambda _value: _snapshot(_state(thread_id, status="failed")),
    )
    _install_storage(monkeypatch, checkpointer, graph)
    _install_runtime_dependencies(monkeypatch)
    _install_tracing(monkeypatch)
    queue = _Queue()

    resume_research(
        thread_id, user_id=1, settings=settings, pool=TEST_POOL, redis=redis, enqueue=queue
    )
    run_research(thread_id, user_id=1, resume=None, settings=settings, pool=TEST_POOL, redis=redis)

    assert queue.runs == [(thread_id, 1, None)]
    assert graph.stream_calls[0].value is None
    assert get_research_status(thread_id, user_id=1, pool=TEST_POOL, redis=redis).status == (
        "failed"
    )


def test_a_redelivered_run_skips_a_task_another_worker_holds(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    redis: fakeredis.FakeRedis,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _snapshot(_state(thread_id, status="running")))
    _install_storage(monkeypatch, checkpointer, graph)
    task_runs.mark_running(redis, thread_id, user_id=1)

    @contextmanager
    def busy(_pool: DatabasePool, _thread_id: str) -> Iterator[None]:
        raise TaskBusyError("research task is already running")
        yield  # pragma: no cover

    monkeypatch.setattr(research_module, "_task_lock", busy)

    run_research(thread_id, user_id=1, resume=None, settings=settings, pool=TEST_POOL, redis=redis)

    assert graph.stream_calls == []
    assert task_runs.run_state(redis, thread_id) == "running"


def test_goal_parser_failure_has_safe_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    thread_id = str(uuid4())
    state = _state(thread_id)
    state.update(
        {
            "status": "failed",
            "failure_stage": "goal_parsing",
            "failure_summary": "The research goal could not be parsed safely.",
            "draft_goal": None,
            "missing_goal_fields": [],
            "defaulted_fields": [],
        }
    )
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(
        checkpointer,
        _snapshot(validate_graph_state(state).model_dump(mode="json")),
    )
    _install_storage(monkeypatch, checkpointer, graph)
    _install_tracing(monkeypatch)

    view = get_research_status(thread_id, user_id=1, pool=TEST_POOL, redis=fakeredis.FakeRedis())

    assert view.safe_error == (
        "Goal Parser could not parse this request. Start a new task to try again."
    )


def test_terminal_resume_returns_the_view_without_queueing_a_run(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    redis: fakeredis.FakeRedis,
    object_store: dict[str, bytes],
) -> None:
    thread_id = str(uuid4())
    object_store[f"research/{thread_id}/report.json"] = b"{}"
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _snapshot(_state(thread_id, status="failed")))
    _install_storage(monkeypatch, checkpointer, graph)
    queue = _Queue()

    view = resume_research(
        thread_id, user_id=1, settings=settings, pool=TEST_POOL, redis=redis, enqueue=queue
    )

    assert view.status == "failed"
    assert view.report_exists is True
    assert queue.runs == []
    assert task_runs.run_state(redis, thread_id) is None


def test_status_is_pure_read_with_disabled_tracing_and_no_runtime_or_report_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _snapshot(_state(thread_id, status="running")))
    _install_storage(monkeypatch, checkpointer, graph)
    trace_calls = _install_tracing(monkeypatch)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("status must stay a read-only checkpoint operation")

    monkeypatch.setattr(research_module, "build_llm", forbidden)
    monkeypatch.setattr(research_module, "YFinanceProvider", forbidden)
    monkeypatch.setattr(finalization_module, "generate_report", forbidden)

    view = get_research_status(thread_id, user_id=1, pool=TEST_POOL, redis=fakeredis.FakeRedis())

    assert isinstance(view, ResearchTaskView)
    assert view.status == "running"
    assert graph.stream_calls == []
    assert len(graph.get_state_calls) == 1
    assert checkpointer.create_calls == 1
    assert checkpointer.setup_calls == 0
    assert trace_calls == [{"enabled": False}]
    assert not (tmp_path / "artifacts").exists()


def test_list_uses_store_pagination_and_returns_safe_updated_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = str(uuid4())
    updated_at = datetime(2026, 8, 5, 8, 30, tzinfo=UTC)
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _state_snapshot({}))
    store = _StoreSpy()
    store.items = [
        SimpleNamespace(
            key=thread_id,
            updated_at=updated_at,
            value={
                "thread_id": thread_id,
                "request_summary": "Research AAPL daily strategies.",
                "status": "running",
                "agent": "research",
                "last_action": "run_backtest",
                "critic_verdict": None,
                "can_resume": True,
                "report_exists": False,
            },
        )
    ]
    _install_storage(monkeypatch, checkpointer, graph, store=store)

    summaries = list_research_tasks(user_id=7, pool=TEST_POOL, limit=7, offset=3)

    assert summaries == (
        ResearchTaskSummary(
            thread_id=thread_id,
            request_summary="Research AAPL daily strategies.",
            status="running",
            agent="research",
            last_action="run_backtest",
            critic_verdict=None,
            updated_at=updated_at,
            can_resume=True,
            report_exists=False,
        ),
    )
    assert store.search_calls == [{"namespace": ("research_tasks", "7"), "limit": 7, "offset": 3}]
    assert checkpointer.create_calls == 0


def test_task_summary_stores_a_bounded_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = str(uuid4())
    unsafe_request = (
        "Research AAPL with sk-summary-secret and "
        "postgresql://user:database-secret@db.internal/research."
    )
    state = _state(thread_id, status="running", request=unsafe_request)
    snapshot = _snapshot(state)
    validated = validate_graph_state(state)
    view = build_task_view(snapshot, validated, report_exists=False, run_state=None)
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, snapshot)
    store = _install_storage(monkeypatch, checkpointer, graph)

    research_module._save_task_summary(TEST_POOL, validated, view)

    stored = store.put_calls[0].value
    assert set(stored) == {
        "thread_id",
        "request_summary",
        "status",
        "agent",
        "last_action",
        "critic_verdict",
        "can_resume",
        "report_exists",
    }
    assert stored["request_summary"] == unsafe_request


def test_status_returns_a_fixed_safe_view_without_raw_task_or_state_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    thread_id = str(uuid4())
    task_sentinel = "unsafe-task-error postgresql://user:secret@db/research"
    state_sentinel = "unsafe-state-failure sk-secret-value"
    parent_state = _state(thread_id, status="running")
    child_workspace = copy.deepcopy(parent_state["workspace"])
    child_workspace.update(
        {
            "agent_steps": 1,
            "tool_calls": 1,
            "fatal_reason": state_sentinel,
            "journal": [
                {
                    "step": 1,
                    "tool_name": "calculate_asset_statistics",
                    "arguments": {},
                    "observation": {
                        "tool_name": "calculate_asset_statistics",
                        "status": "rejected",
                        "summary": "Market data must be loaded first.",
                        "identifiers": {},
                        "metrics": {},
                        "warnings": [],
                    },
                }
            ],
        }
    )
    checkpointer = _CheckpointerSpy()
    checkpointer.agent_checkpoints["research_agent:active-research-task"] = {
        "messages": [],
        "workspace": child_workspace,
    }
    graph = _GraphSpy(
        checkpointer,
        _state_snapshot(
            parent_state,
            tasks=(
                PregelTask(
                    path=(),
                    name="research_agent",
                    id="active-research-task",
                    error=RuntimeError(task_sentinel),
                ),
            ),
            next_nodes=("research_agent",),
        ),
    )
    _install_storage(monkeypatch, checkpointer, graph)
    _install_tracing(monkeypatch)

    view = get_research_status(thread_id, user_id=1, pool=TEST_POOL, redis=fakeredis.FakeRedis())

    assert isinstance(view, ResearchTaskView)
    assert view.thread_id == thread_id
    assert view.status == "running"
    assert view.can_resume is True
    assert view.goal == {
        "objective": "Research AAPL with the supported long-only strategies.",
        "symbol": "AAPL",
        "bar_interval": "1d",
        "start_date": "2021-07-13",
        "end_date": "2026-07-13",
        "training_sessions": None,
        "test_sessions": None,
        "allowed_strategies": ("moving_average", "rsi", "macd"),
        "max_drawdown_limit": 0.2,
        "position_tranches": 5,
        "benchmark_symbol": "SPY",
        "request_issue": None,
        "transaction_cost_bps": 7.5,
        "slippage_bps": 2.5,
    }
    assert view.defaulted_fields == tuple(parent_state["defaulted_fields"])
    assert view.budget_used.model_dump() == {
        "agent_steps": 1,
        "tool_calls": 1,
        "training_backtests": 0,
    }
    assert view.budget_remaining.model_dump() == {
        "agent_steps": 39,
        "tool_calls": 29,
        "training_backtests": 11,
    }
    assert view.active_agent == "research"
    assert view.critic_round == 0
    assert view.critic_verdict is None
    assert view.critic_budget_used.model_dump() == {"agent_steps": 0, "tool_calls": 0}
    assert view.critic_budget_remaining.model_dump() == {"agent_steps": 0, "tool_calls": 0}
    assert view.last_agent == "research"
    assert view.last_action == "calculate_asset_statistics"
    assert view.last_observation == "Market data must be loaded first."
    assert view.selected_fingerprint is None
    assert view.stop_reason is None
    assert view.report_exists is False
    assert view.safe_error == "The research task can be resumed."
    serialized_view = repr(asdict(view))
    assert task_sentinel not in serialized_view
    assert state_sentinel not in serialized_view
    assert "postgresql://" not in serialized_view
    assert "sk-secret-value" not in serialized_view


def test_report_endpoint_does_not_generate_a_missing_report(
    monkeypatch: pytest.MonkeyPatch,
    object_store: dict[str, bytes],
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(
        checkpointer,
        _snapshot(_state(thread_id, status="failed")),
    )
    _install_storage(monkeypatch, checkpointer, graph)
    _install_tracing(monkeypatch)
    with pytest.raises(ValueError, match="report file was not found"):
        get_research_report_view(thread_id, user_id=1, pool=TEST_POOL)

    assert f"research/{thread_id}/report.json" not in object_store


def test_unknown_thread_is_not_treated_as_a_completed_task_or_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _state_snapshot({}))
    _install_storage(monkeypatch, checkpointer, graph)
    _install_tracing(monkeypatch)

    with pytest.raises(ValueError, match="not found"):
        get_research_status(thread_id, user_id=1, pool=TEST_POOL, redis=fakeredis.FakeRedis())

    assert graph.stream_calls == []
    assert checkpointer.setup_calls == 0


@pytest.mark.parametrize("operation", ["status", "resume", "report"])
def test_task_access_is_limited_to_its_owner(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    operation: str,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _snapshot(_state(thread_id)))
    _install_storage(monkeypatch, checkpointer, graph)

    redis = fakeredis.FakeRedis()
    with pytest.raises(ValueError, match="research task was not found"):
        if operation == "status":
            get_research_status(thread_id, user_id=2, pool=TEST_POOL, redis=redis)
        elif operation == "resume":
            resume_research(
                thread_id,
                user_id=2,
                settings=settings,
                pool=TEST_POOL,
                redis=redis,
                enqueue=_Queue(),
            )
        else:
            get_research_report_view(thread_id, user_id=2, pool=TEST_POOL)

    assert graph.stream_calls == []


def test_exhausted_model_retries_leave_a_resumable_task_with_a_safe_view(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
    redis: fakeredis.FakeRedis,
) -> None:
    thread_id = str(uuid4())
    unsafe_error = "unsafe-provider-error sk-recoverable-secret"
    state = _state(thread_id, status="running")
    checkpointer = _CheckpointerSpy()

    class RecoverableGraphSpy(_GraphSpy):
        def stream(self, *args: object, **kwargs: object) -> Iterator[dict[str, object]]:
            yield from super().stream(*args, **kwargs)
            try:
                raise RuntimeError(unsafe_error)
            except RuntimeError as exc:
                raise RecoverableResearchError() from exc

    graph = RecoverableGraphSpy(
        checkpointer,
        _snapshot(state),
        after_stream=lambda _value: _snapshot(state, recoverable_error=RuntimeError(unsafe_error)),
    )
    _install_storage(monkeypatch, checkpointer, graph)
    _install_runtime_dependencies(monkeypatch)
    _install_tracing(monkeypatch)

    run_research(thread_id, user_id=1, resume=None, settings=settings, pool=TEST_POOL, redis=redis)
    view = get_research_status(thread_id, user_id=1, pool=TEST_POOL, redis=redis)

    assert view.status == "running"
    assert view.run_state is None
    assert view.can_resume is True
    assert view.safe_error == "The research task can be resumed."
    serialized_view = repr(asdict(view))
    assert unsafe_error not in serialized_view
    assert "sk-recoverable-secret" not in serialized_view


@pytest.mark.parametrize(
    ("next_nodes", "has_active_agent_task", "has_task_error", "queued", "expected_can_resume"),
    [
        ((), False, False, False, False),
        ((), True, False, False, True),
        ((), False, True, False, True),
        ((), False, True, True, False),
    ],
)
def test_running_can_resume_is_derived_from_pending_work_or_task_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    redis: fakeredis.FakeRedis,
    next_nodes: tuple[str, ...],
    has_active_agent_task: bool,
    has_task_error: bool,
    queued: bool,
    expected_can_resume: bool,
) -> None:
    thread_id = str(uuid4())
    tasks: tuple[PregelTask, ...] = ()
    if has_active_agent_task:
        tasks = (PregelTask(id="critic-task", name="critic_agent", path=()),)
    elif has_task_error:
        tasks = (
            PregelTask(
                id="failed-task",
                name="research_agent",
                path=(),
                error=RuntimeError("unsafe-recoverable-detail"),
            ),
        )
    snapshot = _state_snapshot(
        _state(thread_id, status="running"),
        tasks=tasks,
        next_nodes=next_nodes,
    )
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, snapshot)
    _install_storage(monkeypatch, checkpointer, graph)
    if queued:
        task_runs.claim_run(redis, thread_id, user_id=1, limit=2)

    view = get_research_status(thread_id, user_id=1, pool=TEST_POOL, redis=redis)

    assert view.can_resume is expected_can_resume
    assert view.safe_error == (
        "The research task can be resumed." if has_task_error and not queued else None
    )


def test_clarification_guidance_explains_missing_agent_goal_fields() -> None:
    hints, example = task_view_module._clarification_guidance(
        "awaiting_clarification",
        ("training_sessions", "test_sessions"),
        {"symbol": "SNDK", "bar_interval": "1m", "request_issue": None},
    )

    assert hints == (
        "Complete in-sample trading days, at most 20 in total / "
        "样本内完整交易日数量，与样本外合计不超过 20，例如 2",
        "Complete out-of-sample trading days, at most 20 in total / "
        "样本外完整交易日数量，与样本内合计不超过 20，例如 1",
    )
    assert example == "2个样本内，1个样本外"
