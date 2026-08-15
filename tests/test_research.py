"""Application-boundary tests for durable research task commands."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import langsmith
import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command
from pydantic import SecretStr

import trading.agent.finalization as finalization_module
import trading.research as research_module
import trading.task_view as task_view_module
from trading.agent.events import AgentEvent
from trading.agent.graph import RecoverableResearchError
from trading.agent.models import AgentBudget, GoalDraft, ResearchGoal
from trading.agent.state import GraphState, validate_graph_state
from trading.checkpoint import DatabasePool
from trading.config import Settings
from trading.quant import BacktestCosts
from trading.research import (
    ResearchTaskStarted,
    ResearchTaskSummary,
    ResearchTaskView,
    get_research_report_view,
    get_research_status,
    list_research_tasks,
    stream_resume_research,
    stream_start_research,
)

TEST_POOL = cast(DatabasePool, object())


def _last_view(items: Iterator[object]) -> ResearchTaskView:
    views = [item for item in items if isinstance(item, ResearchTaskView)]
    assert len(views) == 1
    return views[0]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_api_key=SecretStr("test-llm-key"),
        llm_model="test-model",
        llm_base_url="https://llm.example/v1",
        langsmith_api_key=SecretStr("test-langsmith-key"),
        langsmith_project="test-project",
        postgres_host="localhost",
        postgres_port=5432,
        postgres_user="test",
        postgres_password=SecretStr("test"),
        postgres_database="test",
        jwt_secret=SecretStr("test-jwt-secret-with-at-least-32-characters"),
        default_transaction_cost_bps=7.5,
        default_slippage_bps=2.5,
    )


@dataclass(slots=True)
class _Snapshot:
    values: Mapping[str, object]
    tasks: tuple[object, ...] = ()
    next: tuple[str, ...] = ()


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
        self.active = False
        self.open_calls = 0
        self.setup_calls = 0
        self.child_states: dict[str, GraphState] = {}

    def setup(self) -> None:
        assert self.active, "setup must run inside the checkpointer context"
        self.setup_calls += 1

    def get_tuple(self, config: Mapping[str, object]) -> object | None:
        assert self.active, "checkpoint reads must stay inside the storage context"
        configurable = config.get("configurable")
        if not isinstance(configurable, Mapping):
            return None
        namespace = configurable.get("checkpoint_ns")
        state = self.child_states.get(namespace) if isinstance(namespace, str) else None
        if state is None:
            return None
        return SimpleNamespace(
            checkpoint={"channel_values": {"workspace_state": copy.deepcopy(state)}}
        )


@dataclass(slots=True)
class _StorePut:
    namespace: tuple[str, ...]
    key: str
    value: dict[str, object]
    index: bool | None


class _StoreSpy:
    def __init__(self) -> None:
        self.active = False
        self.open_calls = 0
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
        assert self.active
        self.put_calls.append(_StorePut(namespace, key, value, index))

    def search(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int,
        offset: int,
    ) -> list[object]:
        assert self.active
        self.search_calls.append({"namespace": namespace, "limit": limit, "offset": offset})
        return self.items


class _FakeLangSmithClient:
    def __init__(
        self,
        runtime: _TracingRuntime,
        identifier: int,
        kwargs: Mapping[str, object],
    ) -> None:
        self.runtime = runtime
        self.identifier = identifier
        self.kwargs = dict(kwargs)
        self.flush_calls = 0
        self.close_calls = 0
        self.flush_timeouts: list[float | None] = []
        self.close_timeouts: list[float | None] = []
        self.closed = False

    def flush(self, timeout: float | None = None) -> None:
        assert not self.closed
        assert self not in self.runtime.active_clients
        self.flush_calls += 1
        self.flush_timeouts.append(timeout)
        self.runtime.events.append(("flush", self.identifier))

    def close(self, timeout: float | None = None) -> None:
        assert not self.closed
        assert self not in self.runtime.active_clients
        self.close_calls += 1
        self.close_timeouts.append(timeout)
        self.closed = True
        self.runtime.events.append(("close", self.identifier))


class _TracingRuntime:
    """No-network LangSmith Client and tracing-context seam for application tests."""

    def __init__(self) -> None:
        self.clients: list[_FakeLangSmithClient] = []
        self.context_calls: list[dict[str, object]] = []
        self.active_clients: list[_FakeLangSmithClient] = []
        self.events: list[tuple[str, int | None]] = []
        self.context_depth = 0

    def build_client(self, **kwargs: object) -> _FakeLangSmithClient:
        client = _FakeLangSmithClient(self, len(self.clients) + 1, kwargs)
        self.clients.append(client)
        self.events.append(("client", client.identifier))
        return client

    @contextmanager
    def tracing_context(self, **kwargs: object) -> Iterator[None]:
        captured = dict(kwargs)
        self.context_calls.append(captured)
        client = captured.get("client")
        if client is not None:
            assert isinstance(client, _FakeLangSmithClient)
            assert not client.closed
            self.active_clients.append(client)
            identifier: int | None = client.identifier
        else:
            identifier = None
        self.context_depth += 1
        self.events.append(("enter", identifier))
        try:
            yield
        finally:
            self.events.append(("exit", identifier))
            self.context_depth -= 1
            if isinstance(client, _FakeLangSmithClient):
                assert self.active_clients[-1] is client
                self.active_clients.pop()


@pytest.fixture(autouse=True)
def no_network_langsmith(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_TracingRuntime]:
    runtime = _TracingRuntime()
    monkeypatch.setattr(langsmith, "Client", runtime.build_client)
    monkeypatch.setattr(langsmith, "tracing_context", runtime.tracing_context)
    monkeypatch.setattr(research_module, "Client", runtime.build_client, raising=False)
    monkeypatch.setattr(research_module, "tracing_context", runtime.tracing_context)
    yield runtime
    assert runtime.context_depth == 0
    assert runtime.active_clients == []


class _GraphSpy:
    """Only the compile/stream/get_state seam used by the application module."""

    def __init__(
        self,
        checkpointer: _CheckpointerSpy,
        snapshot: _Snapshot,
        *,
        after_stream: Callable[[object], _Snapshot] | None = None,
        events: tuple[AgentEvent, ...] = (),
    ) -> None:
        self.checkpointer = checkpointer
        self.snapshot = snapshot
        self.after_stream = after_stream
        self.events = events
        self.stream_calls: list[_StreamCall] = []
        self.get_state_calls: list[Mapping[str, object]] = []

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
        assert self.checkpointer.active, "stream must run before the checkpointer closes"
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

    def get_state(self, config: Mapping[str, object]) -> _Snapshot:
        assert self.checkpointer.active, "get_state must run before the checkpointer closes"
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
) -> GraphState:
    goal = _goal(request)
    researching = status == "running"
    terminal = status in {"selected", "no_strategy", "failed", "budget_exhausted"}
    rejected = status == "rejected"
    terminal_result: dict[str, object] | None = None
    if terminal:
        terminal_result = {
            "status": status,
            "selected_fingerprint": None,
            "evaluated_fingerprint": None,
            "training_metrics": {},
            "test_metrics": {},
            "conclusion": "The research run stopped safely without selecting a strategy.",
            "stop_reason": "The research run stopped safely.",
            "warnings": [],
            "limitations": [],
            "budget_usage": {
                "agent_steps": 0,
                "tool_calls": 0,
                "training_backtests": 0,
            },
        }
    state: GraphState = {
        "thread_id": thread_id,
        "user_id": 1,
        "request": request,
        "reference_date": "2026-07-13",
        "status": status,  # type: ignore[typeddict-item]
        "failure_stage": None,
        "confirmation_error": None,
        "failure_summary": None,
        "cost_defaults": {
            "transaction_cost_bps": 7.5,
            "slippage_bps": 2.5,
        },
        "position_tranches_default": 5,
        "draft_goal": _goal_draft(request),
        "confirmed_goal": goal if researching or terminal else None,
        "missing_goal_fields": [],
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
        "phase": "researching" if researching else "completed" if terminal else None,
        "active_agent": "research" if researching else None,
        "critic_rounds": [],
        "agent_steps": 0,
        "tool_calls": 0,
        "training_backtests": 0,
        "download_failures": 0,
        "message_journal": [],
        "unresolved_issues": [],
        "runtime_warnings": [],
        "fatal_reason": None,
        "data_snapshot": None,
        "current_candidate": None,
        "validated_configs": {},
        "training_evaluations": [],
        "final_evaluations": [],
        "locked_fingerprint": None,
        "final_batch_completed": False,
        "terminal_result": terminal_result,  # type: ignore[typeddict-item]
    }
    if rejected:
        state["confirmed_goal"] = None
    return validate_graph_state(state)


def _snapshot(state: GraphState, *, recoverable_error: object | None = None) -> _Snapshot:
    tasks: tuple[object, ...] = ()
    next_nodes: tuple[str, ...] = ()
    if recoverable_error is not None:
        tasks = (SimpleNamespace(error=recoverable_error),)
        next_nodes = ("research_agent",)
    return _Snapshot(values=state, tasks=tasks, next=next_nodes)


def _install_storage(
    monkeypatch: pytest.MonkeyPatch,
    checkpointer: _CheckpointerSpy,
    graph: _GraphSpy,
    *,
    store: _StoreSpy | None = None,
) -> _StoreSpy:
    store = store or _StoreSpy()

    @contextmanager
    def open_fake(_settings: Settings) -> Iterator[_CheckpointerSpy]:
        assert not checkpointer.active
        checkpointer.open_calls += 1
        checkpointer.active = True
        try:
            yield checkpointer
        finally:
            checkpointer.active = False

    @contextmanager
    def open_store_fake(_settings: Settings) -> Iterator[_StoreSpy]:
        assert not checkpointer.active
        assert not store.active
        store.open_calls += 1
        store.active = True
        try:
            yield store
        finally:
            store.active = False

    def build_fake(*, checkpointer: object = None) -> _GraphSpy:
        assert graph.checkpointer.active, "compile must run inside the checkpointer context"
        assert checkpointer is graph.checkpointer
        return graph

    monkeypatch.setattr(research_module, "open_checkpointer", open_fake)
    monkeypatch.setattr(research_module, "open_store", open_store_fake)
    monkeypatch.setattr(research_module, "build_research_graph", build_fake)
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


def _install_tracing_spy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    @contextmanager
    def tracing_spy(**kwargs: object) -> Iterator[None]:
        calls.append(dict(kwargs))
        assert kwargs.get("enabled") is enabled
        yield

    monkeypatch.setattr(research_module, "tracing_context", tracing_spy)
    return calls


def _install_disabled_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    return _install_tracing_spy(monkeypatch, enabled=False)


def _install_enabled_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    return _install_tracing_spy(monkeypatch, enabled=True)


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


def test_start_streams_agent_events_with_sync_durability_and_limit_96(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
) -> None:
    checkpointer = _CheckpointerSpy()

    def after_start(value: object) -> _Snapshot:
        assert isinstance(value, Mapping)
        thread_id = value["thread_id"]
        assert isinstance(thread_id, str)
        return _snapshot(_state(thread_id, request=str(value["request"])))

    event = AgentEvent(
        source="goal_parser",
        kind="status",
        message="Goal Parser · Parsing the research request...",
    )
    graph = _GraphSpy(
        checkpointer,
        _Snapshot(values={}),
        after_stream=after_start,
        events=(event,),
    )
    store = _install_storage(monkeypatch, checkpointer, graph)
    model, provider, model_calls, provider_calls = _install_runtime_dependencies(monkeypatch)
    trace_calls = _install_enabled_tracing(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    items = tuple(
        stream_start_research(
            "Research AAPL with the supported long-only strategies.",
            user_id=1,
            settings=settings,
            pool=TEST_POOL,
            reference_date=date(2026, 7, 13),
        )
    )
    view = _last_view(iter(items))

    assert isinstance(view, ResearchTaskView)
    parsed_thread = UUID(view.thread_id)
    assert parsed_thread.version == 4
    assert str(parsed_thread) == view.thread_id
    assert checkpointer.open_calls == 1
    assert checkpointer.setup_calls == 0
    assert checkpointer.active is False
    assert model_calls == [settings]
    assert provider_calls == [None]
    assert len(trace_calls) == 1
    assert trace_calls[0]["enabled"] is True
    assert len(graph.stream_calls) == 1

    call = graph.stream_calls[0]
    assert call.durability == "sync"
    assert call.stream_mode == "custom"
    assert call.version == "v2"
    assert call.subgraphs is True
    assert call.config["recursion_limit"] == 96
    assert _thread_from_config(call.config) == view.thread_id
    assert isinstance(call.value, Mapping)
    assert call.value == {
        "thread_id": view.thread_id,
        "user_id": 1,
        "request": "Research AAPL with the supported long-only strategies.",
        "reference_date": "2026-07-13",
        "cost_defaults": {
            "transaction_cost_bps": 7.5,
            "slippage_bps": 2.5,
        },
        "position_tranches_default": 5,
        "budget": AgentBudget().model_dump(mode="json"),
    }
    assert _context_item(call.context, "model") is model
    assert _context_item(call.context, "provider") is provider
    assert _context_item(call.context, "output_dir") == (
        tmp_path / "artifacts" / "reports" / view.thread_id
    )
    assert not hasattr(call.context, "progress")
    assert event in items
    assert store.open_calls == 1
    assert len(store.put_calls) == 1
    stored = store.put_calls[0]
    assert stored.namespace == ("research_tasks", "1")
    assert stored.key == view.thread_id
    assert stored.index is False
    assert stored.value == {
        "thread_id": view.thread_id,
        "request_summary": "Research AAPL with the supported long-only strategies.",
        "status": "awaiting_confirmation",
        "agent": None,
        "last_action": None,
        "critic_verdict": None,
        "can_resume": False,
        "report_exists": False,
    }


def test_closing_start_stream_stops_before_terminal_state_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
) -> None:
    checkpointer = _CheckpointerSpy()

    def after_start(value: object) -> _Snapshot:
        assert isinstance(value, Mapping)
        return _snapshot(_state(str(value["thread_id"]), request=str(value["request"])))

    graph = _GraphSpy(
        checkpointer,
        _Snapshot(values={}),
        after_stream=after_start,
        events=(
            AgentEvent(
                source="goal_parser",
                kind="status",
                message="Goal Parser · Parsing the research request...",
            ),
        ),
    )
    store = _install_storage(monkeypatch, checkpointer, graph)
    _install_runtime_dependencies(monkeypatch)
    _install_enabled_tracing(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    items = stream_start_research(
        "Research AAPL daily strategies.", user_id=1, settings=settings, pool=TEST_POOL
    )
    started = next(items)
    items.close()

    assert isinstance(started, ResearchTaskStarted)
    assert checkpointer.active is False
    assert graph.get_state_calls == []
    assert store.put_calls == []


@pytest.mark.parametrize(
    ("action", "answer", "resume_payload", "result_status"),
    [
        ("approve", None, {"action": "approve"}, "running"),
        (
            None,
            "Research MSFT with the supported long-only strategies.",
            {
                "answer": "Research MSFT with the supported long-only strategies.",
            },
            "awaiting_confirmation",
        ),
        ("reject", None, {"action": "reject"}, "rejected"),
    ],
)
def test_resume_accepts_only_valid_confirmation_actions_for_awaiting_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
    action: str,
    answer: str | None,
    resume_payload: dict[str, str],
    result_status: str,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    initial = _snapshot(_state(thread_id))

    def after_resume(value: object) -> _Snapshot:
        assert isinstance(value, Command)
        assert value.resume == resume_payload
        next_request = answer or "Research AAPL with the supported long-only strategies."
        return _snapshot(_state(thread_id, status=result_status, request=next_request))

    graph = _GraphSpy(checkpointer, initial, after_stream=after_resume)
    store = _install_storage(monkeypatch, checkpointer, graph)
    _install_runtime_dependencies(monkeypatch)
    trace_calls = _install_enabled_tracing(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    view = _last_view(
        stream_resume_research(
            thread_id,
            user_id=1,
            settings=settings,
            pool=TEST_POOL,
            action=action,
            answer=answer,
        )
    )

    assert view.status == result_status
    assert len(graph.stream_calls) == 1
    call = graph.stream_calls[0]
    assert call.durability == "sync"
    assert call.config["recursion_limit"] == 96
    assert _thread_from_config(call.config) == thread_id
    assert checkpointer.setup_calls == 0
    assert checkpointer.active is False
    assert len(trace_calls) == 1
    assert trace_calls[0]["enabled"] is True
    assert len(store.put_calls) == 1
    assert store.put_calls[0].value["status"] == result_status


@pytest.mark.parametrize(
    ("status", "action", "answer"),
    [
        ("awaiting_confirmation", None, None),
        ("awaiting_confirmation", "approve", "unexpected request"),
        ("awaiting_confirmation", "reject", "unexpected request"),
        ("awaiting_confirmation", "invalid", None),
        ("running", "approve", None),
        ("running", None, "Research MSFT."),
    ],
)
def test_resume_rejects_invalid_action_request_and_phase_before_invoke(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    status: str,
    action: str | None,
    answer: str | None,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _snapshot(_state(thread_id, status=status)))
    _install_storage(monkeypatch, checkpointer, graph)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid resume must not construct runtime dependencies")

    monkeypatch.setattr(research_module, "build_llm", forbidden)
    monkeypatch.setattr(research_module, "YFinanceProvider", forbidden)

    with pytest.raises(ValueError, match="research task|action|answer|confirmation"):
        tuple(
            stream_resume_research(
                thread_id,
                user_id=1,
                settings=settings,
                pool=TEST_POOL,
                action=action,
                answer=answer,
            )
        )

    assert graph.stream_calls == []
    assert checkpointer.setup_calls == 0
    assert checkpointer.active is False


def test_running_task_resumes_with_empty_input_and_no_hitl_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
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
    _install_enabled_tracing(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    view = _last_view(
        stream_resume_research(thread_id, user_id=1, settings=settings, pool=TEST_POOL)
    )

    assert view.status == "failed"
    assert len(graph.stream_calls) == 1
    assert graph.stream_calls[0].value is None
    assert graph.stream_calls[0].durability == "sync"
    assert graph.stream_calls[0].config["recursion_limit"] == 96
    assert checkpointer.setup_calls == 0
    assert checkpointer.active is False


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
            "confirmed_goal": None,
            "missing_goal_fields": [],
            "defaulted_fields": [],
        }
    )
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _snapshot(validate_graph_state(state)))
    _install_storage(monkeypatch, checkpointer, graph)
    _install_disabled_tracing(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    view = get_research_status(thread_id, user_id=1, pool=TEST_POOL)

    assert view.safe_error == (
        "Goal Parser could not parse this request. Start a new task to try again."
    )


def test_terminal_resume_skips_agent_execution_and_refreshes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
) -> None:
    thread_id = str(uuid4())
    report_dir = tmp_path / "artifacts" / "reports" / thread_id
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text("{}", encoding="utf-8")
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(
        checkpointer,
        _snapshot(_state(thread_id, status="failed")),
    )
    store = _install_storage(monkeypatch, checkpointer, graph)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("terminal resume must not construct runtime dependencies")

    monkeypatch.setattr(research_module, "build_llm", forbidden)
    monkeypatch.setattr(research_module, "YFinanceProvider", forbidden)

    view = _last_view(
        stream_resume_research(thread_id, user_id=1, settings=settings, pool=TEST_POOL)
    )

    assert view.status == "failed"
    assert graph.stream_calls == []
    assert len(store.put_calls) == 1
    assert store.put_calls[0].value["status"] == "failed"
    assert store.put_calls[0].value["report_exists"] is True
    assert checkpointer.setup_calls == 0
    assert checkpointer.active is False


def test_status_is_pure_read_with_disabled_tracing_and_no_runtime_or_report_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _snapshot(_state(thread_id, status="running")))
    _install_storage(monkeypatch, checkpointer, graph)
    trace_calls = _install_disabled_tracing(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("status must stay a read-only checkpoint operation")

    monkeypatch.setattr(research_module, "build_llm", forbidden)
    monkeypatch.setattr(research_module, "YFinanceProvider", forbidden)
    monkeypatch.setattr(finalization_module, "generate_report", forbidden)

    view = get_research_status(thread_id, user_id=1, pool=TEST_POOL)

    assert isinstance(view, ResearchTaskView)
    assert view.status == "running"
    assert graph.stream_calls == []
    assert len(graph.get_state_calls) == 1
    assert checkpointer.open_calls == 1
    assert checkpointer.setup_calls == 0
    assert checkpointer.active is False
    assert trace_calls == [{"enabled": False}]
    assert not (tmp_path / "artifacts").exists()


def test_list_uses_store_pagination_and_returns_safe_updated_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = str(uuid4())
    updated_at = datetime(2026, 8, 5, 8, 30, tzinfo=UTC)
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _Snapshot(values={}))
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
    assert checkpointer.open_calls == 0


@pytest.mark.parametrize(("limit", "offset"), [(0, 0), (101, 0), (20, -1), (True, 0)])
def test_list_rejects_invalid_pagination_before_opening_store(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
    offset: int,
) -> None:
    def forbidden(_pool: object) -> object:
        raise AssertionError("invalid pagination must not open PostgreSQL")

    monkeypatch.setattr(research_module, "open_store", forbidden)

    with pytest.raises(ValueError, match="limit|offset"):
        list_research_tasks(user_id=1, pool=TEST_POOL, limit=limit, offset=offset)


def test_task_summary_stores_a_bounded_request(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    thread_id = str(uuid4())
    unsafe_request = (
        "Research AAPL with sk-summary-secret and "
        "postgresql://user:database-secret@db.internal/research."
    )
    state = _state(thread_id, status="running", request=unsafe_request)
    snapshot = _snapshot(state)
    view = research_module._task_view(
        snapshot,
        state=state,
    )
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, snapshot)
    store = _install_storage(monkeypatch, checkpointer, graph)

    research_module._save_task_summary(settings, state, view)

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
    child_state = copy.deepcopy(parent_state)
    child_state["failure_summary"] = state_sentinel
    child_state.update(
        {
            "agent_steps": 1,
            "tool_calls": 1,
            "message_journal": [
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
                    "tool_call_id": "call-1",
                    "replayable": True,
                }
            ],
        }
    )
    child_state = validate_graph_state(child_state)
    checkpointer = _CheckpointerSpy()
    task_id = "active-research-task"
    checkpointer.child_states[f"research_agent:{task_id}"] = child_state
    graph = _GraphSpy(
        checkpointer,
        _Snapshot(
            values=parent_state,
            tasks=(
                SimpleNamespace(
                    name="research_agent",
                    id=task_id,
                    error=RuntimeError(task_sentinel),
                ),
            ),
            next=("research_agent",),
        ),
    )
    _install_storage(monkeypatch, checkpointer, graph)
    _install_disabled_tracing(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    view = get_research_status(thread_id, user_id=1, pool=TEST_POOL)

    assert isinstance(view, ResearchTaskView)
    assert view.thread_id == thread_id
    assert view.status == "running"
    assert view.can_resume is True
    assert view.goal == child_state["confirmed_goal"]
    assert view.defaulted_fields == tuple(child_state["defaulted_fields"])
    assert view.budget_used == {
        "agent_steps": 1,
        "tool_calls": 1,
        "training_backtests": 0,
    }
    assert view.budget_remaining == {
        "agent_steps": 39,
        "tool_calls": 29,
        "training_backtests": 8,
    }
    assert view.active_agent == "research"
    assert view.critic_round == 0
    assert view.critic_verdict is None
    assert view.critic_budget_used == {"agent_steps": 0, "tool_calls": 0}
    assert view.critic_budget_remaining == {"agent_steps": 0, "tool_calls": 0}
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
    tmp_path: Path,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(
        checkpointer,
        _snapshot(_state(thread_id, status="failed")),
    )
    _install_storage(monkeypatch, checkpointer, graph)
    _install_disabled_tracing(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)
    report_dir = tmp_path / "artifacts" / "reports" / thread_id
    report_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="report file was not found"):
        get_research_report_view(thread_id, user_id=1, pool=TEST_POOL)

    assert not (report_dir / "report.json").exists()


def test_unknown_thread_is_not_treated_as_a_completed_task_or_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    thread_id = str(uuid4())
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, _Snapshot(values={}))
    _install_storage(monkeypatch, checkpointer, graph)
    _install_disabled_tracing(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="not found"):
        get_research_status(thread_id, user_id=1, pool=TEST_POOL)

    assert graph.stream_calls == []
    assert checkpointer.setup_calls == 0
    assert checkpointer.active is False


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

    with pytest.raises(ValueError, match="research task was not found"):
        if operation == "status":
            get_research_status(thread_id, user_id=2, pool=TEST_POOL)
        elif operation == "resume":
            tuple(
                stream_resume_research(
                    thread_id,
                    user_id=2,
                    settings=settings,
                    pool=TEST_POOL,
                )
            )
        else:
            get_research_report_view(thread_id, user_id=2, pool=TEST_POOL)

    assert graph.stream_calls == []


def test_recoverable_stream_error_reads_latest_state_before_closing_checkpointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
) -> None:
    thread_id = str(uuid4())
    unsafe_error = "unsafe-provider-error sk-recoverable-secret"
    state = _state(thread_id, status="running")
    state["failure_summary"] = "unsafe-checkpoint-detail postgresql://user:secret@db/task"
    checkpointer = _CheckpointerSpy()

    class RecoverableGraphSpy(_GraphSpy):
        def stream(self, *args: object, **kwargs: object) -> Iterator[dict[str, object]]:
            yield from super().stream(*args, **kwargs)
            try:
                raise RuntimeError(unsafe_error)
            except RuntimeError as exc:
                raise RecoverableResearchError(stage="model") from exc

    graph = RecoverableGraphSpy(
        checkpointer,
        _snapshot(state),
        after_stream=lambda _value: _snapshot(
            state,
            recoverable_error=RuntimeError(unsafe_error),
        ),
    )
    _install_storage(monkeypatch, checkpointer, graph)
    _install_runtime_dependencies(monkeypatch)
    _install_enabled_tracing(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    view = _last_view(
        stream_resume_research(thread_id, user_id=1, settings=settings, pool=TEST_POOL)
    )

    assert view.status == "running"
    assert view.can_resume is True
    assert view.safe_error == "The research task can be resumed."
    assert len(graph.stream_calls) == 1
    assert len(graph.get_state_calls) == 2
    assert checkpointer.open_calls == 1
    assert checkpointer.setup_calls == 0
    assert checkpointer.active is False
    serialized_view = repr(asdict(view))
    assert unsafe_error not in serialized_view
    assert "sk-recoverable-secret" not in serialized_view
    assert "postgresql://" not in serialized_view


@pytest.mark.parametrize("operation", ["status", "resume"])
def test_task_commands_reject_noncanonical_uuid4_before_opening_storage(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    operation: str,
) -> None:
    invalid_thread_id = "not-a-uuid"
    storage_calls: list[Settings] = []
    tracing_calls: list[dict[str, object]] = []

    @contextmanager
    def forbidden_storage(supplied: Settings) -> Iterator[None]:
        storage_calls.append(supplied)
        yield

    @contextmanager
    def tracing_spy(**kwargs: object) -> Iterator[None]:
        tracing_calls.append(dict(kwargs))
        yield

    monkeypatch.setattr(research_module, "open_checkpointer", forbidden_storage)
    monkeypatch.setattr(research_module, "tracing_context", tracing_spy)

    with pytest.raises(ValueError) as captured:
        if operation == "status":
            get_research_status(invalid_thread_id, user_id=1, pool=TEST_POOL)
        else:
            tuple(
                stream_resume_research(
                    invalid_thread_id,
                    user_id=1,
                    settings=settings,
                    pool=TEST_POOL,
                )
            )

    assert str(captured.value) == "research thread ID must be a canonical UUID4"
    assert storage_calls == []
    assert tracing_calls == []


def test_start_and_resume_use_independent_same_thread_clients_and_trace_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
    no_network_langsmith: _TracingRuntime,
) -> None:
    checkpointer = _CheckpointerSpy()
    started_thread: list[str] = []

    def after_stream(value: object) -> _Snapshot:
        if isinstance(value, Mapping):
            thread_id = value["thread_id"]
            assert isinstance(thread_id, str)
            started_thread[:] = [thread_id]
            return _snapshot(_state(thread_id))
        assert isinstance(value, Command)
        assert value.resume == {"action": "reject"}
        return _snapshot(_state(started_thread[0], status="rejected"))

    graph = _GraphSpy(
        checkpointer,
        _Snapshot(values={}),
        after_stream=after_stream,
    )
    _install_storage(monkeypatch, checkpointer, graph)
    _install_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    started = _last_view(
        stream_start_research(
            "Research AAPL with the supported long-only strategies.",
            user_id=1,
            settings=settings,
            pool=TEST_POOL,
            reference_date=date(2026, 7, 13),
        )
    )
    resumed = _last_view(
        stream_resume_research(
            started.thread_id,
            user_id=1,
            settings=settings,
            pool=TEST_POOL,
            action="reject",
        )
    )

    assert resumed.thread_id == started.thread_id
    assert resumed.status == "rejected"
    assert len(no_network_langsmith.clients) == 2
    first_client, second_client = no_network_langsmith.clients
    assert first_client is not second_client
    for client in (first_client, second_client):
        assert client.kwargs["api_key"] == settings.langsmith_api_key.get_secret_value()
        assert client.kwargs["omit_traced_runtime_info"] is True
        assert client.flush_calls == 1
        assert client.close_calls == 1
        assert client.flush_timeouts == [5.0]
        assert client.close_timeouts == [0.0]
        assert client.closed is True
        positions = {
            event: no_network_langsmith.events.index((event, client.identifier))
            for event in ("client", "enter", "exit", "flush", "close")
        }
        assert (
            positions["client"]
            < positions["enter"]
            < positions["exit"]
            < positions["flush"]
            < positions["close"]
        )

    assert len(no_network_langsmith.context_calls) == 2
    for context_call, client in zip(
        no_network_langsmith.context_calls,
        (first_client, second_client),
        strict=True,
    ):
        assert context_call["enabled"] is True
        assert context_call["client"] is client
        assert context_call["project_name"] == settings.langsmith_project
        metadata = context_call["metadata"]
        assert isinstance(metadata, Mapping)
        assert metadata["thread_id"] == started.thread_id
        assert context_call.get("parent") in {None, False}
        assert context_call.get("distributed_parent_id") is None

    assert len(graph.stream_calls) == 2
    assert all(call.config["run_name"] == "research_workflow" for call in graph.stream_calls)
    assert all(_thread_from_config(call.config) == started.thread_id for call in graph.stream_calls)
    assert checkpointer.open_calls == 2
    assert checkpointer.setup_calls == 0
    assert no_network_langsmith.context_depth == 0


def test_start_does_not_create_report_directories_before_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
) -> None:
    checkpointer = _CheckpointerSpy()

    def after_start(value: object) -> _Snapshot:
        assert isinstance(value, Mapping)
        return _snapshot(_state(str(value["thread_id"]), request=str(value["request"])))

    graph = _GraphSpy(checkpointer, _Snapshot(values={}), after_stream=after_start)
    _install_storage(monkeypatch, checkpointer, graph)
    _install_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    view = _last_view(
        stream_start_research(
            "Research AAPL with the supported long-only strategies.",
            user_id=1,
            settings=settings,
            pool=TEST_POOL,
            reference_date=date(2026, 7, 13),
        )
    )

    assert view.status == "awaiting_confirmation"
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("next_nodes", "has_active_agent_task", "has_task_error", "expected_can_resume"),
    [
        ((), False, False, False),
        ((), True, False, True),
        ((), False, True, True),
    ],
)
def test_running_can_resume_is_derived_from_pending_work_or_task_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    next_nodes: tuple[str, ...],
    has_active_agent_task: bool,
    has_task_error: bool,
    expected_can_resume: bool,
) -> None:
    thread_id = str(uuid4())
    tasks: tuple[object, ...] = ()
    if has_active_agent_task:
        tasks = (SimpleNamespace(name="critic_agent", result=None, error=None),)
    elif has_task_error:
        tasks = (SimpleNamespace(error=RuntimeError("unsafe-recoverable-detail")),)
    snapshot = _Snapshot(
        values=_state(thread_id, status="running"),
        tasks=tasks,
        next=next_nodes,
    )
    checkpointer = _CheckpointerSpy()
    graph = _GraphSpy(checkpointer, snapshot)
    _install_storage(monkeypatch, checkpointer, graph)
    monkeypatch.setattr(research_module, "PROJECT_ROOT", tmp_path)

    view = get_research_status(thread_id, user_id=1, pool=TEST_POOL)

    assert view.can_resume is expected_can_resume
    assert view.safe_error == ("The research task can be resumed." if has_task_error else None)


def test_clarification_guidance_explains_missing_agent_goal_fields() -> None:
    hints, example = task_view_module._clarification_guidance(
        "awaiting_clarification",
        ("training_sessions", "test_sessions"),
        {"symbol": "SNDK", "bar_interval": "1m", "request_issue": None},
    )

    assert hints == (
        "Complete in-sample trading days / 样本内完整交易日数量，例如 2",
        "Complete out-of-sample trading days / 样本外完整交易日数量，例如 1",
    )
    assert example == "2个样本内，1个样本外"
