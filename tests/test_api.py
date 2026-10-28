"""Focused REST and SSE contracts for the Research Agent Web boundary."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import fakeredis
import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient
from pydantic import SecretStr

import tragent.api.app as app_module
import tragent.api.auth as auth_module
import tragent.api.doctor as doctor_api
import tragent.research.api as research_api
from tragent import redis_client
from tragent.config import Settings
from tragent.doctor import CheckResult, CheckStatus
from tragent.research import task_runs
from tragent.research.agent.events import AgentEvent, CriticBudgetUsage
from tragent.research.agent.models import BudgetUsage
from tragent.research.reporting import ResearchReportView
from tragent.research.service import (
    ReportNotFoundError,
    ResearchApplicationError,
    ResearchTaskSummary,
    TaskBusyError,
    TaskLimitError,
    TaskNotFoundError,
)
from tragent.research.service import ResearchTaskView as TaskView
from tragent.worker import enqueue_research_run


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
        cors_origins=["https://trading.yzking.cn"],
    )


@pytest.fixture
def redis_server(monkeypatch: pytest.MonkeyPatch) -> fakeredis.FakeServer:
    server = fakeredis.FakeServer()
    monkeypatch.setattr(
        redis_client,
        "connect",
        lambda _settings: fakeredis.FakeRedis(server=server, decode_responses=True),
    )
    monkeypatch.setattr(
        redis_client,
        "connect_async",
        lambda _settings: fakeredis.FakeAsyncRedis(server=server, decode_responses=True),
    )
    return server


@pytest.fixture
def api_client(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    redis_server: fakeredis.FakeServer,
) -> Iterator[tuple[TestClient, dict[str, str]]]:
    pool = SimpleNamespace(open=lambda **_kwargs: None, close=lambda: None)
    monkeypatch.setattr(app_module, "build_connection_pool", lambda _settings: pool)
    monkeypatch.setattr(app_module, "setup_persistence", lambda *_args: None)
    monkeypatch.setattr(auth_module, "authenticate_user", lambda *_args: 1)
    app = app_module.create_app(settings)

    with TestClient(app, raise_server_exceptions=False) as client:
        login = client.post(
            "/api/auth/login",
            json={"username": "researcher", "password": "correct-password"},
        )
        token = login.json()["access_token"]
        yield client, {"Authorization": f"Bearer {token}"}


def _task_view(thread_id: str, *, run_state: task_runs.RunState | None = None) -> TaskView:
    return TaskView(
        thread_id=thread_id,
        status="awaiting_confirmation",
        run_state=run_state,
        can_resume=run_state is None,
        goal={
            "objective": "Research AAPL daily strategies.",
            "symbol": "AAPL",
            "bar_interval": None,
            "start_date": None,
            "end_date": None,
            "training_sessions": None,
            "test_sessions": None,
            "allowed_strategies": None,
            "max_drawdown_limit": None,
            "position_tranches": None,
            "benchmark_symbol": None,
            "request_issue": None,
            "transaction_cost_bps": None,
            "slippage_bps": None,
        },
        missing_goal_fields=(),
        clarification_hints=(),
        clarification_example=None,
        defaulted_fields=("benchmark_symbol",),
        budget_used=BudgetUsage(agent_steps=1, tool_calls=0, training_backtests=0),
        budget_remaining=BudgetUsage(agent_steps=39, tool_calls=30, training_backtests=11),
        active_agent=None,
        critic_round=0,
        critic_verdict=None,
        critic_budget_used=CriticBudgetUsage(agent_steps=0, tool_calls=0),
        critic_budget_remaining=CriticBudgetUsage(agent_steps=0, tool_calls=0),
        last_agent=None,
        last_action=None,
        last_observation=None,
        selected_fingerprint=None,
        stop_reason=None,
        report_exists=False,
        safe_error=None,
    )


def _sse_events(response_text: str) -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = []
    for block in response_text.replace("\r\n", "\n").split("\n\n"):
        fields = {}
        for line in block.splitlines():
            if ": " in line:
                name, value = line.split(": ", maxsplit=1)
                fields[name] = value
        if "event" in fields and "data" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def _report_view() -> ResearchReportView:
    return ResearchReportView(
        research_objective="Research AAPL daily strategies.",
        symbol="AAPL",
        benchmark_symbol="SPY",
        bar_interval="1d",
        data_start="2026-01-01T00:00:00",
        data_end="2026-06-30T00:00:00",
        data_source="yfinance",
        selected_candidate_id=None,
        stop_reason="No strategy met the constraints.",
        research_incomplete=False,
        references=(),
        candidates=(),
        initial_capital=10_000.0,
        warnings=(),
        limitations=(),
        critic_reviews=(),
        critic_review_status="accepted",
        market_context_summary=None,
        market_context_sources=(),
    )


def test_lifespan_owns_and_exposes_database_pool_and_redis(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    redis_server: fakeredis.FakeServer,
) -> None:
    events: list[object] = []

    class PoolSpy:
        def open(self, *, wait: bool, timeout: float) -> None:
            events.append(("open", wait, timeout))

        def close(self) -> None:
            events.append("close")

    pool = PoolSpy()
    monkeypatch.setattr(app_module, "build_connection_pool", lambda _settings: pool)
    monkeypatch.setattr(
        app_module,
        "setup_persistence",
        lambda supplied_pool, supplied_settings: events.append(
            ("setup", supplied_pool, supplied_settings)
        ),
    )

    app = app_module.create_app(settings)
    with TestClient(app):
        assert app.state.settings is settings
        assert app.state.database_pool is pool
        assert app.state.redis.ping()
        assert events == [("open", True, 30.0), ("setup", pool, settings)]

    assert events == [("open", True, 30.0), ("setup", pool, settings), "close"]


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/api/research", None),
        ("POST", "/api/research", {"request": "Research AAPL."}),
        ("GET", f"/api/research/{uuid4()}", None),
        ("POST", f"/api/research/{uuid4()}/resume", {}),
        ("GET", f"/api/research/{uuid4()}/events", None),
        ("GET", f"/api/research/{uuid4()}/report", None),
    ],
)
def test_all_research_endpoints_require_bearer_token(
    api_client: tuple[TestClient, dict[str, str]],
    method: str,
    path: str,
    body: dict[str, str] | None,
) -> None:
    client, _headers = api_client

    response = client.request(method, path, json=body)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_login_and_access_token_failures_have_distinct_messages(
    monkeypatch: pytest.MonkeyPatch,
    api_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, _headers = api_client
    invalid_token = client.get(
        "/api/research",
        headers={"Authorization": "Bearer invalid"},
    )
    monkeypatch.setattr(auth_module, "authenticate_user", lambda *_args: None)
    invalid_login = client.post(
        "/api/auth/login",
        json={"username": "researcher", "password": "wrong"},
    )

    assert invalid_token.json()["detail"] == "Invalid or expired access token"
    assert invalid_login.json()["detail"] == "Incorrect username or password"


def test_research_rest_and_sse_success_contracts(
    monkeypatch: pytest.MonkeyPatch,
    api_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, headers = api_client
    thread_id = str(uuid4())
    view = _task_view(thread_id)
    queued = _task_view(thread_id, run_state="queued")
    summary = ResearchTaskSummary(
        thread_id=thread_id,
        request_summary="Research AAPL daily strategies.",
        status="awaiting_confirmation",
        agent=None,
        last_action=None,
        critic_verdict=None,
        updated_at=datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
        can_resume=True,
        report_exists=False,
    )
    report_view = _report_view()
    calls: dict[str, object] = {}

    def start(request: str, **kwargs: object) -> TaskView:
        calls["start"] = (request, kwargs["user_id"], kwargs["enqueue"])
        return queued

    def resume(supplied_thread_id: str, **kwargs: object) -> TaskView:
        calls["resume"] = (
            supplied_thread_id,
            kwargs["user_id"],
            kwargs["enqueue"],
            kwargs["action"],
            kwargs["answer"],
        )
        return queued

    def list_tasks(**kwargs: object) -> tuple[ResearchTaskSummary, ...]:
        calls["list_user_id"] = kwargs["user_id"]
        return (summary,)

    def get_status(supplied_thread_id: str, **kwargs: object) -> TaskView:
        calls["status"] = (supplied_thread_id, kwargs["user_id"])
        return view

    def get_report(supplied_thread_id: str, **kwargs: object) -> ResearchReportView:
        calls["report"] = (supplied_thread_id, kwargs["user_id"])
        return report_view

    monkeypatch.setattr(research_api, "start_research", start)
    monkeypatch.setattr(research_api, "resume_research", resume)
    monkeypatch.setattr(research_api, "list_research_tasks", list_tasks)
    monkeypatch.setattr(research_api, "get_research_status", get_status)
    monkeypatch.setattr(research_api, "get_research_report_view", get_report)

    listed = client.get("/api/research?limit=20&offset=0", headers=headers)
    started = client.post(
        "/api/research",
        headers=headers,
        json={"request": "  Research AAPL daily strategies.  "},
    )
    status_response = client.get(f"/api/research/{thread_id}", headers=headers)
    resumed = client.post(
        f"/api/research/{thread_id}/resume",
        headers=headers,
        json={"action": "approve"},
    )
    report = client.get(f"/api/research/{thread_id}/report", headers=headers)

    assert listed.status_code == 200
    assert listed.json()[0]["thread_id"] == thread_id
    assert status_response.status_code == 200
    assert status_response.json()["goal"] == view.goal
    assert calls == {
        "list_user_id": 1,
        "start": ("Research AAPL daily strategies.", 1, enqueue_research_run),
        "status": (thread_id, 1),
        "resume": (thread_id, 1, enqueue_research_run, "approve", None),
        "report": (thread_id, 1),
    }

    assert started.status_code == 202
    assert started.json() == jsonable_encoder(queued)
    assert resumed.status_code == 202
    assert resumed.json()["run_state"] == "queued"
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("application/json")
    assert report.headers["cache-control"] == "no-store"
    assert report.json()["symbol"] == "AAPL"
    assert report.json()["critic_review_status"] == "accepted"


@pytest.mark.parametrize(
    ("command", "path", "error", "expected_status"),
    [
        ("list_research_tasks", "/api/research", ResearchApplicationError("Invalid request"), 400),
        (
            "get_research_status",
            f"/api/research/{uuid4()}",
            TaskNotFoundError("Task not found"),
            404,
        ),
        (
            "get_research_report_view",
            f"/api/research/{uuid4()}/report",
            ReportNotFoundError("Report not found"),
            404,
        ),
    ],
)
def test_research_rest_preserves_business_errors(
    monkeypatch: pytest.MonkeyPatch,
    api_client: tuple[TestClient, dict[str, str]],
    command: str,
    path: str,
    error: ResearchApplicationError,
    expected_status: int,
) -> None:
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(research_api, command, fail)
    client, headers = api_client

    response = client.get(path, headers=headers)

    assert response.status_code == expected_status
    assert response.json() == {"detail": str(error)}


@pytest.mark.parametrize(
    ("command", "path"),
    [
        ("list_research_tasks", "/api/research"),
        ("get_research_status", f"/api/research/{uuid4()}"),
        ("get_research_report_view", f"/api/research/{uuid4()}/report"),
    ],
)
@pytest.mark.parametrize("validation_error", [False, True])
def test_research_rest_does_not_treat_internal_errors_as_bad_requests(
    monkeypatch: pytest.MonkeyPatch,
    api_client: tuple[TestClient, dict[str, str]],
    command: str,
    path: str,
    validation_error: bool,
) -> None:
    def fail(*_args, **_kwargs):
        if validation_error:
            ResearchTaskSummary.model_validate({})
        raise ValueError("Internal data failure")

    monkeypatch.setattr(research_api, command, fail)
    client, headers = api_client

    response = client.get(path, headers=headers)

    assert response.status_code == 500
    assert response.text == "Internal Server Error"


def test_resuming_a_queued_or_running_task_is_a_conflict(
    monkeypatch: pytest.MonkeyPatch,
    api_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, headers = api_client

    def busy(*_args: object, **_kwargs: object) -> TaskView:
        raise TaskBusyError("research task is already queued or running")

    monkeypatch.setattr(research_api, "resume_research", busy)
    response = client.post(f"/api/research/{uuid4()}/resume", headers=headers, json={})

    assert response.status_code == 409
    assert response.json() == {"detail": "research task is already queued or running"}


def test_starting_beyond_the_users_slots_is_too_many_requests(
    monkeypatch: pytest.MonkeyPatch,
    api_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, headers = api_client

    def full(*_args: object, **_kwargs: object) -> TaskView:
        raise TaskLimitError("you already have 2 research tasks queued or running")

    monkeypatch.setattr(research_api, "start_research", full)
    response = client.post("/api/research", headers=headers, json={"request": "Research AAPL."})

    assert response.status_code == 429
    assert response.json() == {"detail": "you already have 2 research tasks queued or running"}


def test_events_relay_live_agent_events_until_the_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
    api_client: tuple[TestClient, dict[str, str]],
    redis_server: fakeredis.FakeServer,
) -> None:
    client, headers = api_client
    thread_id = str(uuid4())
    queued = _task_view(thread_id, run_state="running")
    finished = _task_view(thread_id)
    event = AgentEvent(
        source="research_agent",
        kind="decision",
        message="Research Agent · Deciding the next Action...",
    )
    worker = fakeredis.FakeRedis(server=redis_server, decode_responses=True)
    views = iter((queued, finished))

    def get_status(_thread_id: str, **_kwargs: object) -> TaskView:
        view = next(views)
        if view is queued:
            # The worker advances after the stream has subscribed and read the checkpoint.
            task_runs.publish(worker, thread_id, "agent", event.model_dump_json(), user_id=1)
            task_runs.finish_run(worker, thread_id, user_id=1)
        return view

    monkeypatch.setattr(research_api, "get_research_status", get_status)

    response = client.get(f"/api/research/{thread_id}/events", headers=headers)

    assert response.headers["content-type"].startswith("text/event-stream")
    assert _sse_events(response.text) == [
        ("result", jsonable_encoder(queued)),
        ("agent", event.model_dump(mode="json")),
        ("result", jsonable_encoder(finished)),
    ]


def test_events_report_user_facing_application_errors(
    monkeypatch: pytest.MonkeyPatch,
    api_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, headers = api_client

    def missing(*_args: object, **_kwargs: object) -> TaskView:
        raise TaskNotFoundError("research task was not found")

    monkeypatch.setattr(research_api, "get_research_status", missing)
    response = client.get(f"/api/research/{uuid4()}/events", headers=headers)

    assert response.status_code == 200
    assert _sse_events(response.text) == [("error", {"detail": "research task was not found"})]


def test_doctor_is_authenticated_and_returns_live_checks(
    monkeypatch: pytest.MonkeyPatch,
    api_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, headers = api_client
    monkeypatch.setattr(
        doctor_api,
        "run_doctor",
        lambda _settings, _pool: (
            CheckResult("PostgreSQL", CheckStatus.PASS, 0.01, "Checkpoint tables are ready."),
        ),
    )

    assert client.post("/api/doctor").status_code == 401
    response = client.post("/api/doctor", headers=headers)

    assert response.status_code == 200
    assert response.json()[0]["status"] == "PASS"


def test_research_api_rejects_invalid_uuid_and_conflicting_hitl_input(
    api_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, headers = api_client
    thread_id = str(uuid4())

    invalid_uuid = client.get("/api/research/not-a-uuid", headers=headers)
    invalid_limit = client.get("/api/research?limit=0", headers=headers)
    invalid_offset = client.get("/api/research?offset=-1", headers=headers)
    blank_request = client.post("/api/research", headers=headers, json={"request": "   "})
    conflict = client.post(
        f"/api/research/{thread_id}/resume",
        headers=headers,
        json={"action": "approve", "answer": "Change the date range."},
    )
    assert invalid_uuid.status_code == 422
    assert invalid_limit.status_code == 422
    assert invalid_offset.status_code == 422
    assert blank_request.status_code == 422
    assert conflict.status_code == 422
    too_long_answer = client.post(
        f"/api/research/{thread_id}/resume", headers=headers, json={"answer": "A" * 1_001}
    )
    assert too_long_answer.status_code == 422


def test_resume_request_allows_running_task_without_hitl_input() -> None:
    request = research_api.ResumeResearchRequest.model_validate({})

    assert request.action is None
    assert request.answer is None


def test_cors_allows_only_the_production_web_origin(
    api_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, _headers = api_client
    preflight_headers = {
        "Origin": "https://trading.yzking.cn",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    }

    allowed = client.options("/api/research", headers=preflight_headers)
    denied = client.options(
        "/api/research",
        headers={**preflight_headers, "Origin": "https://untrusted.example"},
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://trading.yzking.cn"
    # The refresh cookie rides along from an app on another subdomain.
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers
