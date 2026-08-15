"""Focused REST and SSE contracts for the Research Agent Web boundary."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient
from pydantic import SecretStr

import trading.api.app as app_module
import trading.api.auth as auth_module
import trading.api.doctor as doctor_api
import trading.api.research as research_api
from trading.agent.events import AgentEvent
from trading.config import Settings
from trading.doctor import CheckResult, CheckStatus
from trading.reporting import ResearchReportView
from trading.research import ResearchStreamItem, ResearchTaskStarted, ResearchTaskSummary
from trading.research import ResearchTaskView as TaskView


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
    )


@pytest.fixture
def api_client(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> Iterator[tuple[TestClient, dict[str, str]]]:
    pool = SimpleNamespace(open=lambda **_kwargs: None, close=lambda: None)
    monkeypatch.setattr(app_module, "build_connection_pool", lambda _settings: pool)
    monkeypatch.setattr(app_module, "setup_persistence", lambda _settings: None)
    monkeypatch.setattr(auth_module, "authenticate_user", lambda *_args: 1)
    app = app_module.create_app(settings)

    with TestClient(app) as client:
        login = client.post(
            "/api/auth/login",
            json={"username": "researcher", "password": "correct-password"},
        )
        token = login.json()["access_token"]
        yield client, {"Authorization": f"Bearer {token}"}


def _task_view(thread_id: str) -> TaskView:
    return TaskView(
        thread_id=thread_id,
        status="awaiting_confirmation",
        can_resume=True,
        goal={"objective": "Research AAPL daily strategies.", "symbol": "AAPL"},
        missing_goal_fields=(),
        clarification_hints=(),
        clarification_example=None,
        defaulted_fields=("benchmark_symbol",),
        budget_used={"agent_steps": 1, "tool_calls": 0, "training_backtests": 0},
        budget_remaining={"agent_steps": 39, "tool_calls": 30, "training_backtests": 8},
        active_agent=None,
        critic_round=0,
        critic_verdict=None,
        critic_budget_used={"agent_steps": 0, "tool_calls": 0},
        critic_budget_remaining={"agent_steps": 0, "tool_calls": 0},
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
    )


def test_lifespan_owns_and_exposes_database_pool(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
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
        lambda supplied_pool: events.append(("setup", supplied_pool)),
    )

    app = app_module.create_app(settings)
    with TestClient(app):
        assert app.state.settings is settings
        assert app.state.database_pool is pool
        assert events == [("open", True, 30.0), ("setup", pool)]

    assert events == [("open", True, 30.0), ("setup", pool), "close"]


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/api/research", None),
        ("POST", "/api/research", {"request": "Research AAPL."}),
        ("GET", f"/api/research/{uuid4()}", None),
        ("POST", f"/api/research/{uuid4()}/resume", {}),
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
    event = AgentEvent(
        source="goal_parser",
        kind="status",
        message="Goal Parser · Parsing the research request...",
    )
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

    def start_stream(request: str, **kwargs: object) -> Iterator[ResearchStreamItem]:
        calls["start"] = (request, kwargs["user_id"])
        yield ResearchTaskStarted(thread_id=thread_id)
        yield event
        yield view

    def resume_stream(
        supplied_thread_id: str,
        **kwargs: object,
    ) -> Iterator[ResearchStreamItem]:
        calls["resume"] = (
            supplied_thread_id,
            kwargs["user_id"],
            kwargs["action"],
            kwargs["answer"],
        )
        yield event
        yield view

    def list_tasks(**kwargs: object) -> tuple[ResearchTaskSummary, ...]:
        calls["list_user_id"] = kwargs["user_id"]
        return (summary,)

    def get_status(supplied_thread_id: str, **kwargs: object) -> TaskView:
        calls["status"] = (supplied_thread_id, kwargs["user_id"])
        return view

    def get_report(supplied_thread_id: str, **kwargs: object) -> ResearchReportView:
        calls["report"] = (supplied_thread_id, kwargs["user_id"])
        return report_view

    monkeypatch.setattr(research_api, "stream_start_research", start_stream)
    monkeypatch.setattr(research_api, "stream_resume_research", resume_stream)
    monkeypatch.setattr(research_api, "list_research_tasks", list_tasks)
    monkeypatch.setattr(research_api, "get_research_status", get_status)
    monkeypatch.setattr(research_api, "get_research_report_view", get_report)

    listed = client.get("/api/research?limit=20&offset=0", headers=headers)
    started = client.post(
        "/api/research",
        headers=headers,
        json={"request": "Research AAPL daily strategies."},
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
        "start": ("Research AAPL daily strategies.", 1),
        "status": (thread_id, 1),
        "resume": (thread_id, 1, "approve", None),
        "report": (thread_id, 1),
    }

    assert started.headers["content-type"].startswith("text/event-stream")
    start_events = _sse_events(started.text)
    assert [name for name, _data in start_events] == ["thread", "agent", "result"]
    assert start_events[0][1] == {"thread_id": thread_id}
    assert start_events[1][1] == event.model_dump(mode="json")
    assert start_events[2][1] == jsonable_encoder(view)

    assert resumed.headers["content-type"].startswith("text/event-stream")
    assert [name for name, _data in _sse_events(resumed.text)] == ["agent", "result"]
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("application/json")
    assert report.headers["cache-control"] == "no-store"
    assert report.json()["symbol"] == "AAPL"
    assert report.json()["critic_review_status"] == "accepted"


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
    conflict = client.post(
        f"/api/research/{thread_id}/resume",
        headers=headers,
        json={"action": "approve", "answer": "Change the date range."},
    )

    assert invalid_uuid.status_code == 422
    assert conflict.status_code == 422


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
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers
