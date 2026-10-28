"""Focused tests for the Web doctor checks."""

from contextlib import contextmanager
from types import SimpleNamespace

from conftest import make_test_settings

import tragent.doctor as doctor
from tragent.config import Settings
from tragent.doctor import CheckStatus


def settings() -> Settings:
    return make_test_settings()


def test_run_doctor_returns_the_service_checks(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "check_postgres", lambda _settings: "postgres")
    monkeypatch.setattr(doctor, "check_workers", lambda: "workers")
    monkeypatch.setattr(doctor, "check_langsmith", lambda _settings: "langsmith")
    monkeypatch.setattr(doctor, "check_llm", lambda _settings: "llm")
    monkeypatch.setattr(doctor, "check_embedding", lambda _settings: "embedding")

    assert doctor.run_doctor(settings(), object()) == (
        "postgres",
        "workers",
        "langsmith",
        "llm",
        "embedding",
    )


def test_postgres_check_requires_pgvector(monkeypatch) -> None:
    class FakeCheckpointer:
        def list(self, _config, *, limit: int):
            assert limit == 1
            return iter(())

    class FakeConnection:
        def execute(self, query: str):
            assert "pg_extension" in query
            return SimpleNamespace(fetchone=lambda: {"extversion": "0.8.6"})

    class FakePool:
        @contextmanager
        def connection(self):
            yield FakeConnection()

    monkeypatch.setattr(doctor, "PostgresSaver", lambda _pool: FakeCheckpointer())

    result = doctor.check_postgres(FakePool())

    assert result.status is CheckStatus.PASS
    assert result.detail == "Checkpoint tables and pgvector 0.8.6 are ready."


def _fake_celery(monkeypatch, replies: list[dict[str, object]], waiting: int) -> None:
    @contextmanager
    def connection_for_read():
        channel = SimpleNamespace(
            queue_declare=lambda queue: SimpleNamespace(queue=queue, message_count=waiting)
        )
        yield SimpleNamespace(default_channel=channel)

    monkeypatch.setattr(
        doctor,
        "celery_app",
        SimpleNamespace(
            control=SimpleNamespace(ping=lambda timeout: replies),
            connection_for_read=connection_for_read,
            conf=SimpleNamespace(task_default_queue="celery"),
        ),
    )


def test_worker_check_counts_online_workers_and_waiting_runs(monkeypatch) -> None:
    _fake_celery(monkeypatch, [{"celery@a": {"ok": "pong"}}, {"celery@b": {"ok": "pong"}}], 3)

    result = doctor.check_workers()

    assert result.status is CheckStatus.PASS
    assert result.detail == "2 worker(s) online; 3 runs waiting."


def test_worker_check_explains_how_to_start_a_missing_worker(monkeypatch) -> None:
    _fake_celery(monkeypatch, [], 1)

    result = doctor.check_workers()

    assert result.status is CheckStatus.FAIL
    assert result.detail == (
        "No worker replied; 1 runs are waiting. Start one with: celery -A tragent.worker worker"
    )


def test_llm_check_returns_a_generic_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "build_llm",
        lambda _settings: SimpleNamespace(
            invoke=lambda _prompt: (_ for _ in ()).throw(RuntimeError("secret detail"))
        ),
    )

    result = doctor.check_llm(settings())

    assert result.status is CheckStatus.FAIL
    assert result.detail == "Check failed."


def test_llm_check_reports_success(monkeypatch) -> None:
    response = SimpleNamespace(
        text="TRADING_DOCTOR_OK",
        response_metadata={"object": "response", "model": "test-model"},
        usage_metadata={"input_tokens": 2, "output_tokens": 1},
    )
    monkeypatch.setattr(
        doctor,
        "build_llm",
        lambda _settings: SimpleNamespace(invoke=lambda _prompt: response),
    )

    result = doctor.check_llm(settings())

    assert result.status is CheckStatus.PASS
    assert "Responses API" in result.detail


def test_embedding_check_verifies_vector_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "build_embeddings",
        lambda _settings: SimpleNamespace(embed_query=lambda _text: [0.0] * 1536),
    )

    result = doctor.check_embedding(settings())

    assert result.status is CheckStatus.PASS
    assert result.detail == "Received a 1536-dimensional vector."
