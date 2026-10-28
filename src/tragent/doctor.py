"""Small live checks exposed by the FastAPI application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

from langgraph.checkpoint.postgres import PostgresSaver
from langsmith import Client

from tragent.config import Settings
from tragent.db import DatabasePool
from tragent.research.llm import build_llm
from tragent.research.semantic_store import build_embeddings
from tragent.worker import celery_app

_LLM_PROBE_REPLY = "TRADING_DOCTOR_OK"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class CheckFailure(Exception):
    """A failure whose message is safe and useful to show; other errors stay generic."""


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    duration: float
    detail: str


def _check(name: str, operation: Callable[[], str]) -> CheckResult:
    started = perf_counter()
    try:
        detail = operation()
        status = CheckStatus.PASS
    except CheckFailure as failure:
        detail = str(failure)
        status = CheckStatus.FAIL
    except Exception:
        detail = "Check failed."
        status = CheckStatus.FAIL
    return CheckResult(
        name=name,
        status=status,
        duration=perf_counter() - started,
        detail=detail,
    )


def check_postgres(pool: DatabasePool) -> CheckResult:
    def operation() -> str:
        checkpointer = PostgresSaver(pool)
        next(checkpointer.list(None, limit=1), None)
        with pool.connection() as connection:
            vector = connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
        if vector is None:
            raise RuntimeError("pgvector extension is not installed")
        return f"Checkpoint tables and pgvector {vector['extversion']} are ready."

    return _check("PostgreSQL", operation)


def check_langsmith(settings: Settings) -> CheckResult:
    def operation() -> str:
        client = Client(api_key=settings.langsmith_api_key.get_secret_value())
        try:
            next(iter(client.list_projects(limit=1)), None)
        finally:
            client.close()
        return "Authentication succeeded."

    return _check("LangSmith", operation)


def check_llm(settings: Settings) -> CheckResult:
    def operation() -> str:
        response = build_llm(settings).invoke(
            f"Reply with exactly {_LLM_PROBE_REPLY} and nothing else."
        )
        response_text = str(getattr(response, "text", "")).strip()
        if not response_text:
            raise ValueError

        metadata = getattr(response, "response_metadata", {}) or {}
        api = "Responses API" if metadata.get("object") == "response" else "Chat Completions API"
        model = metadata.get("model_name") or metadata.get("model") or settings.llm_model

        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            raise ValueError

        reply_status = "matched" if response_text == _LLM_PROBE_REPLY else "changed"
        return (
            f"Model: {model}; API: {api}; tokens: {input_tokens} input / "
            f"{output_tokens} output; instructions field: "
            f"{_instruction_field_status(response)}; probe reply {reply_status}."
        )

    return _check("LLM", operation)


def _instruction_field_status(response: object) -> str:
    metadata = getattr(response, "response_metadata", {}) or {}
    if "instructions" in metadata:
        value = metadata["instructions"]
        return "empty" if not value else f"present (length {len(str(value))})"
    return "not exposed"


def check_embedding(settings: Settings) -> CheckResult:
    def operation() -> str:
        vector = build_embeddings(settings).embed_query("Embedding connectivity check.")
        if len(vector) != settings.embedding_dimensions:
            raise ValueError
        return f"Received a {len(vector)}-dimensional vector."

    return _check("Embedding", operation)


def check_workers() -> CheckResult:
    """Ping the Celery workers through the Redis broker and read the waiting runs."""

    def operation() -> str:
        replies = celery_app.control.ping(timeout=1.0)
        with celery_app.connection_for_read() as connection:
            waiting = connection.default_channel.queue_declare(
                queue=celery_app.conf.task_default_queue
            ).message_count
        if not replies:
            raise CheckFailure(
                f"No worker replied; {waiting} runs are waiting. "
                "Start one with: celery -A tragent.worker worker"
            )
        return f"{len(replies)} worker(s) online; {waiting} runs waiting."

    return _check("Workers", operation)


def run_doctor(settings: Settings, pool: DatabasePool) -> tuple[CheckResult, ...]:
    return (
        check_postgres(pool),
        check_workers(),
        check_langsmith(settings),
        check_llm(settings),
        check_embedding(settings),
    )
