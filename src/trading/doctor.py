"""Small live checks exposed by the FastAPI application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

from langsmith import Client

from trading.checkpoint import DatabasePool, open_checkpointer
from trading.config import Settings
from trading.llm import build_llm


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


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
        with open_checkpointer(pool) as checkpointer:
            next(checkpointer.list(None, limit=1), None)
        return "Checkpoint tables are ready."

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
        response = build_llm(settings).invoke("Reply with OK.")
        if not getattr(response, "content", None):
            raise ValueError
        return "Received a response."

    return _check("LLM", operation)


def run_doctor(settings: Settings, pool: DatabasePool) -> tuple[CheckResult, ...]:
    return (
        check_postgres(pool),
        check_langsmith(settings),
        check_llm(settings),
    )
