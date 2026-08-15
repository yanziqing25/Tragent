"""PostgreSQL connection-pool construction and borrowing contracts."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from pydantic import SecretStr

import trading.checkpoint as checkpoint_module
from trading.config import Settings


def _settings() -> Settings:
    return Settings(
        llm_api_key=SecretStr("test-llm-key"),
        llm_model="test-model",
        llm_base_url="https://llm.example/v1",
        langsmith_api_key=SecretStr("test-langsmith-key"),
        langsmith_project="test-project",
        postgres_host="db.internal",
        postgres_port=5433,
        postgres_user="trading",
        postgres_password=SecretStr("database-secret"),
        postgres_database="research",
        postgres_pool_min_size=2,
        postgres_pool_max_size=12,
        postgres_pool_timeout_seconds=15.0,
        jwt_secret=SecretStr("test-jwt-secret-with-at-least-32-characters"),
    )


def test_build_connection_pool_is_lazy_and_configured(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    class FakeConnectionPool:
        check_connection = object()

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(checkpoint_module, "ConnectionPool", FakeConnectionPool)

    pool = checkpoint_module.build_connection_pool(_settings())

    assert isinstance(pool, FakeConnectionPool)
    assert captured["min_size"] == 2
    assert captured["max_size"] == 12
    assert captured["timeout"] == 15.0
    assert captured["open"] is False
    assert captured["check"] is FakeConnectionPool.check_connection
    assert captured["name"] == "trading-postgres"
    assert captured["kwargs"] == {
        "host": "db.internal",
        "port": 5433,
        "user": "trading",
        "password": "database-secret",
        "dbname": "research",
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": checkpoint_module.dict_row,
    }


def test_open_connection_returns_borrowed_connection_to_pool() -> None:
    events: list[str] = []
    connection = object()

    class FakePool:
        @contextmanager
        def connection(self):
            events.append("borrow")
            try:
                yield connection
            finally:
                events.append("return")

    with checkpoint_module.open_connection(FakePool()) as borrowed:  # type: ignore[arg-type]
        assert borrowed is connection
        assert events == ["borrow"]

    assert events == ["borrow", "return"]
