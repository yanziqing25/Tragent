"""PostgreSQL connection-pool configuration."""

from __future__ import annotations

from typing import Any

from conftest import make_test_settings
from pydantic import SecretStr

import tragent.db as db_module
from tragent.config import Settings


def _settings() -> Settings:
    return make_test_settings(
        postgres_host="db.internal",
        postgres_port=5433,
        postgres_user="trading",
        postgres_password=SecretStr("database-secret"),
        postgres_database="research",
        redis_host="localhost",
        redis_port=6379,
        postgres_pool_min_size=2,
        postgres_pool_max_size=12,
        postgres_pool_timeout_seconds=15.0,
    )


def test_build_connection_pool_is_lazy_and_configured(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    class FakeConnectionPool:
        check_connection = object()

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(db_module, "ConnectionPool", FakeConnectionPool)

    pool = db_module.build_connection_pool(_settings())

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
        "row_factory": db_module.dict_row,
    }
