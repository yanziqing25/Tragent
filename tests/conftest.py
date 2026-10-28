"""Shared pytest fixtures and utilities for Tragent tests."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from tragent import storage
from tragent.config import Settings
from tragent.trading import containers, messaging


def make_test_settings(**overrides: object) -> Settings:
    """Factory function for test settings."""
    values: dict[str, object] = {
        "llm_api_key": SecretStr("test-llm-key"),
        "llm_model": "test-model",
        "llm_base_url": "https://llm.example/v1",
        "embedding_api_key": SecretStr("test-embedding-key"),
        "embedding_model": "test-embedding-model",
        "embedding_base_url": "https://embeddings.example/v1",
        "embedding_dimensions": 1536,
        "sec_user_agent": "Test User test@example.test",
        "langsmith_api_key": SecretStr("test-langsmith-key"),
        "langsmith_project": "test-project",
        "postgres_host": "localhost",
        "postgres_port": 5432,
        "postgres_user": "test",
        "postgres_password": SecretStr("test"),
        "postgres_database": "test",
        "redis_host": "localhost",
        "redis_port": 6379,
        "jwt_secret": SecretStr("test-jwt-secret-with-at-least-32-characters"),
        "s3_access_key": "test-s3-key",
        "s3_secret_key": SecretStr("test-s3-secret"),
        "default_transaction_cost_bps": 7.5,
        "default_slippage_bps": 2.5,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.fixture
def settings() -> Settings:
    """Default test settings fixture."""
    return make_test_settings()


class FakeMessage:
    def __init__(self, key: str, value: bytes, topic: str = messaging.ORDER_EVENTS) -> None:
        self._key = key.encode()
        self._value = value
        self._topic = topic

    def topic(self) -> str:
        return self._topic

    def key(self) -> bytes:
        return self._key

    def value(self) -> bytes:
        return self._value

    def error(self) -> None:
        return None


class FakeProducer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, object]] = []

    def produce(self, topic: str, key: str, value: object) -> None:
        self.messages.append((topic, key, value))

    def flush(self, _timeout: float | None = None) -> int:
        return 0

    def poll(self, _timeout: float) -> int:
        return 0


class FakeConsumer:
    def __init__(self) -> None:
        self.messages: list[FakeMessage] = []
        self.committed: list[FakeMessage] = []
        self.closed = False

    def poll(self, timeout: float) -> FakeMessage | None:
        if self.messages:
            return self.messages.pop(0)
        time.sleep(min(timeout, 0.01))
        return None

    def commit(self, message: FakeMessage, asynchronous: bool) -> None:
        self.committed.append(message)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def kafka(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stand-ins for Kafka, so the API starts without a broker; tests can inspect and feed them."""

    fake = SimpleNamespace(producer=FakeProducer(), consumers=[])

    def consumer(*_args: object, **_kwargs: object) -> FakeConsumer:
        fake.consumers.append(FakeConsumer())
        return fake.consumers[-1]

    monkeypatch.setattr(messaging, "ensure_topics", lambda _settings: None)
    monkeypatch.setattr(messaging, "producer", lambda _settings: fake.producer)
    monkeypatch.setattr(messaging, "consumer", consumer)
    return fake


@pytest.fixture(autouse=True)
def object_store(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """An in-memory object store in place of SeaweedFS; tests read and seed it by key."""

    objects: dict[str, bytes] = {}

    def put(key: str, data: bytes, *_args: object, **_kwargs: object) -> None:
        objects[key] = data

    monkeypatch.setattr(storage, "put", put)
    monkeypatch.setattr(storage, "get", lambda key: objects[key])
    monkeypatch.setattr(storage, "exists", lambda key: key in objects)
    monkeypatch.setattr(storage, "download_url", lambda key, seconds=600: f"https://store/{key}")
    return objects


@pytest.fixture(autouse=True)
def no_gateway_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateways published on localhost, as in development, rather than read from a real .env."""

    monkeypatch.setattr(containers, "_network", lambda: None)
