"""Celery broker configuration and the worker's shared run resources."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from conftest import make_test_settings
from pydantic import SecretStr

from tragent import worker
from tragent.worker import broker_url


def test_broker_url_is_built_from_the_redis_settings() -> None:
    assert broker_url(make_test_settings()) == "redis://localhost:6379/0"
    assert (
        broker_url(make_test_settings(redis_password=SecretStr("p@ss/word"), redis_database=2))
        == "redis://:p%40ss%2Fword@localhost:6379/2"
    )


def test_runs_starting_together_open_the_worker_resources_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[object] = []

    class SlowPool:
        def open(self, **_kwargs: object) -> None:
            time.sleep(0.05)

    def build_pool(_settings: object) -> SlowPool:
        opened.append(SlowPool())
        return opened[-1]

    monkeypatch.setattr(worker, "load_settings", make_test_settings)
    monkeypatch.setattr(worker, "build_connection_pool", build_pool)
    monkeypatch.setattr(worker.redis_client, "connect", lambda _settings: object())
    worker._open_resources.cache_clear()
    barrier = Barrier(8)

    def start_run(_index: int) -> object:
        barrier.wait()
        return worker._resources()

    with ThreadPoolExecutor(8) as executor:
        resources = set(map(id, executor.map(start_run, range(8))))

    worker._open_resources.cache_clear()
    assert len(opened) == 1
    assert len(resources) == 1
