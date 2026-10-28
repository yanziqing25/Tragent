"""LangGraph's own research tables are created at startup."""

from __future__ import annotations

from typing import Any

from conftest import make_test_settings

import tragent.research.persistence as persistence_module
from tragent.config import Settings


def test_setup_persistence_runs_vector_store_migrations(
    monkeypatch: Any,
) -> None:
    events: list[tuple[str, object]] = []
    pool = object()
    settings = make_test_settings()

    class SetupSpy:
        def __init__(self, name: str) -> None:
            self.name = name

        def setup(self) -> None:
            events.append((self.name, pool))

    monkeypatch.setattr(
        persistence_module,
        "PostgresSaver",
        lambda supplied_pool: SetupSpy("checkpointer"),
    )

    def fake_build_store(supplied_pool: object, supplied_settings: Settings) -> SetupSpy:
        assert supplied_pool is pool
        assert supplied_settings is settings
        return SetupSpy("semantic_store")

    monkeypatch.setattr(persistence_module, "build_semantic_store", fake_build_store)

    persistence_module.setup_persistence(pool, settings)  # type: ignore[arg-type]

    assert events == [("checkpointer", pool), ("semantic_store", pool)]
