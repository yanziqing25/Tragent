"""Semantic PostgreSQL Store construction."""

from types import SimpleNamespace

import pytest

from tragent.research import semantic_store


def test_build_semantic_store_configures_vector_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_arguments: dict[str, object] = {}
    store_arguments: dict[str, object] = {}
    embedding = object()
    store = object()

    def fake_embeddings(**kwargs: object) -> object:
        embedding_arguments.update(kwargs)
        return embedding

    def fake_store(pool: object, *, index: dict[str, object]) -> object:
        store_arguments.update(pool=pool, index=index)
        return store

    settings = SimpleNamespace(
        embedding_api_key=SimpleNamespace(get_secret_value=lambda: "secret"),
        embedding_model="BAAI/bge-m3",
        embedding_base_url="https://api.siliconflow.cn/v1",
        embedding_dimensions=1024,
    )
    pool = object()
    monkeypatch.setattr(semantic_store, "OpenAIEmbeddings", fake_embeddings)
    monkeypatch.setattr(semantic_store, "PostgresStore", fake_store)

    result = semantic_store.build_semantic_store(pool, settings)  # type: ignore[arg-type]

    assert result is store
    assert embedding_arguments == {
        "api_key": "secret",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "BAAI/bge-m3",
        "check_embedding_ctx_length": False,
    }
    assert store_arguments == {
        "pool": pool,
        "index": {"dims": 1024, "embed": embedding, "fields": ["text"]},
    }
