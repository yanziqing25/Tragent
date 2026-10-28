"""Configured LangGraph semantic store and embedding client."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import OpenAIEmbeddings
from langgraph.store.postgres import PostgresStore

from tragent.config import Settings

if TYPE_CHECKING:
    from tragent.db import DatabasePool


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.embedding_api_key.get_secret_value(),
        base_url=str(settings.embedding_base_url),
        model=settings.embedding_model,
        check_embedding_ctx_length=False,
    )


def build_semantic_store(pool: DatabasePool, settings: Settings) -> PostgresStore:
    return PostgresStore(
        pool,
        index={
            "dims": settings.embedding_dimensions,
            "embed": build_embeddings(settings),
            "fields": ["text"],
        },
    )


__all__ = ["build_embeddings", "build_semantic_store"]
