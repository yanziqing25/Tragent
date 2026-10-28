"""LangGraph's own PostgreSQL tables for research checkpoints and the semantic store."""

from langgraph.checkpoint.postgres import PostgresSaver

from tragent.config import Settings
from tragent.db import DatabasePool
from tragent.research.semantic_store import build_semantic_store


def setup_persistence(pool: DatabasePool, settings: Settings) -> None:
    """Run the official LangGraph PostgreSQL migrations once at application startup."""
    PostgresSaver(pool).setup()
    build_semantic_store(pool, settings).setup()


__all__ = ["setup_persistence"]
