"""LangGraph PostgreSQL persistence lifecycle management."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from trading.config import Settings

type DatabaseConnection = Connection[DictRow]
type DatabasePool = ConnectionPool[DatabaseConnection]


def build_connection_pool(settings: Settings) -> DatabasePool:
    """Create the application pool without opening connections at import time."""

    return ConnectionPool(
        kwargs={
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "user": settings.postgres_user,
            "password": settings.postgres_password.get_secret_value(),
            "dbname": settings.postgres_database,
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=settings.postgres_pool_min_size,
        max_size=settings.postgres_pool_max_size,
        timeout=settings.postgres_pool_timeout_seconds,
        open=False,
        check=ConnectionPool.check_connection,
        name="trading-postgres",
    )


@contextmanager
def open_connection(pool: DatabasePool) -> Iterator[Connection[Mapping[str, Any]]]:
    """Borrow one connection and return it to the application pool after use."""

    with pool.connection() as connection:
        yield connection


@contextmanager
def open_checkpointer(pool: DatabasePool) -> Iterator[PostgresSaver]:
    """Create a checkpointer that borrows connections from the application pool."""

    yield PostgresSaver(pool)


@contextmanager
def open_store(pool: DatabasePool) -> Iterator[PostgresStore]:
    """Create a Store that borrows connections from the application pool."""

    yield PostgresStore(pool)


def setup_persistence(pool: DatabasePool) -> None:
    """Run the official LangGraph PostgreSQL migrations once at application startup."""
    with open_checkpointer(pool) as checkpointer:
        checkpointer.setup()
    with open_store(pool) as store:
        store.setup()


__all__ = [
    "DatabasePool",
    "build_connection_pool",
    "open_checkpointer",
    "open_connection",
    "open_store",
    "setup_persistence",
]
