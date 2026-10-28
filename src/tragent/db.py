"""The PostgreSQL connection pool shared by every module."""

from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from tragent.config import Settings

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


__all__ = [
    "DatabasePool",
    "build_connection_pool",
]
