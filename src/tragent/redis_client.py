"""Redis clients shared by every module: sessions, research runs, and broker status."""

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from tragent.config import Settings


def connect(settings: Settings) -> Redis:
    return Redis(**_connection_options(settings))


def connect_async(settings: Settings) -> AsyncRedis:
    return AsyncRedis(**_connection_options(settings))


def _connection_options(settings: Settings) -> dict[str, object]:
    password = settings.redis_password
    return {
        "host": settings.redis_host,
        "port": settings.redis_port,
        "password": None if password is None else password.get_secret_value(),
        "db": settings.redis_database,
        "decode_responses": True,
    }


__all__ = ["connect", "connect_async"]
