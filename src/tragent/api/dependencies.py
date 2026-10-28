"""Shared FastAPI dependencies backed by application lifespan state."""

from typing import Annotated

from confluent_kafka import Producer
from fastapi import Depends, Request
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from tragent.config import Settings
from tragent.db import DatabasePool


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database_pool(request: Request) -> DatabasePool:
    return request.app.state.database_pool


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


def get_async_redis(request: Request) -> AsyncRedis:
    return request.app.state.async_redis


def get_kafka_producer(request: Request) -> Producer:
    return request.app.state.kafka_producer


SettingsDependency = Annotated[Settings, Depends(get_settings)]
DatabasePoolDependency = Annotated[DatabasePool, Depends(get_database_pool)]
RedisDependency = Annotated[Redis, Depends(get_redis)]
AsyncRedisDependency = Annotated[AsyncRedis, Depends(get_async_redis)]
KafkaProducerDependency = Annotated[Producer, Depends(get_kafka_producer)]


__all__ = [
    "AsyncRedisDependency",
    "DatabasePoolDependency",
    "KafkaProducerDependency",
    "RedisDependency",
    "SettingsDependency",
]
