"""Shared FastAPI dependencies backed by application lifespan state."""

from typing import Annotated

from fastapi import Depends, Request

from trading.checkpoint import DatabasePool
from trading.config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database_pool(request: Request) -> DatabasePool:
    return request.app.state.database_pool


SettingsDependency = Annotated[Settings, Depends(get_settings)]
DatabasePoolDependency = Annotated[DatabasePool, Depends(get_database_pool)]


__all__ = ["DatabasePoolDependency", "SettingsDependency"]
