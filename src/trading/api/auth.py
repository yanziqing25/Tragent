"""Minimal PostgreSQL user and JWT authentication boundary."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel

from trading.api.dependencies import DatabasePoolDependency, SettingsDependency
from trading.checkpoint import DatabasePool, open_connection
from trading.config import Settings

_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = 30
router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer_scheme = HTTPBearer()


class Token(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class LoginRequest(BaseModel):
    username: str
    password: str


def authenticate_user(pool: DatabasePool, username: str, password: str) -> int | None:
    with open_connection(pool) as connection:
        row = connection.execute(
            "SELECT id FROM users WHERE username = %s AND password = %s",
            (username, password),
        ).fetchone()
    return None if row is None else row["id"]


def create_access_token(user_id: int, settings: Settings) -> str:
    expires_at = datetime.now(UTC) + timedelta(minutes=_ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user_id), "exp": expires_at},
        settings.jwt_secret.get_secret_value(),
        algorithm=_ALGORITHM,
    )


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
    settings: SettingsDependency,
) -> int:
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
        user_id = int(payload["sub"])
        if user_id < 1:
            raise InvalidTokenError
    except (InvalidTokenError, KeyError, TypeError, ValueError):
        raise _authentication_error("Invalid or expired access token")
    return user_id


@router.post("/login", response_model=Token)
def login(
    body: LoginRequest,
    pool: DatabasePoolDependency,
    settings: SettingsDependency,
) -> Token:
    user_id = authenticate_user(pool, body.username, body.password)
    if user_id is None:
        raise _authentication_error("Incorrect username or password")
    return Token(access_token=create_access_token(user_id, settings))


def _authentication_error(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


__all__ = [
    "LoginRequest",
    "Token",
    "get_current_user",
    "router",
]
