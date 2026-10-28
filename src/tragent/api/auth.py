"""Users, short-lived JWT access tokens, and single-use refresh tokens in an httpOnly cookie."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Annotated, Literal

import jwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import InvalidTokenError
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, Field
from redis import Redis

from tragent.api.dependencies import DatabasePoolDependency, RedisDependency, SettingsDependency
from tragent.config import Settings
from tragent.db import DatabasePool

_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = 30
_REFRESH_COOKIE = "tragent_refresh"
_REFRESH_SECONDS = 14 * 24 * 60 * 60
router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer_scheme = HTTPBearer()


class Token(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1)


def authenticate_user(pool: DatabasePool, username: str, password: str) -> int | None:
    with pool.connection() as connection:
        row = connection.execute(
            "SELECT id FROM users WHERE username = %s AND password = %s",
            (username, password),
        ).fetchone()
    return row["id"] if row else None


def create_user(pool: DatabasePool, username: str, password: str) -> int:
    with pool.connection() as connection:
        row = connection.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id",
            (username, password),
        ).fetchone()
    return row["id"]


def create_access_token(user_id: int, settings: Settings) -> str:
    expires_at = datetime.now(UTC) + timedelta(minutes=_ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user_id), "exp": expires_at},
        settings.jwt_secret.get_secret_value(),
        algorithm=_ALGORITHM,
    )


def user_from_token(token: str, settings: Settings) -> int | None:
    """The user an access token was issued to, or None if it is invalid or expired."""

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except InvalidTokenError:
        return None
    # Only this API signs tokens, and it puts the user's ID in sub.
    return int(payload["sub"])


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
    settings: SettingsDependency,
) -> int:
    user_id = user_from_token(credentials.credentials, settings)
    if user_id is None:
        raise _authentication_error("Invalid or expired access token")
    return user_id


CurrentUserDependency = Annotated[int, Depends(get_current_user)]


# The browser sends the cookie by itself, so clients need not see it in the schema.
RefreshCookie = Annotated[str | None, Cookie(alias=_REFRESH_COOKIE, include_in_schema=False)]


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    body: LoginRequest,
    pool: DatabasePoolDependency,
    settings: SettingsDependency,
    redis: RedisDependency,
    response: Response,
) -> Token:
    try:
        user_id = create_user(pool, body.username, body.password)
    except UniqueViolation:
        raise HTTPException(status.HTTP_409_CONFLICT, "This username is already taken") from None
    return _start_session(user_id, settings, redis, response)


@router.post("/login")
def login(
    body: LoginRequest,
    pool: DatabasePoolDependency,
    settings: SettingsDependency,
    redis: RedisDependency,
    response: Response,
) -> Token:
    user_id = authenticate_user(pool, body.username, body.password)
    if user_id is None:
        raise _authentication_error("Incorrect username or password")
    return _start_session(user_id, settings, redis, response)


@router.post("/refresh")
def refresh(
    refresh_token: RefreshCookie,
    settings: SettingsDependency,
    redis: RedisDependency,
    response: Response,
) -> Token:
    # Each refresh token works once: a replayed or stolen copy finds nothing.
    user_id = redis.getdel(_refresh_key(refresh_token)) if refresh_token else None
    if user_id is None:
        raise _authentication_error("The session has ended; sign in again")
    return _start_session(int(user_id), settings, redis, response)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(refresh_token: RefreshCookie, redis: RedisDependency, response: Response) -> None:
    if refresh_token:
        redis.delete(_refresh_key(refresh_token))
    response.delete_cookie(_REFRESH_COOKIE, path="/api/auth", secure=True, httponly=True)


def _start_session(user_id: int, settings: Settings, redis: Redis, response: Response) -> Token:
    """Issue an access token and a fresh refresh token that only the auth routes can read."""

    refresh_token = secrets.token_urlsafe(32)
    redis.set(_refresh_key(refresh_token), user_id, ex=_REFRESH_SECONDS)
    response.set_cookie(
        _REFRESH_COOKIE,
        refresh_token,
        max_age=_REFRESH_SECONDS,
        path="/api/auth",
        secure=True,
        httponly=True,
        samesite="strict",
    )
    return Token(access_token=create_access_token(user_id, settings))


def _refresh_key(refresh_token: str) -> str:
    # Only a digest is stored, so reading Redis does not reveal usable tokens.
    return f"tragent:refresh:{sha256(refresh_token.encode()).hexdigest()}"


def _authentication_error(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


__all__ = [
    "CurrentUserDependency",
    "LoginRequest",
    "Token",
    "get_current_user",
    "router",
    "user_from_token",
]
