"""Plaintext user accounts, JWT access tokens, and single-use refresh tokens."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Annotated

import fakeredis
import jwt
import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from psycopg.errors import UniqueViolation
from pydantic import SecretStr

import tragent.api.app as app_module
import tragent.api.auth as auth_module
from tragent import redis_client
from tragent.api.auth import get_current_user
from tragent.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_api_key=SecretStr("test-llm-key"),
        llm_model="test-model",
        llm_base_url="https://llm.example/v1",
        embedding_api_key=SecretStr("test-embedding-key"),
        embedding_model="test-embedding-model",
        embedding_base_url="https://embeddings.example/v1",
        embedding_dimensions=1536,
        sec_user_agent="Test User test@example.test",
        langsmith_api_key=SecretStr("test-langsmith-key"),
        langsmith_project="test-project",
        postgres_host="localhost",
        postgres_port=5432,
        postgres_user="test",
        postgres_password=SecretStr("test"),
        postgres_database="test",
        redis_host="localhost",
        redis_port=6379,
        jwt_secret=SecretStr("test-jwt-secret-with-at-least-32-characters"),
        s3_access_key="test-s3-key",
        s3_secret_key=SecretStr("test-s3-secret"),
    )


@pytest.fixture
def redis(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> fakeredis.FakeRedis:
    server = fakeredis.FakeServer()
    monkeypatch.setattr(
        redis_client,
        "connect",
        lambda _settings: fakeredis.FakeRedis(server=server, decode_responses=True),
    )
    monkeypatch.setattr(
        redis_client,
        "connect_async",
        lambda _settings: fakeredis.FakeAsyncRedis(server=server, decode_responses=True),
    )
    pool = SimpleNamespace(open=lambda **_kwargs: None, close=lambda: None)
    monkeypatch.setattr(app_module, "build_connection_pool", lambda _settings: pool)
    monkeypatch.setattr(app_module, "setup_persistence", lambda *_args: None)
    return fakeredis.FakeRedis(server=server, decode_responses=True)


def test_json_login_and_bearer_dependency_reject_bad_tokens(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    redis: fakeredis.FakeRedis,
) -> None:
    monkeypatch.setattr(
        auth_module,
        "authenticate_user",
        lambda _pool, username, password: (
            42 if username == "researcher" and password == "correct-password" else None
        ),
    )
    app = app_module.create_app(settings)

    @app.get("/protected")
    def protected(
        current_user_id: Annotated[int, Depends(get_current_user)],
    ) -> dict[str, int]:
        return {"user_id": current_user_id}

    with TestClient(app) as client:
        login = client.post(
            "/api/auth/login",
            json={"username": "researcher", "password": "correct-password"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
        )
        assert payload["sub"] == "42"
        assert 29 * 60 <= payload["exp"] - datetime.now(UTC).timestamp() <= 30 * 60

        protected_response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert protected_response.status_code == 200
        assert protected_response.json() == {"user_id": 42}

        wrong_password = client.post(
            "/api/auth/login",
            json={"username": "researcher", "password": "wrong-password"},
        )
        assert wrong_password.status_code == 401

        expired = jwt.encode(
            {"sub": "42", "exp": datetime.now(UTC) - timedelta(seconds=1)},
            settings.jwt_secret.get_secret_value(),
            algorithm="HS256",
        )
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {expired}"},
        )
        assert response.status_code == 401


def _refresh_cookie(response) -> str:
    return response.cookies[auth_module._REFRESH_COOKIE]


def test_register_signs_in_and_rejects_a_taken_username(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    redis: fakeredis.FakeRedis,
) -> None:
    def create_user(_pool: object, username: str, _password: str) -> int:
        if username == "taken":
            raise UniqueViolation()
        return 7

    monkeypatch.setattr(auth_module, "create_user", create_user)

    with TestClient(app_module.create_app(settings)) as client:
        created = client.post("/api/auth/register", json={"username": "new", "password": "pw"})
        taken = client.post("/api/auth/register", json={"username": "taken", "password": "pw"})

    assert created.status_code == 201
    assert (
        jwt.decode(
            created.json()["access_token"], settings.jwt_secret.get_secret_value(), ["HS256"]
        )["sub"]
        == "7"
    )
    cookie = created.headers["set-cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "Path=/api/auth" in cookie
    assert taken.status_code == 409


def test_refresh_tokens_rotate_once_and_logout_revokes_them(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    redis: fakeredis.FakeRedis,
) -> None:
    monkeypatch.setattr(auth_module, "authenticate_user", lambda *_args: 42)

    def refresh(client: TestClient, token: str):
        return client.post("/api/auth/refresh", headers={"Cookie": f"tragent_refresh={token}"})

    with TestClient(app_module.create_app(settings)) as client:
        first = _refresh_cookie(
            client.post("/api/auth/login", json={"username": "u", "password": "p"})
        )
        rotated = refresh(client, first)
        replayed = refresh(client, first)
        second = _refresh_cookie(rotated)
        client.post("/api/auth/logout", headers={"Cookie": f"tragent_refresh={second}"})
        after_logout = refresh(client, second)

    assert rotated.status_code == 200
    assert second != first
    assert replayed.status_code == 401
    assert after_logout.status_code == 401
    assert redis.keys("tragent:refresh:*") == []
