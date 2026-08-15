"""Focused tests for plaintext user login and JWT authentication."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Annotated

import jwt
import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from pydantic import SecretStr

import trading.api.app as app_module
import trading.api.auth as auth_module
from trading.api.auth import get_current_user
from trading.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_api_key=SecretStr("test-llm-key"),
        llm_model="test-model",
        llm_base_url="https://llm.example/v1",
        langsmith_api_key=SecretStr("test-langsmith-key"),
        langsmith_project="test-project",
        postgres_host="localhost",
        postgres_port=5432,
        postgres_user="test",
        postgres_password=SecretStr("test"),
        postgres_database="test",
        jwt_secret=SecretStr("test-jwt-secret-with-at-least-32-characters"),
    )


def test_json_login_and_bearer_dependency_reject_bad_tokens(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    pool = SimpleNamespace(open=lambda **_kwargs: None, close=lambda: None)
    monkeypatch.setattr(app_module, "build_connection_pool", lambda _settings: pool)
    monkeypatch.setattr(app_module, "setup_persistence", lambda _settings: None)
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
