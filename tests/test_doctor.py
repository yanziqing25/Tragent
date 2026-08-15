"""Focused tests for the Web doctor checks."""

from types import SimpleNamespace

from pydantic import SecretStr

import trading.doctor as doctor
from trading.config import Settings
from trading.doctor import CheckStatus


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


def test_run_doctor_returns_the_three_service_checks(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "check_postgres", lambda _settings: "postgres")
    monkeypatch.setattr(doctor, "check_langsmith", lambda _settings: "langsmith")
    monkeypatch.setattr(doctor, "check_llm", lambda _settings: "llm")

    assert doctor.run_doctor(settings(), object()) == ("postgres", "langsmith", "llm")


def test_llm_check_returns_a_generic_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "build_llm",
        lambda _settings: SimpleNamespace(
            invoke=lambda _prompt: (_ for _ in ()).throw(RuntimeError("secret detail"))
        ),
    )

    result = doctor.check_llm(settings())

    assert result.status is CheckStatus.FAIL
    assert result.detail == "Check failed."
