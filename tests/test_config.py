import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from trading.config import ConfigurationError, load_settings

ENVIRONMENT_VARIABLES = (
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DATABASE",
    "POSTGRES_POOL_MIN_SIZE",
    "POSTGRES_POOL_MAX_SIZE",
    "POSTGRES_POOL_TIMEOUT_SECONDS",
    "JWT_SECRET",
    "DEFAULT_TRANSACTION_COST_BPS",
    "DEFAULT_SLIPPAGE_BPS",
    "DEFAULT_POSITION_TRANCHES",
)

VALID_VALUES = {
    "LLM_API_KEY": "test-llm-secret",
    "LLM_MODEL": "test-model",
    "LLM_BASE_URL": "https://api.example.test/v1",
    "LANGSMITH_API_KEY": "test-langsmith-secret",
    "LANGSMITH_PROJECT": "test-project",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "test_user",
    "POSTGRES_PASSWORD": "test-db-secret",
    "POSTGRES_DATABASE": "test_db",
    "JWT_SECRET": "test-jwt-secret-with-at-least-32-characters",
}


@pytest.fixture(autouse=True)
def isolated_environment() -> Iterator[None]:
    original_values = {name: os.environ.get(name) for name in ENVIRONMENT_VARIABLES}
    for name in ENVIRONMENT_VARIABLES:
        os.environ.pop(name, None)

    yield

    for name, value in original_values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def write_env_file(tmp_path: Path, **changes: str | None) -> Path:
    values = VALID_VALUES.copy()
    for key, value in changes.items():
        if value is None:
            values.pop(key)
        else:
            values[key] = value

    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    return env_file


def test_loads_valid_settings_from_supplied_env_file(tmp_path: Path) -> None:
    settings = load_settings(write_env_file(tmp_path))

    assert settings.llm_api_key.get_secret_value() == VALID_VALUES["LLM_API_KEY"]
    assert settings.llm_model == "test-model"
    assert str(settings.llm_base_url) == "https://api.example.test/v1"
    assert settings.langsmith_api_key.get_secret_value() == VALID_VALUES["LANGSMITH_API_KEY"]
    assert settings.langsmith_project == "test-project"
    assert settings.postgres_host == "localhost"
    assert settings.postgres_port == 5432
    assert settings.postgres_user == "test_user"
    assert settings.postgres_password.get_secret_value() == "test-db-secret"
    assert settings.postgres_database == "test_db"
    assert settings.postgres_pool_min_size == 1
    assert settings.postgres_pool_max_size == 10
    assert settings.postgres_pool_timeout_seconds == 30.0
    assert settings.jwt_secret.get_secret_value() == VALID_VALUES["JWT_SECRET"]
    assert settings.default_transaction_cost_bps == 5.0
    assert settings.default_slippage_bps == 5.0
    assert settings.default_position_tranches == 5


def test_loading_env_file_does_not_mutate_process_environment(tmp_path: Path) -> None:
    load_settings(write_env_file(tmp_path))

    assert all(name not in os.environ for name in ENVIRONMENT_VARIABLES)


def test_process_environment_overrides_supplied_env_file(tmp_path: Path) -> None:
    os.environ["LLM_MODEL"] = "environment-model"

    settings = load_settings(write_env_file(tmp_path, LLM_MODEL="file-model"))

    assert settings.llm_model == "environment-model"


def test_loads_backtest_defaults_overrides_from_supplied_env_file(tmp_path: Path) -> None:
    settings = load_settings(
        write_env_file(
            tmp_path,
            DEFAULT_TRANSACTION_COST_BPS="2.5",
            DEFAULT_SLIPPAGE_BPS="7.25",
            DEFAULT_POSITION_TRANCHES="8",
        )
    )

    assert settings.default_transaction_cost_bps == 2.5
    assert settings.default_slippage_bps == 7.25
    assert settings.default_position_tranches == 8


def test_loads_postgres_pool_overrides_from_supplied_env_file(tmp_path: Path) -> None:
    settings = load_settings(
        write_env_file(
            tmp_path,
            POSTGRES_POOL_MIN_SIZE="2",
            POSTGRES_POOL_MAX_SIZE="20",
            POSTGRES_POOL_TIMEOUT_SECONDS="12.5",
        )
    )

    assert settings.postgres_pool_min_size == 2
    assert settings.postgres_pool_max_size == 20
    assert settings.postgres_pool_timeout_seconds == 12.5


def test_reports_a_missing_field_without_showing_other_values(tmp_path: Path) -> None:
    env_file = write_env_file(tmp_path, LLM_API_KEY=None)

    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file)

    message = str(error.value)
    assert "LLM_API_KEY: Field required" in message
    assert all(secret not in message for secret in VALID_VALUES.values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("LLM_BASE_URL", "not-a-url/hidden-value"),
        ("POSTGRES_PORT", "70000"),
        ("POSTGRES_POOL_MAX_SIZE", "0"),
        ("POSTGRES_POOL_TIMEOUT_SECONDS", "-1"),
        ("JWT_SECRET", "too-short"),
        ("DEFAULT_POSITION_TRANCHES", "-1"),
    ],
)
def test_rejects_invalid_settings_without_echoing_input(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    env_file = write_env_file(tmp_path, **{field: value})

    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file)

    message = str(error.value)
    assert f"{field}:" in message
    assert value not in message
    assert "test-llm-secret" not in message
    assert "test-langsmith-secret" not in message
    assert "test-db-secret" not in message


def test_secret_fields_are_redacted_in_settings_representation(tmp_path: Path) -> None:
    settings = load_settings(write_env_file(tmp_path))
    representation = repr(settings)

    assert "test-llm-secret" not in representation
    assert "test-langsmith-secret" not in representation
    assert "test-db-secret" not in representation
    assert VALID_VALUES["JWT_SECRET"] not in representation
    assert "**********" in representation


def test_allows_an_http_llm_endpoint(tmp_path: Path) -> None:
    settings = load_settings(write_env_file(tmp_path, LLM_BASE_URL="http://localhost:8000/v1"))

    assert str(settings.llm_base_url) == "http://localhost:8000/v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("DEFAULT_TRANSACTION_COST_BPS", "-0.1"),
        ("DEFAULT_TRANSACTION_COST_BPS", "nan"),
        ("DEFAULT_SLIPPAGE_BPS", "inf"),
    ],
)
def test_rejects_non_finite_or_negative_backtest_cost(
    tmp_path: Path, field: str, value: str
) -> None:
    env_file = write_env_file(tmp_path, **{field: value})

    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file)

    assert f"{field}: must be a finite non-negative number" in str(error.value)
    assert value not in str(error.value)


def test_rejects_backtest_cost_total_of_one_hundred_percent(tmp_path: Path) -> None:
    env_file = write_env_file(
        tmp_path,
        DEFAULT_TRANSACTION_COST_BPS="5000",
        DEFAULT_SLIPPAGE_BPS="5000",
    )

    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file)

    assert "CONFIGURATION: backtest costs must total less than 10000 bps" in str(error.value)


def test_rejects_postgres_pool_minimum_above_maximum(tmp_path: Path) -> None:
    env_file = write_env_file(
        tmp_path,
        POSTGRES_POOL_MIN_SIZE="11",
        POSTGRES_POOL_MAX_SIZE="10",
    )

    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file)

    assert (
        "CONFIGURATION: PostgreSQL pool minimum size must not exceed maximum size"
        in str(error.value)
    )
