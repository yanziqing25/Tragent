"""Typed configuration loaded from the project ``.env`` file."""

from math import isfinite
from pathlib import Path
from typing import Self

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class ConfigurationError(ValueError):
    """A configuration problem that is safe to show at the application boundary."""


class Settings(BaseSettings):
    """External-service settings and deterministic backtest defaults."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=None,
        env_prefix="",
        extra="ignore",
        str_strip_whitespace=True,
    )

    llm_api_key: SecretStr
    llm_model: str = Field(min_length=1)
    llm_base_url: AnyHttpUrl
    langsmith_api_key: SecretStr
    langsmith_project: str = Field(min_length=1)
    postgres_host: str = Field(min_length=1)
    postgres_port: int = Field(ge=1, le=65535)
    postgres_user: str = Field(min_length=1)
    postgres_password: SecretStr
    postgres_database: str = Field(min_length=1)
    postgres_pool_min_size: int = Field(default=1, ge=1, le=100)
    postgres_pool_max_size: int = Field(default=10, ge=1, le=100)
    postgres_pool_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    jwt_secret: SecretStr
    default_transaction_cost_bps: float = 5.0
    default_slippage_bps: float = 5.0
    default_position_tranches: int = Field(default=5, ge=1, le=20)

    @field_validator("llm_api_key", "langsmith_api_key")
    @classmethod
    def secret_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("postgres_password")
    @classmethod
    def postgres_password_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        raw_value = value.get_secret_value().strip()
        if not raw_value:
            raise ValueError("must not be empty")
        return SecretStr(raw_value)

    @field_validator("jwt_secret")
    @classmethod
    def jwt_secret_must_be_long_enough(cls, value: SecretStr) -> SecretStr:
        raw_value = value.get_secret_value().strip()
        if len(raw_value) < 32:
            raise ValueError("must contain at least 32 characters")
        return SecretStr(raw_value)

    @field_validator("default_transaction_cost_bps", "default_slippage_bps")
    @classmethod
    def backtest_cost_must_be_finite_and_non_negative(cls, value: float) -> float:
        if not isfinite(value) or value < 0:
            raise ValueError("must be a finite non-negative number")
        return value

    @model_validator(mode="after")
    def settings_relationships_are_valid(self) -> Self:
        if self.postgres_pool_min_size > self.postgres_pool_max_size:
            raise ValueError("PostgreSQL pool minimum size must not exceed maximum size")
        if self.default_transaction_cost_bps + self.default_slippage_bps >= 10_000:
            raise ValueError("backtest costs must total less than 10000 bps")
        return self


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Load and validate settings, returning only safe error messages on failure."""

    selected_file = DEFAULT_ENV_FILE if env_file is None else Path(env_file)

    try:
        return Settings(_env_file=selected_file)
    except ValidationError as exc:
        issues = [_safe_issue(error) for error in exc.errors(include_input=False)]
        raise ConfigurationError("Invalid configuration: " + "; ".join(issues)) from None


def _safe_issue(error: dict[str, object]) -> str:
    """Describe a validation error without including its input value."""

    location = error.get("loc", ())
    field_name = str(location[0]).upper() if location else "CONFIGURATION"
    detail = str(error.get("msg", "is invalid")).removeprefix("Value error, ")
    return f"{field_name}: {detail}"
