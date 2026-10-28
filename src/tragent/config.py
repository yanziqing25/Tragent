"""Typed configuration loaded from the project ``.env`` file."""

from pathlib import Path

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class ConfigurationError(ValueError):
    """A configuration problem that is safe to show at the application boundary."""


class Settings(BaseSettings):
    """External-service settings and deterministic backtest defaults."""

    model_config = SettingsConfigDict(str_strip_whitespace=True)

    llm_api_key: SecretStr = Field(min_length=1)
    llm_model: str = Field(min_length=1)
    llm_base_url: AnyHttpUrl
    llm_request_timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    embedding_api_key: SecretStr = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_base_url: AnyHttpUrl
    embedding_dimensions: int = Field(ge=1, le=10_000)
    sec_user_agent: str = Field(min_length=1)
    langsmith_api_key: SecretStr = Field(min_length=1)
    langsmith_project: str = Field(min_length=1)
    postgres_host: str = Field(min_length=1)
    postgres_port: int = Field(ge=1, le=65535)
    postgres_user: str = Field(min_length=1)
    postgres_password: SecretStr = Field(min_length=1)
    postgres_database: str = Field(min_length=1)
    postgres_pool_min_size: int = Field(default=1, ge=1, le=100)
    postgres_pool_max_size: int = Field(default=10, ge=1, le=100)
    postgres_pool_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    redis_host: str = Field(min_length=1)
    redis_port: int = Field(ge=1, le=65535)
    # Leave unset for a Redis server without authentication.
    redis_password: SecretStr | None = None
    redis_database: int = Field(default=0, ge=0)
    # Queued or running research tasks one user may have at a time.
    max_active_runs_per_user: int = Field(default=2, ge=1)
    # Carries order commands to the broker process and order events back to the API.
    kafka_bootstrap_servers: str = Field(default="localhost:9092", min_length=1)
    # S3-compatible object store for research artifacts, statements, and exports.
    s3_endpoint_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8333")
    s3_access_key: str = Field(min_length=1)
    s3_secret_key: SecretStr = Field(min_length=1)
    s3_bucket: str = Field(default="tragent", min_length=3)
    # Where browsers reach the store when it differs from the API's address, as behind Nginx.
    s3_public_url: AnyHttpUrl | None = None
    # A Docker network the Gateways join, so a containerized broker reaches them by name.
    gateway_network: str | None = None
    jwt_secret: SecretStr = Field(min_length=32)
    # Browser origins allowed to call the API from another domain, as a JSON list.
    cors_origins: list[str] = Field(default_factory=list)
    default_transaction_cost_bps: float = Field(default=5.0, ge=0, allow_inf_nan=False)
    default_slippage_bps: float = Field(default=5.0, ge=0, allow_inf_nan=False)
    default_position_tranches: int = Field(default=5, ge=1, le=20)


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

    return f"{str(error['loc'][0]).upper()}: {error['msg']}"
