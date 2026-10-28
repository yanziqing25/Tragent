"""Construction of the project's shared chat model."""

import httpx
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from tragent.config import Settings

# Provider failures that may succeed on a later attempt; other errors are bugs or bad requests.
TRANSIENT_MODEL_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    httpx.NetworkError,
    httpx.TimeoutException,
)


def build_llm(settings: Settings) -> ChatOpenAI:
    """Build one stateless model for both Agents."""
    return ChatOpenAI(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=str(settings.llm_base_url),
        model=settings.llm_model,
        streaming=True,
        # The provider defaults to its deepest "max" reasoning, which dominates every run's time.
        reasoning_effort="high",
        # Compatible endpoints only report token usage in a stream when asked to.
        stream_usage=True,
        timeout=settings.llm_request_timeout_seconds,
        # Each caller retries its whole call instead: a stream that drops midway is beyond the
        # SDK's own retries, and two layers would multiply the attempts.
        max_retries=0,
    )
