"""Construction of the project's single Responses API model."""

from langchain_openai import ChatOpenAI

from trading.config import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    """Build one stateless Responses API model for both Agents."""
    return ChatOpenAI(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=str(settings.llm_base_url),
        model=settings.llm_model,
        timeout=15.0,
        streaming=True,
        use_responses_api=True,
        store=False,
        use_previous_response_id=False,
    )
