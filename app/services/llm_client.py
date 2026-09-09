import logging

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.exceptions import ServiceConfigurationError

logger = logging.getLogger(__name__)


def get_llm_client() -> AsyncOpenAI:
    settings = get_settings()
    if settings.llm_provider.lower() != "ollama" and not settings.llm_api_key.strip():
        raise ServiceConfigurationError(
            f"LLM provider is set to hosted {settings.llm_provider}, but no API key is configured. "
            "Set the provider API key in .env, for example GROQ_API_KEY or LLM_API_KEY."
        )
    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
    )
    return _traced(client)


def _traced(client: AsyncOpenAI) -> AsyncOpenAI:
    """Make report and summary calls visible to LangSmith.

    The agent runs through LangChain, which instruments itself. These two
    services call the OpenAI SDK directly, so without this wrapper enabling
    tracing would light up the chat agent and leave /generate-report and
    /summarize looking as though they make no model calls at all.
    """
    from app.core.observability import tracing_enabled

    if not tracing_enabled():
        return client

    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception as exc:  # tracing must never break the request path
        logger.warning("Could not enable LangSmith tracing for the OpenAI client: %s", exc)
        return client
