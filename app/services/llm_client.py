from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.exceptions import ServiceConfigurationError


def get_llm_client() -> AsyncOpenAI:
    settings = get_settings()
    if settings.llm_provider.lower() != "ollama" and not settings.llm_api_key.strip():
        raise ServiceConfigurationError(
            f"LLM provider is set to hosted {settings.llm_provider}, but no API key is configured. "
            "Set the provider API key in .env, for example GROQ_API_KEY or LLM_API_KEY."
        )
    return AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
    )
