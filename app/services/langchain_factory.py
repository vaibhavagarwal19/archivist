from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.exceptions import ServiceConfigurationError


def get_langchain_chat_model(temperature: float = 0) -> ChatOllama | ChatOpenAI:
    settings = get_settings()
    if settings.llm_provider.lower() == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.llm_base_url.removesuffix("/v1"),
            temperature=temperature,
            num_predict=512,
        )

    if not settings.llm_api_key.strip():
        raise ServiceConfigurationError(
            f"LLM provider is set to hosted {settings.llm_provider}, but no API key is configured. "
            "Set the provider API key in .env, for example GROQ_API_KEY or LLM_API_KEY."
        )

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
    )
