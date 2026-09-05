from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Internal AI Agents"
    app_version: str = "1.0.0"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    secret_key: str = Field(default="change-me", min_length=16)
    access_token_expire_minutes: int = 60
    algorithm: str = "HS256"

    llm_provider: str = "groq"
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GROQ_API_KEY", "MISTRAL_API_KEY", "LLM_API_KEY"),
    )
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: int = 120

    # Embeddings are configured independently of the chat model: Groq serves no
    # embeddings endpoint, so the two providers are almost never the same.
    # "fastembed" runs a small ONNX model in-process - no API key, no network.
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/internal_ai_agents"
    redis_url: str = "redis://localhost:6379/0"
    # Celery/Redis is optional; uploads work without it.
    enable_background_tasks: bool = False
    faiss_index_path: str = str(BASE_DIR / "storage" / "faiss" / "index")
    upload_dir: str = str(BASE_DIR / "storage" / "uploads")
    # Hosts with an ephemeral filesystem lose the FAISS index on every restart.
    # Document text lives in the database, so the index can be rebuilt on boot.
    reindex_on_startup: bool = False

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost:8501"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def reject_insecure_production_secret(self) -> "Settings":
        # A default signing key in production means anyone can mint a valid JWT
        # for any user id. Fail loudly at startup rather than silently.
        if not self.debug and self.secret_key in {"change-me", "replace-with-a-long-random-secret"}:
            raise ValueError(
                "SECRET_KEY is still the placeholder value while DEBUG is false. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        return self

    @field_validator("database_url", mode="before")
    @classmethod
    def normalise_database_url(cls, value):
        """Accept the `postgres://` URLs that hosting providers hand out.

        Render, Heroku and Fly all expose connection strings as `postgres://`,
        a scheme SQLAlchemy 2.0 no longer recognises. Rewriting it here means
        the provider's value can be used verbatim.
        """
        if isinstance(value, str):
            if value.startswith("postgres://"):
                return value.replace("postgres://", "postgresql+psycopg://", 1)
            if value.startswith("postgresql://"):
                return value.replace("postgresql://", "postgresql+psycopg://", 1)
        return value

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_value(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
