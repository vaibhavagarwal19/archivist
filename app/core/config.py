from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices
from pydantic import Field
from pydantic import field_validator
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
    llm_model: str = "llama-3.1-8b-instant"
    llm_timeout_seconds: int = 120
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://127.0.0.1:11434"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/internal_ai_agents"
    redis_url: str = "redis://localhost:6379/0"
    faiss_index_path: str = str(BASE_DIR / "storage" / "faiss" / "index")
    upload_dir: str = str(BASE_DIR / "storage" / "uploads")

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost:8501"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

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
