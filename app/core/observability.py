"""LangSmith tracing setup.

LangChain and LangGraph instrument themselves from environment variables, but
this app reads its configuration through pydantic-settings, which parses .env
itself and never exports those values to os.environ. So a LANGSMITH_TRACING
line in .env would be visible to Settings and invisible to LangChain.

This module bridges the two: it reads the settings and exports them, so
tracing can be configured the same way as everything else.
"""
from __future__ import annotations

import logging
import os

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def configure_tracing() -> bool:
    """Turn on LangSmith tracing if it is configured. Returns whether it is on.

    Tracing is opt-in and fails closed: without an API key the app runs exactly
    as before, so local development, tests, and CI are unaffected.
    """
    settings = get_settings()

    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing disabled (LANGSMITH_TRACING is not true)")
        return False

    if not settings.langsmith_api_key.strip():
        logger.warning(
            "LANGSMITH_TRACING is true but LANGSMITH_API_KEY is empty; tracing stays off"
        )
        return False

    # setdefault so a real environment variable always wins over .env.
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)

    # langchain-core still reads the older names in some code paths.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)

    logger.info("LangSmith tracing enabled for project '%s'", settings.langsmith_project)
    return True


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
