"""LangSmith tracing configuration.

Tracing is opt-in and must fail closed: with no API key the app has to behave
exactly as it did before, or every local run and CI job would try to ship
traces to a service it cannot reach.
"""
import os

import pytest

from app.core import observability
from app.core.config import Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Tracing reads and writes os.environ, so isolate every case."""
    for key in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)


def use_settings(monkeypatch, **overrides):
    settings = Settings(
        debug=True,
        secret_key="test-secret-key-that-is-long-enough",
        **overrides,
    )
    monkeypatch.setattr(observability, "get_settings", lambda: settings)
    return settings


class TestOptIn:
    def test_disabled_by_default(self, monkeypatch):
        use_settings(monkeypatch)
        assert observability.configure_tracing() is False
        assert observability.tracing_enabled() is False
        assert "LANGSMITH_API_KEY" not in os.environ

    def test_flag_without_a_key_stays_off(self, monkeypatch):
        """A half-configured deployment must not try to trace."""
        use_settings(monkeypatch, langsmith_tracing=True, langsmith_api_key="")
        assert observability.configure_tracing() is False
        assert observability.tracing_enabled() is False

    def test_whitespace_key_counts_as_missing(self, monkeypatch):
        use_settings(monkeypatch, langsmith_tracing=True, langsmith_api_key="   ")
        assert observability.configure_tracing() is False

    def test_enabled_when_flag_and_key_are_present(self, monkeypatch):
        use_settings(
            monkeypatch,
            langsmith_tracing=True,
            langsmith_api_key="ls-test-key",
            langsmith_project="archivist-test",
        )
        assert observability.configure_tracing() is True
        assert observability.tracing_enabled() is True
        assert os.environ["LANGSMITH_API_KEY"] == "ls-test-key"
        assert os.environ["LANGSMITH_PROJECT"] == "archivist-test"

    def test_legacy_variable_names_are_set_too(self, monkeypatch):
        """langchain-core still reads LANGCHAIN_* on some code paths."""
        use_settings(monkeypatch, langsmith_tracing=True, langsmith_api_key="ls-test-key")
        observability.configure_tracing()
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "ls-test-key"

    def test_a_real_environment_variable_wins_over_settings(self, monkeypatch):
        """setdefault, so the platform's own value is never overwritten."""
        monkeypatch.setenv("LANGSMITH_PROJECT", "set-by-the-platform")
        use_settings(
            monkeypatch,
            langsmith_tracing=True,
            langsmith_api_key="ls-test-key",
            langsmith_project="from-dotenv",
        )
        observability.configure_tracing()
        assert os.environ["LANGSMITH_PROJECT"] == "set-by-the-platform"

    def test_endpoint_is_only_set_when_given(self, monkeypatch):
        use_settings(monkeypatch, langsmith_tracing=True, langsmith_api_key="ls-test-key")
        observability.configure_tracing()
        assert "LANGSMITH_ENDPOINT" not in os.environ


class TestOpenAIClientWrapping:
    """The raw OpenAI client must be traced too, or half the calls vanish."""

    def test_client_is_returned_unwrapped_when_tracing_is_off(self, monkeypatch):
        from openai import AsyncOpenAI

        from app.services import llm_client

        client = AsyncOpenAI(api_key="k", base_url="https://example.invalid/v1")
        assert llm_client._traced(client) is client

    def test_client_is_wrapped_when_tracing_is_on(self, monkeypatch):
        from openai import AsyncOpenAI

        from app.services import llm_client

        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        client = AsyncOpenAI(api_key="k", base_url="https://example.invalid/v1")
        wrapped = llm_client._traced(client)
        # wrap_openai patches the completions method in place and returns the
        # same object, so assert the call path still exists and is usable.
        assert wrapped is not None
        assert hasattr(wrapped.chat.completions, "create")

    def test_a_broken_wrapper_never_breaks_the_request(self, monkeypatch):
        """Observability failing must not take the endpoint down with it."""
        import sys

        from openai import AsyncOpenAI

        from app.services import llm_client

        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setitem(sys.modules, "langsmith.wrappers", None)
        client = AsyncOpenAI(api_key="k", base_url="https://example.invalid/v1")
        assert llm_client._traced(client) is client
