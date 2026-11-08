"""Settings must load from the example file the README tells people to copy."""
from __future__ import annotations

from pathlib import Path

from app.core.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def test_env_example_loads_without_error():
    """`cp .env.example .env` is step one of the quickstart; it must not crash.

    The file carries keys other components own (API_BASE_URL is the Streamlit
    UI's), so the backend's settings have to tolerate keys they do not declare.
    """
    Settings(_env_file=ENV_EXAMPLE)
