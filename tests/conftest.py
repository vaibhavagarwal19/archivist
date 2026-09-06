"""Shared test fixtures.

Tests run against a throwaway SQLite database and never touch the LLM,
the embedding model, or the network: the agent and vector store are replaced
with fakes so the suite is fast and deterministic in CI.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Settings are read at import time and cached, so the environment has to be
# shaped before anything under app.* is imported.
_TMP_DIR = Path(tempfile.mkdtemp(prefix="archivist-tests-"))
os.environ.update(
    {
        "DEBUG": "true",
        "SECRET_KEY": "test-secret-key-that-is-long-enough",
        "DATABASE_URL": f"sqlite:///{_TMP_DIR / 'test.db'}",
        "LLM_PROVIDER": "groq",
        "LLM_API_KEY": "test-key-not-used",
        "EMBEDDING_PROVIDER": "fastembed",
        "FAISS_INDEX_PATH": str(_TMP_DIR / "faiss" / "index"),
        "UPLOAD_DIR": str(_TMP_DIR / "uploads"),
    }
)

from fastapi.testclient import TestClient  # noqa: E402

from app.db.base import Base  # noqa: E402
from app.db.session import engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_database():
    """Give every test an empty schema so cases cannot leak into each other."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def user_credentials() -> dict[str, str]:
    return {
        "email": "analyst@example.com",
        "full_name": "Test Analyst",
        "password": "StrongPass123",
    }


@pytest.fixture
def auth_headers(client: TestClient, user_credentials: dict[str, str]) -> dict[str, str]:
    """Register a user and return headers carrying a valid bearer token."""
    client.post("/api/v1/auth/register", json=user_credentials)
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": user_credentials["email"],
            "password": user_credentials["password"],
        },
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
