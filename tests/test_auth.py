"""Authentication and access-control tests.

The regression these exist to prevent: an earlier version of get_current_user
returned a shared "guest" account when the Authorization header was missing,
so every protected route served real user data to anonymous callers.
"""
import pytest
from fastapi.testclient import TestClient

PROTECTED_ROUTES = [
    ("GET", "/api/v1/auth/me"),
    ("GET", "/api/v1/upload-doc"),
    ("POST", "/api/v1/chat"),
    ("POST", "/api/v1/generate-report"),
    ("POST", "/api/v1/summarize"),
    ("POST", "/api/v1/workflow"),
]


def test_health_is_public(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_register_returns_user_without_password(client: TestClient, user_credentials):
    response = client.post("/api/v1/auth/register", json=user_credentials)
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == user_credentials["email"]
    assert "password" not in body
    assert "hashed_password" not in body


def test_duplicate_registration_is_rejected(client: TestClient, user_credentials):
    client.post("/api/v1/auth/register", json=user_credentials)
    response = client.post("/api/v1/auth/register", json=user_credentials)
    assert response.status_code == 400


def test_short_password_is_rejected(client: TestClient):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "weak@example.com", "full_name": "Weak", "password": "short"},
    )
    assert response.status_code == 422


def test_login_with_wrong_password_is_unauthorized(client: TestClient, user_credentials):
    client.post("/api/v1/auth/register", json=user_credentials)
    response = client.post(
        "/api/v1/auth/login",
        json={"email": user_credentials["email"], "password": "WrongPassword123"},
    )
    assert response.status_code == 401


def test_login_with_unknown_email_is_unauthorized(client: TestClient):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "StrongPass123"},
    )
    assert response.status_code == 401


def test_login_against_non_hash_password_does_not_500(client: TestClient):
    """Accounts seeded with a placeholder instead of a real hash must 401.

    passlib raises on an unparseable hash, which previously surfaced as a 500.
    """
    from app.db.session import SessionLocal
    from app.models.user import User

    with SessionLocal() as db:
        db.add(
            User(
                email="placeholder@example.com",
                full_name="Placeholder",
                hashed_password="access-disabled",
                is_active=True,
            )
        )
        db.commit()

    response = client.post(
        "/api/v1/auth/login",
        json={"email": "placeholder@example.com", "password": "anything"},
    )
    assert response.status_code == 401


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_protected_routes_reject_anonymous_callers(client: TestClient, method: str, path: str):
    """No anonymous fallback: a missing token is always a 401, never guest access."""
    response = client.request(method, path, json={})
    assert response.status_code == 401, f"{method} {path} served an anonymous caller"


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_protected_routes_reject_malformed_tokens(client: TestClient, method: str, path: str):
    response = client.request(
        method, path, json={}, headers={"Authorization": "Bearer not.a.real.token"}
    )
    assert response.status_code == 401


def test_token_for_deleted_user_is_rejected(client: TestClient, auth_headers, user_credentials):
    """A token stays cryptographically valid after its user is gone."""
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.user import User

    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == user_credentials["email"]))
        db.delete(user)
        db.commit()

    response = client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 401


def test_inactive_user_is_rejected(client: TestClient, auth_headers, user_credentials):
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.user import User

    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == user_credentials["email"]))
        user.is_active = False
        db.commit()

    response = client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 401


def test_me_returns_the_authenticated_user(client: TestClient, auth_headers, user_credentials):
    response = client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == user_credentials["email"]
