"""Tests for POST /auth/signup, POST /auth/login, GET /auth/me.

These run standalone (no Postgres, no Redis) using the SQLite `client` fixture
defined in conftest.py.

Naming convention: test_<endpoint>_<scenario>
"""

from httpx import AsyncClient

# ── Helpers ───────────────────────────────────────────────────────────────────

_SIGNUP_URL = "/auth/signup"
_LOGIN_URL = "/auth/login"
_ME_URL = "/auth/me"


async def _register(client: AsyncClient, email: str, password: str = "securepass1") -> None:
    """Sign up a test user; assert it succeeded so failures are obvious."""
    resp = await client.post(_SIGNUP_URL, json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text


async def _login_token(client: AsyncClient, email: str, password: str = "securepass1") -> str:
    """Log in and return the raw JWT string."""
    resp = await client.post(_LOGIN_URL, json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


# ── Signup ────────────────────────────────────────────────────────────────────


async def test_signup_success(client: AsyncClient) -> None:
    resp = await client.post(
        _SIGNUP_URL, json={"email": "alice@example.com", "password": "securepass1"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "alice@example.com"
    assert body["is_active"] is True
    assert "id" in body
    assert "created_at" in body
    # The hashed password must never be exposed
    assert "hashed_password" not in body
    assert "password" not in body


async def test_signup_duplicate_email(client: AsyncClient) -> None:
    await _register(client, "bob@example.com")
    resp = await client.post(
        _SIGNUP_URL, json={"email": "bob@example.com", "password": "securepass1"}
    )
    assert resp.status_code == 400
    assert "already exists" in resp.json()["detail"].lower()


async def test_signup_short_password(client: AsyncClient) -> None:
    """Pydantic enforces min_length=8 before the route handler runs."""
    resp = await client.post(_SIGNUP_URL, json={"email": "short@example.com", "password": "abc"})
    assert resp.status_code == 422


async def test_signup_invalid_email(client: AsyncClient) -> None:
    resp = await client.post(_SIGNUP_URL, json={"email": "not-an-email", "password": "securepass1"})
    assert resp.status_code == 422


# ── Login ─────────────────────────────────────────────────────────────────────


async def test_login_success(client: AsyncClient) -> None:
    await _register(client, "carol@example.com")
    resp = await client.post(
        _LOGIN_URL, json={"email": "carol@example.com", "password": "securepass1"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    # Token should be a non-trivial JWT (three dot-separated segments)
    assert len(body["access_token"].split(".")) == 3


async def test_login_wrong_password(client: AsyncClient) -> None:
    await _register(client, "dave@example.com")
    resp = await client.post(
        _LOGIN_URL, json={"email": "dave@example.com", "password": "wrongpassword"}
    )
    assert resp.status_code == 401
    # Message must not reveal whether the email was found
    assert "incorrect" in resp.json()["detail"].lower()


async def test_login_unknown_email(client: AsyncClient) -> None:
    resp = await client.post(
        _LOGIN_URL, json={"email": "ghost@example.com", "password": "securepass1"}
    )
    # Same 401 as wrong password — never reveal which one failed
    assert resp.status_code == 401


async def test_login_missing_fields(client: AsyncClient) -> None:
    resp = await client.post(_LOGIN_URL, json={"email": "x@example.com"})
    assert resp.status_code == 422


# ── /me ───────────────────────────────────────────────────────────────────────


async def test_me_success(client: AsyncClient) -> None:
    await _register(client, "eve@example.com")
    token = await _login_token(client, "eve@example.com")

    resp = await client.get(_ME_URL, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "eve@example.com"
    assert body["is_active"] is True
    assert "hashed_password" not in body


async def test_me_no_token(client: AsyncClient) -> None:
    resp = await client.get(_ME_URL)
    assert resp.status_code == 401


async def test_me_invalid_token(client: AsyncClient) -> None:
    resp = await client.get(_ME_URL, headers={"Authorization": "Bearer not.a.real.token"})
    assert resp.status_code == 401


async def test_me_malformed_header(client: AsyncClient) -> None:
    """Sending the token without the 'Bearer ' prefix should also fail."""
    await _register(client, "frank@example.com")
    token = await _login_token(client, "frank@example.com")
    resp = await client.get(_ME_URL, headers={"Authorization": token})  # missing "Bearer "
    assert resp.status_code == 401
