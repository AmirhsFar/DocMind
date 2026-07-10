"""Tests for the /documents endpoints.

All tests run standalone — no MinIO, no Postgres, no Redis required.
The DB is an in-memory SQLite instance and storage is a FakeStorageClient.

Fixture hierarchy:
  docs_db      — fresh SQLite DB per test
  docs_client  — authenticated HTTP client + FakeStorageClient + mocked Celery task
"""

import uuid
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.core.database import Base, get_db
from api.documents.dependencies import get_storage
from api.main import app
from tests.fakes import FakeStorageClient

_SQLITE_URL = "sqlite+aiosqlite:///:memory:"

# Small sample file contents
_PDF = b"%PDF-1.4 fake pdf"
_TXT = b"Hello from a plain text document."


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def docs_db() -> AsyncGenerator[AsyncSession, None]:
    """Fresh in-memory SQLite database, created and torn down per test."""
    engine = create_async_engine(_SQLITE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def docs_client(
    docs_db: AsyncSession,
    monkeypatch,
) -> AsyncGenerator[tuple[AsyncClient, FakeStorageClient], None]:
    """Authenticated HTTP client with:
    - SQLite DB override
    - FakeStorageClient override
    - Celery task dispatch mocked (no Redis needed)
    """
    fake_storage = FakeStorageClient()

    # Mock Celery task dispatch — .delay() would otherwise try to connect to Redis.
    monkeypatch.setattr("api.documents.router.process_document", MagicMock())

    async def _get_db() -> AsyncGenerator[AsyncSession, None]:
        yield docs_db

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_storage] = lambda: fake_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.post("/auth/signup", json={"email": "owner@test.com", "password": "ownerpass1"})
        login = await ac.post(
            "/auth/login", json={"email": "owner@test.com", "password": "ownerpass1"}
        )
        ac.headers.update({"Authorization": f"Bearer {login.json()['access_token']}"})
        yield ac, fake_storage

    app.dependency_overrides.clear()


# ── Upload ────────────────────────────────────────────────────────────────────


async def test_upload_pdf(docs_client: tuple) -> None:
    client, storage = docs_client
    resp = await client.post(
        "/documents/",
        files={"file": ("report.pdf", _PDF, "application/pdf")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "report.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["size_bytes"] == len(_PDF)
    assert body["status"] == "pending"
    assert "id" in body
    assert "created_at" in body
    assert "storage_key" not in body
    stored_key = next(k for k in storage.objects.keys() if "report.pdf" in k)
    stored_object = storage.objects[stored_key]
    assert stored_object["data"] == _PDF
    assert stored_object["content_type"] == "application/pdf"


async def test_upload_plain_text(docs_client: tuple) -> None:
    client, _ = docs_client
    resp = await client.post(
        "/documents/",
        files={"file": ("notes.txt", _TXT, "text/plain")},
    )
    assert resp.status_code == 201
    assert resp.json()["filename"] == "notes.txt"


async def test_upload_markdown(docs_client: tuple) -> None:
    client, _ = docs_client
    resp = await client.post(
        "/documents/",
        files={"file": ("readme.md", b"# Hello", "text/markdown")},
    )
    assert resp.status_code == 201


async def test_upload_unsupported_type(docs_client: tuple) -> None:
    client, _ = docs_client
    resp = await client.post(
        "/documents/",
        files={"file": ("photo.png", b"\x89PNG", "image/png")},
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


async def test_upload_too_large(docs_client: tuple, monkeypatch) -> None:
    """Override _MAX_BYTES to avoid allocating a real 10 MB buffer in tests."""
    import api.documents.router as router_mod

    monkeypatch.setattr(router_mod, "_MAX_BYTES", 5)

    client, _ = docs_client
    resp = await client.post(
        "/documents/",
        files={"file": ("big.pdf", b"X" * 10, "application/pdf")},
    )
    assert resp.status_code == 413


async def test_upload_requires_auth(docs_client: tuple) -> None:
    client, _ = docs_client
    resp = await client.post(
        "/documents/",
        files={"file": ("test.pdf", _PDF, "application/pdf")},
        headers={"Authorization": ""},  # overrides the fixture's default token
    )
    assert resp.status_code == 401


# ── List ──────────────────────────────────────────────────────────────────────


async def test_list_empty(docs_client: tuple) -> None:
    client, _ = docs_client
    resp = await client.get("/documents/")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_returns_users_documents(docs_client: tuple) -> None:
    client, _ = docs_client
    await client.post("/documents/", files={"file": ("a.pdf", _PDF, "application/pdf")})
    await client.post("/documents/", files={"file": ("b.txt", _TXT, "text/plain")})

    resp = await client.get("/documents/")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_list_pagination(docs_client: tuple) -> None:
    client, _ = docs_client
    for i in range(5):
        await client.post("/documents/", files={"file": (f"f{i}.txt", _TXT, "text/plain")})

    page1 = await client.get("/documents/?limit=3")
    assert len(page1.json()) == 3

    page2 = await client.get("/documents/?limit=3&skip=3")
    assert len(page2.json()) == 2


# ── Get ───────────────────────────────────────────────────────────────────────


async def test_get_document(docs_client: tuple) -> None:
    client, _ = docs_client
    upload = await client.post("/documents/", files={"file": ("x.pdf", _PDF, "application/pdf")})
    doc_id = upload.json()["id"]

    resp = await client.get(f"/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == doc_id


async def test_get_nonexistent_document(docs_client: tuple) -> None:
    client, _ = docs_client
    resp = await client.get(f"/documents/{uuid.uuid4()}")
    assert resp.status_code == 404


# ── Download redirect ─────────────────────────────────────────────────────────


async def test_download_redirect(docs_client: tuple) -> None:
    """The endpoint should redirect to a presigned URL (don't follow the redirect)."""
    client, _ = docs_client
    upload = await client.post("/documents/", files={"file": ("dl.pdf", _PDF, "application/pdf")})
    doc_id = upload.json()["id"]

    # allow_redirects=False so we can inspect the 307 itself
    resp = await client.get(f"/documents/{doc_id}/download", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"].startswith("http://fake-minio/")


# ── Delete ────────────────────────────────────────────────────────────────────


async def test_delete_document(docs_client: tuple) -> None:
    client, storage = docs_client
    upload = await client.post("/documents/", files={"file": ("del.pdf", _PDF, "application/pdf")})
    doc_id = upload.json()["id"]
    count_before = len(storage.objects)

    resp = await client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 204

    # Object must be gone from storage
    assert len(storage.objects) < count_before
    # And from the DB
    resp = await client.get(f"/documents/{doc_id}")
    assert resp.status_code == 404


async def test_delete_nonexistent(docs_client: tuple) -> None:
    client, _ = docs_client
    resp = await client.delete(f"/documents/{uuid.uuid4()}")
    assert resp.status_code == 404


# ── User isolation ────────────────────────────────────────────────────────────


async def test_user_isolation(docs_client: tuple) -> None:
    """User B must receive 404 when accessing User A's documents."""
    client, _ = docs_client  # client is already authenticated as User A

    upload = await client.post(
        "/documents/", files={"file": ("secret.pdf", _PDF, "application/pdf")}
    )
    doc_id = upload.json()["id"]

    # Register User B and get their token
    await client.post("/auth/signup", json={"email": "intruder@test.com", "password": "intruder1"})
    login_b = await client.post(
        "/auth/login", json={"email": "intruder@test.com", "password": "intruder1"}
    )
    token_b = f"Bearer {login_b.json()['access_token']}"

    # User B tries to GET the document — must get 404, not 403
    resp = await client.get(f"/documents/{doc_id}", headers={"Authorization": token_b})
    assert resp.status_code == 404

    # User B tries to DELETE — also 404
    resp = await client.delete(f"/documents/{doc_id}", headers={"Authorization": token_b})
    assert resp.status_code == 404

    # User B's own list must be empty
    resp = await client.get("/documents/", headers={"Authorization": token_b})
    assert resp.json() == []
