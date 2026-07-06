"""Pytest fixtures for DocMind.

Two fixture tiers:
  - Standalone (SQLite in-memory): `db_session` + `client`
    Run anywhere with: pytest
    Used by: test_auth.py (and future unit tests)

  - Integration (real Postgres + Redis): `integration_client`
    Run inside docker-compose with: make test
    Used by: test_health.py
"""

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.core.database import Base, get_db
from api.main import app

# ── Standalone (SQLite) fixtures ──────────────────────────────────────────────

_SQLITE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Fresh SQLite in-memory database, created and torn down per test.

    Each test gets a clean slate — no cross-test data leakage.
    """
    engine = create_async_engine(_SQLITE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client backed by the FastAPI app with the DB swapped for SQLite.

    Use this for all endpoint tests that don't need Postgres / Redis.
    """

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ── Integration fixtures (require running docker-compose services) ─────────────


@pytest_asyncio.fixture
async def integration_client() -> AsyncGenerator[AsyncClient, None]:
    """HTTP client that uses the real app stack (Postgres + Redis).

    Only used in tests marked with @pytest.mark.integration.
    Run these inside docker-compose: make test
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
