"""Infrastructure health checks.

These tests require live Postgres and Redis — run them inside docker-compose:

    make test

or with the standalone test runner only after `docker compose up`:

    pytest -m integration
"""

import pytest
from httpx import AsyncClient


async def test_root(client: AsyncClient) -> None:
    """No external dependencies needed — just checks the app boots and routes."""
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


@pytest.mark.integration
async def test_health(integration_client: AsyncClient) -> None:
    """Confirms the API can reach Postgres and Redis (needs docker-compose)."""
    response = await integration_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected"
    assert body["redis"] == "connected"
