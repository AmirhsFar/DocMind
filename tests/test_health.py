from httpx import AsyncClient


async def test_root(client: AsyncClient) -> None:
    """No external dependencies needed — just checks the app boots and routes."""
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


async def test_health(client: AsyncClient) -> None:
    """Requires Postgres and Redis to be reachable — run via `make test`
    (inside docker-compose) or against services started by CI.
    """
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected"
    assert body["redis"] == "connected"
