import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import router as auth_router
from api.core.config import get_settings
from api.core.database import get_db
from api.core.logging import configure_logging

settings = get_settings()
logger = logging.getLogger("docmind")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    logger.info("Starting DocMind API (env=%s)", settings.environment)
    yield
    logger.info("Shutting down DocMind API")


app = FastAPI(title="DocMind API", version="0.2.0", lifespan=lifespan)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router)
# Phase 2: app.include_router(documents_router)
# Phase 4: app.include_router(chat_router)


# ── Infrastructure endpoints ──────────────────────────────────────────────────


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    """Unauthenticated sanity check — confirms the API process is up at all."""
    return {"message": "DocMind API", "status": "running"}


@app.get("/health", tags=["meta"])
async def health(db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """Confirms the API can reach both Postgres and Redis.

    This is the integration health check used by docker-compose and CI.
    """
    await db.execute(text("SELECT 1"))

    redis_client = redis.from_url(settings.redis_url)
    try:
        await redis_client.ping()
    finally:
        await redis_client.aclose()

    return {"status": "ok", "database": "connected", "redis": "connected"}
