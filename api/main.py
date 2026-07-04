import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

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


app = FastAPI(title="DocMind API", version="0.1.0", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    """Unauthenticated sanity check — confirms the API process is up at all."""
    return {"message": "DocMind API", "status": "running"}


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """Checks that the API can actually reach Postgres and Redis.

    This is the endpoint Phase 0 is really about: if this returns 200,
    every container in docker-compose is wired up correctly.
    """
    await db.execute(text("SELECT 1"))

    redis_client = redis.from_url(settings.redis_url)
    try:
        await redis_client.ping()
    finally:
        await redis_client.aclose()

    return {"status": "ok", "database": "connected", "redis": "connected"}
