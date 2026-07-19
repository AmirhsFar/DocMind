import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import router as auth_router
from api.chat import router as chat_router
from api.core.config import get_settings
from api.core.database import get_db
from api.core.logging import configure_logging
from api.documents import router as documents_router
from api.documents.embeddings import EMBEDDING_MODEL, EmbeddingClient
from api.documents.storage import StorageClient

settings = get_settings()
logger = logging.getLogger("docmind")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    logger.info("Starting DocMind API (env=%s)", settings.environment)

    # Initialise MinIO client and guarantee the bucket exists before any
    # request is handled.  Routes access the client via the get_storage
    # dependency, which reads from app.state.storage.
    storage = StorageClient.from_settings(settings)
    await storage.ensure_bucket()
    app.state.storage = storage
    logger.info("Storage ready (bucket=%s)", settings.minio_bucket)

    # Built once here (same pattern as storage) so api/chat/router.py can
    # embed the user's question via the get_embedding_client dependency
    # without constructing a fresh OpenAI client on every request. Unlike
    # the chat LLM client (built lazily inside api/chat/rag.py), this is
    # safe to build eagerly: OPENAI_API_KEY has been a hard requirement
    # since Phase 3 already, so this doesn't introduce a new way for a
    # missing key to take down startup.
    app.state.embedding_client = EmbeddingClient.from_settings(settings)
    logger.info("Embedding client ready (model=%s)", EMBEDDING_MODEL)

    yield

    logger.info("Shutting down DocMind API")


app = FastAPI(title="DocMind API", version="0.4.0", lifespan=lifespan)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)


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
