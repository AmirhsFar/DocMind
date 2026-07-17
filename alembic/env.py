"""Alembic migration environment — async SQLAlchemy edition.

Key points:
  - DB URL comes from `settings.database_url` (not alembic.ini) so it
    automatically picks up environment variables / .env files.
  - `run_migrations_online` uses an async engine because our app uses asyncpg.
  - All ORM models must be imported here (even if unused) so Alembic's
    autogenerate can see their table definitions via `Base.metadata`.

To create a new migration:
    alembic revision --autogenerate -m "describe what changed"

To apply all pending migrations:
    alembic upgrade head
"""

import asyncio
import logging
from logging.config import fileConfig

from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# ── Import every model so its table is registered on Base.metadata ────────────
# Add a new import here whenever you create a new model file.
from api.auth.models import User  # noqa: F401
from api.core.config import get_settings
from api.core.database import Base
from api.documents.models import Document, DocumentChunk  # noqa: F401

# Phase 4: from api.chat.models import ChatSession, ChatMessage      # noqa: F401

settings = get_settings()

alembic_config = context.config

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

target_metadata = Base.metadata

logger = logging.getLogger("alembic.env")


def run_migrations_offline() -> None:
    """Generate SQL without a live DB connection (useful for review / audits)."""
    # Use the sync variant of the URL for offline SQL generation.
    url = settings.database_url.replace("+asyncpg", "")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:  # type: ignore[type-arg]
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
