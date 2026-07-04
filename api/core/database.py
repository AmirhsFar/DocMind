from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from api.core.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=settings.environment == "development")

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


class Base(DeclarativeBase):
    """Base class every ORM model (users, documents, chunks, ...) will inherit from."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session and closes it afterwards."""
    async with AsyncSessionLocal() as session:
        yield session
