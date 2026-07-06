"""Database queries for the auth module.

Keeping queries here (rather than inline in the router) means they can be
reused across routes and are easy to unit-test independently.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.models import User


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    """Return the User with the given *email*, or None if not found."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()
