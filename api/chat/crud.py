"""Database queries for the chat domain (sessions + messages)."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.chat.models import ChatMessage, ChatSession, MessageRole


async def create_session(
    db: AsyncSession, user_id: uuid.UUID, title: str | None = None
) -> ChatSession:
    """Insert a new, empty chat session."""
    session = ChatSession(user_id=user_id, title=title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_sessions_by_user(db: AsyncSession, user_id: uuid.UUID) -> Sequence[ChatSession]:
    """Return every session owned by *user_id*, newest first."""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )
    return result.scalars().all()


async def get_session_by_id(
    db: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID
) -> ChatSession | None:
    """Fetch a specific session, ensuring it belongs to the requesting user."""
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def update_session_title(
    db: AsyncSession, session: ChatSession, new_title: str
) -> ChatSession:
    """Update the title of an existing chat session."""
    session.title = new_title
    await db.commit()
    await db.refresh(session)
    return session


async def get_messages_by_session(db: AsyncSession, session_id: uuid.UUID) -> Sequence[ChatMessage]:
    """Return every message in *session_id*, oldest first (conversation order)."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    return result.scalars().all()


async def create_message_pair(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_content: str,
    assistant_content: str,
    assistant_sources: list[dict] | None,
    user_submitted_at: datetime,
) -> tuple[ChatMessage, ChatMessage]:
    """Insert the user's question and the assistant's answer as one transaction.

    Keeping a turn atomic means the DB can never show a question with no
    answer (or vice versa) if the process dies mid-write — mirrors the
    atomicity reasoning behind `replace_document_chunks()` in
    api/documents/crud.py.
    """
    user_msg = ChatMessage(
        session_id=session_id,
        role=MessageRole.USER,
        content=user_content,
        created_at=user_submitted_at,
    )
    response_completed_at = datetime.now(UTC)
    assistant_msg = ChatMessage(
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        content=assistant_content,
        sources=assistant_sources,
        created_at=response_completed_at,
    )
    db.add_all([user_msg, assistant_msg])

    await db.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(updated_at=response_completed_at)
    )

    await db.commit()
    await db.refresh(user_msg)
    await db.refresh(assistant_msg)
    return user_msg, assistant_msg
