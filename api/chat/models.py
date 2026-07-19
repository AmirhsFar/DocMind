"""ORM models for the chat domain (Phase 4 — RAG chat).

Follows the same Mapped[]/UUID-PK/ondelete=CASCADE style as
api/documents/models.py.
"""

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from api.core.database import Base


class MessageRole(StrEnum):
    """Who authored a ChatMessage.

    Stored as a plain VARCHAR (see `ChatMessage.role` below), not a Postgres
    native ENUM like `DocStatus` — there's no ingestion-style state machine
    to protect here, just two fixed values, so the CREATE TYPE ceremony
    isn't worth it. This enum exists purely for type safety in application
    code (avoids typo'd string literals when building messages/queries).
    """

    USER = "user"
    ASSISTANT = "assistant"


class ChatSession(Base):
    """A conversation thread between a user and the RAG assistant."""

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # Nullable — a session starts untitled; the frontend (Phase 5) or a
    # future auto-titling step can set this from the first message.
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} user_id={self.user_id} title={self.title!r}>"


class ChatMessage(Base):
    """A single turn (user question or assistant answer) within a ChatSession.

    `sources` is populated only on assistant turns whose answer drew on
    retrieved chunks — it stays NULL for user messages and is `[]` (not
    NULL) for assistant answers that had no relevant context to cite.

    Uses the generic `sqlalchemy.types.JSON` type rather than
    `postgresql.JSONB` directly, so `Base.metadata.create_all()` keeps
    working on SQLite for standalone tests — same cross-dialect reasoning
    as `EmbeddingVector` in api/documents/models.py, just via a type that
    already has native SQLite support instead of a custom TypeDecorator.
    The real migration (0004) still declares the column as native JSONB,
    since that's what actually gets deployed against Postgres.
    """

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    role: Mapped[MessageRole] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} session_id={self.session_id} role={self.role}>"
