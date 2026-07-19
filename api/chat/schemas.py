import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from api.chat.models import MessageRole

# ── Inbound ──────────────────────────────────────────────────────────────────


class ChatSessionCreate(BaseModel):
    """Body for POST /chat/sessions. An empty (or omitted) body is fine —
    a session starts untitled and can be named later."""

    title: str | None = Field(default=None, max_length=255)


class ChatSessionUpdate(BaseModel):
    """Body for PATCH /chat/sessions/{id}. Used by the frontend to set the auto-title."""

    title: str = Field(min_length=1, max_length=255)


class MessageIn(BaseModel):
    """Body for POST /chat/sessions/{id}/messages."""

    content: str = Field(min_length=1, description="The user's question")
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description=(
            "Optional: restrict retrieval to these document IDs only. "
            "Omit (or send an empty list) to search across all of the user's documents."
        ),
    )


# ── Outbound ─────────────────────────────────────────────────────────────────


class ChatSessionOut(BaseModel):
    """ChatSession fields safe to expose to the client."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class SourceOut(BaseModel):
    """One cited chunk backing an assistant answer."""

    document_id: uuid.UUID
    filename: str
    chunk_index: int
    excerpt: str


class MessageOut(BaseModel):
    """ChatMessage fields safe to expose to the client."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    sources: list[SourceOut] | None
    created_at: datetime
