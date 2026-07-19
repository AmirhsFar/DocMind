"""Chat endpoints: create/list sessions, list messages, and the
SSE-streamed RAG turn.

POST .../messages is the interesting one: it authenticates and verifies
session ownership synchronously, using the normal request-scoped `db`
(safe — that work finishes before the response is returned). It then
hands off to `rag.retrieve_and_stream`, which opens its own DB session for
everything that happens once the SSE body actually starts streaming.
See rag.py's docstring for why that split matters.
"""

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from api.auth.dependencies import get_current_user
from api.auth.models import User
from api.chat.crud import (
    create_session,
    get_messages_by_session,
    get_session_by_id,
    get_sessions_by_user,
    update_session_title,
)
from api.chat.models import ChatMessage, ChatSession
from api.chat.rag import retrieve_and_stream
from api.chat.schemas import (
    ChatSessionCreate,
    ChatSessionOut,
    ChatSessionUpdate,
    MessageIn,
    MessageOut,
)
from api.core.database import get_db
from api.documents.dependencies import get_embedding_client
from api.documents.embeddings import EmbeddingClient

logger = logging.getLogger("docmind.chat")

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Sessions ──────────────────────────────────────────────────────────────────


@router.post(
    "/sessions",
    response_model=ChatSessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new chat session",
)
async def create_chat_session(
    body: ChatSessionCreate | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatSession:
    """Create an (optionally titled) empty session. A bare POST with no
    body is fine — the session just starts untitled."""
    title = body.title if body is not None else None
    return await create_session(db, current_user.id, title=title)


@router.get(
    "/sessions",
    response_model=list[ChatSessionOut],
    summary="List the current user's chat sessions",
)
async def list_chat_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ChatSession]:
    return await get_sessions_by_user(db, current_user.id)


@router.patch(
    "/sessions/{session_id}",
    response_model=ChatSessionOut,
    summary="Update an existing chat session (e.g. applying an auto-generated title)",
)
async def update_chat_session(
    session_id: uuid.UUID,
    body: ChatSessionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatSession:
    # 1. Verify the session exists and belongs to the user
    session = await _owned_session_or_404(db, session_id, current_user)

    # 2. Apply the update using our new CRUD method
    return await update_session_title(db, session, body.title)


# ── Messages ─────────────────────────────────────────────────────────────────


@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[MessageOut],
    summary="List every message in a session, oldest first",
)
async def list_messages(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ChatMessage]:
    await _owned_session_or_404(db, session_id, current_user)
    return await get_messages_by_session(db, session_id)


@router.post(
    "/sessions/{session_id}/messages",
    summary="Ask a question; the answer streams back as Server-Sent Events",
)
async def send_message(
    session_id: uuid.UUID,
    body: MessageIn,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    embeddings: EmbeddingClient = Depends(get_embedding_client),
) -> EventSourceResponse:
    """Retrieve relevant chunks, call the LLM, and stream the answer back.

    The user's question and the assistant's answer are both persisted only
    *after* the stream completes (see `retrieve_and_stream`), so a client
    that disconnects mid-answer never leaves a half-written turn in the DB.

    Retrieval is always scoped to the current user
    (`DocumentChunk.user_id == current_user.id` inside `retrieve_and_stream`)
    regardless of what `document_ids` is passed — narrowing to specific
    documents can only ever shrink the result set, never cross into
    another user's chunks, so no separate ownership check is needed for it.
    """
    session = await _owned_session_or_404(db, session_id, current_user)
    history = await get_messages_by_session(db, session.id)

    user_submitted_at = datetime.now(UTC)

    async def event_generator():
        async for token in retrieve_and_stream(
            session_id=session.id,
            user_id=current_user.id,
            user_message=body.content,
            history=history,
            embeddings=embeddings,
            user_submitted_at=user_submitted_at,
            document_ids=body.document_ids,
        ):
            yield {"data": token}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


# ── Private helpers ───────────────────────────────────────────────────────────


async def _owned_session_or_404(
    db: AsyncSession, session_id: uuid.UUID, current_user: User
) -> ChatSession:
    """Fetch a session and verify ownership.

    404 (not 403) either way — same ID-enumeration protection as
    `_owned_or_404` in api/documents/router.py.
    """
    session = await get_session_by_id(db, session_id, current_user.id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    return session
