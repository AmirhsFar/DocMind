"""Retrieval-augmented generation: embed the question, retrieve the most
relevant chunks the user owns, build a grounded prompt, and stream the
LLM's answer back token by token.

Kept separate from router.py for testability. This file splits cleanly
into two halves:

  - Pure / SQLite-friendly: RetrievedChunk, format_context_block,
    build_messages, chunks_to_sources. No I/O, or DB access that works on
    any dialect — exercised by the standalone test suite.
  - Postgres-only: retrieve_relevant_chunks, which uses pgvector's `<=>`
    operator. SQLite has no equivalent, so this path is only exercised by
    @pytest.mark.integration tests (see tests/test_chat.py).

retrieve_and_stream() ties both halves together plus the LLM call itself.
"""

import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime

from anthropic import AsyncAnthropic
from sqlalchemy import literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.chat.crud import create_message_pair
from api.chat.models import ChatMessage, MessageRole
from api.core.config import Settings, get_settings
from api.core.database import AsyncSessionLocal
from api.documents.embeddings import EmbeddingClient
from api.documents.models import Document, DocumentChunk, EmbeddingVector

logger = logging.getLogger("docmind.rag")

CHAT_MODEL = "claude-sonnet-5"
MAX_RESPONSE_TOKENS = 1024
TOP_K_CHUNKS = 5
_EXCERPT_MAX_CHARS = 240

SYSTEM_PROMPT = (
    "You are DocMind, an assistant that answers questions using only the "
    "context inside <context> tags below, which was retrieved from the "
    "user's own uploaded documents. The user's question is inside "
    "<question> tags.\n\n"
    "Rules:\n"
    "- Answer using only the provided context — never outside knowledge.\n"
    "- If the context doesn't contain enough information to answer, say so "
    "plainly instead of guessing.\n"
    '- Cite the source filename for every claim you make, e.g. "(source: report.pdf)".\n'
    "- Be concise and direct."
)


# ── LLM client ───────────────────────────────────────────────────────────────


class LLMClient:
    """Thin async wrapper around the Anthropic Messages API.

    Mirrors EmbeddingClient's shape (api/documents/embeddings.py): built
    from Settings via `from_settings`, with one narrow async method that
    hides the SDK's exact call shape. That narrow surface is what makes
    this trivial to fake in tests — a `FakeLLMClient` only has to be an
    async generator function, not a replica of Anthropic's streaming
    context-manager protocol.
    """

    def __init__(self, api_key: str | None, model: str = CHAT_MODEL) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        """Convenience constructor that reads config from the Settings object."""
        return cls(api_key=settings.anthropic_api_key)

    async def stream_reply(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield response text tokens as they arrive from the model."""
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=MAX_RESPONSE_TOKENS,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text


def _get_llm_client() -> LLMClient:
    """Build a fresh LLMClient from the current settings.

    Constructed lazily, at call time — not at import time or app startup —
    so a missing ANTHROPIC_API_KEY only breaks the chat endpoint, not the
    whole API. Mirrors worker/tasks.py's `_get_storage_client()` /
    `_get_embedding_client()` helpers; tests monkeypatch this function the
    same way.
    """
    return LLMClient.from_settings(get_settings())


# ── Retrieval ────────────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """One chunk of a document surfaced by similarity search — just enough
    metadata to build both the LLM context block and the citation shown
    to the user."""

    document_id: uuid.UUID
    filename: str
    chunk_index: int
    content: str


async def retrieve_relevant_chunks(
    db: AsyncSession,
    user_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int = TOP_K_CHUNKS,
    document_ids: Sequence[uuid.UUID] | None = None,
) -> list[RetrievedChunk]:
    """Return the *top_k* chunks (owned by *user_id*) closest to *query_embedding*.

    Uses pgvector's cosine-distance operator (`<=>`) directly via `.op()`,
    matching the ivfflat/vector_cosine_ops index created in migration 0003
    — smaller distance means more similar, so the default ascending
    ORDER BY is already correct.

    The query vector is bound with an explicit `EmbeddingVector` type
    (rather than relying on type inference through `.op()`) so it's
    unambiguous which bind/result processor serialises it — since
    `document_chunks.embedding` goes through our custom TypeDecorator
    rather than pgvector's `Vector` directly, being explicit here removes
    any doubt about how the literal gets bound.

    Postgres-only: SQLite has no `<=>` operator, so this function is
    exercised by @pytest.mark.integration tests, not the standalone suite
    — see tests/test_chat.py.
    """
    query_vector = literal(query_embedding, type_=EmbeddingVector())

    stmt = (
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.user_id == user_id)
    )
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    stmt = stmt.order_by(DocumentChunk.embedding.op("<=>")(query_vector)).limit(top_k)

    result = await db.execute(stmt)
    return [
        RetrievedChunk(
            document_id=chunk.document_id,
            filename=filename,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
        )
        for chunk, filename in result.all()
    ]


# ── Prompt building (pure — no I/O) ─────────────────────────────────────────


def format_context_block(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks into the context block injected into the prompt.

    Pure string formatting — fully unit-testable without a database.
    """
    if not chunks:
        return "(No relevant content was found in the user's documents.)"

    parts = [
        f"[{i}] Source: {chunk.filename} (chunk {chunk.chunk_index})\n{chunk.content}"
        for i, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(parts)


def build_messages(
    history: Sequence[ChatMessage], user_message: str, context_block: str
) -> list[dict[str, str]]:
    """Assemble the Anthropic `messages` list: prior turns + the new question,
    with retrieved context folded into the final user turn.

    Only plain history content is replayed — the context retrieved for
    *previous* turns isn't re-injected on every subsequent turn, which
    would make the prompt grow unbounded over a long conversation. Each
    turn does its own retrieval instead.

    `history` rows are always inserted in user/assistant pairs (see
    `create_message_pair`), so this naturally alternates roles and starts
    with "user" — exactly what the Messages API expects — before this
    function appends one more "user" turn for the live question.
    """
    messages: list[dict[str, str]] = [{"role": msg.role, "content": msg.content} for msg in history]
    messages.append(
        {
            "role": MessageRole.USER,
            "content": (
                f"<context>\n{context_block}\n</context>\n\n<question>\n{user_message}\n</question>"
            ),
        }
    )
    return messages


def _truncate(text: str, max_chars: int = _EXCERPT_MAX_CHARS) -> str:
    """Shorten *text* to *max_chars*, appending an ellipsis if it was cut."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def chunks_to_sources(chunks: Sequence[RetrievedChunk]) -> list[dict]:
    """Convert retrieved chunks into the JSON-serialisable `sources` shape
    persisted on the assistant's ChatMessage row."""
    return [
        {
            "document_id": str(chunk.document_id),
            "filename": chunk.filename,
            "chunk_index": chunk.chunk_index,
            "excerpt": _truncate(chunk.content),
        }
        for chunk in chunks
    ]


# ── Orchestration ────────────────────────────────────────────────────────────


async def retrieve_and_stream(
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    user_message: str,
    history: Sequence[ChatMessage],
    embeddings: EmbeddingClient,
    user_submitted_at: datetime,
    document_ids: Sequence[uuid.UUID] | None = None,
) -> AsyncIterator[str]:
    """The full RAG turn: embed -> retrieve -> build prompt -> stream -> persist.

    Opens its own DB session rather than accepting one via FastAPI's
    Depends(get_db). By the time an SSE response body is actually being
    streamed, FastAPI has already torn down the request-scoped dependency
    (its cleanup runs as the route handler returns the Response object,
    which happens *before* Starlette ever iterates that response's body —
    i.e. before this generator is iterated). Using a request-scoped
    session in here would mean reusing an already-closed AsyncSession.
    This mirrors how worker/tasks.py opens `AsyncSessionLocal()` directly
    for the same reason: no framework-managed scope covers its whole
    lifetime. `embeddings` is safe to receive as a parameter, though —
    `get_embedding_client` has no teardown (it just reads app.state), so
    it stays usable for as long as the app is running.

    Any failure anywhere in this turn (embedding, retrieval, or the LLM
    call) is caught, logged, and turned into a short apology yielded to
    the client — plus a matching assistant turn persisted with empty
    sources — rather than left to crash the stream mid-response.
    """
    try:
        [query_embedding] = await embeddings.embed_texts([user_message])
        llm = _get_llm_client()

        async with AsyncSessionLocal() as db:
            chunks = await retrieve_relevant_chunks(
                db, user_id, query_embedding, document_ids=document_ids
            )
            context_block = format_context_block(chunks)
            messages = build_messages(history, user_message, context_block)

            reply_parts: list[str] = []
            async for text in llm.stream_reply(system=SYSTEM_PROMPT, messages=messages):
                reply_parts.append(text)
                yield text

            full_reply = "".join(reply_parts).strip()
            sources = chunks_to_sources(chunks)
            await create_message_pair(
                db, session_id, user_message, full_reply, sources, user_submitted_at
            )
            logger.info(
                "RAG turn complete for session %s (%d chunks retrieved, %d chars generated)",
                session_id,
                len(chunks),
                len(full_reply),
            )
    except Exception:
        logger.exception("RAG turn failed for session %s", session_id)
        apology = "Sorry — something went wrong generating a response. Please try again."
        yield apology
        async with AsyncSessionLocal() as db:
            await create_message_pair(db, session_id, user_message, apology, [], user_submitted_at)
