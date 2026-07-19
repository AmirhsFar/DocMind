"""Tests for the /chat endpoints and the RAG pipeline (api/chat/*).

Three tiers, in order of increasing infrastructure need:
  - Pure functions (format_context_block, build_messages, chunks_to_sources,
    _truncate): no DB, no fakes, just plain assertions.
  - Session/message CRUD + retrieve_and_stream orchestration: standalone,
    SQLite + FakeEmbeddingClient + FakeLLMClient. retrieve_relevant_chunks
    itself is monkeypatched out in this tier since it relies on the
    pgvector `<=>` operator, which SQLite doesn't have.
  - @pytest.mark.integration: the real similarity-search SQL, which does
    need a live Postgres with pgvector — run via `make test` inside
    docker-compose, not the standalone suite.
"""

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import api.chat.rag as rag_module
from api.auth.models import User
from api.chat.models import ChatMessage, ChatSession
from api.core.database import AsyncSessionLocal, Base, get_db
from api.documents.dependencies import get_embedding_client
from api.documents.embeddings import EMBEDDING_DIMENSIONS
from api.documents.models import DocStatus, Document, DocumentChunk
from api.main import app
from tests.fakes import FakeEmbeddingClient, FakeLLMClient

_SQLITE_URL = "sqlite+aiosqlite:///:memory:"


# ── Fixtures — sessions/messages endpoints (never touch rag.py) ────────────────


@pytest_asyncio.fixture
async def chat_db() -> AsyncGenerator[AsyncSession, None]:
    """Fresh in-memory SQLite database, created and torn down per test."""
    engine = create_async_engine(_SQLITE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def chat_client(chat_db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Authenticated client for the session/listing endpoints. These never
    call get_embedding_client or touch rag.py, so a single shared SQLite
    session (like test_documents.py's docs_client) is all that's needed.
    """

    async def _get_db() -> AsyncGenerator[AsyncSession, None]:
        yield chat_db

    app.dependency_overrides[get_db] = _get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.post("/auth/signup", json={"email": "chatter@test.com", "password": "chatpass1"})
        login = await ac.post(
            "/auth/login", json={"email": "chatter@test.com", "password": "chatpass1"}
        )
        ac.headers.update({"Authorization": f"Bearer {login.json()['access_token']}"})
        yield ac

    app.dependency_overrides.clear()


async def _seed_session(
    factory: async_sessionmaker, user_id: uuid.UUID | None = None
) -> tuple[uuid.UUID, uuid.UUID]:
    """Insert a bare ChatSession directly (bypassing crud.py, same pattern
    as test_ingestion.py's _seed_document) and return (user_id, session_id).
    """
    async with factory() as db:
        user_id = user_id or uuid.uuid4()
        session = ChatSession(user_id=user_id)
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return user_id, session.id


# ── Fixtures — retrieve_and_stream orchestration (rag.py, no HTTP layer) ───────


@pytest_asyncio.fixture
async def rag_session() -> AsyncGenerator[async_sessionmaker, None]:
    """Yields the session *factory* itself (not one open session), so the
    test and retrieve_and_stream can each open their own session against
    the same database — mirrors test_ingestion.py's ingest_session.
    """
    engine = create_async_engine(_SQLITE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
def rag_fakes(monkeypatch, rag_session: async_sessionmaker) -> SimpleNamespace:
    """Patch rag.py's session factory, LLM client, and retrieval SQL.

    retrieve_relevant_chunks uses the pgvector `<=>` operator, which has no
    SQLite equivalent, so it's stubbed here to return whatever `.chunks`
    currently holds — tests append to that list to control what "was
    retrieved". Real similarity search is covered separately, further down
    this file, by the @pytest.mark.integration tests.
    """
    embeddings = FakeEmbeddingClient()
    llm = FakeLLMClient()
    chunks: list[rag_module.RetrievedChunk] = []

    async def _fake_retrieve(db, user_id, query_embedding, top_k=5, document_ids=None):
        return chunks

    monkeypatch.setattr(rag_module, "AsyncSessionLocal", rag_session)
    monkeypatch.setattr(rag_module, "_get_llm_client", lambda: llm)
    monkeypatch.setattr(rag_module, "retrieve_relevant_chunks", _fake_retrieve)

    return SimpleNamespace(embeddings=embeddings, llm=llm, chunks=chunks)


# ── Fixtures — full SSE endpoint (HTTP layer + rag.py together) ────────────────


@pytest_asyncio.fixture
async def rag_client(monkeypatch) -> AsyncGenerator[tuple[AsyncClient, FakeLLMClient], None]:
    """Full-stack client for POST .../messages: the httpx client's get_db
    override and rag.py's independently-opened AsyncSessionLocal both
    point at the same SQLite database. Embeddings, the LLM call, and
    retrieval are all faked, the same way as the `rag_fakes` tier above.
    """
    engine = create_async_engine(_SQLITE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    fake_embeddings = FakeEmbeddingClient()
    fake_llm = FakeLLMClient()

    async def _fake_retrieve(db, user_id, query_embedding, top_k=5, document_ids=None):
        return []

    monkeypatch.setattr(rag_module, "AsyncSessionLocal", factory)
    monkeypatch.setattr(rag_module, "_get_llm_client", lambda: fake_llm)
    monkeypatch.setattr(rag_module, "retrieve_relevant_chunks", _fake_retrieve)

    async def _get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_embedding_client] = lambda: fake_embeddings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.post("/auth/signup", json={"email": "rag@test.com", "password": "ragpassword1"})
        login = await ac.post(
            "/auth/login", json={"email": "rag@test.com", "password": "ragpassword1"}
        )
        ac.headers.update({"Authorization": f"Bearer {login.json()['access_token']}"})
        yield ac, fake_llm

    app.dependency_overrides.clear()
    await engine.dispose()


# ── Prompt building (pure functions — no DB, no fakes) ──────────────────────


def test_format_context_block_empty_says_so() -> None:
    assert "No relevant content" in rag_module.format_context_block([])


def test_format_context_block_numbers_and_labels_each_chunk() -> None:
    chunks = [
        rag_module.RetrievedChunk(uuid.uuid4(), "a.pdf", 0, "Alpha content"),
        rag_module.RetrievedChunk(uuid.uuid4(), "b.txt", 2, "Beta content"),
    ]
    block = rag_module.format_context_block(chunks)
    assert "[1] Source: a.pdf (chunk 0)" in block
    assert "Alpha content" in block
    assert "[2] Source: b.txt (chunk 2)" in block
    assert "Beta content" in block


def test_build_messages_no_history_is_a_single_user_turn() -> None:
    messages = rag_module.build_messages([], "hello?", "some context")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "hello?" in messages[0]["content"]
    assert "some context" in messages[0]["content"]


def test_build_messages_preserves_history_roles_and_order() -> None:
    history = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="second"),
    ]
    messages = rag_module.build_messages(history, "third", "ctx")
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "first"
    assert messages[1]["content"] == "second"
    assert "third" in messages[2]["content"]


def test_chunks_to_sources_shape() -> None:
    doc_id = uuid.uuid4()
    chunks = [rag_module.RetrievedChunk(doc_id, "x.pdf", 1, "short text")]
    assert rag_module.chunks_to_sources(chunks) == [
        {
            "document_id": str(doc_id),
            "filename": "x.pdf",
            "chunk_index": 1,
            "excerpt": "short text",
        }
    ]


def test_chunks_to_sources_truncates_long_content() -> None:
    long_text = "word " * 200
    chunks = [rag_module.RetrievedChunk(uuid.uuid4(), "x.pdf", 0, long_text)]
    excerpt = rag_module.chunks_to_sources(chunks)[0]["excerpt"]
    assert excerpt.endswith("…")
    assert len(excerpt) <= rag_module._EXCERPT_MAX_CHARS + 1


def test_truncate_leaves_short_text_untouched() -> None:
    assert rag_module._truncate("short") == "short"


# ── Sessions ──────────────────────────────────────────────────────────────────


async def test_create_session_default_has_no_title(chat_client: AsyncClient) -> None:
    resp = await chat_client.post("/chat/sessions")
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] is None
    assert "id" in body
    assert "created_at" in body
    assert "updated_at" in body


async def test_create_session_with_title(chat_client: AsyncClient) -> None:
    resp = await chat_client.post("/chat/sessions", json={"title": "Trip planning"})
    assert resp.status_code == 201
    assert resp.json()["title"] == "Trip planning"


async def test_create_session_requires_auth(chat_client: AsyncClient) -> None:
    resp = await chat_client.post("/chat/sessions", headers={"Authorization": ""})
    assert resp.status_code == 401


async def test_list_sessions_empty(chat_client: AsyncClient) -> None:
    resp = await chat_client.get("/chat/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_sessions_newest_first(chat_client: AsyncClient) -> None:
    await chat_client.post("/chat/sessions", json={"title": "First"})
    await chat_client.post("/chat/sessions", json={"title": "Second"})
    resp = await chat_client.get("/chat/sessions")
    assert [s["title"] for s in resp.json()] == ["Second", "First"]


async def test_update_session_title(chat_client: AsyncClient) -> None:
    create = await chat_client.post("/chat/sessions")
    session_id = create.json()["id"]
    resp = await chat_client.patch(
        f"/chat/sessions/{session_id}", json={"title": "Auto-Generated Title"}
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "Auto-Generated Title"
    fetch = await chat_client.get("/chat/sessions")
    assert fetch.json()[0]["title"] == "Auto-Generated Title"


# ── Message listing ──────────────────────────────────────────────────────────


async def test_list_messages_empty_session(chat_client: AsyncClient) -> None:
    create = await chat_client.post("/chat/sessions")
    session_id = create.json()["id"]
    resp = await chat_client.get(f"/chat/sessions/{session_id}/messages")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_messages_nonexistent_session(chat_client: AsyncClient) -> None:
    resp = await chat_client.get(f"/chat/sessions/{uuid.uuid4()}/messages")
    assert resp.status_code == 404


# ── User isolation ────────────────────────────────────────────────────────────


async def test_session_isolation(chat_client: AsyncClient) -> None:
    """User B must get 404 for User A's session — never 403 — and must
    never see it in their own list."""
    create = await chat_client.post("/chat/sessions", json={"title": "Private"})
    session_id = create.json()["id"]

    await chat_client.post(
        "/auth/signup", json={"email": "intruder@test.com", "password": "intruder1"}
    )
    login_b = await chat_client.post(
        "/auth/login", json={"email": "intruder@test.com", "password": "intruder1"}
    )
    token_b = f"Bearer {login_b.json()['access_token']}"

    resp = await chat_client.get(
        f"/chat/sessions/{session_id}/messages", headers={"Authorization": token_b}
    )
    assert resp.status_code == 404

    resp = await chat_client.get("/chat/sessions", headers={"Authorization": token_b})
    assert resp.json() == []


# ── retrieve_and_stream orchestration ───────────────────────────────────────


async def test_retrieve_and_stream_happy_path(rag_fakes: SimpleNamespace, rag_session) -> None:
    rag_fakes.llm.reply = "Paris is the capital of France."
    rag_fakes.chunks.append(
        rag_module.RetrievedChunk(uuid.uuid4(), "geo.txt", 0, "Paris is the capital of France.")
    )
    user_id, session_id = await _seed_session(rag_session)

    tokens = [
        token
        async for token in rag_module.retrieve_and_stream(
            session_id=session_id,
            user_id=user_id,
            user_message="What is the capital of France?",
            history=[],
            embeddings=rag_fakes.embeddings,
            user_submitted_at=datetime.now(UTC),
        )
    ]
    assert "".join(tokens).strip() == "Paris is the capital of France."

    async with rag_session() as db:
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        messages = result.scalars().all()

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "What is the capital of France?"
    assert messages[0].sources is None
    assert messages[1].role == "assistant"
    assert messages[1].content == "Paris is the capital of France."
    assert messages[1].sources[0]["filename"] == "geo.txt"

    assert rag_fakes.embeddings.embedded_batches == [["What is the capital of France?"]]
    assert len(rag_fakes.llm.calls) == 1
    assert "geo.txt" in rag_fakes.llm.calls[0]["messages"][-1]["content"]


async def test_retrieve_and_stream_with_no_matching_chunks(
    rag_fakes: SimpleNamespace, rag_session
) -> None:
    user_id, session_id = await _seed_session(rag_session)

    async for _ in rag_module.retrieve_and_stream(
        session_id=session_id,
        user_id=user_id,
        user_message="Anything in here?",
        history=[],
        embeddings=rag_fakes.embeddings,
        user_submitted_at=datetime.now(UTC),
    ):
        pass

    assert "No relevant content" in rag_fakes.llm.calls[0]["messages"][-1]["content"]

    async with rag_session() as db:
        result = await db.execute(select(ChatMessage).where(ChatMessage.session_id == session_id))
        messages = result.scalars().all()
    assistant_msg = next(m for m in messages if m.role == "assistant")
    assert assistant_msg.sources == []


async def test_retrieve_and_stream_includes_prior_history(
    rag_fakes: SimpleNamespace, rag_session
) -> None:
    user_id, session_id = await _seed_session(rag_session)
    history = [
        ChatMessage(session_id=session_id, role="user", content="Hi"),
        ChatMessage(session_id=session_id, role="assistant", content="Hello! How can I help?"),
    ]

    async for _ in rag_module.retrieve_and_stream(
        session_id=session_id,
        user_id=user_id,
        user_message="Follow-up question",
        history=history,
        embeddings=rag_fakes.embeddings,
        user_submitted_at=datetime.now(UTC),
    ):
        pass

    sent = rag_fakes.llm.calls[0]["messages"]
    assert sent[0] == {"role": "user", "content": "Hi"}
    assert sent[1] == {"role": "assistant", "content": "Hello! How can I help?"}
    assert sent[2]["role"] == "user"
    assert "Follow-up question" in sent[2]["content"]


async def test_retrieve_and_stream_handles_llm_failure_gracefully(
    rag_fakes: SimpleNamespace, rag_session
) -> None:
    rag_fakes.llm.should_fail = True
    user_id, session_id = await _seed_session(rag_session)

    tokens = [
        token
        async for token in rag_module.retrieve_and_stream(
            session_id=session_id,
            user_id=user_id,
            user_message="Will this work?",
            history=[],
            embeddings=rag_fakes.embeddings,
            user_submitted_at=datetime.now(UTC),
        )
    ]
    assert tokens  # something was yielded — the client isn't left hanging

    async with rag_session() as db:
        result = await db.execute(select(ChatMessage).where(ChatMessage.session_id == session_id))
        messages = result.scalars().all()
    assert len(messages) == 2
    assert {m.role for m in messages} == {"user", "assistant"}
    assistant_msg = next(m for m in messages if m.role == "assistant")
    assert assistant_msg.sources == []


# ── SSE endpoint (full HTTP stack) ──────────────────────────────────────────


async def test_send_message_streams_and_persists(rag_client: tuple) -> None:
    client, fake_llm = rag_client
    fake_llm.reply = "The answer is 42."

    create = await client.post("/chat/sessions")
    session_id = create.json()["id"]

    resp = await client.post(
        f"/chat/sessions/{session_id}/messages", json={"content": "What is the answer?"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "42" in resp.text
    assert "[DONE]" in resp.text

    history = await client.get(f"/chat/sessions/{session_id}/messages")
    messages = history.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is the answer?"
    assert messages[1]["role"] == "assistant"
    assert "42" in messages[1]["content"]


async def test_send_message_requires_auth(rag_client: tuple) -> None:
    client, _ = rag_client
    create = await client.post("/chat/sessions")
    session_id = create.json()["id"]
    resp = await client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"content": "hi"},
        headers={"Authorization": ""},
    )
    assert resp.status_code == 401


async def test_send_message_nonexistent_session(rag_client: tuple) -> None:
    client, _ = rag_client
    resp = await client.post(f"/chat/sessions/{uuid.uuid4()}/messages", json={"content": "hi"})
    assert resp.status_code == 404


async def test_send_message_empty_content_rejected(rag_client: tuple) -> None:
    client, _ = rag_client
    create = await client.post("/chat/sessions")
    session_id = create.json()["id"]
    resp = await client.post(f"/chat/sessions/{session_id}/messages", json={"content": ""})
    assert resp.status_code == 422


# ── Integration: real pgvector similarity search ────────────────────────────
#
# These need a live Postgres with the vector extension. Run via `make test`
# inside docker-compose — not part of the standalone suite.


def _one_hot(index: int, dims: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """A unit vector with a 1.0 in exactly one position — makes cosine
    distance unambiguous: 0 against itself, 1 against any other _one_hot."""
    vector = [0.0] * dims
    vector[index] = 1.0
    return vector


@pytest.mark.integration
async def test_retrieve_relevant_chunks_orders_by_similarity() -> None:
    async with AsyncSessionLocal() as db:
        user = User(email=f"vec-{uuid.uuid4()}@test.com", hashed_password="x")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        doc = Document(
            user_id=user.id,
            filename="vec.txt",
            storage_key=f"{user.id}/vec/vec.txt",
            content_type="text/plain",
            size_bytes=1,
            status=DocStatus.READY,
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        db.add_all(
            [
                DocumentChunk(
                    document_id=doc.id,
                    user_id=user.id,
                    chunk_index=0,
                    content="near",
                    embedding=_one_hot(0),
                ),
                DocumentChunk(
                    document_id=doc.id,
                    user_id=user.id,
                    chunk_index=1,
                    content="far",
                    embedding=_one_hot(1),
                ),
            ]
        )
        await db.commit()

        results = await rag_module.retrieve_relevant_chunks(db, user.id, _one_hot(0), top_k=2)
        assert [r.content for r in results] == ["near", "far"]

        await db.delete(user)  # cascades to doc + both chunks
        await db.commit()


@pytest.mark.integration
async def test_retrieve_relevant_chunks_respects_document_ids_filter() -> None:
    async with AsyncSessionLocal() as db:
        user = User(email=f"vec-{uuid.uuid4()}@test.com", hashed_password="x")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        doc_a = Document(
            user_id=user.id,
            filename="a.txt",
            storage_key=f"{user.id}/a/a.txt",
            content_type="text/plain",
            size_bytes=1,
            status=DocStatus.READY,
        )
        doc_b = Document(
            user_id=user.id,
            filename="b.txt",
            storage_key=f"{user.id}/b/b.txt",
            content_type="text/plain",
            size_bytes=1,
            status=DocStatus.READY,
        )
        db.add_all([doc_a, doc_b])
        await db.commit()
        await db.refresh(doc_a)
        await db.refresh(doc_b)

        # Both chunks are equally close to the query — the filter, not
        # distance, is what should decide which one comes back.
        db.add_all(
            [
                DocumentChunk(
                    document_id=doc_a.id,
                    user_id=user.id,
                    chunk_index=0,
                    content="from a",
                    embedding=_one_hot(0),
                ),
                DocumentChunk(
                    document_id=doc_b.id,
                    user_id=user.id,
                    chunk_index=0,
                    content="from b",
                    embedding=_one_hot(0),
                ),
            ]
        )
        await db.commit()

        results = await rag_module.retrieve_relevant_chunks(
            db, user.id, _one_hot(0), top_k=5, document_ids=[doc_a.id]
        )
        assert [r.content for r in results] == ["from a"]

        await db.delete(user)  # cascades to both docs + both chunks
        await db.commit()
