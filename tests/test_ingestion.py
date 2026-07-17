"""Tests for the Phase 3 ingestion pipeline (worker/tasks.py).

All tests run standalone — no Redis, no real Postgres, no real OpenAI
account required. `_ingest_document` (the async core of `process_document`)
is awaited directly so it shares an event loop with the test, storage is
`FakeStorageClient`, and embeddings are `FakeEmbeddingClient`.

`process_document` itself (the thin `@celery_app.task` wrapper that bridges
into `_ingest_document` via `asyncio.run`) is intentionally not exercised
here — testing it would mean either nesting `asyncio.run` inside a running
event loop or crossing event-loop boundaries with a SQLite connection,
neither of which is a reliable thing to assert on. Its job is a two-line
sync/async bridge; `_ingest_document` is where the actual logic lives.
"""

import uuid
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import worker.tasks as tasks_module
from api.core.database import Base
from api.documents.models import DocStatus, Document, DocumentChunk
from tests.fakes import FakeEmbeddingClient, FakeStorageClient

_SQLITE_URL = "sqlite+aiosqlite:///:memory:"
_LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 300

# Hand-built minimal PDF (no real xref table — pypdf's recovery mode reads
# it anyway). Verified separately to extract to exactly "Hello DocMind".
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]"
    b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
    b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"5 0 obj<</Length 45>>stream\n"
    b"BT /F1 24 Tf 72 100 Td (Hello DocMind) Tj ET\n"
    b"endstream\n"
    b"endobj\n"
    b"xref\n"
    b"0 6\n"
    b"trailer<</Size 6/Root 1 0 R>>\n"
    b"startxref\n"
    b"0\n"
    b"%%EOF"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def ingest_session() -> AsyncGenerator[async_sessionmaker, None]:
    """Fresh in-memory SQLite database; yields the session *factory* itself
    (not a single session) so the test and `_ingest_document` can each open
    their own session against the same database — exactly how
    `AsyncSessionLocal` is used in the real application.
    """
    engine = create_async_engine(_SQLITE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
def fakes(monkeypatch, ingest_session: async_sessionmaker):
    """Patch the worker's session factory + storage + embedding clients."""
    storage = FakeStorageClient()
    embeddings = FakeEmbeddingClient()
    monkeypatch.setattr(tasks_module, "AsyncSessionLocal", ingest_session)
    monkeypatch.setattr(tasks_module, "_get_storage_client", lambda: storage)
    monkeypatch.setattr(tasks_module, "_get_embedding_client", lambda: embeddings)
    return storage, embeddings


async def _seed_document(
    factory: async_sessionmaker,
    *,
    content_type: str = "text/plain",
) -> Document:
    async with factory() as db:
        doc = Document(
            user_id=uuid.uuid4(),
            filename="notes.txt",
            storage_key="user/doc/notes.txt",
            content_type=content_type,
            size_bytes=len(_LONG_TEXT),
            status=DocStatus.PENDING,
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        return doc


# ── Chunking ──────────────────────────────────────────────────────────────────


def test_split_into_chunks_empty_text_returns_nothing() -> None:
    assert tasks_module._split_into_chunks("") == []


def test_split_into_chunks_short_text_is_a_single_chunk() -> None:
    chunks = tasks_module._split_into_chunks("just a handful of words")
    assert len(chunks) == 1
    assert "handful" in chunks[0]


def test_split_into_chunks_long_text_produces_multiple_chunks() -> None:
    chunks = tasks_module._split_into_chunks(_LONG_TEXT)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) and chunk for chunk in chunks)


# ── Text extraction ───────────────────────────────────────────────────────────


def test_extract_text_plain_utf8() -> None:
    text = tasks_module._extract_text("doc-1", "text/plain", b"hello there")
    assert text == "hello there"


def test_extract_text_pdf() -> None:
    text = tasks_module._extract_text("doc-1", "application/pdf", _MINIMAL_PDF)
    assert "Hello DocMind" in text


def test_extract_text_invalid_utf8_is_unrecoverable() -> None:
    with pytest.raises(tasks_module._UnrecoverableIngestionError):
        tasks_module._extract_text("doc-1", "text/plain", b"\xff\xfe\x00bad")


def test_extract_text_corrupt_pdf_is_unrecoverable() -> None:
    with pytest.raises(tasks_module._UnrecoverableIngestionError):
        tasks_module._extract_text("doc-1", "application/pdf", b"not a real pdf")


# ── End-to-end ingestion ──────────────────────────────────────────────────────


async def test_ingest_document_happy_path(fakes, ingest_session) -> None:
    storage, embeddings = fakes
    doc = await _seed_document(ingest_session)
    await storage.put_object(doc.storage_key, _LONG_TEXT.encode(), doc.content_type)

    await tasks_module._ingest_document(str(doc.id))

    async with ingest_session() as db:
        refreshed = await db.get(Document, doc.id)
        assert refreshed.status == DocStatus.READY

        result = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc.id)
            .order_by(DocumentChunk.chunk_index)
        )
        chunks = result.scalars().all()

    assert len(chunks) > 1
    assert all(c.user_id == doc.user_id for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Every stored chunk was actually sent to the embedding client.
    assert sum(len(batch) for batch in embeddings.embedded_batches) == len(chunks)


async def test_ingest_document_empty_file_marks_failed(fakes, ingest_session) -> None:
    storage, _ = fakes
    doc = await _seed_document(ingest_session)
    await storage.put_object(doc.storage_key, b"   \n\t  ", doc.content_type)

    with pytest.raises(tasks_module._UnrecoverableIngestionError):
        await tasks_module._ingest_document(str(doc.id))

    async with ingest_session() as db:
        refreshed = await db.get(Document, doc.id)
        assert refreshed.status == DocStatus.FAILED


async def test_ingest_document_missing_row_raises(fakes) -> None:
    with pytest.raises(tasks_module._UnrecoverableIngestionError):
        await tasks_module._ingest_document(str(uuid.uuid4()))


async def test_ingest_document_rerun_replaces_rather_than_duplicates(fakes, ingest_session) -> None:
    """A second run over the same document must replace, not duplicate, chunks."""
    storage, _ = fakes
    doc = await _seed_document(ingest_session)
    await storage.put_object(doc.storage_key, _LONG_TEXT.encode(), doc.content_type)

    await tasks_module._ingest_document(str(doc.id))
    await tasks_module._ingest_document(str(doc.id))

    async with ingest_session() as db:
        result = await db.execute(select(DocumentChunk).where(DocumentChunk.document_id == doc.id))
        chunks = result.scalars().all()

    assert len(chunks) == len(tasks_module._split_into_chunks(_LONG_TEXT))
