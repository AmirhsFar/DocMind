"""Celery tasks for DocMind.

Phase 3 replaces the process_document stub with the real ingestion
pipeline: extract text -> split into overlapping chunks -> embed each
chunk -> store the vectors in Postgres via pgvector.
"""

import asyncio
import io
import logging
import uuid

import pypdf
import tiktoken
from celery.exceptions import MaxRetriesExceededError
from pypdf.errors import PyPdfError

from api.core.config import get_settings
from api.core.database import AsyncSessionLocal
from api.documents.crud import replace_document_chunks, update_document_status
from api.documents.embeddings import EmbeddingClient
from api.documents.models import DocStatus, Document, DocumentChunk
from api.documents.storage import StorageClient
from worker.celery_app import celery_app

logger = logging.getLogger("docmind.worker")

# Resolved as plain module attributes so tests can monkeypatch them directly
# (same pattern as `_MAX_BYTES` in api/documents/router.py). `AsyncSessionLocal`
# is already a bare, swappable factory; storage/embeddings need a thin wrapper
# function because constructing them requires the *current* Settings.
_CHUNK_SIZE_TOKENS = 512
_CHUNK_OVERLAP_TOKENS = 50
_TOKENIZER_ENCODING = "cl100k_base"


def _get_storage_client() -> StorageClient:
    return StorageClient.from_settings(get_settings())


def _get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient.from_settings(get_settings())


class _UnrecoverableIngestionError(Exception):
    """The document itself is the problem — retrying the task won't help."""


def _split_into_chunks(text: str) -> list[str]:
    """Split *text* into overlapping windows of `_CHUNK_SIZE_TOKENS` tokens.

    Chunking on tokens (via tiktoken) rather than characters or words keeps
    each chunk close to the embedding model's effective context size
    regardless of how dense the source text is.
    """
    encoding = tiktoken.get_encoding(_TOKENIZER_ENCODING)
    tokens = encoding.encode(text)
    if not tokens:
        return []

    stride = _CHUNK_SIZE_TOKENS - _CHUNK_OVERLAP_TOKENS
    chunks: list[str] = []
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + _CHUNK_SIZE_TOKENS]
        chunks.append(encoding.decode(window))
        if start + _CHUNK_SIZE_TOKENS >= len(tokens):
            break
    return chunks


def _extract_text(document_id: str, content_type: str, data: bytes) -> str:
    """Extract raw text from uploaded file bytes.

    Raises `_UnrecoverableIngestionError` for content that can never be
    parsed — a corrupt PDF or invalid encoding won't fix itself on retry.
    """
    if content_type == "application/pdf":
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except PyPdfError as exc:
            raise _UnrecoverableIngestionError(
                f"Document {document_id} is not a readable PDF: {exc}"
            ) from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _UnrecoverableIngestionError(
            f"Document {document_id} is not valid UTF-8 text: {exc}"
        ) from exc


async def _ingest_document(document_id: str) -> None:
    """The real Phase 3 pipeline, run inside its own DB session.

    Kept as a plain async function (rather than being the task body itself)
    so tests can `await` it directly instead of going through the sync/async
    bridge that `process_document` uses to call it from Celery.
    """
    doc_uuid = uuid.UUID(document_id)

    async with AsyncSessionLocal() as db:
        document = await db.get(Document, doc_uuid)
        if document is None:
            raise _UnrecoverableIngestionError(f"Document {document_id} not found")

        await update_document_status(db, document, DocStatus.PROCESSING)

        try:
            storage = _get_storage_client()
            raw_bytes = await storage.get_object(document.storage_key)
            text = _extract_text(document_id, document.content_type, raw_bytes)

            if not text.strip():
                raise _UnrecoverableIngestionError(
                    f"Document {document_id} contains no extractable text"
                )

            chunk_texts = _split_into_chunks(text)
            if not chunk_texts:
                raise _UnrecoverableIngestionError(f"Document {document_id} produced zero chunks")

            embeddings = _get_embedding_client()
            vectors = await embeddings.embed_texts(chunk_texts)

            chunk_rows = [
                DocumentChunk(
                    document_id=doc_uuid,
                    user_id=document.user_id,
                    chunk_index=index,
                    content=chunk_text,
                    embedding=vector,
                )
                for index, (chunk_text, vector) in enumerate(zip(chunk_texts, vectors, strict=True))
            ]
            await replace_document_chunks(db, doc_uuid, chunk_rows)

            await update_document_status(db, document, DocStatus.READY)
            logger.info("Ingested document %s into %d chunks", document_id, len(chunk_rows))
        except _UnrecoverableIngestionError:
            await update_document_status(db, document, DocStatus.FAILED)
            raise


async def _mark_document_failed(document_id: str) -> None:
    """Best-effort status flip used once Celery's automatic retries run out."""
    async with AsyncSessionLocal() as db:
        document = await db.get(Document, uuid.UUID(document_id))
        if document is not None:
            await update_document_status(db, document, DocStatus.FAILED)


@celery_app.task(name="worker.process_document", bind=True, max_retries=3)
def process_document(self, document_id: str) -> None:
    """Ingest a newly uploaded document: extract, chunk, embed, store.

    `bind=True` + `max_retries=3`: transient failures (MinIO hiccup, OpenAI
    rate limit, ...) are retried up to three times with exponential
    back-off. Failures caused by the document itself (corrupt file, no
    extractable text) are not retried — `_ingest_document` marks the
    document "failed" immediately and re-raises.

    Celery tasks are plain sync callables, so `asyncio.run` bridges into the
    async pipeline — the same bridge alembic/env.py uses for its migrations.
    """
    try:
        asyncio.run(_ingest_document(document_id))
    except _UnrecoverableIngestionError as exc:
        logger.error("process_document: unrecoverable error for %s: %s", document_id, exc)
        raise
    except Exception as exc:
        logger.warning(
            "process_document: transient error for %s (attempt %d/%d): %s",
            document_id,
            self.request.retries + 1,
            self.max_retries,
            exc,
        )
        try:
            # self.retry() raises Retry (caught by the worker to reschedule
            # this task) unless retries are exhausted, in which case it
            # raises MaxRetriesExceededError instead.
            raise self.retry(exc=exc, countdown=2**self.request.retries) from exc
        except MaxRetriesExceededError:
            logger.error("process_document: retries exhausted for %s — marking failed", document_id)
            asyncio.run(_mark_document_failed(document_id))
            raise
