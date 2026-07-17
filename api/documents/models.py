import uuid
from datetime import datetime
from enum import StrEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON, TypeDecorator

from api.core.database import Base

EMBEDDING_DIMENSIONS = 1536


class DocStatus(StrEnum):
    """Lifecycle states a document moves through as the ingestion pipeline runs.

    pending    → uploaded to MinIO, DB record created, Celery task queued
    processing → worker picked it up, text extraction underway
    ready      → chunks embedded and stored in pgvector, available for chat
    failed     → ingestion pipeline encountered an unrecoverable error
    """

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    """Uploaded PDF document metadata."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # MinIO object key — format: {user_id}/{document_id}/{filename}
    storage_key: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[DocStatus] = mapped_column(
        ENUM(DocStatus, name="doc_status_enum", create_type=False),
        default=DocStatus.PENDING,
        nullable=False,
    )
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
        return f"<Document id={self.id} filename={self.filename!r} status={self.status}>"


class EmbeddingVector(TypeDecorator):
    """Vector column that degrades to JSON on dialects without pgvector.

    Production talks to Postgres (with the `vector` extension), where this
    delegates entirely to `pgvector.sqlalchemy.Vector` — real similarity
    search, real storage format. Standalone tests run on SQLite, which has
    no vector type at all, so there we fall back to a JSON-encoded float
    array — enough for ingestion tests to insert and read back chunks.
    Real cosine-distance search only ever runs against Postgres (Phase 4).
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(EMBEDDING_DIMENSIONS))
        return dialect.type_descriptor(JSON())


class DocumentChunk(Base):
    """An embedded slice of a document's extracted text.

    `process_document` (worker/tasks.py) fans a Document out into many of
    these: one row per chunk, each with its own vector so Phase 4 can run
    a per-user similarity search directly against this table.
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    # Denormalised from documents.user_id so Phase 4's retrieval query can
    # filter WHERE user_id = :current_user without joining back to documents.
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        EmbeddingVector(EMBEDDING_DIMENSIONS), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk id={self.id} document_id={self.document_id} index={self.chunk_index}>"
        )
