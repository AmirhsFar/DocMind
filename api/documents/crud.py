import uuid
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.documents.models import DocStatus, Document, DocumentChunk


async def create_document(
    db: AsyncSession,
    user_id: uuid.UUID,
    filename: str,
    storage_key: str,
    content_type: str,
    size_bytes: int,
) -> Document:
    """Inserts a new document record into the database."""
    db_doc = Document(
        user_id=user_id,
        filename=filename,
        storage_key=storage_key,
        content_type=content_type,
        size_bytes=size_bytes,
        status=DocStatus.PENDING,
    )
    db.add(db_doc)
    await db.commit()
    await db.refresh(db_doc)
    return db_doc


async def get_documents_by_user(
    db: AsyncSession, user_id: uuid.UUID, skip: int = 0, limit: int = 20
) -> Sequence[Document]:
    """Return *limit* documents owned by *user_id*, newest first (paginated)."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def get_document_by_id(
    db: AsyncSession, doc_id: uuid.UUID, user_id: uuid.UUID
) -> Document | None:
    """Fetches a specific document, ensuring it belongs to the requesting user."""
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def delete_document(db: AsyncSession, document: Document) -> None:
    """Removes the document record from the database."""
    await db.delete(document)
    await db.commit()


async def update_document_status(
    db: AsyncSession, document: Document, status: DocStatus
) -> Document:
    """Updates the ingestion status of a document."""
    document.status = status
    await db.commit()
    await db.refresh(document)
    return document


async def replace_document_chunks(
    db: AsyncSession, document_id: uuid.UUID, chunks: Sequence[DocumentChunk]
) -> None:
    """Atomically swap out all chunks belonging to *document_id* for *chunks*.

    Deletes any existing rows first so a retried ingestion run overwrites
    its own partial output instead of accumulating duplicates.
    """
    await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    db.add_all(chunks)
    await db.commit()
