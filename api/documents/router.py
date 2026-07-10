"""Document management endpoints.

Upload flow (the key teaching moment for Phase 2):
  1. Validate content-type and file size
  2. PUT the raw bytes to MinIO  ← if this fails, nothing is in the DB yet
  3. INSERT a Document row with status="pending"
  4. Dispatch the Celery task (Phase 3 adds the real ingestion logic)
  5. Return 201 with document metadata

Download flow:
  - GET /documents/{id}/download  →  307 redirect to a 1-hour presigned MinIO URL
  - The client follows the redirect directly to MinIO — the API server never
    proxies the file bytes, keeping memory usage flat regardless of file size.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import get_current_user
from api.auth.models import User
from api.core.config import get_settings
from api.core.database import get_db
from api.documents.crud import (
    create_document,
    delete_document,
    get_document_by_id,
    get_documents_by_user,
)
from api.documents.dependencies import get_storage
from api.documents.models import Document
from api.documents.schemas import DocumentOut
from api.documents.storage import StorageClient
from worker.tasks import process_document

logger = logging.getLogger("docmind.documents")
settings = get_settings()

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/x-markdown",
    }
)

# Resolved once at import time from settings so tests can monkeypatch this module attribute.
_MAX_BYTES: int = settings.max_upload_size_mb * 1024 * 1024


# ── Upload ────────────────────────────────────────────────────────────────────


@router.post(
    "/",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
)
async def upload_document(
    file: UploadFile,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageClient = Depends(get_storage),
) -> Document:
    """Accept a PDF or plain-text file (max 10 MB) and start the ingestion pipeline.

    The file is stored in MinIO and a metadata record is created with
    `status="pending"`. The Celery worker (Phase 3) will update the status
    as it processes the file.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    content = await file.read()

    if len(content) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File is {len(content) / 1_048_576:.1f} MB — "
                f"the limit is {settings.max_upload_size_mb} MB."
            ),
        )

    document_id = uuid.uuid4()
    # Key encodes ownership so it's easy to audit access in MinIO logs.
    storage_key = f"{current_user.id}/{document_id}/{file.filename}"

    # Upload to MinIO before touching the DB.
    # If this raises, no DB record is created — nothing to roll back.
    await storage.put_object(key=storage_key, data=content, content_type=file.content_type)

    doc = await create_document(
        db,
        current_user.id,
        file.filename or "unnamed",
        storage_key,
        file.content_type,
        len(content),
    )

    # Fire-and-forget: queue the ingestion task.
    # .delay() puts a message on Redis without blocking; the worker picks it up
    # asynchronously.  Phase 3 replaces the stub with real extraction logic.
    process_document.delay(str(doc.id))

    logger.info(
        "Uploaded document %s (%d bytes) by user %s — ingestion queued",
        doc.id,
        doc.size_bytes,
        current_user.id,
    )
    return doc


# ── List ──────────────────────────────────────────────────────────────────────


@router.get(
    "/",
    response_model=list[DocumentOut],
    summary="List the current user's documents",
)
async def list_documents(
    skip: int = Query(default=0, ge=0, description="Offset for pagination"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results to return"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[Document]:
    return await get_documents_by_user(db, current_user.id, skip=skip, limit=limit)


# ── Get ───────────────────────────────────────────────────────────────────────


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Get metadata for a single document",
)
async def get_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    return await _owned_or_404(db, document_id, current_user)


# ── Download redirect ─────────────────────────────────────────────────────────


@router.get(
    "/{document_id}/download",
    summary="Redirect to a short-lived presigned download URL",
    response_class=RedirectResponse,
)
async def download_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageClient = Depends(get_storage),
) -> RedirectResponse:
    """Return a 307 redirect to a 1-hour presigned MinIO URL.

    The client follows the redirect and downloads the file directly from MinIO.
    The API server never touches the file bytes after the initial upload,
    keeping memory consumption constant regardless of file size.
    """
    doc = await _owned_or_404(db, document_id, current_user)
    url = await storage.presigned_get_url(doc.storage_key, expires_seconds=3600)
    return RedirectResponse(url=url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


# ── Delete ────────────────────────────────────────────────────────────────────


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Permanently delete a document",
)
async def delete_document_endpoint(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageClient = Depends(get_storage),
) -> None:
    """Remove the document from MinIO and delete its DB record.

    MinIO is cleaned up first — if that succeeds but the DB delete fails,
    the DB record will be an orphan (no file behind it). A Phase 6 cleanup
    job can sweep these up. The reverse order (DB first) would be worse:
    an inaccessible DB record pointing at a file that still exists.
    """
    doc = await _owned_or_404(db, document_id, current_user)
    await storage.delete_object(doc.storage_key)
    await delete_document(db, doc)
    logger.info("Deleted document %s (owner %s)", document_id, current_user.id)


# ── Private helpers ───────────────────────────────────────────────────────────


async def _owned_or_404(
    db: AsyncSession,
    document_id: uuid.UUID,
    current_user: User,
) -> Document:
    """Fetch a document and verify ownership.

    Returns 404 for both "not found" and "wrong owner" — never 403 — so
    callers cannot enumerate other users' document IDs by probing for 403s.
    """
    doc = await get_document_by_id(db, document_id, current_user.id)
    if doc is None or doc.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc
