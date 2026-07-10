"""Celery tasks for DocMind.

Phase 2: `process_document` is a stub that logs receipt and returns.
Phase 3: replaces the stub body with:
    1. Pull the Document record from Postgres and set status → "processing"
    2. Fetch the raw file from MinIO
    3. Extract text (pdfplumber / unstructured)
    4. Split into overlapping chunks
    5. Call the embedding API for each chunk
    6. Write chunk vectors to pgvector
    7. Set status → "ready"  (or "failed" on any unrecoverable error)
"""

import logging

from worker.celery_app import celery_app

logger = logging.getLogger("docmind.worker")


@celery_app.task(name="worker.process_document", bind=True, max_retries=3)
def process_document(self, document_id: str) -> None:
    """Ingest a newly uploaded document.

    `bind=True` gives access to `self` for retry logic (Phase 3).
    `max_retries=3` means transient failures (network blip, API rate limit)
    will be retried up to three times with exponential back-off.
    """
    logger.info(
        "process_document received document_id=%s — ingestion stub (Phase 3 adds real work)",
        document_id,
    )
