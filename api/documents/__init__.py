"""Documents package — upload, list, retrieve, and delete user documents.

Storage:           MinIO (S3-compatible object store)
Metadata:          PostgreSQL (via SQLAlchemy)
Background work:   Celery worker (Phase 3 adds real text extraction / embedding)
"""

from api.documents.router import router

__all__ = ["router"]
