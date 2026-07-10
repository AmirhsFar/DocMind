"""MinIO object storage client.

The official `minio` Python SDK is synchronous. We make it async-compatible
by offloading every blocking call to a thread pool via `asyncio.to_thread`.

This is the standard pattern for using sync I/O libraries inside an async
FastAPI app — important to understand because many mature Python libraries
(AWS SDK, some DB clients) don't yet have async variants.

Design note: the storage client is initialised once in `main.py`'s lifespan
and attached to `app.state.storage`. All routes access it through the
`get_storage` dependency (see `dependencies.py`), which makes it trivially
swappable in tests.
"""

import asyncio
import io
import logging
from datetime import timedelta

from minio import Minio
from minio.error import S3Error

from api.core.config import Settings

logger = logging.getLogger("docmind.storage")


class StorageClient:
    """Async-friendly wrapper around the MinIO SDK."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        self._client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._bucket = bucket

    @classmethod
    def from_settings(cls, settings: Settings) -> "StorageClient":
        """Convenience constructor that reads config from the Settings object."""
        return cls(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
        )

    async def ensure_bucket(self) -> None:
        """Create the configured bucket if it does not already exist (idempotent).

        Called once during application startup so the rest of the code can
        assume the bucket is always present.
        """
        exists = await asyncio.to_thread(self._client.bucket_exists, self._bucket)
        if not exists:
            await asyncio.to_thread(self._client.make_bucket, self._bucket)
            logger.info("Created MinIO bucket: %s", self._bucket)
        else:
            logger.debug("MinIO bucket already exists: %s", self._bucket)

    async def put_object(self, key: str, data: bytes, content_type: str) -> None:
        """Upload *data* to the bucket under *key*.

        Uses `asyncio.to_thread` so the sync MinIO call doesn't block the
        event loop while the bytes are streaming to MinIO.
        """
        try:
            await asyncio.to_thread(
                self._client.put_object,
                self._bucket,
                key,
                io.BytesIO(data),
                len(data),
                content_type=content_type,
            )
        except S3Error as exc:
            logger.error("MinIO put_object failed for key=%s: %s", key, exc)
            raise
        logger.debug("Stored object: %s (%d bytes)", key, len(data))

    async def delete_object(self, key: str) -> None:
        """Remove *key* from the bucket.

        Succeeds silently if the object does not exist — safe to call during
        cleanup even if the upload never completed.
        """
        try:
            await asyncio.to_thread(self._client.remove_object, self._bucket, key)
        except S3Error as exc:
            logger.error("MinIO delete_object failed for key=%s: %s", key, exc)
            raise
        logger.debug("Deleted object: %s", key)

    async def presigned_get_url(self, key: str, expires_seconds: int = 3600) -> str:
        """Return a time-limited presigned URL for direct download of *key*.

        The URL works without any auth headers — MinIO validates the embedded
        HMAC signature instead. Default expiry is 1 hour.
        """
        url: str = await asyncio.to_thread(
            self._client.presigned_get_object,
            self._bucket,
            key,
            expires=timedelta(seconds=expires_seconds),
        )
        return url
