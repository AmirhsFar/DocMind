"""Fake implementations of external service clients.

These are in-memory replacements used by tests to avoid spinning up real
MinIO (or other external services) for every test run.

Pattern: each fake implements the same interface as the real client, so
FastAPI's dependency_overrides can swap them in transparently.
"""

from typing import Any


class FakeStorageClient:
    """In-memory substitute for `api.documents.storage.StorageClient`.

    Stores uploaded bytes in a plain dict keyed by the object key.
    Useful for asserting that uploads and deletes happened without
    needing a running MinIO instance.
    """

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}

    async def ensure_bucket(self) -> None:
        pass

    async def put_object(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = {"data": data, "content_type": content_type}

    async def delete_object(self, key: str) -> None:
        self.objects.pop(key, None)

    async def presigned_get_url(self, key: str, expires_seconds: int = 3600) -> str:
        # Return a deterministic fake URL so tests can assert on its shape.
        return f"http://fake-minio/{key}?expires={expires_seconds}"
