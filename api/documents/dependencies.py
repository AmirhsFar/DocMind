"""FastAPI dependency for the storage client.

The real client is stored on `app.state.storage` (set during lifespan startup).
In tests, override this dependency to inject a `FakeStorageClient`:

    from api.documents.dependencies import get_storage
    from tests.fakes import FakeStorageClient

    app.dependency_overrides[get_storage] = lambda: FakeStorageClient()
"""

from fastapi import Request

from api.documents.storage import StorageClient


async def get_storage(request: Request) -> StorageClient:
    storage: StorageClient | None = getattr(request.app.state, "storage", None)
    if storage is None:
        raise RuntimeError(
            "StorageClient not found on app.state — "
            "check that the lifespan startup completed without errors."
        )
    return storage
