"""FastAPI dependencies for the storage and embedding clients.

Both real clients are built once in main.py's lifespan and stored on
app.state (`app.state.storage`, `app.state.embedding_client`). In tests,
override these dependencies to inject fakes:

    from api.documents.dependencies import get_storage, get_embedding_client
    from tests.fakes import FakeStorageClient, FakeEmbeddingClient

    app.dependency_overrides[get_storage] = lambda: FakeStorageClient()
    app.dependency_overrides[get_embedding_client] = lambda: FakeEmbeddingClient()
"""

from fastapi import Request

from api.documents.embeddings import EmbeddingClient
from api.documents.storage import StorageClient


async def get_storage(request: Request) -> StorageClient:
    storage: StorageClient | None = getattr(request.app.state, "storage", None)
    if storage is None:
        raise RuntimeError(
            "StorageClient not found on app.state — "
            "check that the lifespan startup completed without errors."
        )
    return storage


async def get_embedding_client(request: Request) -> EmbeddingClient:
    """Used by api/chat/router.py to embed the user's question (Phase 4).

    The ingestion worker (worker/tasks.py) builds its own EmbeddingClient
    directly instead — it has no FastAPI request/app.state to read from.
    """
    embedding_client: EmbeddingClient | None = getattr(request.app.state, "embedding_client", None)
    if embedding_client is None:
        raise RuntimeError(
            "EmbeddingClient not found on app.state — "
            "check that the lifespan startup completed without errors."
        )
    return embedding_client
