"""Fake implementations of external service clients.

These are in-memory replacements used by tests to avoid spinning up real
MinIO (or other external services) for every test run.

Pattern: each fake implements the same interface as the real client, so
FastAPI's dependency_overrides can swap them in transparently.
"""

from collections.abc import AsyncIterator
from typing import Any

from api.documents.embeddings import EMBEDDING_DIMENSIONS


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

    async def get_object(self, key: str) -> bytes:
        return self.objects[key]["data"]


class FakeEmbeddingClient:
    """In-memory substitute for `api.documents.embeddings.EmbeddingClient`.

    Returns a short, deterministic vector per input text — no network calls,
    no real OpenAI account needed — and records every batch it was asked to
    embed so tests can assert on what the ingestion pipeline actually sent.
    """

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.embedded_batches: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embedded_batches.append(list(texts))
        return [self._fake_vector(text) for text in texts]

    def _fake_vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        vector[0] = float(len(text))
        return vector


class FakeLLMClient:
    """In-memory substitute for `api.chat.rag.LLMClient`.

    Streams a fixed reply word-by-word (no network, no real Anthropic
    account needed) and records every call so tests can assert on the
    system prompt / messages that were actually sent. Set `should_fail`
    to exercise retrieve_and_stream's error-handling path.
    """

    def __init__(self, reply: str = "This is a fake answer.") -> None:
        self.reply = reply
        self.should_fail = False
        self.calls: list[dict[str, object]] = []

    async def stream_reply(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls.append({"system": system, "messages": messages})
        if self.should_fail:
            raise RuntimeError("simulated LLM failure")
        for word in self.reply.split(" "):
            yield word + " "
