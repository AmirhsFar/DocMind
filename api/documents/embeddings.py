"""OpenAI embedding client.

Mirrors the `StorageClient` pattern in `storage.py`: a small async wrapper
constructed from `Settings`, with a `from_settings` classmethod so it is
trivially swappable for a fake in tests.

Anthropic does not (yet) offer a standalone embeddings API, so OpenAI is
used here even though Phase 4's chat completions may call Anthropic instead.
"""

import logging

from openai import AsyncOpenAI

from api.core.config import Settings

logger = logging.getLogger("docmind.embeddings")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # must match EmbeddingVector()'s size in models.py


class EmbeddingClient:
    """Async wrapper around OpenAI's embeddings endpoint."""

    def __init__(self, api_key: str | None, model: str = EMBEDDING_MODEL) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "EmbeddingClient":
        """Convenience constructor that reads config from the Settings object."""
        return cls(api_key=settings.openai_api_key)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings, returning one vector per input, in order.

        A single batched request is used instead of one call per chunk —
        OpenAI's embeddings endpoint accepts a list natively, which means
        fewer round trips and one rate-limit hit instead of many.
        """
        response = await self._client.embeddings.create(model=self._model, input=texts)
        logger.debug("Embedded %d chunks with model=%s", len(texts), self._model)
        return [item.embedding for item in response.data]
