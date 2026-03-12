"""Async embedding client for OpenLLM's OpenAI-compatible /v1/embeddings endpoint."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from autoknowledge.config import EmbeddingConfig

logger = logging.getLogger(__name__)

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5  # seconds


class EmbeddingClient:
    """Async client that wraps OpenLLM's /v1/embeddings endpoint."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(60.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> EmbeddingClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def healthcheck(self) -> bool:
        """Return True if the embedding service is reachable and returns correct dimensions."""
        try:
            vectors = await self.embed(["healthcheck"])
            if len(vectors[0]) != self._config.dimension:
                logger.error(
                    "Embedding dimension mismatch: config says %d but got %d. "
                    "Update embedding.dimension in config.toml.",
                    self._config.dimension, len(vectors[0]),
                )
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding service healthcheck failed: %s", exc)
            return False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one vector per text.

        Handles batching and retries automatically.
        Raises RuntimeError if the service returns mismatched dimensions.
        """
        if not texts:
            return []

        results: list[list[float]] = []
        for batch in _batched(texts, self._config.batch_size):
            vectors = await self._embed_batch(batch)
            results.extend(vectors)

        # Validate dimensions on the full result set
        for i, vec in enumerate(results):
            if len(vec) != self._config.dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch at index {i}: "
                    f"expected {self._config.dimension}, got {len(vec)}. "
                    f"Update embedding.dimension in config.toml."
                )

        return results

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(
                    "/v1/embeddings",
                    json={"model": self._config.model, "input": texts},
                )
                if response.status_code in _RETRY_STATUSES:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "Embedding request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(f"Embedding failed after {_MAX_RETRIES} attempts: {last_exc}")


def _batched(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
