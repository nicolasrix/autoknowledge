"""Tests for the embedding client (mocked HTTP)."""

from __future__ import annotations

import pytest
import respx
import httpx

from autoknowledge.config import EmbeddingConfig
from autoknowledge.embedding.client import EmbeddingClient


def _config(dimension: int = 4, batch_size: int = 2) -> EmbeddingConfig:
    return EmbeddingConfig(
        base_url="http://test-llm:3000",
        model="test-model",
        dimension=dimension,
        batch_size=batch_size,
    )


def _make_response(texts: list[str], dim: int = 4) -> dict:
    return {
        "data": [
            {"embedding": [0.1 * (i + 1)] * dim, "index": i}
            for i in range(len(texts))
        ]
    }


@respx.mock
@pytest.mark.asyncio
async def test_embed_returns_vectors() -> None:
    respx.post("http://test-llm:3000/v1/embeddings").mock(
        return_value=httpx.Response(200, json=_make_response(["hello"]))
    )
    async with EmbeddingClient(_config()) as client:
        result = await client.embed(["hello"])
    assert len(result) == 1
    assert len(result[0]) == 4


@respx.mock
@pytest.mark.asyncio
async def test_embed_batches_requests() -> None:
    """With batch_size=2, three texts should produce two HTTP calls."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        import json
        body = json.loads(request.content)
        return httpx.Response(200, json=_make_response(body["input"]))

    respx.post("http://test-llm:3000/v1/embeddings").mock(side_effect=handler)

    async with EmbeddingClient(_config(batch_size=2)) as client:
        result = await client.embed(["a", "b", "c"])

    assert call_count == 2
    assert len(result) == 3


@respx.mock
@pytest.mark.asyncio
async def test_embed_empty_returns_empty() -> None:
    async with EmbeddingClient(_config()) as client:
        result = await client.embed([])
    assert result == []


@respx.mock
@pytest.mark.asyncio
async def test_dimension_mismatch_raises() -> None:
    respx.post("http://test-llm:3000/v1/embeddings").mock(
        return_value=httpx.Response(200, json=_make_response(["hello"], dim=8))
    )
    async with EmbeddingClient(_config(dimension=4)) as client:
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            await client.embed(["hello"])


@respx.mock
@pytest.mark.asyncio
async def test_retries_on_500() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(500)
        return httpx.Response(200, json=_make_response(["hello"]))

    respx.post("http://test-llm:3000/v1/embeddings").mock(side_effect=handler)

    # Patch asyncio.sleep to avoid real waiting
    import asyncio
    async def no_sleep(_: float) -> None:
        pass

    import autoknowledge.embedding.client as client_module
    original_sleep = asyncio.sleep

    async with EmbeddingClient(_config()) as client:
        client_module_sleep = client_module
        # Monkeypatch sleep at the asyncio level within the module
        import unittest.mock as mock
        with mock.patch("asyncio.sleep", side_effect=no_sleep):
            result = await client.embed(["hello"])

    assert call_count == 3
    assert len(result) == 1


@respx.mock
@pytest.mark.asyncio
async def test_healthcheck_returns_true_on_success() -> None:
    respx.post("http://test-llm:3000/v1/embeddings").mock(
        return_value=httpx.Response(200, json=_make_response(["healthcheck"]))
    )
    async with EmbeddingClient(_config()) as client:
        assert await client.healthcheck() is True


@respx.mock
@pytest.mark.asyncio
async def test_healthcheck_returns_false_on_dim_mismatch() -> None:
    respx.post("http://test-llm:3000/v1/embeddings").mock(
        return_value=httpx.Response(200, json=_make_response(["healthcheck"], dim=8))
    )
    async with EmbeddingClient(_config(dimension=4)) as client:
        assert await client.healthcheck() is False


@respx.mock
@pytest.mark.asyncio
async def test_healthcheck_returns_false_on_connection_error() -> None:
    respx.post("http://test-llm:3000/v1/embeddings").mock(
        side_effect=httpx.ConnectError("refused")
    )
    async with EmbeddingClient(_config()) as client:
        assert await client.healthcheck() is False
