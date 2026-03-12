"""Tests for the ChromaDB store wrapper (ephemeral in-memory client)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import chromadb
import pytest

from autoknowledge.config import EmbeddingConfig
from autoknowledge.index.chroma_store import ChromaStore
from autoknowledge.types import Chunk


def _cfg(model: str = "test-model", dim: int = 4) -> EmbeddingConfig:
    return EmbeddingConfig(model=model, dimension=dim)


def _make_chunk(
    chunk_id: str = "abc",
    doc_path: str = "test.md",
    content: str = "some content",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_path=doc_path,
        heading_path=["Introduction"],
        content=content,
        token_count=3,
        start_line=1,
        end_line=5,
        metadata={"title": "Test", "tags": "ml", "wikilinks": "", "doc_path": doc_path},
    )


def _vec(dim: int = 4, val: float = 0.1) -> list[float]:
    return [val] * dim


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    """ChromaStore backed by a real temp dir (uses PersistentClient)."""
    return ChromaStore(tmp_path, _cfg())


def test_upsert_and_count(store: ChromaStore) -> None:
    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    store.upsert_chunks(chunks, [_vec(), _vec(val=0.2)])
    assert store.count() == 2


def test_query_returns_results(store: ChromaStore) -> None:
    chunk = _make_chunk("c1", content="transformer attention mechanism")
    store.upsert_chunks([chunk], [_vec()])
    results = store.query(_vec(), top_k=1)
    assert len(results) == 1
    chunk_id, distance, metadata, document = results[0]
    assert chunk_id == "c1"
    assert isinstance(distance, float)
    assert document == "transformer attention mechanism"


def test_delete_by_doc_path(store: ChromaStore) -> None:
    c1 = _make_chunk("c1", doc_path="paper_a.md")
    c2 = _make_chunk("c2", doc_path="paper_b.md")
    store.upsert_chunks([c1, c2], [_vec(), _vec(val=0.2)])
    store.delete_by_doc_path("paper_a.md")
    assert store.count() == 1


def test_query_returns_empty_on_empty_store(store: ChromaStore) -> None:
    results = store.query(_vec(), top_k=5)
    assert results == []


def test_get_all_chunks_text(store: ChromaStore) -> None:
    chunks = [_make_chunk("c1", content="foo"), _make_chunk("c2", content="bar")]
    store.upsert_chunks(chunks, [_vec(), _vec(val=0.2)])
    all_text = store.get_all_chunks_text()
    texts = {text for _, text in all_text}
    assert "foo" in texts
    assert "bar" in texts


def test_model_mismatch_raises(tmp_path: Path) -> None:
    # Create store with model A
    ChromaStore(tmp_path, _cfg(model="model-a"))
    # Try to open with model B — should raise
    with pytest.raises(RuntimeError, match="model mismatch"):
        ChromaStore(tmp_path, _cfg(model="model-b"))


def test_drop_and_recreate_clears_data(store: ChromaStore) -> None:
    store.upsert_chunks([_make_chunk("c1")], [_vec()])
    assert store.count() == 1
    store.drop_and_recreate()
    assert store.count() == 0
