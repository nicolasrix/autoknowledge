"""Integration tests for the full indexing pipeline."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from autoknowledge.config import Config, EmbeddingConfig, IndexConfig, VaultConfig
from autoknowledge.index.bm25_store import BM25Store
from autoknowledge.index.chroma_store import ChromaStore
from autoknowledge.index.hash_tracker import HashTracker
from autoknowledge.indexer.pipeline import _index_file, _run_pipeline


def _config(vault: Path, data: Path) -> Config:
    return Config(
        vault=VaultConfig(path=vault),
        embedding=EmbeddingConfig(dimension=4, batch_size=8),
        index=IndexConfig(data_dir=data),
    )


def _fake_embedder(dim: int = 4) -> AsyncMock:
    """Mock EmbeddingClient that returns deterministic random-ish vectors."""
    mock = AsyncMock()
    mock.embed = AsyncMock(
        side_effect=lambda texts: [[float(hash(t) % 100) / 100.0] * dim for t in texts]
    )
    return mock


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    (v / "Paper A.md").write_text(textwrap.dedent("""\
        ---
        title: Paper A
        tags: [ml, attention]
        ---
        # Introduction
        This paper discusses multi-head attention.

        # Methods
        We apply transformers.
    """))
    (v / "Paper B.md").write_text(textwrap.dedent("""\
        ---
        title: Paper B
        tags: [cnn]
        ---
        # Abstract
        A convolutional approach for image recognition.
    """))
    return v


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def config(vault: Path, data: Path) -> Config:
    return _config(vault, data)


# ── _index_file ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_index_file_creates_chunks(config: Config, vault: Path, data: Path) -> None:
    chroma = ChromaStore(data, config.embedding)
    embedder = _fake_embedder()
    path = vault / "Paper A.md"

    chunks, sha256 = await _index_file(config, path, embedder, chroma)

    assert len(chunks) > 0
    assert len(sha256) == 64  # SHA-256 hex
    assert chroma.count() == len(chunks)


@pytest.mark.asyncio
async def test_index_file_replaces_old_chunks(config: Config, vault: Path, data: Path) -> None:
    chroma = ChromaStore(data, config.embedding)
    embedder = _fake_embedder()
    path = vault / "Paper A.md"

    await _index_file(config, path, embedder, chroma)
    count_before = chroma.count()

    # Re-index same file — chunk count should stay the same (old deleted, new inserted)
    await _index_file(config, path, embedder, chroma)
    assert chroma.count() == count_before


# ── _run_pipeline ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_pipeline_indexes_all_files(
    config: Config, vault: Path, data: Path
) -> None:
    chroma = ChromaStore(data, config.embedding)
    embedder = _fake_embedder()

    with HashTracker(data / "hashes.db") as tracker:
        stats = await _run_pipeline(config, vault, embedder, chroma, tracker)

    assert stats.total_files == 2
    assert stats.indexed_files == 2
    assert stats.skipped_unchanged == 0
    assert stats.deleted_files == 0
    assert stats.total_chunks > 0
    assert stats.errors == []


@pytest.mark.asyncio
async def test_incremental_skips_unchanged(
    config: Config, vault: Path, data: Path
) -> None:
    chroma = ChromaStore(data, config.embedding)
    embedder = _fake_embedder()

    with HashTracker(data / "hashes.db") as tracker:
        stats1 = await _run_pipeline(config, vault, embedder, chroma, tracker)
        # Second run: nothing changed
        stats2 = await _run_pipeline(config, vault, embedder, chroma, tracker)

    assert stats2.indexed_files == 0
    assert stats2.skipped_unchanged == 2


@pytest.mark.asyncio
async def test_modified_file_is_reindexed(
    config: Config, vault: Path, data: Path
) -> None:
    chroma = ChromaStore(data, config.embedding)
    embedder = _fake_embedder()
    paper_a = vault / "Paper A.md"

    with HashTracker(data / "hashes.db") as tracker:
        await _run_pipeline(config, vault, embedder, chroma, tracker)
        # Modify Paper A
        paper_a.write_text("---\ntitle: Paper A v2\n---\n# New content\nUpdated.\n")
        stats = await _run_pipeline(config, vault, embedder, chroma, tracker)

    assert stats.indexed_files == 1
    assert stats.skipped_unchanged == 1


@pytest.mark.asyncio
async def test_deleted_file_chunks_removed(
    config: Config, vault: Path, data: Path
) -> None:
    chroma = ChromaStore(data, config.embedding)
    embedder = _fake_embedder()
    paper_b = vault / "Paper B.md"

    with HashTracker(data / "hashes.db") as tracker:
        await _run_pipeline(config, vault, embedder, chroma, tracker)
        chunks_before = chroma.count()

        paper_b.unlink()
        stats = await _run_pipeline(config, vault, embedder, chroma, tracker)

    assert stats.deleted_files == 1
    assert chroma.count() < chunks_before
