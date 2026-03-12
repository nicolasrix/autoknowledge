"""Full and incremental indexing pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from autoknowledge.config import Config
from autoknowledge.embedding.client import EmbeddingClient
from autoknowledge.index.bm25_store import BM25Store
from autoknowledge.index.chroma_store import ChromaStore
from autoknowledge.index.hash_tracker import HashTracker
from autoknowledge.types import IndexStats
from autoknowledge.vault.chunker import chunk_document
from autoknowledge.vault.parser import parse_document
from autoknowledge.vault.scanner import scan_vault

logger = logging.getLogger(__name__)


async def run_index(
    config: Config,
    full: bool = False,
    vault_override: Path | None = None,
) -> IndexStats:
    """Run a full or incremental index of the vault.

    Args:
        config: loaded Config object
        full: if True, drop and rebuild the entire index
        vault_override: override vault path from config
    """
    vault_path = (vault_override or config.vault_path).expanduser().resolve()
    data_dir = config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    async with EmbeddingClient(config.embedding) as embedder:
        # Pre-flight: verify the embedding service is reachable
        if not await embedder.healthcheck():
            raise RuntimeError(
                f"Embedding service not reachable at {config.embedding.base_url}. "
                "Start OpenLLM and retry, or check your config."
            )

        if full:
            logger.info("Full reindex requested — dropping existing index")
        chroma = ChromaStore(data_dir, config.embedding, full=full)
        bm25 = BM25Store()

        with HashTracker(data_dir / "file_hashes.db") as tracker:
            stats = await _run_pipeline(config, vault_path, embedder, chroma, tracker)

    # Rebuild BM25 from all stored chunks
    logger.info("Rebuilding BM25 index from %d stored chunks …", chroma.count())
    bm25.build(chroma.get_all_chunks_text())

    print(stats.summary(), file=sys.stderr)
    return stats


async def _run_pipeline(
    config: Config,
    vault_path: Path,
    embedder: EmbeddingClient,
    chroma: ChromaStore,
    tracker: HashTracker,
) -> IndexStats:
    current_paths = set(scan_vault(vault_path))
    tracked_paths = tracker.get_all_tracked_paths()

    # Detect and remove deleted files
    deleted_paths = tracked_paths - current_paths
    deleted_count = 0
    for path in deleted_paths:
        chroma.delete_by_doc_path(str(path))
        tracker.remove(path)
        deleted_count += 1
        logger.info("Removed deleted file: %s", path)

    total = len(current_paths)
    indexed = 0
    skipped = 0
    total_chunks = 0
    errors: list[str] = []

    for i, path in enumerate(sorted(current_paths), start=1):
        _progress(i, total, path)

        if not tracker.has_changed(path):
            skipped += 1
            continue

        try:
            chunks, sha256 = await _index_file(config, path, embedder, chroma)
            tracker.update(path, sha256, chunk_count=len(chunks))
            total_chunks += len(chunks)
            indexed += 1
        except Exception as exc:  # noqa: BLE001
            msg = f"{path}: {exc}"
            logger.error("Failed to index %s: %s", path, exc)
            errors.append(msg)

    return IndexStats(
        total_files=total,
        indexed_files=indexed,
        skipped_unchanged=skipped,
        deleted_files=deleted_count,
        total_chunks=chroma.count(),
        errors=errors,
    )


async def _index_file(
    config: Config,
    path: Path,
    embedder: EmbeddingClient,
    chroma: ChromaStore,
) -> tuple[list, str]:
    """Parse, chunk, embed, and upsert a single file. Returns (chunks, sha256)."""
    sha256 = HashTracker.compute_hash(path)

    doc = parse_document(path)
    chunks = chunk_document(doc, config.index)

    if not chunks:
        # File exists but produced no chunks (empty/whitespace-only)
        chroma.delete_by_doc_path(str(path))
        return [], sha256

    # Remove old chunks for this file before upserting new ones
    chroma.delete_by_doc_path(str(path))

    # Embed in batches
    texts = [c.content for c in chunks]
    embeddings = await embedder.embed(texts)

    chroma.upsert_chunks(chunks, embeddings)
    logger.debug("Indexed %d chunks from %s", len(chunks), path.name)

    return chunks, sha256


def _progress(i: int, total: int, path: Path) -> None:
    print(f"\rIndexing {i}/{total}: {path.name}   ", end="", file=sys.stderr, flush=True)
    if i == total:
        print(file=sys.stderr)  # newline at end
