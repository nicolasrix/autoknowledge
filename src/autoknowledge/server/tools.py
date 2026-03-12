"""MCP tool implementations: search_knowledge, get_document, reindex."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from autoknowledge.config import Config
from autoknowledge.index.bm25_store import BM25Store
from autoknowledge.index.chroma_store import ChromaStore
from autoknowledge.retrieval.hybrid import HybridRetriever
from autoknowledge.types import SearchResult

logger = logging.getLogger(__name__)


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results as readable Markdown for Claude Code."""
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        c = r.chunk
        section = " > ".join(c.heading_path) if c.heading_path else "(no heading)"
        lines.extend([
            f"## Result {i} (score: {r.combined_score:.3f})",
            f"**Source**: `{c.doc_path}`",
            f"**Section**: {section}",
        ])
        tags = c.metadata.get("tags", "")
        if tags:
            lines.append(f"**Tags**: {tags}")
        lines.extend(["", c.content, "", "---"])

    return "\n".join(lines)


async def search_knowledge(
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
    retriever: HybridRetriever,
) -> str:
    """Execute hybrid search and return formatted results."""
    results = await retriever.search(query, top_k=top_k, filters=filters or None)
    return format_search_results(results)


def get_document(path: str, vault_path: Path) -> str:
    """Return the full Markdown content of a document.

    Accepts paths relative to vault_root or absolute paths within vault_root.
    Raises ValueError on path traversal attempts.
    """
    requested = Path(path)

    # Resolve relative to vault root if not absolute
    if not requested.is_absolute():
        candidate = (vault_path / requested).resolve()
    else:
        candidate = requested.resolve()

    # Security: ensure the resolved path is inside the vault
    try:
        candidate.relative_to(vault_path.resolve())
    except ValueError:
        raise ValueError(
            f"Path '{path}' is outside the vault directory. "
            "Only paths within the vault are accessible."
        )

    if not candidate.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not candidate.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return candidate.read_text(encoding="utf-8", errors="replace")


async def reindex(
    scope: str | None,
    config: Config,
    chroma: ChromaStore,
    bm25: BM25Store,
) -> str:
    """Trigger an incremental reindex and return a stats summary."""
    from autoknowledge.embedding.client import EmbeddingClient
    from autoknowledge.index.hash_tracker import HashTracker
    from autoknowledge.indexer.pipeline import _run_pipeline

    vault_override: Path | None = None
    if scope:
        candidate = config.vault_path / scope
        if not candidate.exists():
            raise ValueError(f"Scope path does not exist: {scope}")
        vault_override = candidate if candidate.is_dir() else candidate.parent

    vault_path = vault_override or config.vault_path

    async with EmbeddingClient(config.embedding) as embedder:
        if not await embedder.healthcheck():
            return "ERROR: Embedding service not reachable. Cannot reindex."
        with HashTracker(config.data_dir / "file_hashes.db") as tracker:
            stats = await _run_pipeline(config, vault_path, embedder, chroma, tracker)

    # Rebuild BM25 with new data
    logger.info("Rebuilding BM25 after reindex …")
    bm25.build(chroma.get_all_chunks_text())

    return stats.summary()
