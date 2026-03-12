"""ChromaDB vector store wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from autoknowledge.config import EmbeddingConfig
from autoknowledge.types import Chunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "autoknowledge"


class ChromaStore:
    """Wraps a persistent ChromaDB collection for chunk embedding storage."""

    def __init__(self, data_dir: Path, embedding_config: EmbeddingConfig, full: bool = False) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._cfg = embedding_config
        self._client = chromadb.PersistentClient(
            path=str(data_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        if full:
            try:
                self._client.delete_collection(_COLLECTION_NAME)
                logger.info("ChromaDB collection dropped for full reindex")
            except Exception:  # noqa: BLE001
                pass
        self._collection = self._get_or_create_collection()

    # ── Collection lifecycle ──────────────────────────────────────────────────

    def _get_or_create_collection(self) -> chromadb.Collection:
        existing = {c.name for c in self._client.list_collections()}

        if _COLLECTION_NAME in existing:
            col = self._client.get_collection(_COLLECTION_NAME)
            self._validate_collection_metadata(col)
            return col

        return self._client.create_collection(
            name=_COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self._cfg.model,
                "embedding_dimension": self._cfg.dimension,
            },
        )

    def _validate_collection_metadata(self, col: chromadb.Collection) -> None:
        meta = col.metadata or {}
        stored_model = meta.get("embedding_model")
        stored_dim = meta.get("embedding_dimension")

        if stored_model and stored_model != self._cfg.model:
            raise RuntimeError(
                f"Embedding model mismatch: collection was built with '{stored_model}' "
                f"but config says '{self._cfg.model}'. Run 'autoknowledge index --full' "
                "to rebuild the index with the new model."
            )
        if stored_dim and stored_dim != self._cfg.dimension:
            raise RuntimeError(
                f"Embedding dimension mismatch: collection has dim={stored_dim} "
                f"but config says dim={self._cfg.dimension}. Run 'autoknowledge index --full'."
            )

    def drop_and_recreate(self) -> None:
        """Delete the collection and create a fresh one (used by --full reindex)."""
        try:
            self._client.delete_collection(_COLLECTION_NAME)
        except Exception:  # noqa: BLE001
            pass
        self._collection = self._get_or_create_collection()
        logger.info("ChromaDB collection recreated")

    # ── Write operations ──────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[_serialise_metadata(c.metadata | {
                "doc_path": c.doc_path,
                "heading": " > ".join(c.heading_path),
                "heading_path_json": str(c.heading_path),
                "start_line": c.start_line,
                "end_line": c.end_line,
                "token_count": c.token_count,
            }) for c in chunks],
        )

    def delete_by_doc_path(self, doc_path: str) -> None:
        results = self._collection.get(where={"doc_path": {"$eq": doc_path}})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.debug("Deleted %d chunks for %s", len(ids), doc_path)

    # ── Read operations ───────────────────────────────────────────────────────

    def query(
        self,
        embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any], str]]:
        """Return list of (chunk_id, distance, metadata, document_text)."""
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, self.count() or 1),
            "include": ["metadatas", "distances", "documents"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        if not results["ids"] or not results["ids"][0]:
            return []

        return [
            (chunk_id, dist, meta, doc)
            for chunk_id, dist, meta, doc in zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0],
                results["documents"][0],
            )
        ]

    def get_all_chunks_text(self) -> list[tuple[str, str]]:
        """Return (chunk_id, document_text) for all chunks (used to rebuild BM25)."""
        results = self._collection.get(include=["documents"])
        return list(zip(results["ids"], results["documents"]))

    def count(self) -> int:
        return self._collection.count()


def _serialise_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB metadata values must be str, int, float, or bool."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = ", ".join(str(i) for i in v)
        else:
            out[k] = str(v)
    return out
