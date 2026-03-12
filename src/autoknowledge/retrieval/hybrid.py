"""Hybrid semantic + BM25 retrieval with weighted score fusion."""

from __future__ import annotations

import ast
import logging
from typing import Any

from autoknowledge.config import RetrievalConfig
from autoknowledge.embedding.client import EmbeddingClient
from autoknowledge.index.bm25_store import BM25Store
from autoknowledge.index.chroma_store import ChromaStore
from autoknowledge.types import Chunk, SearchResult

logger = logging.getLogger(__name__)

_OVERFETCH_FACTOR = 3  # fetch top_k * this from each store before merging


class HybridRetriever:
    """Combines ChromaDB semantic search and BM25 keyword search."""

    def __init__(
        self,
        chroma: ChromaStore,
        bm25: BM25Store,
        embedder: EmbeddingClient,
        config: RetrievalConfig,
    ) -> None:
        self._chroma = chroma
        self._bm25 = bm25
        self._embedder = embedder
        self._config = config

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return hybrid-ranked SearchResult list for *query*.

        Args:
            query: natural language search query
            top_k: number of results to return (defaults to config.default_top_k)
            filters: optional metadata filters (see retrieval.filters)
        """
        k = top_k or self._config.default_top_k
        fetch_k = k * _OVERFETCH_FACTOR

        # 1. Embed query
        query_vec = (await self._embedder.embed([query]))[0]

        # 2. Semantic search from ChromaDB
        from autoknowledge.retrieval.filters import build_chroma_filter
        where = build_chroma_filter(filters)
        semantic_hits = self._chroma.query(query_vec, top_k=fetch_k, where=where)

        # 3. Keyword search from BM25 (no metadata filtering)
        bm25_hits = self._bm25.query(query, top_k=fetch_k)

        # 4. Collect all unique chunk IDs
        all_ids: set[str] = {h[0] for h in semantic_hits} | {h[0] for h in bm25_hits}
        if not all_ids:
            return []

        # 5. Build score maps
        semantic_raw: dict[str, float] = {cid: dist for cid, dist, _, _ in semantic_hits}
        bm25_raw: dict[str, float] = {cid: score for cid, score in bm25_hits}

        # ChromaDB cosine distance is in [0, 2]; convert to similarity [0, 1]
        semantic_sim = {cid: 1.0 - (dist / 2.0) for cid, dist in semantic_raw.items()}

        # 6. Normalise both score sets to [0, 1]
        sem_norm = _min_max_normalise(semantic_sim)
        bm25_norm = _min_max_normalise(bm25_raw)

        alpha = self._config.alpha

        # 7. Merge: combine scores, zero-fill missing
        combined: dict[str, tuple[float, float, float]] = {}
        for cid in all_ids:
            s = sem_norm.get(cid, 0.0)
            b = bm25_norm.get(cid, 0.0)
            combined[cid] = (alpha * s + (1.0 - alpha) * b, s, b)

        # 8. Sort by combined score
        ranked = sorted(combined.items(), key=lambda x: x[1][0], reverse=True)[:k]

        # 9. Fetch chunk content from ChromaDB for IDs not already in semantic_hits
        semantic_data: dict[str, tuple[dict[str, Any], str]] = {
            cid: (meta, doc) for cid, _, meta, doc in semantic_hits
        }
        missing_ids = [cid for cid, _ in ranked if cid not in semantic_data]
        if missing_ids:
            for cid, meta, doc in self._chroma.get_by_ids(missing_ids):
                semantic_data[cid] = (meta, doc)

        # 10. Build SearchResult objects
        results: list[SearchResult] = []
        for cid, (combined_score, sem_score, bm25_score) in ranked:
            if cid not in semantic_data:
                continue
            meta, doc = semantic_data[cid]
            chunk = _meta_to_chunk(cid, doc, meta)
            results.append(SearchResult(
                chunk=chunk,
                semantic_score=sem_score,
                bm25_score=bm25_score,
                combined_score=combined_score,
            ))

        return results


def _min_max_normalise(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _meta_to_chunk(chunk_id: str, content: str, meta: dict[str, Any]) -> Chunk:
    heading_raw = meta.get("heading_path_json", "[]")
    try:
        heading_path: list[str] = ast.literal_eval(heading_raw)
    except Exception:  # noqa: BLE001
        heading_path = []

    return Chunk(
        chunk_id=chunk_id,
        doc_path=str(meta.get("doc_path", "")),
        heading_path=heading_path,
        content=content,
        token_count=int(meta.get("token_count", 0)),
        start_line=int(meta.get("start_line", 0)),
        end_line=int(meta.get("end_line", 0)),
        metadata=dict(meta),
    )
