"""In-memory BM25 index built from ChromaDB chunk texts."""

from __future__ import annotations

import logging
import re
import threading

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """Thread-safe in-memory BM25 index over chunk texts.

    Built from (chunk_id, text) pairs. Rebuilt after each indexing run.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._index: BM25Okapi | None = None

    def build(self, chunks: list[tuple[str, str]]) -> None:
        """Build (or rebuild) the BM25 index from a list of (chunk_id, text) pairs."""
        if not chunks:
            with self._lock:
                self._ids = []
                self._index = None
            return

        ids = [c[0] for c in chunks]
        tokenized = [_tokenize(c[1]) for c in chunks]

        new_index = BM25Okapi(tokenized)
        with self._lock:
            self._ids = ids
            self._index = new_index

        logger.debug("BM25 index built with %d chunks", len(ids))

    def query(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to *top_k* (chunk_id, bm25_score) pairs, highest score first."""
        with self._lock:
            if self._index is None or not self._ids:
                return []
            tokens = _tokenize(query_text)
            if not tokens:
                return []
            scores = self._index.get_scores(tokens)

        # Pair with IDs and sort descending
        paired = sorted(zip(self._ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(score)) for cid, score in paired[:top_k]]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._ids)
