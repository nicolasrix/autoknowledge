"""Build ChromaDB `where` filter clauses from user-facing filter dicts."""

from __future__ import annotations

from typing import Any

_SUPPORTED_FILTERS = frozenset({"tags", "path_prefix", "title"})


def build_chroma_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate user-facing filter keys into a ChromaDB `where` clause.

    Supported keys:
      tags (list[str])    — chunk must include at least one of these tags
      path_prefix (str)   — doc_path must start with this prefix
      title (str)         — title must contain this string (case-insensitive)

    Returns None if no filters are provided or all values are empty.
    Raises ValueError for unknown filter keys.
    """
    if not filters:
        return None

    unknown = set(filters.keys()) - _SUPPORTED_FILTERS
    if unknown:
        raise ValueError(
            f"Unknown filter key(s): {sorted(unknown)}. "
            f"Supported: {sorted(_SUPPORTED_FILTERS)}"
        )

    clauses: list[dict[str, Any]] = []

    if "tags" in filters:
        tags = filters["tags"]
        if not isinstance(tags, list) or not tags:
            raise ValueError("'tags' filter must be a non-empty list of strings")
        # ChromaDB: tags are stored as a comma-joined string; use $contains for each
        tag_clauses = [{"tags": {"$contains": str(t)}} for t in tags]
        if len(tag_clauses) == 1:
            clauses.append(tag_clauses[0])
        else:
            clauses.append({"$or": tag_clauses})

    if path_prefix := filters.get("path_prefix"):
        clauses.append({"doc_path": {"$contains": str(path_prefix)}})

    if title := filters.get("title"):
        clauses.append({"title": {"$contains": str(title)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
