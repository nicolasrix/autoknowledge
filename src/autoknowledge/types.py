"""Shared dataclasses used across the autoknowledge package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    path: Path
    title: str
    tags: list[str]
    front_matter: dict[str, Any]
    raw_content: str
    wikilinks: list[str]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_path: str
    heading_path: list[str]
    content: str
    token_count: int
    start_line: int
    end_line: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    semantic_score: float
    bm25_score: float
    combined_score: float


@dataclass(frozen=True)
class IndexStats:
    total_files: int
    indexed_files: int
    skipped_unchanged: int
    deleted_files: int
    total_chunks: int
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Indexed:   {self.indexed_files}/{self.total_files} files",
            f"Skipped:   {self.skipped_unchanged} unchanged",
            f"Deleted:   {self.deleted_files} removed files",
            f"Chunks:    {self.total_chunks} total",
        ]
        if self.errors:
            lines.append(f"Errors:    {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)
