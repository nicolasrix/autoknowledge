"""Tests for the document chunker."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from autoknowledge.config import IndexConfig
from autoknowledge.types import Document
from autoknowledge.vault.chunker import chunk_document


def _doc(content: str, path: str = "test.md") -> Document:
    return Document(
        path=Path(path),
        title="Test",
        tags=["test"],
        front_matter={},
        raw_content=textwrap.dedent(content),
        wikilinks=[],
    )


def _config(min_t: int = 10, max_t: int = 200, overlap_t: int = 20) -> IndexConfig:
    return IndexConfig(
        chunk_min_tokens=min_t,
        chunk_max_tokens=max_t,
        chunk_overlap_tokens=overlap_t,
    )


def test_empty_document_returns_no_chunks() -> None:
    doc = _doc("")
    assert chunk_document(doc, _config()) == []


def test_whitespace_only_returns_no_chunks() -> None:
    doc = _doc("   \n\n   ")
    assert chunk_document(doc, _config()) == []


def test_single_section_produces_one_chunk() -> None:
    doc = _doc("# Intro\nSome text here.\n")
    chunks = chunk_document(doc, _config())
    assert len(chunks) >= 1
    assert any("Some text here" in c.content for c in chunks)


def test_heading_path_captured() -> None:
    doc = _doc("# Introduction\nText.\n\n## Background\nMore text.\n")
    chunks = chunk_document(doc, _config())
    heading_paths = [c.heading_path for c in chunks]
    assert any("Introduction" in hp for hp in heading_paths)
    assert any("Background" in hp for hp in heading_paths)


def test_chunk_ids_are_unique() -> None:
    doc = _doc("# A\nText A.\n\n# B\nText B.\n\n# C\nText C.\n")
    chunks = chunk_document(doc, _config())
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_are_deterministic() -> None:
    doc = _doc("# Intro\nSome text.\n")
    chunks1 = chunk_document(doc, _config())
    chunks2 = chunk_document(doc, _config())
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_token_count_within_max(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a generous max so we can test token cap with real content
    cfg = _config(max_t=50, overlap_t=0)
    long_text = " ".join(["word"] * 200)
    doc = _doc(f"# Section\n{long_text}\n")
    chunks = chunk_document(doc, cfg)
    for chunk in chunks:
        assert chunk.token_count <= 60  # small tolerance for sentence splitting


def test_metadata_fields_present() -> None:
    doc = _doc("# Intro\nText.\n")
    chunks = chunk_document(doc, _config())
    for chunk in chunks:
        assert "title" in chunk.metadata
        assert "doc_path" in chunk.metadata
        assert "tags" in chunk.metadata


def test_no_heading_document() -> None:
    """Documents without headings should still be chunked."""
    doc = _doc("Just some plain text without any headings.\n")
    chunks = chunk_document(doc, _config())
    assert len(chunks) >= 1
