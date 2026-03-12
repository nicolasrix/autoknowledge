"""Tests for MCP tool implementations."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from autoknowledge.server.tools import (
    format_search_results,
    get_document,
    search_knowledge,
)
from autoknowledge.types import Chunk, SearchResult


def _chunk(
    chunk_id: str = "abc",
    doc_path: str = "vault/paper.md",
    heading: list[str] | None = None,
    content: str = "Some relevant content.",
    tags: str = "ml, attention",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_path=doc_path,
        heading_path=["Introduction"] if heading is None else heading,
        content=content,
        token_count=5,
        start_line=1,
        end_line=5,
        metadata={"title": "Paper", "tags": tags, "doc_path": doc_path},
    )


def _result(score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk=_chunk(),
        semantic_score=score,
        bm25_score=score * 0.8,
        combined_score=score,
    )


# ── format_search_results ─────────────────────────────────────────────────────

def test_format_empty_results() -> None:
    assert format_search_results([]) == "No results found."


def test_format_includes_source() -> None:
    output = format_search_results([_result()])
    assert "vault/paper.md" in output


def test_format_includes_content() -> None:
    output = format_search_results([_result()])
    assert "Some relevant content." in output


def test_format_includes_score() -> None:
    output = format_search_results([_result(0.85)])
    assert "0.850" in output


def test_format_multiple_results_numbered() -> None:
    results = [_result(0.9), _result(0.7)]
    output = format_search_results(results)
    assert "## Result 1" in output
    assert "## Result 2" in output


def test_format_no_heading_shows_fallback() -> None:
    r = SearchResult(
        chunk=_chunk(heading=[]),
        semantic_score=0.9,
        bm25_score=0.7,
        combined_score=0.85,
    )
    output = format_search_results([r])
    assert "(no heading)" in output


# ── get_document ──────────────────────────────────────────────────────────────

def test_get_document_relative_path(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "paper.md").write_text("# Paper\nContent.")

    result = get_document("paper.md", vault)
    assert "Content." in result


def test_get_document_absolute_path_inside_vault(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    doc = vault / "paper.md"
    doc.write_text("# Paper\nContent.")

    result = get_document(str(doc), vault)
    assert "Content." in result


def test_get_document_path_traversal_raises(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET")

    with pytest.raises(ValueError, match="outside the vault"):
        get_document("../secret.txt", vault)


def test_get_document_missing_file_raises(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    with pytest.raises(FileNotFoundError):
        get_document("nonexistent.md", vault)


# ── search_knowledge ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_knowledge_calls_retriever() -> None:
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[_result(0.9)])

    result = await search_knowledge("transformers", top_k=5, filters=None, retriever=mock_retriever)

    mock_retriever.search.assert_called_once_with("transformers", top_k=5, filters=None)
    assert "Some relevant content." in result


@pytest.mark.asyncio
async def test_search_knowledge_empty_results() -> None:
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])

    result = await search_knowledge("query", top_k=5, filters=None, retriever=mock_retriever)
    assert result == "No results found."
