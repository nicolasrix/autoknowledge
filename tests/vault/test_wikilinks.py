"""Tests for wikilink extraction and resolution."""

from __future__ import annotations

from pathlib import Path

from autoknowledge.vault.wikilinks import extract_wikilinks, resolve_wikilinks


# ── extract_wikilinks ────────────────────────────────────────────────────────

def test_extract_simple() -> None:
    assert extract_wikilinks("See [[Paper A]] for details.") == ["Paper A"]


def test_extract_alias_stripped() -> None:
    assert extract_wikilinks("See [[Paper A|display text]].") == ["Paper A"]


def test_extract_multiple() -> None:
    result = extract_wikilinks("[[Foo]] and [[Bar]] are cited.")
    assert result == ["Foo", "Bar"]


def test_extract_no_links() -> None:
    assert extract_wikilinks("No links here.") == []


def test_extract_empty_string() -> None:
    assert extract_wikilinks("") == []


def test_extract_preserves_spaces() -> None:
    assert extract_wikilinks("[[My Paper Title]]") == ["My Paper Title"]


# ── resolve_wikilinks ────────────────────────────────────────────────────────

def test_resolve_exact_match(tmp_path: Path) -> None:
    p = tmp_path / "Paper A.md"
    p.write_text("# A")
    result = resolve_wikilinks(["Paper A"], [p], tmp_path)
    assert result["Paper A"] == p


def test_resolve_case_insensitive(tmp_path: Path) -> None:
    p = tmp_path / "Paper A.md"
    p.write_text("# A")
    result = resolve_wikilinks(["paper a"], [p], tmp_path)
    assert result["paper a"] == p


def test_resolve_unresolved_returns_none(tmp_path: Path) -> None:
    result = resolve_wikilinks(["Unknown"], [], tmp_path)
    assert result["Unknown"] is None


def test_resolve_shortest_path_wins(tmp_path: Path) -> None:
    """When there are two files with the same stem, pick the shallowest."""
    deep = tmp_path / "sub" / "deep" / "Note.md"
    deep.parent.mkdir(parents=True)
    deep.write_text("# Deep")
    shallow = tmp_path / "Note.md"
    shallow.write_text("# Shallow")

    result = resolve_wikilinks(["Note"], [deep, shallow], tmp_path)
    assert result["Note"] == shallow


def test_resolve_deduplicates_targets(tmp_path: Path) -> None:
    p = tmp_path / "Paper A.md"
    p.write_text("# A")
    result = resolve_wikilinks(["Paper A", "Paper A"], [p], tmp_path)
    assert len(result) == 1
