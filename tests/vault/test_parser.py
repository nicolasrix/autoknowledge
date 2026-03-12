"""Tests for document parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

from autoknowledge.vault.parser import parse_document


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


def test_extracts_title(tmp_path: Path) -> None:
    p = _write(tmp_path, "paper.md", """\
        ---
        title: "My Paper"
        ---
        Body text.
    """)
    doc = parse_document(p)
    assert doc.title == "My Paper"


def test_falls_back_to_stem_when_no_title(tmp_path: Path) -> None:
    p = _write(tmp_path, "my-notes.md", "# Heading\nBody.")
    doc = parse_document(p)
    assert doc.title == "my-notes"


def test_extracts_tags_list(tmp_path: Path) -> None:
    p = _write(tmp_path, "paper.md", """\
        ---
        tags: [transformer, attention]
        ---
        Body.
    """)
    doc = parse_document(p)
    assert doc.tags == ["transformer", "attention"]


def test_extracts_tags_string(tmp_path: Path) -> None:
    p = _write(tmp_path, "paper.md", """\
        ---
        tags: "single-tag"
        ---
        Body.
    """)
    doc = parse_document(p)
    assert doc.tags == ["single-tag"]


def test_empty_tags_when_missing(tmp_path: Path) -> None:
    p = _write(tmp_path, "paper.md", "No front-matter.\n")
    doc = parse_document(p)
    assert doc.tags == []


def test_extracts_wikilinks(tmp_path: Path) -> None:
    p = _write(tmp_path, "paper.md", """\
        ---
        title: A
        ---
        See [[Paper B]] and [[Paper C|alias]].
    """)
    doc = parse_document(p)
    assert "Paper B" in doc.wikilinks
    assert "Paper C" in doc.wikilinks


def test_no_wikilinks_when_absent(tmp_path: Path) -> None:
    p = _write(tmp_path, "paper.md", "No links.\n")
    doc = parse_document(p)
    assert doc.wikilinks == []


def test_front_matter_preserved_in_dict(tmp_path: Path) -> None:
    p = _write(tmp_path, "paper.md", """\
        ---
        title: A
        year: 2023
        ---
        Body.
    """)
    doc = parse_document(p)
    assert doc.front_matter["year"] == 2023


def test_empty_file_returns_document(tmp_path: Path) -> None:
    p = tmp_path / "empty.md"
    p.write_text("")
    doc = parse_document(p)
    assert doc.title == "empty"
    assert doc.raw_content == ""


def test_broken_yaml_falls_back_gracefully(tmp_path: Path) -> None:
    p = _write(tmp_path, "broken.md", """\
        ---
        title: [unclosed
        ---
        Body text.
    """)
    doc = parse_document(p)
    # Should not raise; title falls back to stem or whatever was parsed
    assert doc.path == p
