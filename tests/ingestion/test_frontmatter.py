"""Tests for YAML frontmatter builder."""

from __future__ import annotations

from pathlib import Path

import frontmatter

from autoknowledge.ingestion.frontmatter import build_frontmatter
from autoknowledge.types import PdfMetadata


def _make_meta(**kwargs) -> PdfMetadata:
    defaults = dict(
        title="Test Title",
        author="Test Author",
        creation_date="2024-01-15",
        page_count=5,
        source_path=Path("/vault/papers/test.pdf"),
    )
    defaults.update(kwargs)
    return PdfMetadata(**defaults)


def test_frontmatter_is_parseable() -> None:
    result = build_frontmatter(_make_meta())
    post = frontmatter.loads(result)
    assert post.metadata is not None


def test_frontmatter_contains_title() -> None:
    result = build_frontmatter(_make_meta(title="My Paper"))
    post = frontmatter.loads(result)
    assert post["title"] == "My Paper"


def test_frontmatter_contains_author() -> None:
    result = build_frontmatter(_make_meta())
    post = frontmatter.loads(result)
    assert post["author"] == "Test Author"


def test_frontmatter_contains_date() -> None:
    import datetime
    result = build_frontmatter(_make_meta())
    post = frontmatter.loads(result)
    # python-frontmatter parses ISO dates as datetime.date objects
    assert post["date"] in ("2024-01-15", datetime.date(2024, 1, 15))


def test_frontmatter_no_date_when_none() -> None:
    result = build_frontmatter(_make_meta(creation_date=None))
    post = frontmatter.loads(result)
    assert "date" not in post.metadata


def test_frontmatter_source_type_is_pdf() -> None:
    result = build_frontmatter(_make_meta())
    post = frontmatter.loads(result)
    assert post["source_type"] == "pdf"


def test_frontmatter_page_count() -> None:
    result = build_frontmatter(_make_meta(page_count=12))
    post = frontmatter.loads(result)
    assert post["page_count"] == 12


def test_frontmatter_has_pdf_tag() -> None:
    result = build_frontmatter(_make_meta())
    assert "pdf" in result


def test_frontmatter_extra_tags() -> None:
    result = build_frontmatter(_make_meta(), tags=["machine-learning", "forecasting"])
    assert "machine-learning" in result
    assert "forecasting" in result


def test_frontmatter_special_chars_in_title() -> None:
    result = build_frontmatter(_make_meta(title="Title: A Study of [Things]"))
    assert "---" in result  # still valid frontmatter structure
