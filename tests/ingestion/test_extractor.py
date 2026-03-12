"""Tests for the PDF extractor."""

from __future__ import annotations

from pathlib import Path

from autoknowledge.ingestion.extractor import _parse_pdf_date, extract_pdf


def test_extract_returns_markdown_text(sample_pdf: Path) -> None:
    markdown, _, _ = extract_pdf(sample_pdf)
    assert "Introduction" in markdown or "Results" in markdown


def test_extract_metadata_title(sample_pdf: Path) -> None:
    _, meta, _ = extract_pdf(sample_pdf)
    assert meta.title == "Test Paper"


def test_extract_metadata_author(sample_pdf: Path) -> None:
    _, meta, _ = extract_pdf(sample_pdf)
    assert meta.author == "Test Author"


def test_extract_metadata_creation_date(sample_pdf: Path) -> None:
    _, meta, _ = extract_pdf(sample_pdf)
    assert meta.creation_date == "2024-01-15"


def test_extract_metadata_page_count(sample_pdf: Path) -> None:
    _, meta, _ = extract_pdf(sample_pdf)
    assert meta.page_count == 2


def test_extract_metadata_source_path(sample_pdf: Path) -> None:
    _, meta, _ = extract_pdf(sample_pdf)
    assert meta.source_path == sample_pdf


def test_fallback_title_from_filename(sample_pdf_no_metadata: Path) -> None:
    _, meta, _ = extract_pdf(sample_pdf_no_metadata)
    assert meta.title == "no meta"  # stem with spaces replacing separators


def test_fallback_author_unknown(sample_pdf_no_metadata: Path) -> None:
    _, meta, _ = extract_pdf(sample_pdf_no_metadata)
    assert meta.author == "Unknown"


def test_fallback_creation_date_none(sample_pdf_no_metadata: Path) -> None:
    _, meta, _ = extract_pdf(sample_pdf_no_metadata)
    assert meta.creation_date is None


def test_images_is_list(sample_pdf: Path) -> None:
    _, _, images = extract_pdf(sample_pdf)
    assert isinstance(images, list)


def test_parse_pdf_date_valid() -> None:
    assert _parse_pdf_date("D:20240115120000") == "2024-01-15"


def test_parse_pdf_date_none() -> None:
    assert _parse_pdf_date(None) is None


def test_parse_pdf_date_empty() -> None:
    assert _parse_pdf_date("") is None


def test_parse_pdf_date_invalid() -> None:
    assert _parse_pdf_date("not-a-date") is None
