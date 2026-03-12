"""Fixtures for ingestion tests, including a minimal programmatic PDF."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal 2-page PDF with text and metadata for testing."""
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    doc.set_metadata({
        "title": "Test Paper",
        "author": "Test Author",
        "creationDate": "D:20240115120000",
    })

    page1 = doc.new_page()
    page1.insert_text((72, 72), "Introduction\n\nThis is the first paragraph of the test document.")

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Results\n\nThis section contains the results.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_pdf_no_metadata(tmp_path: Path) -> Path:
    """Create a PDF with no metadata fields set."""
    pdf_path = tmp_path / "no_meta.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Some content without metadata.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
