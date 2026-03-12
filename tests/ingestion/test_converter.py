"""Tests for the PDF converter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autoknowledge.ingestion.converter import convert_pdf
from autoknowledge.types import ImageRef, PdfMetadata


def _make_meta(path: Path) -> PdfMetadata:
    return PdfMetadata(
        title="Test Paper",
        author="Test Author",
        creation_date="2024-01-15",
        page_count=2,
        source_path=path,
    )


def _make_image(description: str | None = None) -> ImageRef:
    return ImageRef(page=1, index=0, image_bytes=b"fakebytes", mime_type="image/png", description=description)


@pytest.mark.asyncio
async def test_convert_creates_md_file(tmp_path: Path, sample_pdf: Path) -> None:
    output = await convert_pdf(sample_pdf, tmp_path, describe_images=False, describer=None)
    assert output.exists()
    assert output.suffix == ".md"


@pytest.mark.asyncio
async def test_convert_output_next_to_pdf_when_no_output_dir(sample_pdf: Path) -> None:
    output = await convert_pdf(sample_pdf, None, describe_images=False, describer=None)
    assert output.parent == sample_pdf.parent
    output.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_convert_md_has_frontmatter(tmp_path: Path, sample_pdf: Path) -> None:
    output = await convert_pdf(sample_pdf, tmp_path, describe_images=False, describer=None)
    content = output.read_text()
    assert content.startswith("---")
    assert "title:" in content
    assert "source_type: pdf" in content


@pytest.mark.asyncio
async def test_convert_with_described_images(tmp_path: Path, sample_pdf: Path) -> None:
    mock_describer = MagicMock()
    described = [_make_image(description="A bar chart showing accuracy over epochs.")]
    mock_describer.describe = AsyncMock(return_value=described)

    with patch("autoknowledge.ingestion.converter.extract_pdf") as mock_extract:
        mock_extract.return_value = ("# Body text", _make_meta(sample_pdf), [_make_image()])
        output = await convert_pdf(sample_pdf, tmp_path, describe_images=True, describer=mock_describer)

    content = output.read_text()
    assert "A bar chart showing accuracy over epochs." in content
    assert "Figures" in content


@pytest.mark.asyncio
async def test_convert_skips_api_when_no_images(tmp_path: Path, sample_pdf: Path) -> None:
    mock_describer = MagicMock()
    mock_describer.describe = AsyncMock()

    with patch("autoknowledge.ingestion.converter.extract_pdf") as mock_extract:
        mock_extract.return_value = ("# Body text", _make_meta(sample_pdf), [])
        await convert_pdf(sample_pdf, tmp_path, describe_images=True, describer=mock_describer)

    mock_describer.describe.assert_not_called()


@pytest.mark.asyncio
async def test_convert_no_figures_section_without_described_images(tmp_path: Path, sample_pdf: Path) -> None:
    with patch("autoknowledge.ingestion.converter.extract_pdf") as mock_extract:
        mock_extract.return_value = ("# Body text", _make_meta(sample_pdf), [_make_image(description=None)])
        output = await convert_pdf(sample_pdf, tmp_path, describe_images=False, describer=None)

    content = output.read_text()
    assert "## Figures" not in content
