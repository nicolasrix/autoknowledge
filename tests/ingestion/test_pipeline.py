"""Integration tests for the PDF ingestion pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from autoknowledge.config import load_config
from autoknowledge.ingestion.pipeline import run_ingest
from autoknowledge.vault.parser import parse_document


@pytest.mark.asyncio
async def test_pipeline_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    config = load_config()
    stats = await run_ingest(empty, None, False, config)
    assert stats.total_pdfs == 0


@pytest.mark.asyncio
async def test_pipeline_converts_single_pdf(tmp_path: Path, sample_pdf: Path) -> None:
    output_dir = tmp_path / "out"
    config = load_config()
    stats = await run_ingest(sample_pdf, output_dir, False, config)

    assert stats.total_pdfs == 1
    assert stats.converted == 1
    assert stats.skipped == 0
    assert stats.errors == []

    output = output_dir / "sample.md"
    assert output.exists()


@pytest.mark.asyncio
async def test_pipeline_output_parseable_by_vault(tmp_path: Path, sample_pdf: Path) -> None:
    output_dir = tmp_path / "out"
    config = load_config()
    await run_ingest(sample_pdf, output_dir, False, config)

    output = output_dir / "sample.md"
    doc = parse_document(output)
    assert doc.title == "Test Paper"
    assert doc.front_matter.get("source_type") == "pdf"
    assert "pdf" in doc.tags


@pytest.mark.asyncio
async def test_pipeline_skips_up_to_date(tmp_path: Path, sample_pdf: Path) -> None:
    output_dir = tmp_path / "out"
    config = load_config()

    await run_ingest(sample_pdf, output_dir, False, config)

    # Touch the output to make it newer
    output = output_dir / "sample.md"
    output.touch()

    stats = await run_ingest(sample_pdf, output_dir, False, config)
    assert stats.skipped == 1
    assert stats.converted == 0


@pytest.mark.asyncio
async def test_pipeline_reconverts_stale(tmp_path: Path, sample_pdf: Path) -> None:
    output_dir = tmp_path / "out"
    config = load_config()

    await run_ingest(sample_pdf, output_dir, False, config)

    # Make the PDF newer than the output
    time.sleep(0.01)
    sample_pdf.touch()

    stats = await run_ingest(sample_pdf, output_dir, False, config)
    assert stats.converted == 1
    assert stats.skipped == 0


@pytest.mark.asyncio
async def test_pipeline_batch_converts_directory(tmp_path: Path) -> None:
    import fitz

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    for name in ["a.pdf", "b.pdf", "c.pdf"]:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), f"Content of {name}")
        doc.save(str(pdf_dir / name))
        doc.close()

    output_dir = tmp_path / "out"
    config = load_config()
    stats = await run_ingest(pdf_dir, output_dir, False, config)

    assert stats.total_pdfs == 3
    assert stats.converted == 3
    assert len(list(output_dir.glob("*.md"))) == 3
