"""Batch PDF ingestion pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from autoknowledge.config import Config
from autoknowledge.ingestion.converter import convert_pdf
from autoknowledge.ingestion.scanner import scan_pdfs
from autoknowledge.types import IngestStats

logger = logging.getLogger(__name__)


async def run_ingest(
    input_path: Path,
    output_dir: Path | None,
    describe_images: bool,
    config: Config,
) -> IngestStats:
    """Scan input_path for PDFs and convert each to Markdown.

    Skips PDFs whose output .md already exists and is newer than the source PDF.
    """
    pdfs = scan_pdfs(input_path)
    total = len(pdfs)

    if total == 0:
        print("No PDF files found.", file=sys.stderr)
        return IngestStats(total_pdfs=0, converted=0, skipped=0)

    describer = None
    if describe_images:
        from autoknowledge.ingestion.image_describer import ImageDescriber
        describer = ImageDescriber(
            model=config.ingest.anthropic_model,
            max_image_dimension=config.ingest.max_image_dimension,
        )

    converted = 0
    skipped = 0
    errors: list[str] = []

    for i, pdf_path in enumerate(pdfs, start=1):
        _progress(i, total, pdf_path)

        dest_dir = output_dir or pdf_path.parent
        output_path = dest_dir / f"{pdf_path.stem}.md"

        if _is_up_to_date(pdf_path, output_path):
            skipped += 1
            continue

        try:
            await convert_pdf(pdf_path, output_dir, describe_images, describer)
            converted += 1
        except Exception as exc:
            msg = f"{pdf_path.name}: {exc}"
            logger.error("Failed to convert %s: %s", pdf_path, exc)
            errors.append(msg)

    print(file=sys.stderr)  # newline after progress
    stats = IngestStats(total_pdfs=total, converted=converted, skipped=skipped, errors=errors)
    print(stats.summary(), file=sys.stderr)
    return stats


def _is_up_to_date(pdf_path: Path, output_path: Path) -> bool:
    """Return True if the output .md exists and is newer than the source PDF."""
    if not output_path.exists():
        return False
    return output_path.stat().st_mtime >= pdf_path.stat().st_mtime


def _progress(i: int, total: int, path: Path) -> None:
    print(f"\rConverting {i}/{total}: {path.name}   ", end="", file=sys.stderr, flush=True)
