"""Convert a single PDF file to Markdown."""

from __future__ import annotations

import logging
from pathlib import Path

from autoknowledge.ingestion.extractor import extract_pdf
from autoknowledge.ingestion.frontmatter import build_frontmatter
from autoknowledge.types import ImageRef

logger = logging.getLogger(__name__)


async def convert_pdf(
    path: Path,
    output_dir: Path | None,
    describe_images: bool,
    describer: object | None,
) -> Path:
    """Convert a PDF to Markdown and write it to disk. Returns the output path."""
    markdown, metadata, images = extract_pdf(path)

    described_images: list[ImageRef] = images
    if describe_images and describer is not None and images:
        logger.info("Describing %d image(s) from %s …", len(images), path.name)
        described_images = await describer.describe(images)  # type: ignore[union-attr]

    body = _insert_image_descriptions(markdown, described_images)
    frontmatter = build_frontmatter(metadata)
    content = frontmatter + body

    dest_dir = output_dir or path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / f"{path.stem}.md"
    output_path.write_text(content, encoding="utf-8")

    logger.debug("Wrote %s", output_path)
    return output_path


def _insert_image_descriptions(markdown: str, images: list[ImageRef]) -> str:
    """Append image descriptions as blockquotes at the end of the document."""
    if not images:
        return markdown

    described = [img for img in images if img.description]
    if not described:
        return markdown

    lines = [markdown.rstrip(), "", "## Figures", ""]
    for img in described:
        lines.append(f"> **[Figure, p.{img.page}]**: {img.description}")
        lines.append("")

    return "\n".join(lines)
