"""Extract text, metadata, and images from a PDF file."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm

from autoknowledge.types import ImageRef, PdfMetadata

logger = logging.getLogger(__name__)

# Supported image types by xref colorspace
_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}
_FALLBACK_MIME = "image/png"

# Minimum image area to bother describing (skip tiny icons/decorations)
_MIN_IMAGE_PIXELS = 50 * 50


def extract_pdf(path: Path) -> tuple[str, PdfMetadata, list[ImageRef]]:
    """Extract markdown text, metadata, and images from a PDF.

    Returns:
        (markdown_text, metadata, images)
    """
    doc = fitz.open(str(path))
    try:
        markdown = pymupdf4llm.to_markdown(str(path))
        metadata = _extract_metadata(doc, path)
        images = _extract_images(doc)
    finally:
        doc.close()

    return markdown, metadata, images


def _extract_metadata(doc: fitz.Document, path: Path) -> PdfMetadata:
    meta = doc.metadata or {}

    title = _clean(meta.get("title")) or path.stem.replace("-", " ").replace("_", " ")
    author = _clean(meta.get("author")) or "Unknown"
    creation_date = _parse_pdf_date(meta.get("creationDate") or meta.get("modDate"))

    return PdfMetadata(
        title=title,
        author=author,
        creation_date=creation_date,
        page_count=doc.page_count,
        source_path=path,
    )


def _extract_images(doc: fitz.Document) -> list[ImageRef]:
    images: list[ImageRef] = []
    seen_xrefs: set[int] = set()

    for page_num, page in enumerate(doc):
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                logger.debug("Could not extract image xref=%d on page %d", xref, page_num + 1)
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            if width * height < _MIN_IMAGE_PIXELS:
                continue

            ext = base_image.get("ext", "png").lower()
            mime = _MIME_BY_EXT.get(ext, _FALLBACK_MIME)
            image_bytes = base_image["image"]

            images.append(ImageRef(
                page=page_num + 1,
                index=img_index,
                image_bytes=image_bytes,
                mime_type=mime,
            ))

    return images


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _parse_pdf_date(raw: str | None) -> str | None:
    """Parse PDF date format D:YYYYMMDDHHmmSS into ISO date string."""
    if not raw:
        return None
    match = re.search(r"D:(\d{4})(\d{2})(\d{2})", raw)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None
