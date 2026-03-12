"""Markdown document parser: front-matter + wikilink extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import frontmatter

from autoknowledge.types import Document
from autoknowledge.vault.wikilinks import extract_wikilinks

logger = logging.getLogger(__name__)


def parse_document(path: Path) -> Document:
    """Parse a Markdown file into a Document.

    - Extracts YAML front-matter (empty dict if missing or malformed).
    - Reads the body as raw_content.
    - Extracts all [[wikilink]] targets from the body.
    - Falls back to the file stem as title if front-matter has no 'title'.
    """
    try:
        post = frontmatter.load(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse front-matter in %s: %s", path, exc)
        try:
            raw_content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw_content = ""
        return Document(
            path=path,
            title=path.stem,
            tags=[],
            front_matter={},
            raw_content=raw_content,
            wikilinks=[],
        )

    meta: dict[str, Any] = dict(post.metadata) if post.metadata else {}
    raw_content: str = post.content or ""

    title = str(meta.get("title", path.stem))
    raw_tags = meta.get("tags", [])
    tags: list[str] = (
        [str(t) for t in raw_tags] if isinstance(raw_tags, list)
        else [str(raw_tags)]
    )

    wikilinks = extract_wikilinks(raw_content)

    return Document(
        path=path,
        title=title,
        tags=tags,
        front_matter=meta,
        raw_content=raw_content,
        wikilinks=wikilinks,
    )
