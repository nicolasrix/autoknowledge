"""Build YAML frontmatter for converted PDF documents."""

from __future__ import annotations

from autoknowledge.types import PdfMetadata


def build_frontmatter(meta: PdfMetadata, tags: list[str] | None = None) -> str:
    """Return a YAML frontmatter block string for the given PDF metadata."""
    lines = ["---"]
    lines.append(f"title: {_yaml_str(meta.title)}")
    lines.append(f"author: {_yaml_str(meta.author)}")
    if meta.creation_date:
        lines.append(f"date: {meta.creation_date}")
    lines.append(f"source_type: pdf")
    lines.append(f"page_count: {meta.page_count}")
    lines.append(f"source_path: {meta.source_path}")
    all_tags = ["pdf"] + (tags or [])
    lines.append(f"tags: [{', '.join(all_tags)}]")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _yaml_str(value: str) -> str:
    """Quote a string for YAML if it contains special characters."""
    needs_quoting = any(c in value for c in ('"', "'", ":", "#", "[", "]", "{", "}", "&", "*"))
    if needs_quoting:
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value
