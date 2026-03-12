"""Scan a path for PDF files to ingest."""

from __future__ import annotations

from pathlib import Path

_SKIP_DIRS = {".obsidian", ".trash", ".git", "__pycache__", ".DS_Store"}


def scan_pdfs(input_path: Path) -> list[Path]:
    """Return a sorted list of PDF files at input_path (file or directory tree)."""
    input_path = input_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {input_path}")
        return [input_path]

    pdfs: list[Path] = []
    for item in input_path.rglob("*.pdf"):
        if not any(part in _SKIP_DIRS for part in item.parts):
            pdfs.append(item)

    return sorted(pdfs)
