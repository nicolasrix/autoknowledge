"""Tests for the PDF scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoknowledge.ingestion.scanner import scan_pdfs


def test_single_pdf_file(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    assert scan_pdfs(pdf) == [pdf]


def test_directory_finds_pdfs_recursively(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.pdf").touch()
    (sub / "notes.md").touch()

    result = scan_pdfs(tmp_path)
    assert len(result) == 2
    assert all(p.suffix == ".pdf" for p in result)


def test_empty_directory_returns_empty(tmp_path: Path) -> None:
    assert scan_pdfs(tmp_path) == []


def test_returns_sorted(tmp_path: Path) -> None:
    for name in ["c.pdf", "a.pdf", "b.pdf"]:
        (tmp_path / name).touch()
    result = scan_pdfs(tmp_path)
    assert result == sorted(result)


def test_nonexistent_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_pdfs(tmp_path / "does_not_exist")


def test_non_pdf_file_raises(tmp_path: Path) -> None:
    md = tmp_path / "notes.md"
    md.touch()
    with pytest.raises(ValueError, match="not a PDF"):
        scan_pdfs(md)


def test_skips_hidden_directories(tmp_path: Path) -> None:
    (tmp_path / "visible.pdf").touch()
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "hidden.pdf").touch()

    result = scan_pdfs(tmp_path)
    assert len(result) == 1
    assert result[0].name == "visible.pdf"
