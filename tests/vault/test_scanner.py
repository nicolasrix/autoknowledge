"""Tests for vault scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoknowledge.vault.scanner import scan_vault


def test_finds_markdown_files(tmp_vault: Path) -> None:
    results = scan_vault(tmp_vault)
    stems = {p.stem for p in results}
    assert "Paper A" in stems
    assert "Paper B" in stems
    assert "Notes" in stems


def test_excludes_obsidian_dir(tmp_vault: Path) -> None:
    results = scan_vault(tmp_vault)
    assert not any(".obsidian" in str(p) for p in results)


def test_excludes_non_markdown(tmp_vault: Path) -> None:
    results = scan_vault(tmp_vault)
    assert all(p.suffix == ".md" for p in results)


def test_results_are_sorted(tmp_vault: Path) -> None:
    results = scan_vault(tmp_vault)
    assert results == sorted(results)


def test_results_are_absolute(tmp_vault: Path) -> None:
    results = scan_vault(tmp_vault)
    assert all(p.is_absolute() for p in results)


def test_nonexistent_vault_raises() -> None:
    with pytest.raises(FileNotFoundError):
        scan_vault(Path("/nonexistent/path"))


def test_file_as_vault_raises(tmp_path: Path) -> None:
    f = tmp_path / "file.md"
    f.write_text("# Hello")
    with pytest.raises(NotADirectoryError):
        scan_vault(f)


def test_nested_files_found(tmp_vault: Path) -> None:
    results = scan_vault(tmp_vault)
    doc_paths = [str(p) for p in results]
    assert any("subfolder" in p for p in doc_paths)
