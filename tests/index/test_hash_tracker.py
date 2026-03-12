"""Tests for the SHA-256 hash tracker."""

from __future__ import annotations

from pathlib import Path

from autoknowledge.index.hash_tracker import HashTracker


def test_new_file_has_changed(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("Hello")
    with HashTracker(tmp_path / "hashes.db") as t:
        assert t.has_changed(f) is True


def test_unchanged_file_after_update(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("Hello")
    h = HashTracker.compute_hash(f)
    with HashTracker(tmp_path / "hashes.db") as t:
        t.update(f, h, chunk_count=3)
        assert t.has_changed(f) is False


def test_changed_file_after_modification(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("Hello")
    h = HashTracker.compute_hash(f)
    with HashTracker(tmp_path / "hashes.db") as t:
        t.update(f, h, chunk_count=3)
        f.write_text("World")  # content changed
        assert t.has_changed(f) is True


def test_remove_deletes_entry(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("Hello")
    h = HashTracker.compute_hash(f)
    with HashTracker(tmp_path / "hashes.db") as t:
        t.update(f, h, chunk_count=1)
        t.remove(f)
        assert t.has_changed(f) is True  # removed = treated as new


def test_get_all_tracked_paths(tmp_path: Path) -> None:
    files = [tmp_path / f"note{i}.md" for i in range(3)]
    for f in files:
        f.write_text("Content")
    with HashTracker(tmp_path / "hashes.db") as t:
        for f in files:
            t.update(f, HashTracker.compute_hash(f), chunk_count=1)
        tracked = t.get_all_tracked_paths()
        assert tracked == set(files)


def test_update_is_idempotent(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("Hello")
    h = HashTracker.compute_hash(f)
    with HashTracker(tmp_path / "hashes.db") as t:
        t.update(f, h, chunk_count=2)
        t.update(f, h, chunk_count=5)  # update chunk count
        assert t.has_changed(f) is False


def test_compute_hash_is_stable(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_bytes(b"Fixed content")
    h1 = HashTracker.compute_hash(f)
    h2 = HashTracker.compute_hash(f)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex
