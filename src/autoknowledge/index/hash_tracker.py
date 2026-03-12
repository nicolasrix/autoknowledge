"""SHA-256 hash tracker for incremental indexing (SQLite-backed)."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class HashTracker:
    """Tracks file path → SHA-256 hash to detect changes between index runs."""

    _CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS file_hashes (
            path        TEXT    PRIMARY KEY,
            sha256      TEXT    NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            last_indexed TEXT   NOT NULL
        )
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._CREATE_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> HashTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def has_changed(self, path: Path) -> bool:
        """Return True if *path* is new or its content differs from the stored hash."""
        current_hash = self.compute_hash(path)
        row = self._conn.execute(
            "SELECT sha256 FROM file_hashes WHERE path = ?", (str(path),)
        ).fetchone()
        return row is None or row["sha256"] != current_hash

    def update(self, path: Path, sha256: str, chunk_count: int) -> None:
        """Upsert the hash record for *path*."""
        self._conn.execute(
            """
            INSERT INTO file_hashes (path, sha256, chunk_count, last_indexed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                sha256 = excluded.sha256,
                chunk_count = excluded.chunk_count,
                last_indexed = excluded.last_indexed
            """,
            (str(path), sha256, chunk_count, _now()),
        )
        self._conn.commit()

    def remove(self, path: Path) -> None:
        """Delete the hash record for *path*."""
        self._conn.execute("DELETE FROM file_hashes WHERE path = ?", (str(path),))
        self._conn.commit()

    def get_all_tracked_paths(self) -> set[Path]:
        """Return the set of all currently tracked file paths."""
        rows = self._conn.execute("SELECT path FROM file_hashes").fetchall()
        return {Path(row["path"]) for row in rows}

    @staticmethod
    def compute_hash(path: Path) -> str:
        """Compute SHA-256 of raw file bytes, returned as a hex string."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
