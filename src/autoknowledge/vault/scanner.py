"""Vault filesystem scanner."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Obsidian system directories that should never be indexed
_IGNORED_DIRS = frozenset({".obsidian", ".trash", ".git"})


def scan_vault(vault_path: Path) -> list[Path]:
    """Return all .md files in *vault_path*, excluding Obsidian system dirs.

    Results are sorted for stable ordering across runs.
    """
    if not vault_path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {vault_path}")
    if not vault_path.is_dir():
        raise NotADirectoryError(f"Vault path is not a directory: {vault_path}")

    found: list[Path] = []
    for path in vault_path.rglob("*.md"):
        if any(part in _IGNORED_DIRS for part in path.parts):
            continue
        try:
            if path.is_file():
                found.append(path.resolve())
        except PermissionError:
            logger.warning("Permission denied, skipping: %s", path)

    return sorted(found)
