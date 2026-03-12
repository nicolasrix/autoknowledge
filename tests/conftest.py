"""Shared pytest fixtures."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Create a minimal Obsidian-style vault in a temp directory."""
    vault = tmp_path / "vault"
    vault.mkdir()

    (vault / "Paper A.md").write_text(textwrap.dedent("""\
        ---
        title: "Paper A"
        tags: [transformer, attention]
        year: 2022
        ---
        # Introduction
        This paper introduces a new approach.

        ## Background
        Prior work has shown [[Paper B]] is relevant.

        # Methods
        We use multi-head attention.
    """))

    (vault / "Paper B.md").write_text(textwrap.dedent("""\
        ---
        title: "Paper B"
        tags: [cnn]
        year: 2021
        ---
        # Abstract
        A convolutional approach.
    """))

    subdir = vault / "subfolder"
    subdir.mkdir()
    (subdir / "Notes.md").write_text(textwrap.dedent("""\
        # My Notes
        Some unstructured notes about [[Paper A]].
    """))

    # Files that should be ignored
    obsidian = vault / ".obsidian"
    obsidian.mkdir()
    (obsidian / "config.json").write_text("{}")
    (vault / "not-markdown.txt").write_text("ignore me")

    return vault
