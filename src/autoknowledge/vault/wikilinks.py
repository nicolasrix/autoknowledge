"""Obsidian [[wikilink]] extraction and resolution."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches [[Target]] and [[Target|Display alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|#\n]+?)(?:\|[^\]]+)?\]\]")


def extract_wikilinks(text: str) -> list[str]:
    """Return the raw target names of all [[wikilinks]] in *text*.

    Aliases are stripped: [[Foo|Bar]] → "Foo".
    Heading anchors are stripped: [[Foo#Section]] → "Foo".
    """
    targets = []
    for match in _WIKILINK_RE.finditer(text):
        target = match.group(1).strip()
        # Strip heading anchor if present (the regex already excludes # in group)
        targets.append(target)
    return targets


def resolve_wikilinks(
    targets: list[str],
    all_paths: list[Path],
    vault_root: Path,
) -> dict[str, Path | None]:
    """Resolve wikilink target names to absolute paths using Obsidian's rules.

    Resolution order (matches Obsidian behaviour):
    1. Exact stem match (case-insensitive).
    2. Among matches, prefer the shortest relative path from vault_root.
    3. Unresolvable targets map to None.

    Logs a warning when a target is ambiguous (multiple same-length shortest paths).
    """
    # Build stem → [paths] index (case-insensitive)
    stem_index: dict[str, list[Path]] = {}
    for p in all_paths:
        key = p.stem.lower()
        stem_index.setdefault(key, []).append(p)

    resolved: dict[str, Path | None] = {}
    for target in targets:
        if target in resolved:
            continue

        # Strip any heading anchor the caller may have left in (e.g. "Foo#Bar")
        stem = target.split("#")[0].strip().lower()
        candidates = stem_index.get(stem)

        if not candidates:
            resolved[target] = None
            continue

        if len(candidates) == 1:
            resolved[target] = candidates[0]
            continue

        # Multiple candidates: pick the shortest relative path (Obsidian heuristic)
        def _rel_len(p: Path) -> int:
            try:
                return len(p.relative_to(vault_root).parts)
            except ValueError:
                return 999

        candidates_sorted = sorted(candidates, key=_rel_len)
        shortest_len = _rel_len(candidates_sorted[0])
        tied = [p for p in candidates_sorted if _rel_len(p) == shortest_len]

        if len(tied) > 1:
            logger.warning(
                "Ambiguous wikilink [[%s]]: %d equally-short candidates, using first",
                target, len(tied),
            )
        resolved[target] = tied[0]

    return resolved
