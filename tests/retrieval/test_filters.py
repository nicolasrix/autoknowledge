"""Tests for filter builder."""

from __future__ import annotations

import pytest

from autoknowledge.retrieval.filters import build_chroma_filter


def test_none_returns_none() -> None:
    assert build_chroma_filter(None) is None


def test_empty_dict_returns_none() -> None:
    assert build_chroma_filter({}) is None


def test_tags_single() -> None:
    result = build_chroma_filter({"tags": ["transformer"]})
    assert result == {"tags": {"$contains": "transformer"}}


def test_tags_multiple_uses_or() -> None:
    result = build_chroma_filter({"tags": ["transformer", "attention"]})
    assert result is not None
    assert "$or" in result


def test_path_prefix() -> None:
    result = build_chroma_filter({"path_prefix": "papers/"})
    assert result == {"doc_path": {"$contains": "papers/"}}


def test_title() -> None:
    result = build_chroma_filter({"title": "attention"})
    assert result == {"title": {"$contains": "attention"}}


def test_multiple_filters_uses_and() -> None:
    result = build_chroma_filter({"tags": ["ml"], "path_prefix": "papers/"})
    assert result is not None
    assert "$and" in result


def test_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown filter key"):
        build_chroma_filter({"unknown_field": "value"})


def test_empty_tags_raises() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        build_chroma_filter({"tags": []})
