"""Tests for hybrid retrieval score fusion."""

from __future__ import annotations

from autoknowledge.retrieval.hybrid import _min_max_normalise


def test_min_max_normalise_basic() -> None:
    result = _min_max_normalise({"a": 0.0, "b": 0.5, "c": 1.0})
    assert result["a"] == pytest.approx(0.0)
    assert result["b"] == pytest.approx(0.5)
    assert result["c"] == pytest.approx(1.0)


def test_min_max_normalise_all_same() -> None:
    result = _min_max_normalise({"a": 0.7, "b": 0.7, "c": 0.7})
    for v in result.values():
        assert v == pytest.approx(1.0)


def test_min_max_normalise_empty() -> None:
    assert _min_max_normalise({}) == {}


def test_min_max_normalise_single() -> None:
    result = _min_max_normalise({"only": 0.42})
    assert result["only"] == pytest.approx(1.0)


import pytest
