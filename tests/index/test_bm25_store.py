"""Tests for the in-memory BM25 store."""

from __future__ import annotations

from autoknowledge.index.bm25_store import BM25Store


def test_query_returns_best_match() -> None:
    store = BM25Store()
    store.build([
        ("c1", "transformer attention mechanism deep learning"),
        ("c2", "random forest decision tree classification"),
        ("c3", "neural network backpropagation gradient"),
    ])
    results = store.query("transformer attention", top_k=3)
    assert results[0][0] == "c1"  # best match


def test_query_returns_top_k() -> None:
    store = BM25Store()
    store.build([(f"c{i}", f"document number {i}") for i in range(10)])
    results = store.query("document", top_k=3)
    assert len(results) == 3


def test_query_empty_store_returns_empty() -> None:
    store = BM25Store()
    assert store.query("anything", top_k=5) == []


def test_query_empty_query_returns_empty() -> None:
    store = BM25Store()
    store.build([("c1", "some content")])
    assert store.query("", top_k=5) == []


def test_build_empty_clears_index() -> None:
    store = BM25Store()
    store.build([("c1", "content")])
    store.build([])
    assert store.size == 0
    assert store.query("content", top_k=1) == []


def test_scores_are_non_negative() -> None:
    store = BM25Store()
    store.build([("c1", "hello world"), ("c2", "foo bar")])
    results = store.query("hello", top_k=2)
    for _, score in results:
        assert score >= 0.0


def test_rebuild_replaces_index() -> None:
    store = BM25Store()
    store.build([("c1", "old content about cats")])
    store.build([("c2", "new content about dogs")])
    results = store.query("dogs", top_k=1)
    assert results[0][0] == "c2"
    assert store.size == 1
