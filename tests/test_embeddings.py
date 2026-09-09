"""Stage 2 — the pure parts of the embedding client. No network."""

import pytest

from arxiv_rag.retrieval.embeddings import batched, cache_key


def test_cache_key_is_stable():
    assert cache_key("hello", "m") == cache_key("hello", "m")


def test_cache_key_depends_on_model():
    """Vectors from different models are not comparable; the key must separate them."""
    assert cache_key("hello", "text-embedding-3-small") != cache_key("hello", "other")


def test_cache_key_depends_on_text():
    assert cache_key("hello", "m") != cache_key("hello ", "m")


def test_batched_splits_evenly():
    assert list(batched(["a", "b", "c", "d"], 2)) == [["a", "b"], ["c", "d"]]


def test_batched_last_batch_is_short():
    assert list(batched(["a", "b", "c"], 2)) == [["a", "b"], ["c"]]


def test_batched_size_larger_than_input():
    assert list(batched(["a", "b"], 10)) == [["a", "b"]]


def test_batched_empty():
    assert list(batched([], 3)) == []


def test_batched_rejects_zero_size():
    """Otherwise the loop never advances."""
    with pytest.raises(ValueError):
        list(batched(["a"], 0))
