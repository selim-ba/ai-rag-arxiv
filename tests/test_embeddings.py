"""Stage 2 — the pure parts of the embedding client. No network."""

import pytest

from arxiv_rag.retrieval.embeddings import (
    EmbeddingCache,
    batched,
    cache_key,
    get_cache,
)


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


# -- cache I/O ------------------------------------------------------------------------
# Added in Stage 3 after measuring: a warm-cache query cost 479ms against 0.9ms of actual
# search, because the 28MB cache was reloaded and rewritten on every call.


def test_save_is_a_noop_when_nothing_was_added(tmp_path):
    """A pure cache hit must not rewrite 28MB of JSON."""
    cache = EmbeddingCache(tmp_path)
    cache.put("k", [1.0, 2.0])
    cache.save()
    before = (tmp_path / "cache.json").stat().st_mtime_ns

    EmbeddingCache(tmp_path).save()  # loaded, read from, never written to
    assert (tmp_path / "cache.json").stat().st_mtime_ns == before


def test_save_writes_when_something_was_added(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put("k", [1.0, 2.0])
    cache.save()
    assert EmbeddingCache(tmp_path).get("k") == [1.0, 2.0]


def test_save_leaves_no_temporary_file_behind(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put("k", [1.0])
    cache.save()
    assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]


def test_get_cache_returns_the_same_object_for_a_directory(tmp_path):
    """Two callers must share one cache, or their writes do not see each other."""
    get_cache.cache_clear()
    first = get_cache(tmp_path)
    first.put("k", [1.0])
    assert get_cache(tmp_path).get("k") == [1.0]
    assert get_cache(tmp_path) is first
    get_cache.cache_clear()


def test_a_second_save_after_more_writes_still_persists(tmp_path):
    get_cache.cache_clear()
    cache = get_cache(tmp_path)
    cache.put("a", [1.0])
    cache.save()
    cache.put("b", [2.0])
    cache.save()
    get_cache.cache_clear()
    reloaded = get_cache(tmp_path)
    assert reloaded.get("a") == [1.0] and reloaded.get("b") == [2.0]
    get_cache.cache_clear()
