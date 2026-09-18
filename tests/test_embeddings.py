"""Stage 2 — the pure parts of the embedding client. No network.

Stage 6 added the second half: what the cache does when the filesystem says no. In a
container the cache is a write to a layer that dies with the container, and on a hardened
platform to a filesystem that is read-only. Both are now first-class cases rather than
crashes.
"""

from pathlib import Path

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


# -- Stage 6: the cache in a container ---------------------------------------------


def test_writes_can_be_turned_off(tmp_path):
    """`PT_EMBEDDING_CACHE_WRITES=false` in the image. One embedding call per unseen
    question is the price, and a fresh container pays it either way."""
    cache = EmbeddingCache(tmp_path, writes_enabled=False)
    cache.put("k", [0.1, 0.2])
    cache.save()
    assert not (tmp_path / "cache.json").exists()


def test_reading_still_works_when_writing_is_off(tmp_path):
    """Reading and writing are separate permissions. A cache mounted read-only, or baked
    into an image, is still worth every hit it serves."""
    warm = EmbeddingCache(tmp_path)
    warm.put("k", [0.1, 0.2])
    warm.save()

    cold = EmbeddingCache(tmp_path, writes_enabled=False)
    assert cold.get("k") == [0.1, 0.2]


def test_a_read_only_filesystem_does_not_fail_the_request(tmp_path, monkeypatch, caplog):
    """The rule this draws: a write the system depends on fails loudly, a write that only
    makes it cheaper degrades quietly and says so once. Returning 500 because an
    optimisation could not persist trades a real failure for an imaginary one."""
    attempts = []

    def read_only(self, *args, **kwargs):
        attempts.append(self)
        raise OSError(30, "Read-only file system")

    cache = EmbeddingCache(tmp_path)
    cache.put("k", [0.1])
    monkeypatch.setattr(Path, "write_text", read_only)

    cache.save()  # must not raise

    assert len(attempts) == 1
    assert "read-only" in caplog.text.lower()


def test_a_rejected_write_is_not_retried_on_every_later_miss(tmp_path, monkeypatch):
    """One OSError per process, not one per request. A read-only filesystem does not
    become writable while you watch it, and the retry would put a stack trace in the log
    for every question asked."""
    attempts = []

    def read_only(self, *args, **kwargs):
        attempts.append(self)
        raise OSError(30, "Read-only file system")

    cache = EmbeddingCache(tmp_path)
    cache.put("k1", [0.1])
    monkeypatch.setattr(Path, "write_text", read_only)
    cache.save()

    cache.put("k2", [0.2])
    cache.save()
    cache.put("k3", [0.3])
    cache.save()

    assert len(attempts) == 1
