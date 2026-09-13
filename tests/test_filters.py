"""Stage 3 - metadata filtering. Pre-filter, so k still means k."""

import numpy as np

from arxiv_rag.config import Settings
from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.bm25 import BM25Index
from arxiv_rag.retrieval.filters import ChunkFilter, allowed_indices
from arxiv_rag.retrieval.hybrid import DenseRetriever, HybridRetriever
from arxiv_rag.retrieval.store import ChunkStore


def chunk(arxiv_id: str, index: int, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{arxiv_id}::{index}",
        arxiv_id=arxiv_id,
        title=f"paper {arxiv_id}",
        section=section,
        text=text,
        index=index,
        token_count=5,
    )


CHUNKS = [
    chunk("1811.04551", 0, "Method", "planet plans with cross entropy method"),
    chunk("1811.04551", 1, "Results", "planet reaches strong scores"),
    chunk("1912.01603", 0, "Method", "dreamer learns an actor by latent imagination"),
    chunk("1912.01603", 1, "results", "dreamer reaches strong scores"),
]


# -- ChunkFilter.matches --------------------------------------------------------------


def test_no_constraints_allows_everything():
    f = ChunkFilter()
    assert all(f.matches(c) for c in CHUNKS)
    assert f.is_noop


def test_arxiv_ids_restricts_to_those_papers():
    f = ChunkFilter(arxiv_ids=frozenset({"1811.04551"}))
    assert [c.chunk_id for c in CHUNKS if f.matches(c)] == ["1811.04551::0", "1811.04551::1"]


def test_empty_frozenset_is_not_the_same_as_none():
    """None means 'no constraint'; an empty set means 'nothing passes'.

    Conflating them is how a filter silently becomes a no-op.
    """
    assert all(ChunkFilter(arxiv_ids=None).matches(c) for c in CHUNKS)
    assert not any(ChunkFilter(arxiv_ids=frozenset()).matches(c) for c in CHUNKS)


def test_sections_are_matched_case_insensitively():
    """Headings come from PDF extraction: 'Results' and 'results' both occur."""
    f = ChunkFilter(sections=frozenset({"results"}))
    assert [c.chunk_id for c in CHUNKS if f.matches(c)] == ["1811.04551::1", "1912.01603::1"]


def test_exclusion_wins_over_inclusion():
    """A chunk named in both lists is excluded - an exclusion is the stronger statement."""
    f = ChunkFilter(
        arxiv_ids=frozenset({"1811.04551"}),
        exclude_arxiv_ids=frozenset({"1811.04551"}),
    )
    assert not any(f.matches(c) for c in CHUNKS)


def test_filters_combine():
    f = ChunkFilter(arxiv_ids=frozenset({"1912.01603"}), sections=frozenset({"method"}))
    assert [c.chunk_id for c in CHUNKS if f.matches(c)] == ["1912.01603::0"]


# -- allowed_indices ------------------------------------------------------------------


def test_no_filter_means_no_work():
    """None, not a full set: the unfiltered path should not build an 874-element set."""
    assert allowed_indices(CHUNKS, None) is None
    assert allowed_indices(CHUNKS, ChunkFilter()) is None


def test_allowed_indices_are_positions_not_ids():
    assert allowed_indices(CHUNKS, ChunkFilter(arxiv_ids=frozenset({"1912.01603"}))) == {2, 3}


# -- the retrievers respect it --------------------------------------------------------


def store() -> ChunkStore:
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    return ChunkStore(CHUNKS, vectors)


def test_dense_returns_only_permitted_chunks():
    dense = DenseRetriever(store(), Settings(), embed=lambda q, s: [1.0, 0.0])
    f = ChunkFilter(arxiv_ids=frozenset({"1912.01603"}))
    hits = dense.search("planet", k=4, chunk_filter=f)
    assert {h.chunk.arxiv_id for h in hits} == {"1912.01603"}


def test_dense_prefilters_rather_than_postfilters():
    """Filtering to two chunks and asking for two must return two, not two-minus-dropped."""
    dense = DenseRetriever(store(), Settings(), embed=lambda q, s: [1.0, 0.0])
    f = ChunkFilter(arxiv_ids=frozenset({"1912.01603"}))
    assert len(dense.search("planet", k=2, chunk_filter=f)) == 2


def test_dense_forbidden_chunks_never_appear_even_when_most_similar():
    """The query vector points straight at 1811.04551::0; the filter must still win."""
    dense = DenseRetriever(store(), Settings(), embed=lambda q, s: [1.0, 0.0])
    f = ChunkFilter(exclude_arxiv_ids=frozenset({"1811.04551"}))
    assert all(h.chunk.arxiv_id != "1811.04551" for h in dense.search("q", k=4, chunk_filter=f))


def test_bm25_respects_the_filter():
    index = BM25Index(CHUNKS)
    f = ChunkFilter(sections=frozenset({"method"}))
    hits = index.search("strong scores", k=4, chunk_filter=f)
    assert all(h.chunk.section.lower() == "method" for h in hits)


def test_hybrid_passes_the_filter_to_both_retrievers():
    dense = DenseRetriever(store(), Settings(), embed=lambda q, s: [1.0, 0.0])
    hybrid = HybridRetriever([dense, BM25Index(CHUNKS)], depth=10)
    f = ChunkFilter(arxiv_ids=frozenset({"1912.01603"}))
    hits = hybrid.search("planet plans strong scores", k=4, chunk_filter=f)
    assert hits and all(h.chunk.arxiv_id == "1912.01603" for h in hits)


def test_a_filter_matching_nothing_returns_nothing():
    dense = DenseRetriever(store(), Settings(), embed=lambda q, s: [1.0, 0.0])
    f = ChunkFilter(arxiv_ids=frozenset({"9999.99999"}))
    assert dense.search("q", k=4, chunk_filter=f) == []
