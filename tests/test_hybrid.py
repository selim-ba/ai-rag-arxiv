"""Stage 3 - the hybrid retriever. Fake retrievers, so every fused ranking is checkable."""

import numpy as np
import pytest

from arxiv_rag.config import Settings
from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.bm25 import BM25Index
from arxiv_rag.retrieval.hybrid import DenseRetriever, HybridRetriever
from arxiv_rag.retrieval.store import ChunkStore, SearchHit

CHUNKS = {
    name: Chunk(
        chunk_id=name,
        arxiv_id="1234.5678",
        title="A paper",
        section="Method",
        text=f"text of {name}",
        index=i,
        token_count=3,
    )
    for i, name in enumerate(["a", "b", "c", "d", "e"])
}


class FakeRetriever:
    """Returns a fixed ranking, and records the k it was asked for."""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.asked_for: list[int] = []

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self.asked_for.append(k)
        return [SearchHit(CHUNKS[name], 1.0) for name in self.order[:k]]


def ids(hits: list[SearchHit]) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


# -- fusion behaviour ----------------------------------------------------------------


def test_agreement_ranks_first():
    hybrid = HybridRetriever([FakeRetriever(["a", "b"]), FakeRetriever(["a", "c"])])
    assert ids(hybrid.search("q", k=1)) == ["a"]


def test_a_chunk_only_one_retriever_found_still_surfaces():
    """The q030 case: BM25 was the only retriever that found it at all."""
    dense = FakeRetriever(["a", "b", "c"])
    sparse = FakeRetriever(["d"])
    assert "d" in ids(HybridRetriever([dense, sparse]).search("q", k=4))


def test_retrieves_depth_but_returns_k():
    """The whole point of over-retrieving: the useful candidate may sit at rank 8."""
    dense = FakeRetriever(["a", "b", "c", "d", "e"])
    sparse = FakeRetriever(["e", "d", "c", "b", "a"])
    hybrid = HybridRetriever([dense, sparse], depth=30)
    hits = hybrid.search("q", k=2)
    assert dense.asked_for == [30]
    assert sparse.asked_for == [30]
    assert len(hits) == 2


def test_an_empty_retriever_is_not_an_error():
    """BM25 returns [] when no indexed term appears in the query."""
    hybrid = HybridRetriever([FakeRetriever(["a", "b"]), FakeRetriever([])])
    assert ids(hybrid.search("q", k=2)) == ["a", "b"]


def test_single_retriever_passes_its_order_through():
    hybrid = HybridRetriever([FakeRetriever(["c", "a", "b"])])
    assert ids(hybrid.search("q", k=3)) == ["c", "a", "b"]


def test_results_are_search_hits_carrying_the_fused_score():
    """The score is an RRF score, not a cosine or a BM25 score - those were incomparable."""
    hit = HybridRetriever([FakeRetriever(["a"]), FakeRetriever(["a"])]).search("q", k=1)[0]
    assert hit.chunk.chunk_id == "a"
    assert 0.0 < hit.score < 1.0


def test_weights_shift_the_outcome():
    """Weighting is an empirical question, so it has to be expressible."""
    dense = FakeRetriever(["a", "c"])
    sparse = FakeRetriever(["b", "d"])
    assert ids(HybridRetriever([dense, sparse], weights=[10.0, 1.0]).search("q", k=1)) == ["a"]
    assert ids(HybridRetriever([dense, sparse], weights=[1.0, 10.0]).search("q", k=1)) == ["b"]


def test_deterministic_across_calls():
    """Same query, same order, every time - or eval numbers wobble for no reason."""
    hybrid = HybridRetriever([FakeRetriever(["a", "b"]), FakeRetriever(["b", "a"])])
    assert ids(hybrid.search("q", k=2)) == ids(hybrid.search("q", k=2))


# -- construction guards -------------------------------------------------------------


def test_no_retrievers_is_a_configuration_error():
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_weight_count_must_match_retriever_count():
    with pytest.raises(ValueError):
        HybridRetriever([FakeRetriever(["a"])], weights=[1.0, 1.0])


# -- the real components fit the protocol --------------------------------------------


def test_dense_and_bm25_are_interchangeable_under_the_protocol():
    """DenseRetriever and BM25Index share a shape without sharing a base class."""
    chunks = [CHUNKS["a"], CHUNKS["b"]]
    store = ChunkStore(chunks, np.array([[1.0, 0.0], [0.0, 1.0]]))
    dense = DenseRetriever(store, Settings(), embed=lambda q, s: [1.0, 0.0])
    sparse = BM25Index(chunks)

    hybrid = HybridRetriever([dense, sparse])
    hits = hybrid.search("text of a", k=2)
    assert ids(hits)[0] == "a"
