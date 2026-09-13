"""Stage 3 - reranking. A fake reranker with a known rule, so every ordering is checkable.

No model is loaded and nothing is downloaded: `RerankingRetriever` is tested against the
`Reranker` protocol, not against a specific cross-encoder.
"""

import pytest

from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.rerank import RerankingRetriever
from arxiv_rag.retrieval.store import SearchHit


def make_chunk(name: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=name,
        arxiv_id="1234.5678",
        title="A paper",
        section="Method",
        text=text,
        index=0,
        token_count=5,
    )


CHUNKS = {
    "a": make_chunk("a", "padding padding padding"),
    "b": make_chunk("b", "padding padding"),
    "c": make_chunk("c", "the answer to the question"),
    "d": make_chunk("d", "padding"),
}


class FakeBase:
    """Returns a fixed ranking and records the k it was asked for."""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.asked_for: list[int] = []

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self.asked_for.append(k)
        # descending base scores, so base order is obvious in the input
        return [SearchHit(CHUNKS[n], 1.0 - i * 0.1) for i, n in enumerate(self.order[:k])]


class KeywordReranker:
    """Scores a chunk by how much of the query it contains. Deterministic and readable."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        self.calls.append(len(chunks))
        words = set(query.lower().split())
        return [float(len(words & set(c.text.lower().split()))) for c in chunks]


def ids(hits: list[SearchHit]) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


# -- the point of reranking -----------------------------------------------------------


def test_a_candidate_ranked_last_by_the_base_can_reach_first():
    """The whole reason this exists. Measured on the real corpus: q037's gold is at base
    rank 29, so a reranker that cannot promote from deep in the pool is useless."""
    base = FakeBase(["a", "b", "d", "c"])
    hits = RerankingRetriever(base, KeywordReranker(), depth=30).search("the answer", k=1)
    assert ids(hits) == ["c"]


def test_base_ordering_is_discarded_not_blended():
    base = FakeBase(["a", "b", "d", "c"])
    hits = RerankingRetriever(base, KeywordReranker(), depth=30).search("the answer", k=4)
    assert ids(hits)[0] == "c"


def test_scores_come_from_the_reranker_not_the_base():
    base = FakeBase(["a", "c"])
    hits = RerankingRetriever(base, KeywordReranker(), depth=30).search("the answer", k=1)
    assert hits[0].score == 2.0  # "the" and "answer", not the base's 0.9


# -- over-retrieve, return k ----------------------------------------------------------


def test_retrieves_depth_and_returns_k():
    base = FakeBase(["a", "b", "c", "d"])
    reranker = KeywordReranker()
    hits = RerankingRetriever(base, reranker, depth=30).search("the answer", k=2)
    assert base.asked_for == [30]
    assert reranker.calls == [4]  # everything the base returned gets rescored
    assert len(hits) == 2


def test_every_candidate_is_scored_exactly_once():
    base = FakeBase(["a", "b", "c", "d"])
    reranker = KeywordReranker()
    RerankingRetriever(base, reranker, depth=30).search("the answer", k=1)
    assert reranker.calls == [4]


# -- edges ----------------------------------------------------------------------------


def test_empty_pool_does_not_invoke_the_model():
    """A forward pass over nothing is pure latency."""
    reranker = KeywordReranker()
    assert RerankingRetriever(FakeBase([]), reranker, depth=30).search("q", k=5) == []
    assert reranker.calls == []


def test_fewer_candidates_than_k():
    base = FakeBase(["c"])
    hits = RerankingRetriever(base, KeywordReranker(), depth=30).search("the answer", k=5)
    assert ids(hits) == ["c"]


def test_ties_fall_back_to_base_order():
    """Every candidate scores 0: the base ordering must survive, deterministically."""
    base = FakeBase(["a", "b", "d"])
    hits = RerankingRetriever(base, KeywordReranker(), depth=30).search("zzz", k=3)
    assert ids(hits) == ["a", "b", "d"]


def test_deterministic_across_calls():
    base = FakeBase(["a", "b", "d", "c"])
    retriever = RerankingRetriever(base, KeywordReranker(), depth=30)
    assert ids(retriever.search("the answer", k=3)) == ids(retriever.search("the answer", k=3))


def test_depth_is_what_is_asked_of_the_base():
    base = FakeBase(["a", "b", "c", "d"])
    RerankingRetriever(base, KeywordReranker(), depth=7).search("the answer", k=2)
    assert base.asked_for == [7]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_k_is_respected(k):
    base = FakeBase(["a", "b", "c", "d"])
    hits = RerankingRetriever(base, KeywordReranker(), depth=30).search("the answer", k=k)
    assert len(hits) == k


# -- listwise: ordering -> scores -----------------------------------------------------
# A listwise reranker returns a ranking, not scores. This is the mapping, and it is where
# a model's invented passage numbers have to be caught.

from arxiv_rag.retrieval.rerank import order_to_scores  # noqa: E402


def ranked_order(scores: list[float]) -> list[int]:
    """1-based candidate numbers, best score first, base order breaking ties."""
    return [i + 1 for i, _ in sorted(enumerate(scores), key=lambda p: (-p[1], p[0]))]


def test_full_ranking_is_followed():
    assert ranked_order(order_to_scores([3, 1, 2], 3)) == [3, 1, 2]


def test_unranked_candidates_fall_below_ranked_ones():
    """The model mentioned 3 and 1; 2 and 4 keep base order underneath."""
    assert ranked_order(order_to_scores([3, 1], 4)) == [3, 1, 2, 4]


def test_an_empty_ranking_keeps_the_base_order():
    """A parse failure must degrade to the retriever's opinion, not to chaos."""
    assert ranked_order(order_to_scores([], 4)) == [1, 2, 3, 4]


def test_invented_passage_numbers_are_ignored():
    """The model returning 47 out of 30 must not shift anything."""
    assert ranked_order(order_to_scores([47, 2, 0, -1], 3)) == [2, 1, 3]


def test_repeats_keep_their_first_position():
    assert ranked_order(order_to_scores([2, 2, 1], 3)) == [2, 1, 3]


def test_non_integers_are_ignored():
    assert ranked_order(order_to_scores(["x", None, 2], 3)) == [2, 1, 3]


def test_every_candidate_gets_a_score():
    assert len(order_to_scores([2], 5)) == 5


def test_ranked_always_outscores_unranked():
    scores = order_to_scores([5], 5)
    assert scores[4] > max(scores[:4])
