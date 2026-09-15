"""Phase 0 of two-stage retrieval: scoring papers from chunk hits. Pure; no index.

Built before the retriever that will use it, because the coarse pass is the part that can
LOSE questions - a gold chunk in a paper that misses the top N becomes unreachable, which
is worse than not narrowing at all. The oracle measurement had `lost = 0` by construction;
a real two-stage pass will not.
"""

import pytest

from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.hierarchical import score_papers, top_papers
from arxiv_rag.retrieval.store import SearchHit


def hit(arxiv_id: str, index: int, score: float) -> SearchHit:
    return SearchHit(
        Chunk(
            chunk_id=f"{arxiv_id}::{index}",
            arxiv_id=arxiv_id,
            title=f"Paper {arxiv_id}",
            section="Method",
            text="text",
            index=index,
            token_count=5,
        ),
        score,
    )


# -- the aggregations, each hand-computable -------------------------------------------

MIXED = [
    hit("aaa", 0, 0.9),
    hit("aaa", 1, 0.1),
    hit("bbb", 0, 0.5),
    hit("bbb", 1, 0.5),
    hit("bbb", 2, 0.5),
]


def test_max_takes_the_best_chunk():
    assert score_papers(MIXED, "max") == [("aaa", 0.9), ("bbb", 0.5)]


def test_sum_rewards_breadth():
    """bbb totals 1.5 against aaa's 1.0 and overtakes it - three decent chunks beating one
    excellent one. Defensible, and the opposite of what `max` says."""
    assert score_papers(MIXED, "sum") == [("bbb", 1.5), ("aaa", 1.0)]


def test_mean_normalises_length_to_the_point_of_a_tie():
    """One excellent chunk plus one poor one averages to the same as three mediocre ones.

    That is `mean` doing exactly what it promises and losing the distinction in the
    process: `max` ranks aaa first for its 0.9, `sum` ranks bbb first for its breadth, and
    `mean` cannot separate them at all. The tie then breaks on arXiv id, deterministically.
    """
    assert score_papers(MIXED, "mean") == [("aaa", 0.5), ("bbb", 0.5)]


def test_count_ignores_scores_entirely():
    assert score_papers(MIXED, "count") == [("bbb", 3.0), ("aaa", 2.0)]


def test_the_aggregations_disagree_on_the_same_hits():
    """The reason `--paper-recall` sweeps them instead of picking one. Same five hits,
    three different winners."""
    winners = {agg: score_papers(MIXED, agg)[0][0] for agg in ("max", "sum", "mean", "count")}
    assert winners["max"] == "aaa", "best single chunk"
    assert winners["sum"] == "bbb", "breadth across chunks"
    assert winners["count"] == "bbb", "most chunks in the deep pool"
    assert len(set(winners.values())) > 1


# -- length bias, the measured risk ---------------------------------------------------


def test_sum_lets_a_long_weak_paper_beat_a_short_strong_one():
    """Papers here hold 6 to 43 chunks. A long paper accumulates score from mediocre
    chunks that a short one cannot, so `sum` can promote breadth over relevance."""
    long_weak = [hit("long", i, 0.2) for i in range(20)]
    short_strong = [hit("short", i, 0.95) for i in range(3)]
    assert top_papers(long_weak + short_strong, 1, "sum") == ["long"]
    assert top_papers(long_weak + short_strong, 1, "max") == ["short"]


def test_mean_punishes_one_excellent_chunk_among_many_poor_ones():
    """What a long paper answering a narrow question looks like."""
    hits = [hit("long", 0, 0.95)] + [hit("long", i, 0.05) for i in range(1, 20)]
    hits += [hit("short", 0, 0.4), hit("short", 1, 0.4)]
    assert top_papers(hits, 1, "mean") == ["short"]
    assert top_papers(hits, 1, "max") == ["long"]


# -- determinism and shape ------------------------------------------------------------


def test_ties_break_on_arxiv_id():
    """A coarse pass returning different papers for the same query between runs would make
    every downstream number unreproducible. Same reason RRF breaks ties on id."""
    hits = [hit("zzz", 0, 0.5), hit("aaa", 0, 0.5)]
    assert score_papers(hits, "max") == [("aaa", 0.5), ("zzz", 0.5)]


def test_every_paper_with_a_hit_appears():
    hits = [hit("aaa", 0, 0.1), hit("bbb", 0, 0.2), hit("ccc", 0, 0.3)]
    assert {a for a, _ in score_papers(hits)} == {"aaa", "bbb", "ccc"}


def test_no_hits_gives_no_papers():
    assert score_papers([]) == []
    assert top_papers([], 3) == []


def test_top_papers_does_not_pad_when_fewer_papers_exist():
    """Padding to n would widen the fine pass for no reason."""
    assert top_papers([hit("aaa", 0, 0.9)], 5) == ["aaa"]


def test_top_papers_rejects_a_nonsense_n():
    with pytest.raises(ValueError):
        top_papers(MIXED, 0)


def test_scores_can_be_negative():
    """Cosine similarity lives in [-1, 1] and RRF scores are positive; a retriever that
    returns raw cosine can hand us negatives, and `max` must still mean best."""
    hits = [hit("aaa", 0, -0.9), hit("aaa", 1, -0.1), hit("bbb", 0, -0.5)]
    assert score_papers(hits, "max") == [("aaa", -0.1), ("bbb", -0.5)]
