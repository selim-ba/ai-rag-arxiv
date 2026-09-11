"""Stage 3 - reciprocal rank fusion. Small hand-built rankings, checkable by reading."""

from arxiv_rag.retrieval.fusion import reciprocal_rank_fusion


def test_agreement_wins():
    """A document both retrievers ranked first must come first."""
    assert reciprocal_rank_fusion([["a", "b"], ["a", "c"]])[0] == "a"


def test_consensus_beats_a_single_favourite():
    """The point of k.

    'b' is second for both retrievers; 'a' is first for one and absent from the other.
    With k=60 the rank-1/rank-2 gap is 1.6%, so two second places beat one first place.
    Without k, rank 1 would score 1.0 against rank 2's 0.5 and 'a' would win.
    """
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
    assert fused[0] == "b"


def test_only_ranks_matter_not_how_far_apart_they_were():
    """Neither retriever's scores are passed in at all - that is the design."""
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]


def test_every_id_survives():
    fused = reciprocal_rank_fusion([["a"], ["b"], ["c"]])
    assert sorted(fused) == ["a", "b", "c"]


def test_rankings_of_different_lengths():
    """BM25 returns fewer results than asked for when few documents match."""
    fused = reciprocal_rank_fusion([["a", "b", "c", "d"], ["b"]])
    assert fused[0] == "b"


def test_empty_ranking_is_ignored_not_an_error():
    assert reciprocal_rank_fusion([["a", "b"], []]) == ["a", "b"]


def test_no_rankings_at_all():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_ties_break_deterministically():
    """Same input, same output, every run - or eval numbers wobble for no reason."""
    rankings = [["a", "b"], ["b", "a"]]
    first = reciprocal_rank_fusion(rankings)
    assert first == reciprocal_rank_fusion(rankings)
    assert sorted(first) == ["a", "b"]


def test_a_document_found_by_one_retriever_still_ranks():
    """Scoring once is not scoring zero - q030 was found only by BM25."""
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["d"]])
    assert "d" in fused


def test_k_controls_how_much_rank_one_is_worth():
    """Small k makes fusion behave like 'whoever ranked something first wins'."""
    rankings = [["a", "b"], ["c", "b"]]
    assert reciprocal_rank_fusion(rankings, k=60)[0] == "b"
    assert reciprocal_rank_fusion(rankings, k=0)[0] in ("a", "c")
