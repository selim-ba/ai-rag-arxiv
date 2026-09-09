"""Stage 2 — retrieval metrics. Pure functions, exact expectations."""

from arxiv_rag.evaluation.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank

RETRIEVED = ["a", "b", "c", "d", "e"]


def test_hit_at_k_true_when_gold_is_inside_k():
    assert hit_at_k(RETRIEVED, ["c"], k=3) is True


def test_hit_at_k_false_when_gold_is_past_k():
    assert hit_at_k(RETRIEVED, ["d"], k=3) is False


def test_hit_at_k_any_gold_counts():
    assert hit_at_k(RETRIEVED, ["zzz", "b"], k=2) is True


def test_hit_at_k_no_gold_retrieved():
    assert hit_at_k(RETRIEVED, ["zzz"], k=5) is False


def test_recall_at_k_counts_the_fraction_found():
    """The difference from hit@k: comparison questions need more than one passage."""
    assert recall_at_k(RETRIEVED, ["a", "b", "zzz", "yyy"], k=5) == 0.5


def test_recall_at_k_all_found():
    assert recall_at_k(RETRIEVED, ["a", "b"], k=5) == 1.0


def test_recall_at_k_respects_k():
    assert recall_at_k(RETRIEVED, ["a", "e"], k=2) == 0.5


def test_recall_at_k_with_no_gold_ids():
    assert recall_at_k(RETRIEVED, [], k=5) == 0.0


def test_reciprocal_rank_is_one_based():
    assert reciprocal_rank(RETRIEVED, ["a"]) == 1.0
    assert reciprocal_rank(RETRIEVED, ["b"]) == 0.5
    assert reciprocal_rank(RETRIEVED, ["d"]) == 0.25


def test_reciprocal_rank_uses_the_first_gold_found():
    assert reciprocal_rank(RETRIEVED, ["e", "b"]) == 0.5


def test_reciprocal_rank_zero_when_nothing_found():
    assert reciprocal_rank(RETRIEVED, ["zzz"]) == 0.0


def test_mean():
    assert mean([1.0, 0.0, 0.5]) == 0.5
    assert mean([]) == 0.0
