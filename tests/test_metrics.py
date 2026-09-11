"""Stage 2 - retrieval metrics. Pure functions; every expected value is checkable by hand.

Gold chunks are grouped: ``[[a], [b]]`` means both a and b are needed, ``[[a, b]]`` means
either one will do. The alternative case is the reason groups exist - see the module
docstring in metrics.py.
"""

import pytest

from arxiv_rag.evaluation.metrics import (
    flatten,
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a", "b", "c", "d", "e"]


# -- flatten -------------------------------------------------------------------------


def test_flatten_keeps_every_acceptable_chunk():
    assert flatten([["a"], ["b", "c"]]) == ["a", "b", "c"]


def test_flatten_of_nothing_is_empty():
    assert flatten([]) == []


# -- hit_at_k ------------------------------------------------------------------------


def test_hit_when_gold_is_first():
    assert hit_at_k(RETRIEVED, [["a"]], 5)


def test_hit_when_gold_is_last_in_range():
    assert hit_at_k(RETRIEVED, [["e"]], 5)


def test_no_hit_when_gold_is_just_outside_k():
    assert not hit_at_k(RETRIEVED, [["c"]], 2)


def test_no_hit_when_gold_was_never_retrieved():
    assert not hit_at_k(RETRIEVED, [["z"]], 5)


def test_hit_ignores_grouping():
    """One relevant chunk anywhere in the top k is a hit, whatever group it came from."""
    assert hit_at_k(RETRIEVED, [["z"], ["y", "b"]], 5)


def test_no_gold_groups_is_not_a_hit():
    assert not hit_at_k(RETRIEVED, [], 5)


# -- recall_at_k ---------------------------------------------------------------------


def test_recall_is_one_when_every_group_is_satisfied():
    assert recall_at_k(RETRIEVED, [["a"], ["b"]], 5) == 1.0


def test_recall_counts_groups_not_chunks():
    """Three complementary facts, one found."""
    assert recall_at_k(RETRIEVED, [["a"], ["y"], ["z"]], 5) == pytest.approx(1 / 3)


def test_alternatives_in_one_group_score_full_recall():
    """The whole point of grouping.

    'Does PlaNet train a policy network?' is answered by the PlaNet paper OR by the
    survey. Finding either is a complete answer, and must not be scored as half of one.
    """
    assert recall_at_k(RETRIEVED, [["a", "z"]], 5) == 1.0


def test_finding_both_alternatives_is_still_one():
    """A group is satisfied, not over-satisfied. Recall can never exceed 1.0."""
    assert recall_at_k(RETRIEVED, [["a", "b"]], 5) == 1.0


def test_recall_respects_k():
    assert recall_at_k(RETRIEVED, [["a"], ["d"]], 2) == 0.5


def test_recall_is_zero_when_nothing_matched():
    assert recall_at_k(RETRIEVED, [["y"], ["z"]], 5) == 0.0


def test_recall_with_no_groups_does_not_divide_by_zero():
    assert recall_at_k(RETRIEVED, [], 5) == 0.0


def test_mixed_alternatives_and_complements():
    """Two facts; the second has two acceptable sources, one of which was retrieved."""
    assert recall_at_k(RETRIEVED, [["a"], ["z", "c"]], 5) == 1.0
    assert recall_at_k(RETRIEVED, [["a"], ["z", "y"]], 5) == 0.5


# -- reciprocal_rank -----------------------------------------------------------------


def test_rr_is_one_for_first_place():
    assert reciprocal_rank(RETRIEVED, [["a"]]) == 1.0


def test_rr_is_a_half_for_second_place():
    assert reciprocal_rank(RETRIEVED, [["b"]]) == 0.5


def test_rr_uses_the_earliest_gold_chunk_from_any_group():
    assert reciprocal_rank(RETRIEVED, [["d"], ["b"]]) == 0.5


def test_rr_is_zero_when_nothing_matched():
    assert reciprocal_rank(RETRIEVED, [["z"]]) == 0.0


def test_rr_with_no_groups_is_zero():
    assert reciprocal_rank(RETRIEVED, []) == 0.0


# -- mean ----------------------------------------------------------------------------


def test_mean_of_values():
    assert mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)


def test_mean_of_empty_is_zero_not_an_exception():
    assert mean([]) == 0.0
