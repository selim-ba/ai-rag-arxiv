"""Stage 2 - refusal scoring. A 2x2 table, so every expected value is checkable by hand."""

import pytest

from arxiv_rag.evaluation.judge import RefusalRecord, refusal_scores


def rec(qid: str, should: bool, did: bool) -> RefusalRecord:
    return RefusalRecord(question_id=qid, should_answer=should, did_answer=did)


def test_perfect_behaviour():
    records = [rec("q1", True, True), rec("q2", False, False)]
    scores = refusal_scores(records)
    assert scores["answer_rate"] == 1.0
    assert scores["refusal_rate"] == 1.0
    assert scores["false_refusal_rate"] == 0.0
    assert scores["accuracy"] == 1.0


def test_answers_everything_hallucinates_on_the_unanswerable():
    """The failure mode the six unanswerable questions exist to catch."""
    records = [rec("q1", True, True), rec("q2", False, True), rec("q3", False, True)]
    scores = refusal_scores(records)
    assert scores["answer_rate"] == 1.0
    assert scores["refusal_rate"] == 0.0
    assert scores["false_refusal_rate"] == 0.0
    assert scores["accuracy"] == pytest.approx(1 / 3)


def test_refuses_everything():
    """Scores a perfect refusal_rate. This is why you never quote it alone."""
    records = [rec("q1", True, False), rec("q2", True, False), rec("q3", False, False)]
    scores = refusal_scores(records)
    assert scores["answer_rate"] == 0.0
    assert scores["refusal_rate"] == 1.0
    assert scores["false_refusal_rate"] == 1.0
    assert scores["accuracy"] == pytest.approx(1 / 3)


def test_mixed():
    records = [
        rec("q1", True, True),
        rec("q2", True, True),
        rec("q3", True, False),
        rec("q4", True, False),
        rec("q5", False, False),
        rec("q6", False, True),
    ]
    scores = refusal_scores(records)
    assert scores["answer_rate"] == 0.5
    assert scores["refusal_rate"] == 0.5
    assert scores["false_refusal_rate"] == 0.5
    assert scores["accuracy"] == 0.5


def test_empty_denominators_do_not_divide_by_zero():
    scores = refusal_scores([rec("q1", True, True)])
    assert scores["refusal_rate"] == 0.0

    scores = refusal_scores([])
    assert scores["accuracy"] == 0.0
