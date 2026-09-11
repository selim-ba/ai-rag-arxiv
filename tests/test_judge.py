"""Stage 2 - refusal scoring. A 2x2 table, so every expected value is checkable by hand."""

import pytest

from arxiv_rag.evaluation.judge import (
    RefusalRecord,
    Verdict,
    refusal_scores,
    verdict_is_consistent,
)


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


# -- verdict_is_consistent -----------------------------------------------------------


def verdict(correct: bool, correct_reason: str, faithful_reason: str = "passage P2 says 'x'"):
    return Verdict(
        faithful=True,
        correct=correct,
        faithful_reason=faithful_reason,
        correct_reason=correct_reason,
    )


def test_pass_with_ok_prefix_is_consistent():
    assert verdict_is_consistent(verdict(True, "OK: names CEM and the re-fitting step."))


def test_fail_with_contradiction_prefix_is_consistent():
    v = verdict(False, "CONTRADICTION: answer says 'degrades', reference says 'does not learn'.")
    assert verdict_is_consistent(v)


def test_fail_with_wrong_system_prefix_is_consistent():
    v = verdict(False, "WRONG SYSTEM: answer describes V-JEPA 2, question asked about V-JEPA.")
    assert verdict_is_consistent(v)


def test_failing_verdict_with_no_failure_prefix_is_inconsistent():
    """The measured bug: correct=false alongside 'no contradiction found'.

    q018 and q023 did exactly this, about 7% of graded answers.
    """
    assert not verdict_is_consistent(verdict(False, "no contradiction found"))


def test_failing_verdict_with_ok_prefix_is_inconsistent():
    assert not verdict_is_consistent(verdict(False, "OK: conveys the main point."))


def test_passing_verdict_with_failure_prefix_is_inconsistent():
    """The mirror image: a pass justified by a contradiction it claims to have found."""
    assert not verdict_is_consistent(verdict(True, "CONTRADICTION: answer says x."))


def test_faithful_reason_citing_the_reference_is_inconsistent():
    """Faithfulness is judged against the passages alone. This was the v1 judge bug."""
    v = verdict(True, "OK: fine.", faithful_reason="matches the reference answer closely")
    assert not verdict_is_consistent(v)


def test_prefix_check_ignores_surrounding_whitespace():
    assert verdict_is_consistent(verdict(True, "  OK: fine."))
