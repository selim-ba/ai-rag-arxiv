"""Stage 2 - refusal scoring. A 2x2 table, so every expected value is checkable by hand."""

import pytest

from arxiv_rag.evaluation.judge import (
    RefusalRecord,
    Verdict,
    failure_is_substantiated,
    quoted_spans,
    refusal_scores,
    spans_overlap,
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


def verdict(
    correct: bool,
    correct_reason: str,
    faithful_reason: str = "SUPPORTED: passage P2 says 'x'",
    faithful: bool = True,
):
    return Verdict(
        faithful=faithful,
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


# -- the faithfulness contract -------------------------------------------------------
#
# Added after measuring the two axes against a blind re-labelling: `correct`, which had
# ordered tests and mandatory prefixes, agreed 10/10 (kappa +1.00); `faithful`, which had
# a paragraph of prose and no contract, agreed 7/10 at kappa -0.15 - below chance for its
# own base rate. These tests are the same forcing function applied to the second axis.


def test_faithful_with_supported_prefix_is_consistent():
    assert verdict_is_consistent(verdict(True, "OK: fine.", "SUPPORTED: 'no explicit policy'"))


def test_faithful_with_no_claims_prefix_is_consistent():
    """A refusal asserts nothing, so it cannot assert something unsupported.

    Measured (q005): the judge marked a refusal unfaithful, and its reason was the
    REFERENCE answer's claim - not the answer's. That is the correctness axis leaking into
    the faithfulness check, and at request time no reference exists for a verify node to
    reach for.
    """
    assert verdict_is_consistent(verdict(True, "OK: fine.", "NO CLAIMS: the answer declined"))


def test_unfaithful_with_contradicted_prefix_is_consistent():
    """Measured (q031): the answer reversed I-JEPA's result - the passage says
    representations degrade in PIXEL space - and the judge quoted an earlier, genuinely
    supported sentence and stopped reading."""
    v = verdict(True, "OK: fine.", "CONTRADICTED: answer says 'representation space'", False)
    assert verdict_is_consistent(v)


def test_unfaithful_with_unsupported_prefix_is_consistent():
    v = verdict(True, "OK: fine.", "UNSUPPORTED: no passage gives a GPU-hour figure", False)
    assert verdict_is_consistent(v)


def test_faithful_without_a_prefix_is_inconsistent():
    """The old contract accepted any prose here. That permissiveness is the measured gap."""
    assert not verdict_is_consistent(verdict(True, "OK: fine.", "passage P2 says 'x'"))


def test_unfaithful_with_a_passing_prefix_is_inconsistent():
    """The mirror image: faithful=false justified by a reason that says it was supported."""
    v = verdict(True, "OK: fine.", "SUPPORTED: 'no explicit policy'", False)
    assert not verdict_is_consistent(v)


def test_faithful_with_a_failing_prefix_is_inconsistent():
    assert not verdict_is_consistent(verdict(True, "OK: fine.", "UNSUPPORTED: nothing found"))


def test_faithful_prefix_check_ignores_surrounding_whitespace():
    assert verdict_is_consistent(verdict(True, "OK: fine.", "   SUPPORTED: 'x'"))


def test_the_reference_check_still_applies_under_a_valid_prefix():
    """A correct prefix must not become a way to smuggle the reference in behind it."""
    v = verdict(True, "OK: fine.", "SUPPORTED: matches the reference answer closely")
    assert not verdict_is_consistent(v)


# -- substantiating a failure ---------------------------------------------------------
#
# The prefix contract alone made the judge WORSE: agreement fell 7/10 -> 5/10 while
# contract violations stayed at 0/31. It learned the format and kept the reasoning, which
# is this project's standing lesson about contracts constraining output rather than
# thought. These tests cover the three measured failures, all of them prefix-compliant.

ANSWER = (
    "No, PlaNet does not train an explicit policy network. Instead, it implements a "
    "policy through model-predictive control (MPC) planning, using the best sequence of "
    "future actions derived from its models [2107.08241]."
)
PASSAGES = (
    "[P1] arXiv:2107.08241 | A survey | Section: Methods\n"
    "In contrast to model-free approaches, no explicit policy or value function network "
    "is used; the policy is implemented as MPC planning with the best sequence of future "
    "actions."
)


# An answer that genuinely contradicts the passage, for the one test that needs a real
# failure to substantiate. Written after the first attempt at that test quoted the PASSAGE
# twice and called it a contradiction - the same mistake as q013, made by the person
# writing the guard against q013.
ANSWER_WRONG = (
    "Yes. PlaNet trains a policy network with an actor-critic objective and uses it "
    "directly at action time [2107.08241]."
)


def unfaithful(reason: str) -> Verdict:
    return Verdict(faithful=False, correct=True, faithful_reason=reason, correct_reason="OK: x")


def test_quoted_spans_finds_straight_and_curly_quotes():
    assert quoted_spans('says "the first thing" and \u201cthe second thing\u201d') == [
        "the first thing",
        "the second thing",
    ]


def test_a_faithful_verdict_needs_no_substantiation():
    v = Verdict(faithful=True, correct=True, faithful_reason="SUPPORTED: 'x'", correct_reason="OK:")
    assert failure_is_substantiated(v, ANSWER, PASSAGES)


def test_a_failure_with_no_quotes_at_all_is_unsubstantiated():
    """q001's actual output: a failure prefix followed by unquoted prose."""
    v = unfaithful("CONTRADICTED: The world model does not have access to the reward signal")
    assert not failure_is_substantiated(v, ANSWER, PASSAGES)


def test_a_failure_quoting_words_the_answer_never_wrote_is_unsubstantiated():
    """q005: the answer refused; the judge took the claim from the QUESTION and failed it."""
    v = unfaithful("UNSUPPORTED: 'the world model is updated during policy learning'")
    assert not failure_is_substantiated(v, ANSWER, PASSAGES)


def test_a_contradiction_the_passages_do_not_state_is_unsubstantiated():
    """q013: the 'contradiction' restated the answer, which the passage supports."""
    v = unfaithful(
        "CONTRADICTED: 'PlaNet does not train an explicit policy network' but passage says "
        "'PlaNet learns an actor network by policy gradient'"
    )
    assert not failure_is_substantiated(v, ANSWER, PASSAGES)


def test_a_real_contradiction_is_substantiated():
    """One quote from the ANSWER, one from a PASSAGE that says the opposite."""
    v = unfaithful(
        "CONTRADICTED: 'PlaNet trains a policy network with an actor-critic objective' "
        "but passage says 'no explicit policy or value function network is used'"
    )
    assert failure_is_substantiated(v, ANSWER_WRONG, PASSAGES)


def test_two_quotes_both_from_the_passages_are_unsubstantiated():
    """The mistake this suite's own author made first time: quoting the passage twice and
    calling it a contradiction. Neither span is a claim the answer made, so there is
    nothing being failed - and here both spans in fact SUPPORT the answer."""
    v = unfaithful(
        "CONTRADICTED: 'no explicit policy or value function network is used' but passage "
        "says 'the policy is implemented as MPC planning'"
    )
    assert not failure_is_substantiated(v, ANSWER, PASSAGES)


def test_unsupported_needs_only_the_answers_own_words():
    """UNSUPPORTED asserts an absence, so there is no passage span to quote - only the
    claim being failed, which must be something the answer actually wrote."""
    v = unfaithful("UNSUPPORTED: 'best sequence of future actions derived from its models'")
    assert failure_is_substantiated(v, ANSWER, PASSAGES)


def test_substantiation_survives_ligatures_and_rewrapping():
    """`quote_supported` normalises PDF text; a judge quoting 'efficiency' against a
    passage containing the 'ﬁ' ligature is quoting correctly."""
    v = unfaithful("UNSUPPORTED: 'model-predictive control (MPC)   planning'")
    assert failure_is_substantiated(v, ANSWER, PASSAGES)


# -- nothing contradicts itself -------------------------------------------------------
#
# Definitional, not tuned: "two overlapping quotes do not contradict each other" is true
# on any dataset and needs no labels. Both measured cases were FULLY substantiated - real
# quotes from the right sources - and still not contradictions.


def test_a_quote_containing_the_other_overlaps():
    """q017, verbatim. The passage span contains the answer span word for word."""
    assert spans_overlap(
        [
            "the agent does not learn without it",
            "the stochastic component is even more important - the agent does not learn without it",
        ]
    )


def test_normalisation_stops_the_same_sentence_passing_as_two():
    """Ligatures and rewrapping must not make one statement look like two."""
    assert spans_overlap(
        ["data-ef\ufb01ciency of PlaNet", "Dreamer inherits the data-efficiency of  PlaNet"]
    )


def test_genuinely_different_statements_do_not_overlap():
    assert not spans_overlap(
        ["PlaNet trains a policy network", "no explicit policy or value function network is used"]
    )


def test_a_single_span_cannot_overlap_itself():
    assert not spans_overlap(["only one quote here"])


def test_a_contradiction_whose_quotes_agree_is_unsubstantiated():
    """q013: both quotes real, both from the right sources, and they say the same thing.
    Roughly a fifth of the UNFAITHFUL list was this shape."""
    v = unfaithful(
        "CONTRADICTED: 'no explicit policy or value function network is used' but passage "
        "says 'the policy is implemented as MPC planning; no explicit policy or value "
        "function network is used'"
    )
    assert not failure_is_substantiated(v, ANSWER, PASSAGES)


def test_the_overlap_check_does_not_touch_unsupported_verdicts():
    """UNSUPPORTED asserts an absence and quotes only the answer's claim. One span, and
    nothing to compare it against."""
    v = unfaithful("UNSUPPORTED: 'best sequence of future actions derived from its models'")
    assert failure_is_substantiated(v, ANSWER, PASSAGES)
