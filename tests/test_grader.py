"""Stage 4 - the grader's evidence check. Pure function; no model, no network.

Written after measuring: the grader passed 10 of 12 retrievals that could not produce a
correct answer (every one of those questions also scored `correct=False` in the full eval
run). It was not lying - it was pattern-matching on topic. The fix is to require a quote
and check it, which is the fifth place in this project where a model's own report is
verified against the source rather than trusted.
"""

from arxiv_rag.agent.grader import Grade, verify_evidence

CONTEXT = (
    "[P1] arXiv:1811.04551 | Learning Latent Dynamics | Section: Method\n"
    "In contrast to model-free approaches, no explicit policy or value function network "
    "is used; the policy is implemented as MPC planning.\n\n"
    "[P2] arXiv:1912.01603 | Dream to Control | Section: Results\n"
    "Dreamer inherits the data-efﬁciency of PlaNet while exceeding the asymptotic "
    "performance of the best model-free agents."
)


def test_a_real_quote_keeps_the_verdict():
    grade = Grade(relevant=True, evidence="no explicit policy or value function network is used")
    assert verify_evidence(grade, CONTEXT).relevant


def test_ligatures_and_wrapping_do_not_break_a_real_quote():
    """PDF text carries 'ﬁ'; a model quoting it silently normalises."""
    grade = Grade(relevant=True, evidence="Dreamer inherits the data-efficiency of PlaNet")
    assert verify_evidence(grade, CONTEXT).relevant


def test_an_invented_quote_flips_the_verdict():
    """The measured failure mode: fluent, on-topic, and nowhere in the passages."""
    grade = Grade(relevant=True, evidence="PlaNet trains a policy network with actor-critic")
    result = verify_evidence(grade, CONTEXT)
    assert not result.relevant
    assert "not in the passages" in result.missing


def test_claiming_relevant_without_quoting_anything_flips_it():
    """An assertion with no pointer is not evidence."""
    result = verify_evidence(Grade(relevant=True, evidence=""), CONTEXT)
    assert not result.relevant
    assert "quoted nothing" in result.missing


def test_a_quote_reassembled_from_the_passages_vocabulary_is_rejected():
    """Every word is present; the sentence never was. Word order is the tell."""
    grade = Grade(relevant=True, evidence="the policy network is used as a value function")
    assert not verify_evidence(grade, CONTEXT).relevant


def test_an_irrelevant_verdict_is_never_touched():
    """Only a *claim* of relevance needs substantiating. Downgrades pass through."""
    grade = Grade(relevant=False, missing="no learning rate anywhere")
    result = verify_evidence(grade, CONTEXT)
    assert not result.relevant
    assert result.missing == "no learning rate anywhere"


def test_the_original_quote_is_kept_for_debugging():
    """When a verdict is flipped, what the grader claimed is worth being able to read."""
    grade = Grade(relevant=True, evidence="something nobody wrote down at all here")
    assert verify_evidence(grade, CONTEXT).evidence == "something nobody wrote down at all here"


def test_naming_something_missing_contradicts_a_relevant_verdict():
    """Measured: relevant=true alongside "the specific loss function is not stated".

    If a correct answer needs something the passages lack, the passages are not
    sufficient. That is the definition, not a judgement call.
    """
    grade = Grade(
        relevant=True,
        evidence="no explicit policy or value function network is used",
        missing="the specific loss function is not stated",
    )
    result = verify_evidence(grade, CONTEXT)
    assert not result.relevant
    assert result.missing == "the specific loss function is not stated"


def test_whitespace_only_missing_is_not_a_contradiction():
    """A model padding the field with a space must not flip an otherwise good verdict."""
    grade = Grade(
        relevant=True,
        evidence="no explicit policy or value function network is used",
        missing="   ",
    )
    assert verify_evidence(grade, CONTEXT).relevant
