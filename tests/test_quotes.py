"""Verifying that a quoted span really is in the text it was attributed to.

Written after the gold-label auditor produced three fabricated quotes - fluent, on topic,
and absent from the chunk they named. The real cases below are taken from that run.
"""

from arxiv_rag.evaluation.quotes import normalise, quote_supported

SURVEY = (
    "PlaNet uses a Recurrent state space model (RSSM) that consists of a transition "
    "model, an observation model, a variational encoder and a reward model.\n\n"
    "Based on these models a Model-predictive control agent is used to adapt its plan, "
    "replanning each step.\n\nIn contrast to model-free approaches, no explicit policy "
    "or value function network is used; the policy is implemented as MPC planning."
)


def test_exact_quote_is_supported():
    assert quote_supported("no explicit policy or value function network is used", SURVEY)


def test_fabricated_quote_is_rejected():
    """The real q015 failure: fluent, plausible, and nowhere in the chunk."""
    invented = "the observation model provides a rich training signal but is not used for planning"
    assert not quote_supported(invented, SURVEY)


def test_quote_built_from_the_chunks_own_vocabulary_is_still_rejected():
    """Every word appears in the source; the sentence does not. Word order is the tell."""
    remix = "the observation model uses a policy network for planning each step"
    assert not quote_supported(remix, SURVEY)


def test_ligatures_and_line_breaks_do_not_break_a_real_quote():
    """PDF text has 'ﬁ' and hard wraps; a model quoting it silently normalises both."""
    source = (
        "Dreamer inherits the data-efﬁciency of PlaNet while\nexceeding the asymptotic performance"
    )
    assert quote_supported("Dreamer inherits the data-efficiency of PlaNet while exceeding", source)


def test_case_and_punctuation_are_ignored():
    assert quote_supported("NO EXPLICIT POLICY, OR VALUE FUNCTION NETWORK IS USED!", SURVEY)


def test_a_dropped_word_still_counts_as_supported():
    assert quote_supported("a Model-predictive control agent is used to adapt plan", SURVEY)


def test_empty_quote_is_not_support():
    assert not quote_supported("", SURVEY)


def test_too_short_to_confirm_anything():
    """Three words can be a coincidence. Four consecutive words is the floor."""
    assert not quote_supported("the observation model", SURVEY)


def test_normalise_collapses_whitespace_and_ligatures():
    assert normalise("The  data-efﬁciency\nof   PlaNet!") == "the data efficiency of planet"
