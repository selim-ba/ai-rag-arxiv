"""Stage 4 step 3 - the rewrite, and the check that stops it destroying retrieval.

The model call is faked throughout. What is under test is the *guard*: a rewrite is a
model output, and this project's standing rule is that a model is a good proposer and an
unreliable reporter. Sixth place that rule is enforced in code.
"""

import pytest

from arxiv_rag.agent import rewrite as rewrite_mod
from arxiv_rag.agent.rewrite import key_terms, preserves_key_terms, rewrite_query
from arxiv_rag.config import Settings


class FakeResponse:
    """Just enough of the OpenAI response shape to reach `.choices[0].message.content`."""

    def __init__(self, content):
        message = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": message})()]


@pytest.fixture
def fake_model(monkeypatch):
    """Make `rewrite_query` return whatever this fixture is told to, without a network."""
    box = {"content": '{"query": "placeholder"}', "raise": None}

    def fake_chat(settings, **kwargs):
        box["kwargs"] = kwargs
        if box["raise"] is not None:
            raise box["raise"]
        return FakeResponse(box["content"])

    # Patches `llm.chat`, the counted seam, rather than the client it wraps: every call
    # site went through `get_client` before Stage 7 and goes through `chat` now, so a
    # fixture that fakes the client would be faking a layer nothing calls any more.
    monkeypatch.setattr(rewrite_mod, "chat", fake_chat)
    return box


# -- key_terms: what must survive -------------------------------------------------------


def test_hyphenated_model_names_are_key_terms():
    assert key_terms("What data is V-JEPA trained on?") == {"v-jepa"}


def test_names_with_digits_are_key_terms():
    assert "dreamerv3" in key_terms("how does DreamerV3 handle discrete actions")


def test_acronyms_are_key_terms():
    assert "rssm" in key_terms("what is an RSSM")


def test_ordinary_words_are_not_key_terms():
    """Losing "data" or "trained" costs nothing - dense retrieval handles meaning. Only
    the rare tokens BM25 leans on matter."""
    assert key_terms("what training data was used for the model") == set()


# -- preserves_key_terms: the guard -----------------------------------------------------


def test_a_rewrite_that_keeps_the_name_passes():
    assert preserves_key_terms(
        "What data is V-JEPA trained on?",
        "V-JEPA pretraining dataset size and video sources",
    )


def test_a_fluent_paraphrase_that_drops_the_name_fails():
    """The exact failure this module exists to prevent. Measured in Stage 3: dense
    retrieval put V-JEPA's own paper at rank 170 and BM25 put it at 2, on that token alone.
    """
    assert not preserves_key_terms(
        "What data is V-JEPA trained on?",
        "what training data was used for the video joint-embedding predictive architecture",
    )


def test_case_does_not_count_as_dropping_a_term():
    assert preserves_key_terms("what is an RSSM", "rssm recurrent state space model details")


def test_adding_terms_is_fine():
    """The guard is one-directional: a rewrite may add rare tokens, it may not lose them."""
    assert preserves_key_terms("what is an RSSM", "RSSM in PlaNet and DreamerV3")


# -- rewrite_query ----------------------------------------------------------------------


QUESTION = "What data is V-JEPA trained on?"
MISSING = "the size of the pretraining dataset"


def test_a_good_rewrite_is_returned(fake_model):
    fake_model["content"] = '{"query": "V-JEPA pretraining dataset size VideoMix2M"}'
    result = rewrite_query(QUESTION, MISSING, Settings())
    assert result == "V-JEPA pretraining dataset size VideoMix2M"


def test_a_rewrite_that_drops_a_key_term_falls_back(fake_model):
    fake_model["content"] = '{"query": "video joint embedding architecture training corpus"}'
    result = rewrite_query(QUESTION, MISSING, Settings())
    assert result == f"{QUESTION} {MISSING}"


def test_the_fallback_cannot_lose_a_key_term(fake_model):
    """It contains the whole question verbatim. That is the entire argument for it."""
    fake_model["content"] = '{"query": "something else entirely"}'
    result = rewrite_query(QUESTION, MISSING, Settings())
    assert preserves_key_terms(QUESTION, result)


def test_a_failed_call_falls_back_rather_than_raising(fake_model):
    """Fails open, like the grader: a rewrite that cannot run must degrade the retry, not
    take down the request."""
    fake_model["raise"] = RuntimeError("connection reset")
    assert rewrite_query(QUESTION, MISSING, Settings()) == f"{QUESTION} {MISSING}"


def test_unparseable_json_falls_back(fake_model):
    fake_model["content"] = "here is your query: V-JEPA dataset"
    assert rewrite_query(QUESTION, MISSING, Settings()) == f"{QUESTION} {MISSING}"


def test_an_empty_query_falls_back(fake_model):
    fake_model["content"] = '{"query": "   "}'
    assert rewrite_query(QUESTION, MISSING, Settings()) == f"{QUESTION} {MISSING}"


def test_a_null_content_falls_back(fake_model):
    """`message.content` is `str | None`. Third time that has bitten this project."""
    fake_model["content"] = None
    assert rewrite_query(QUESTION, MISSING, Settings()) == f"{QUESTION} {MISSING}"


def test_the_call_is_deterministic_and_json_mode(fake_model):
    fake_model["content"] = '{"query": "V-JEPA dataset"}'
    rewrite_query(QUESTION, MISSING, Settings())
    assert fake_model["kwargs"]["temperature"] == 0
    assert fake_model["kwargs"]["response_format"] == {"type": "json_object"}


def test_the_model_is_shown_both_the_question_and_what_was_missing(fake_model):
    fake_model["content"] = '{"query": "V-JEPA dataset"}'
    rewrite_query(QUESTION, MISSING, Settings())
    sent = " ".join(m["content"] for m in fake_model["kwargs"]["messages"])
    assert QUESTION in sent and MISSING in sent
