"""Stage 2 - grounding and citation. Pure functions only; no API key needed."""

from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.answer import (
    REFUSAL_TOKEN,
    SYSTEM_PROMPT,
    extract_citations,
    format_context,
    is_refusal,
    refusal_token_misplaced,
    resolve_citations,
)
from arxiv_rag.retrieval.store import SearchHit


def make_hit(arxiv_id: str, index: int, section: str, text: str, score: float) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            chunk_id=f"{arxiv_id}::{index}",
            arxiv_id=arxiv_id,
            title="World Models",
            section=section,
            text=text,
            index=index,
            token_count=10,
        ),
        score=score,
    )


# -- format_context ------------------------------------------------------------------


def test_context_contains_every_arxiv_id():
    """The model can only cite an id it can see."""
    hits = [
        make_hit("1803.10122", 4, "Method", "The MDN-RNN outputs a mixture.", 0.8),
        make_hit("1811.04551", 7, "Model", "The RSSM splits the state.", 0.7),
    ]
    context = format_context(hits)
    assert "1803.10122" in context
    assert "1811.04551" in context


def test_context_contains_the_text_and_the_section():
    hits = [make_hit("1803.10122", 4, "Method", "The MDN-RNN outputs a mixture.", 0.8)]
    context = format_context(hits)
    assert "The MDN-RNN outputs a mixture." in context
    assert "Method" in context


def test_context_numbers_passages_in_rank_order():
    hits = [
        make_hit("1803.10122", 4, "Method", "first", 0.8),
        make_hit("1811.04551", 7, "Model", "second", 0.7),
    ]
    context = format_context(hits)
    assert context.index("[P1]") < context.index("first")
    assert context.index("first") < context.index("[P2]")


def test_passage_markers_are_prefixed_so_they_cannot_be_confused():
    """[P1], not [1].

    The chunk text is full of the papers' own reference markers ([6], [15]), so a bare
    bracketed integer is ambiguous. Measured: with [1] numbering the model cited passage
    numbers on every answer.
    """
    hits = [make_hit("1803.10122", 4, "Method", "text with no markers", 0.8)]
    context = format_context(hits)
    assert "[P1]" in context


def test_passage_numbers_never_use_square_brackets():
    """Brackets are reserved for citations.

    The chunk text is full of the papers' own reference markers ([6], [15]), so any
    bracketed integer we add teaches the model that brackets hold small numbers - and it
    then cites passage numbers instead of arXiv ids. Measured: it did, on every answer.
    """
    hits = [make_hit("1803.10122", 4, "Method", "text with no markers", 0.8)]
    context = format_context(hits)
    assert "[1]" not in context


def test_empty_hits_give_empty_context():
    assert format_context([]) == ""


# -- extract_citations ---------------------------------------------------------------


def test_extracts_ids_in_order():
    text = "Dreamer [1912.01603] extends PlaNet [1811.04551]."
    assert extract_citations(text) == ["1912.01603", "1811.04551"]


def test_deduplicates_but_keeps_first_appearance_order():
    text = "PlaNet [1811.04551] plans; Dreamer [1912.01603] learns; PlaNet [1811.04551] does not."
    assert extract_citations(text) == ["1811.04551", "1912.01603"]


def test_ignores_passage_numbers_and_reference_markers():
    """[1] is our own passage numbering; [12, 15] is a reference marker from a paper."""
    text = "As passage [1] shows, and unlike [12, 15], the model [1803.10122] predicts."
    assert extract_citations(text) == ["1803.10122"]


def test_handles_five_digit_ids():
    assert extract_citations("see [2402.15391] and [1803.10122]") == [
        "2402.15391",
        "1803.10122",
    ]


def test_no_citations_gives_empty_list():
    assert extract_citations("The paper does not say.") == []


# -- is_refusal ----------------------------------------------------------------------


def test_refusal_is_detected():
    assert is_refusal(f"{REFUSAL_TOKEN} the passages do not give a learning rate.")


def test_token_mid_answer_is_not_a_refusal():
    text = f"The model would emit {REFUSAL_TOKEN} here, but the answer is 15 steps."
    assert not is_refusal(text)


def test_normal_answer_is_not_a_refusal():
    assert not is_refusal("The RNN outputs a mixture of Gaussians [1803.10122].")


# -- resolve_citations ---------------------------------------------------------------


def two_hits() -> list[SearchHit]:
    return [
        make_hit("2107.08241", 17, "Model-Based RL", "survey text", 0.9),
        make_hit("1811.04551", 4, "Method", "planet text", 0.8),
    ]


def test_markers_become_arxiv_ids():
    text = "MPC planning [P1], not a policy network [P2]."
    assert resolve_citations(text, two_hits()) == (
        "MPC planning [2107.08241], not a policy network [1811.04551]."
    )


def test_repeated_marker_resolves_every_time():
    text = "PlaNet plans [P2] and replans each step [P2]."
    assert resolve_citations(text, two_hits()) == (
        "PlaNet plans [1811.04551] and replans each step [1811.04551]."
    )


def test_out_of_range_marker_is_dropped():
    """[P9] when five were retrieved refers to nothing, so it cannot stay."""
    text = "A claim [P9] and a real one [P1]."
    assert resolve_citations(text, two_hits()) == "A claim  and a real one [2107.08241]."


def test_reference_markers_from_the_papers_are_left_alone():
    """[6] is the paper's own bibliography marker, not ours. Noise, but not ours to cut."""
    text = "The V-JEPA [6] architecture [P1] uses a ViT-Base [15]."
    assert resolve_citations(text, two_hits()) == (
        "The V-JEPA [6] architecture [2107.08241] uses a ViT-Base [15]."
    )


def test_text_without_markers_is_unchanged():
    text = f"{REFUSAL_TOKEN} the passages do not say."
    assert resolve_citations(text, two_hits()) == text


def test_resolved_text_feeds_extract_citations():
    """The two functions compose: resolve, then extract. No id can be fabricated."""
    resolved = resolve_citations("First [P2], then [P1], then [P2] again.", two_hits())
    assert extract_citations(resolved) == ["1811.04551", "2107.08241"]


# -- the refusal contract -------------------------------------------------------------
#
# Measured, and the reason the generator prompt now names the trigger condition rather
# than relying on the model to recognise a refusal by feel:
#
#   q027  "What imagination horizon did the original Dreamer use?" - the value is in an
#         un-ingested appendix. The model wrote a CORRECT prose refusal with no token at
#         all, was scored `refused=False`, and dragged refusal_rate to 0.833. The whole
#         justification for Stage 4's verify node was this artefact.
#   q005  ended "Thus, the answer is INSUFFICIENT_CONTEXT." - token present, wrong place.
#   q024  cited a paper, answered substantively, THEN appended the token. Not a refusal.


def test_a_bare_token_is_a_refusal():
    assert is_refusal(REFUSAL_TOKEN)


def test_the_token_with_an_explanation_after_it_is_a_refusal():
    """The contracted shape: token first, colon, one sentence."""
    assert is_refusal(f"{REFUSAL_TOKEN}: the passages never give a value for H.")


def test_leading_whitespace_does_not_hide_a_refusal():
    assert is_refusal(f"   {REFUSAL_TOKEN}: nothing here.")


def test_an_answer_that_mentions_the_token_at_the_end_is_not_a_refusal():
    """q024's shape. It cited a paper and answered; the trailing token does not undo that."""
    text = f"IRIS uses a discrete autoencoder [P1]. {REFUSAL_TOKEN}: the objective is unclear."
    assert not is_refusal(text)


def test_a_misplaced_token_is_counted():
    assert refusal_token_misplaced(f"The passages do not say. Thus, the answer is {REFUSAL_TOKEN}.")


def test_a_correctly_placed_token_is_not_counted():
    assert not refusal_token_misplaced(f"{REFUSAL_TOKEN}: no passage gives a value.")


def test_an_ordinary_answer_is_neither():
    text = "Dreamer uses a value model to extend beyond the imagination horizon [P1]."
    assert not is_refusal(text) and not refusal_token_misplaced(text)


def test_the_refusal_rule_stays_a_single_bullet():
    """MEASURED GOODHART, kept as a regression test. Do not add a second refusal rule.

    `refusal_rate` sat at 0.833 because q027 - "What imagination horizon did the original
    Dreamer use?", whose value lives in an un-ingested appendix - wrote a CORRECT prose
    refusal with no token. Not a hallucination: a formatting quirk on one question. Two
    attempts to close that 0.167:

    | prompt                                  | refusal | false_refusal | answer | accuracy |
    |-----------------------------------------|---------|---------------|--------|----------|
    | committed (one bullet)                  |  0.833  |     0.059     | 0.941  |  0.925   |
    | + "on-topic is not being an answer"     |  1.000  |     0.559     | 0.441  |  0.525   |
    | + "decline only for the MAIN thing"     |  1.000  |     0.382     | 0.618  |  0.675   |

    Both hit the target and destroyed the system - 19 and 13 of 34 answerable questions
    refused, many with the gold chunk retrieved. The SECOND attempt was written to *reduce*
    refusing and still over-refused, which suggests the damage is salience rather than
    content: a prompt with two bullets about declining produces more declining than a
    prompt with one, whatever the second bullet says. Untested - one more run would have
    been one more prompt tuned against 40 questions.

    And note what the answer-quality metrics did while this happened: faithfulness and
    correctness both ROSE, because the judged set shrank from 32 to 15 and only the easy
    questions survived. `accuracy` - the 2x2 - is the only number that caught it.
    """
    bullets = [b for b in SYSTEM_PROMPT.split("\n- ") if REFUSAL_TOKEN in b]
    assert len(bullets) == 1, f"{len(bullets)} refusal bullets; see this test's docstring"
