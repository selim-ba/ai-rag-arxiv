"""Stage 2 - grounding and citation. Pure functions only; no API key needed."""

from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.answer import (
    REFUSAL_TOKEN,
    extract_citations,
    format_context,
    is_refusal,
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
