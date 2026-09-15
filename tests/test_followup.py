"""Stage 5 - the follow-up resolver's verification layer. Pure; no model, no network.

Every rule here fails toward *unchanged*, because the two failure directions are not
symmetric: an under-resolved question retrieves badly and shows it, while an over-resolved
one retrieves well, reads well, and answers something nobody asked.
"""

from arxiv_rag.agent.followup import Resolution, Turn, resolve_followup, verify_resolution
from arxiv_rag.config import Settings

FOLLOWUP = "How does it stop the target encoder from collapsing?"


def test_a_real_resolution_passes():
    resolved = "How does I-JEPA stop the target encoder from collapsing?"
    out = verify_resolution(Resolution(changed=True, resolved=resolved), FOLLOWUP)
    assert out.changed and "I-JEPA" in out.resolved


def test_unchanged_means_untouched():
    """A model that reports it changed nothing and returns different words has done
    something nobody asked for and nobody will notice."""
    out = verify_resolution(
        Resolution(changed=False, resolved="How does the model avoid collapse?"), FOLLOWUP
    )
    assert out.resolved == FOLLOWUP
    assert "claimed unchanged" in out.reason


def test_unchanged_and_identical_is_fine():
    out = verify_resolution(Resolution(changed=False, resolved=FOLLOWUP), FOLLOWUP)
    assert out.resolved == FOLLOWUP and not out.changed


def test_surrounding_whitespace_is_not_a_change():
    out = verify_resolution(Resolution(changed=False, resolved=f"  {FOLLOWUP}  "), FOLLOWUP)
    assert out.resolved == FOLLOWUP


def test_an_empty_resolution_falls_back():
    out = verify_resolution(Resolution(changed=True, resolved="   "), FOLLOWUP)
    assert out.resolved == FOLLOWUP


def test_a_resolution_that_drops_the_followups_own_terms_falls_back():
    """`preserves_key_terms` again - written in Stage 4 to stop a query rewrite
    paraphrasing 'V-JEPA' away. A resolution that loses a name the USER wrote has replaced
    their question rather than completed it."""
    followup = "And what about V-JEPA 2?"
    out = verify_resolution(
        Resolution(changed=True, resolved="What data is the video model trained on?"), followup
    )
    assert out.resolved == followup
    assert "dropped" in out.reason


def test_adding_a_referent_is_not_dropping_one():
    """The check runs one way: the follow-up's terms must survive. Adding the referent is
    the entire job."""
    followup = "And V-JEPA 2?"
    out = verify_resolution(
        Resolution(changed=True, resolved="What data is V-JEPA 2 trained on?"), followup
    )
    assert out.changed and out.resolved == "What data is V-JEPA 2 trained on?"


def test_no_history_returns_the_question_as_asked():
    """The first turn of a conversation has nothing to resolve against, and must not cost
    a model call to discover that."""
    out = resolve_followup([], FOLLOWUP, Settings())
    assert out.resolved == FOLLOWUP and not out.changed


def test_a_turn_carries_its_answer():
    """A third of the specified cases point at something only the ANSWER named."""
    turn = Turn(question="What does Sub-JEPA change?", answer="It freezes random projections.")
    assert "projections" in turn.answer
