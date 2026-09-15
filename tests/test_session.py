"""Stage 5 - conversation history. Pure; no model, no network, no index."""

import pytest

from arxiv_rag.agent.followup import Turn
from arxiv_rag.agent.session import SessionStore, new_conversation_id


def turn(n: int) -> Turn:
    return Turn(question=f"q{n}", answer=f"a{n}")


def test_an_unknown_id_is_an_empty_conversation_not_an_error():
    """Ids are server-generated, so an unrecognised one is a stale client rather than
    something worth a 4xx."""
    assert SessionStore().history("never-seen") == []


def test_no_id_is_an_empty_conversation():
    assert SessionStore().history(None) == []


def test_turns_come_back_oldest_first():
    store = SessionStore()
    for n in range(3):
        store.append("c1", turn(n))
    assert [t.question for t in store.history("c1")] == ["q0", "q1", "q2"]


def test_conversations_are_separate():
    store = SessionStore()
    store.append("c1", turn(1))
    store.append("c2", turn(2))
    assert [t.question for t in store.history("c1")] == ["q1"]
    assert [t.question for t in store.history("c2")] == ["q2"]


def test_history_is_a_copy():
    """A caller that mutates what it is handed would edit the store through a side door."""
    store = SessionStore()
    store.append("c1", turn(1))
    store.history("c1").append(turn(99))
    assert len(store.history("c1")) == 1


def test_turns_are_capped_and_the_newest_survive():
    """The resolver reads the last few turns; keeping a hundred costs memory to serve
    nobody."""
    store = SessionStore(max_turns=3)
    for n in range(6):
        store.append("c1", turn(n))
    assert [t.question for t in store.history("c1")] == ["q3", "q4", "q5"]


def test_conversations_are_capped_least_recently_used_first():
    """An unbounded dict keyed on a client-supplied id is a memory leak with a public
    endpoint in front of it."""
    store = SessionStore(max_conversations=2)
    store.append("a", turn(1))
    store.append("b", turn(2))
    store.append("c", turn(3))
    assert len(store) == 2
    assert store.history("a") == []
    assert [t.question for t in store.history("c")] == ["q3"]


def test_reading_a_conversation_keeps_it_alive():
    """LRU on access, not just on write: a conversation someone is still reading from must
    not be evicted under them."""
    store = SessionStore(max_conversations=2)
    store.append("a", turn(1))
    store.append("b", turn(2))
    store.history("a")  # touch it
    store.append("c", turn(3))
    assert [t.question for t in store.history("a")] == ["q1"]
    assert store.history("b") == []


def test_ids_are_unique():
    assert len({new_conversation_id() for _ in range(200)}) == 200


def test_nonsense_caps_are_rejected():
    with pytest.raises(ValueError):
        SessionStore(max_turns=0)
    with pytest.raises(ValueError):
        SessionStore(max_conversations=0)
