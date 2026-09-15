"""Conversation history, kept so a follow-up has something to resolve against.

**Why this is not a LangGraph checkpointer.** A checkpointer persists `AgentState` so a
run can resume or continue mid-flight, and it is the obvious thing to reach for. It does
not fit this design. Follow-ups are resolved into standalone questions *before* the graph
(`agent.followup`), so the graph sees one self-contained question per turn and has no
in-flight state worth resuming - a request lasts one to three seconds and either finishes
or is retried. What is actually needed is the transcript, and that is a dictionary.

Building the checkpointer anyway would be the verify-node mistake: a component with a
respectable name and no measured need.

**Bounded, deliberately.** An unbounded dict keyed on a client-supplied id is a memory leak
with a public endpoint in front of it. Two caps:

* ``max_turns`` per conversation - the resolver reads the last few turns and nothing else,
  so keeping a hundred costs memory to serve nobody;
* ``max_conversations`` overall, evicting least-recently-used - the cap that stops a
  client minting new ids from consuming the process.

Both are a stopgap for a single process. A second worker has a second dictionary and a
client can land on either, so conversations break under horizontal scaling. That is a real
limitation, written down rather than discovered in Stage 6.
"""

import logging
from collections import OrderedDict
from uuid import uuid4

from arxiv_rag.agent.followup import Turn

log = logging.getLogger(__name__)


def new_conversation_id() -> str:
    """A fresh id. Server-generated, never taken from the client.

    A client that picks its own id can read another client's history by guessing, and can
    fill the store with ids of its choosing. Accepting an unknown id as a *new* empty
    conversation is safe; minting ids for the client is not.
    """
    return uuid4().hex


class SessionStore:
    """Conversation id -> the turns of that conversation. In-memory, bounded, LRU."""

    def __init__(self, max_turns: int = 10, max_conversations: int = 500) -> None:
        if max_turns < 1 or max_conversations < 1:
            raise ValueError("caps must be positive")
        self.max_turns = max_turns
        self.max_conversations = max_conversations
        self._conversations: OrderedDict[str, list[Turn]] = OrderedDict()

    def history(self, conversation_id: str | None) -> list[Turn]:
        """Turns so far, oldest first. An unknown id is an empty conversation, not an error.

        Returning a copy, because a caller that mutates what it is handed would edit the
        store through a side door.
        """
        if conversation_id is None:
            return []
        turns = self._conversations.get(conversation_id)
        if turns is None:
            return []
        self._conversations.move_to_end(conversation_id)
        return list(turns)

    def append(self, conversation_id: str, turn: Turn) -> None:
        """Record a completed turn, evicting as needed."""
        turns = self._conversations.setdefault(conversation_id, [])
        turns.append(turn)
        del turns[: -self.max_turns]  # keep the newest `max_turns`
        self._conversations.move_to_end(conversation_id)
        while len(self._conversations) > self.max_conversations:
            evicted, _ = self._conversations.popitem(last=False)
            log.info("evicted conversation %s (cap %d)", evicted, self.max_conversations)

    def __len__(self) -> int:
        return len(self._conversations)
