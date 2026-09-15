"""The state threaded through every node of the agent graph.

One typed dict, passed to each node and merged with whatever that node returns. A node
does **not** mutate it and does **not** return the whole thing - it returns a *partial*
dict of just the keys it changed, and LangGraph merges.

**Reducers are the part that surprises people.** By default, merging means *overwrite*:
if two nodes both return ``{"hits": [...]}`` the second wins. For a key that should
accumulate, you declare a reducer with ``Annotated``::

    trace: Annotated[list[str], operator.add]

Now returning ``{"trace": ["retrieve: 5 hits"]}`` **appends** rather than replaces. Get
this wrong on a key several nodes write and updates vanish silently - no error, just a
shorter list than you expected. ``trace`` is deliberately the first thing in this project
to use one, because it makes the behaviour visible on every run.

**Loop counters live here, not in a closure.** ``attempts`` is state because the retry
edge reads it to decide whether to give up. A counter held outside the graph resets
between invocations and the cap silently stops capping - and an agent with no exit
condition will spend money all night.
"""

import operator
from typing import Annotated, TypedDict

from arxiv_rag.agent.grader import Grade
from arxiv_rag.retrieval.answer import Answer
from arxiv_rag.retrieval.filters import ChunkFilter
from arxiv_rag.retrieval.store import SearchHit


class AgentState(TypedDict, total=False):
    """What every node sees and may add to.

    ``total=False`` so nodes can return partial dicts without static type complaints;
    LangGraph merges them into the running state either way.
    """

    # -- input
    question: str
    chunk_filter: ChunkFilter | None
    # Per-request override for how many passages to retrieve. `None` means "use the
    # configured default" - NOT zero, and not falsy-checked anywhere, because this project
    # has now shipped five bugs from treating a legitimate empty value as absent.
    k: int | None

    # -- working values, overwritten by whichever node produced them last
    query: str  # what retrieval actually searched for; diverges from `question` on rewrite
    hits: list[SearchHit]
    answer: Answer | None

    # The FIRST retrieval's hits, written once and never overwritten. Kept because
    # "did the retry help?" is unanswerable from the final state alone: a rescue and a
    # retry that changed nothing look identical once `hits` has been replaced.
    first_hits: list[SearchHit]

    # The grader's verdict on the CURRENT hits. Overwritten on every pass, deliberately:
    # the edge condition wants the latest one, and a stale verdict from the previous
    # retrieval would route on evidence that no longer describes what is in `hits`.
    grade: Grade | None

    # Retrieval timing, measured by the node that does the retrieving. The pipeline
    # gets this from `answer_question`; the agent calls `generate_answer` directly, so
    # without this the number is silently 0 and the eval harness drops the whole row.
    retrieve_ms: float

    # The router's decision. In state because an edge reads it, and because a trace that
    # shows which route ran is the difference between an agent and a pipeline with extra
    # latency.
    route: str
    # The ids the router named. Kept because `verify_route` puts UNINDEXED ids here when it
    # sends a question to catalog - that is precisely the paper the answer must name.
    route_ids: list[str]

    # -- loop control. In state, deliberately: the edge condition reads it.
    attempts: int

    # -- observability. The one key with a reducer: every node appends, none overwrite.
    trace: Annotated[list[str], operator.add]


def initial_state(
    question: str,
    chunk_filter: ChunkFilter | None = None,
    k: int | None = None,
) -> AgentState:
    """A fresh state for one question.

    Every key a node might read is initialised here rather than left absent, so a node can
    do ``state["attempts"]`` without a guard. ``query`` starts as the question and only
    diverges once a rewrite node exists.
    """
    return AgentState(
        question=question,
        query=question,
        chunk_filter=chunk_filter,
        k=k,
        hits=[],
        first_hits=[],
        answer=None,
        grade=None,
        route="retrieve",
        route_ids=[],
        attempts=0,
        retrieve_ms=0.0,
        trace=[],
    )
