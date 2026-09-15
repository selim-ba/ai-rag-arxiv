"""Graph nodes. Each takes the state and returns only the keys it changed.

Every node here wraps something that already exists and is already measured: the retrieve
node calls the Stage 3 ``Retriever``, the generate node calls Stage 2's ``generate_answer``.
No retrieval logic and no prompting lives in this file, and that is the point - the agent
reorganises proven components rather than replacing them.
"""

import logging
from time import perf_counter

from arxiv_rag.agent.grader import Grader
from arxiv_rag.agent.rewrite import rewrite_query
from arxiv_rag.agent.state import AgentState
from arxiv_rag.config import Settings
from arxiv_rag.retrieval.answer import generate_answer
from arxiv_rag.retrieval.hybrid import Retriever

log = logging.getLogger(__name__)


def make_retrieve_node(retriever: Retriever, settings: Settings):
    """Build the retrieve node. Given to you - read it as the worked example.

    Returns a *closure* over the retriever rather than a bare function, because LangGraph
    calls nodes with the state and nothing else. Dependencies have to be bound at graph
    construction time, which is also what keeps them swappable: dense, hybrid or
    hybrid-plus-reranking all arrive the same way.
    """

    def retrieve(state: AgentState) -> dict:
        query = state.get("query") or state["question"]
        started = perf_counter()
        # `is None`, not `or`: k=0 is invalid upstream, but reaching for `or` here is how
        # `k or settings.top_k` became this project's first falsy-default bug.
        requested = state.get("k")
        k = settings.top_k if requested is None else requested
        hits = retriever.search(query, k=k, chunk_filter=state.get("chunk_filter"))
        elapsed_ms = (perf_counter() - started) * 1000
        # A partial dict, not the whole state. `trace` has an `operator.add` reducer, so
        # this single-element list is APPENDED to whatever is already there.
        update = {
            "hits": hits,
            # Overwrite rather than accumulate: on a retry this is the latest retrieval's
            # cost, not the sum. Total agent time is measured by the caller.
            "retrieve_ms": elapsed_ms,
            "trace": [f"retrieve({query[:40]!r}) -> {len(hits)} hits in {elapsed_ms:.1f}ms"],
        }
        # Snapshot the first lap, once. On later laps the key is simply absent from the
        # returned dict, and LangGraph leaves the existing value alone - which is why this
        # needs no guard beyond the counter. `attempts` is still 0 here on lap one: the
        # rewrite node is what increments it, and it has not run yet.
        if state.get("attempts", 0) == 0:
            update["first_hits"] = hits
        return update

    return retrieve


def make_generate_node(settings: Settings):
    """Build the generate node."""

    def generate(state: AgentState) -> dict:
        answer = generate_answer(state["question"], state["hits"], settings)
        status = "refused" if answer.refused else f"{len(answer.citations)} citations"
        return {
            "answer": answer,
            "trace": [f"generate -> {status}"],
        }

    return generate


def make_grade_node(grader: Grader):
    """Build the grade node. Given to you.

    Takes a *Grader instance* rather than settings, for the same reason the retrieve node
    takes a retriever: the test suite has to be able to substitute a grader that returns a
    fixed verdict, and a node that constructs its own dependency cannot be substituted.

    The node itself is trivial - the thinking is in `grader.py` and in the edge that reads
    the verdict. Note what it does NOT do: it does not decide anything. Deciding is the
    edge's job, and keeping the two apart is what makes the loop guard testable without a
    model in the loop.
    """

    def grade(state: AgentState) -> dict:
        verdict = grader.grade(state["question"], state["hits"])
        detail = "relevant" if verdict.relevant else f"missing={verdict.missing[:60]!r}"
        return {
            "grade": verdict,
            "trace": [f"grade -> {detail}"],
        }

    return grade


def make_rewrite_node(settings: Settings):
    """Build the rewrite node. Given to you.

    **This node owns the counter.** `attempts` is incremented here and nowhere else,
    because this is the node whose existence causes another lap: every increment
    corresponds to exactly one extra retrieval. Incrementing in the grade node instead
    would count passes rather than retries and the cap would be off by one; incrementing
    in the edge is impossible, since an edge returns a route, not a state update.

    `attempts` has no reducer, so returning it here overwrites. That is right - it is a
    counter, not a log.
    """

    def rewrite(state: AgentState) -> dict:
        grade = state.get("grade")
        missing = grade.missing if grade is not None else ""
        new_query = rewrite_query(state["question"], missing, settings)
        return {
            "query": new_query,
            "attempts": state.get("attempts", 0) + 1,
            "trace": [f"rewrite -> {new_query[:60]!r}"],
        }

    return rewrite
