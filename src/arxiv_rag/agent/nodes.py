"""Graph nodes. Each takes the state and returns only the keys it changed.

Every node here wraps something that already exists and is already measured: the retrieve
node calls the Stage 3 ``Retriever``, the generate node calls Stage 2's ``generate_answer``.
No retrieval logic and no prompting lives in this file, and that is the point - the agent
reorganises proven components rather than replacing them.
"""

import logging
from time import perf_counter

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
        hits = retriever.search(query, k=settings.top_k, chunk_filter=state.get("chunk_filter"))
        elapsed_ms = (perf_counter() - started) * 1000
        # A partial dict, not the whole state. `trace` has an `operator.add` reducer, so
        # this single-element list is APPENDED to whatever is already there.
        return {
            "hits": hits,
            # Overwrite rather than accumulate: on a retry this is the latest retrieval's
            # cost, not the sum. Total agent time is measured by the caller.
            "retrieve_ms": elapsed_ms,
            "trace": [f"retrieve({query[:40]!r}) -> {len(hits)} hits in {elapsed_ms:.1f}ms"],
        }

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
