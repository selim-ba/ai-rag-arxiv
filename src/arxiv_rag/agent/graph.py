"""Assembling the nodes into a runnable graph.

**Why a graph rather than two function calls.** Right now, nothing. The shape earns its
keep once there are cycles: retrieve -> grade -> rewrite -> retrieve is a loop, and loops
are exactly what a chain cannot express. Adding a node to a graph is an edge change;
adding a retry to a hand-written pipeline is a rewrite.
"""

import logging

from langgraph.graph import END, START, StateGraph

from arxiv_rag.agent.grader import Grader
from arxiv_rag.agent.nodes import (
    make_catalog_node,
    make_generate_node,
    make_grade_node,
    make_retrieve_node,
    make_rewrite_node,
    make_route_node,
)
from arxiv_rag.agent.state import AgentState, initial_state
from arxiv_rag.agent.tools import list_indexed_papers
from arxiv_rag.config import Settings
from arxiv_rag.retrieval.answer import Answer
from arxiv_rag.retrieval.filters import ChunkFilter
from arxiv_rag.retrieval.hybrid import Retriever

log = logging.getLogger(__name__)


def make_decide_after_grade(settings: Settings):
    """Build the conditional-edge function: after grading, rewrite or generate?


    Return ``"generate"`` when:

    * the grade says ``relevant`` - retrieval succeeded, there is nothing to retry; **or**
    * ``state["attempts"] >= settings.max_retries`` - the budget is spent.

    Otherwise return ``"rewrite"``.

    """

    def decide(state: AgentState) -> str:
        # The cap is read FIRST, deliberately. Written the other way round the function
        # still behaves identically - but the order on the page is the order of authority,
        # and the thing with authority here is the counter, not the model.
        attempts = state.get("attempts", 0)
        if attempts >= settings.max_retries:
            return "generate"

        grade = state.get("grade")
        # A missing verdict routes to `generate`: absent evidence of failure is not
        # evidence of failure, and this is the branch that costs nothing.
        if grade is None or grade.relevant:
            return "generate"

        return "rewrite"

    return decide


def decide_after_route(state: AgentState) -> str:
    """Conditional edge out of the router. Two destinations, three routes.

    ``filtered`` and ``retrieve`` both go to ``retrieve`` - the difference between them is
    a ``ChunkFilter`` the route node already put in state, not a different path. Only
    ``catalog`` diverges, because it is the one route that answers from the index's
    metadata rather than from its passages.

    Unknown routes fall through to ``retrieve``. A router that returns something
    unexpected should degrade to the behaviour the system had before it existed, not stop
    the request - the same failing-open rule as `route_question` and the grader.
    """
    return "catalog" if state.get("route") == "catalog" else "retrieve"


def build_graph(
    retriever: Retriever,
    settings: Settings,
    grader: Grader | None = None,
    store=None,
):
    """Wire the nodes into a compiled graph.

    The shape as of Stage 4 step 5::

        START -> route -+- catalog ---------------------------------> catalog -> END
                        |
                        +- retrieve / filtered --> retrieve -> grade -+-> generate -> END
                                                      ^               |
                                                      +--- rewrite <--+

    ``filtered`` is not a node: it is ``retrieve`` with a ``ChunkFilter`` the route node
    put in state.

    The shape as of Stage 4 step 3::

        START -> retrieve -> grade -+-- relevant, or out of retries --> generate -> END
                    ^               |
                    |               +-- not relevant -----------------> rewrite
                    +-------------------------------------------------------+

    That back-edge is the whole reason this is a graph. ``retrieve -> grade -> rewrite ->
    retrieve`` is a cycle, and a cycle is what a chain cannot express: LCEL, a hand-written
    pipeline and a list of steps all assume the work flows one way.

    ``grader`` is injected and defaults to a real one, so tests can substitute a verdict
    without a network call.

    Nothing is passed to ``compile()`` yet. A ``checkpointer`` is what gives persistence
    across invocations - conversation memory, resume-after-crash, human-in-the-loop pauses -
    and it belongs in Stage 5 with sessions, not here.
    """
    graph = StateGraph(AgentState)
    # The router needs the catalog, and the catalog node needs the store. Without a store
    # the graph is exactly what it was before Stage 4 step 5 - which keeps every existing
    # test and the pipeline comparison valid.
    routed = store is not None
    if routed:
        graph.add_node("route", make_route_node(list_indexed_papers(store), settings))
        graph.add_node("catalog", make_catalog_node(store, settings))
    graph.add_node("retrieve", make_retrieve_node(retriever, settings))
    graph.add_node("grade", make_grade_node(grader if grader is not None else Grader(settings)))
    graph.add_node("rewrite", make_rewrite_node(settings))
    graph.add_node("generate", make_generate_node(settings))

    if routed:
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route", decide_after_route, {"catalog": "catalog", "retrieve": "retrieve"}
        )
        graph.add_edge("catalog", END)
    else:
        graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    # The conditional edge. The third argument maps the function's return value to a node
    # name; returning a string that is not a key here is a runtime error, not a silent
    # fallthrough, which is the behaviour you want from a router.
    graph.add_conditional_edges(
        "grade",
        make_decide_after_grade(settings),
        {"rewrite": "rewrite", "generate": "generate"},
    )
    graph.add_edge("rewrite", "retrieve")  # the back-edge
    graph.add_edge("generate", END)
    return graph.compile()


def run_agent(
    graph,
    question: str,
    chunk_filter: ChunkFilter | None = None,
    k: int | None = None,
) -> Answer:
    """Invoke a compiled graph for one question and hand back the Answer.

    The adapter that keeps everything upstream ignorant of the graph: ``scripts/eval.py``
    and ``/ask`` care about an ``Answer``, not about ``AgentState``. Same reasoning as the
    ``Retriever`` protocol - the agent should be swappable for the pipeline without either
    side knowing.
    """
    final: AgentState = graph.invoke(initial_state(question, chunk_filter, k))
    log.debug("trace: %s", final.get("trace"))
    answer = final.get("answer")
    if answer is None:  # a graph that ended without generating is a wiring bug, not an edge case
        raise RuntimeError(f"graph produced no answer; trace: {final.get('trace')}")
    # The retrieve node measured this; carry it onto the Answer so the pipeline and the
    # agent report the same fields to the same harness.
    answer.retrieve_ms = final.get("retrieve_ms", 0.0)
    # What the loop actually did. Carried on the Answer because that is the only thing
    # `scripts/eval.py` sees, and a metric the harness cannot see does not exist.
    answer.attempts = final.get("attempts", 0)
    answer.trace = list(final.get("trace", []))
    if answer.attempts:
        answer.first_retrieved_ids = [h.chunk.chunk_id for h in final.get("first_hits", [])]
        answer.final_query = final.get("query", "")
    return answer
