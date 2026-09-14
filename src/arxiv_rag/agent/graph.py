"""Assembling the nodes into a runnable graph.

**Why a graph rather than two function calls.** Right now, nothing. The shape earns its
keep once there are cycles: retrieve -> grade -> rewrite -> retrieve is a loop, and loops
are exactly what a chain cannot express. Adding a node to a graph is an edge change;
adding a retry to a hand-written pipeline is a rewrite.
"""

import logging

from langgraph.graph import END, START, StateGraph

from arxiv_rag.agent.nodes import make_generate_node, make_retrieve_node
from arxiv_rag.agent.state import AgentState, initial_state
from arxiv_rag.config import Settings
from arxiv_rag.retrieval.answer import Answer
from arxiv_rag.retrieval.filters import ChunkFilter
from arxiv_rag.retrieval.hybrid import Retriever

log = logging.getLogger(__name__)


def build_graph(retriever: Retriever, settings: Settings):
    """Wire the nodes into a compiled graph.


    Nothing is passed to ``compile()`` yet. A ``checkpointer`` is what gives persistence
    across invocations - conversation memory, resume-after-crash, human-in-the-loop pauses -
    and it belongs in Stage 5 with sessions, not here.
    """
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", make_retrieve_node(retriever, settings))
    graph.add_node("generate", make_generate_node(settings))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_agent(
    graph,
    question: str,
    chunk_filter: ChunkFilter | None = None,
) -> Answer:
    """Invoke a compiled graph for one question and hand back the Answer. Given to you.

    The adapter that keeps everything upstream ignorant of the graph: ``scripts/eval.py``
    and ``/ask`` care about an ``Answer``, not about ``AgentState``. Same reasoning as the
    ``Retriever`` protocol - the agent should be swappable for the pipeline without either
    side knowing.
    """
    final: AgentState = graph.invoke(initial_state(question, chunk_filter))
    log.debug("trace: %s", final.get("trace"))
    answer = final.get("answer")
    if answer is None:  # a graph that ended without generating is a wiring bug, not an edge case
        raise RuntimeError(f"graph produced no answer; trace: {final.get('trace')}")
    # The retrieve node measured this; carry it onto the Answer so the pipeline and the
    # agent report the same fields to the same harness.
    answer.retrieve_ms = final.get("retrieve_ms", 0.0)
    return answer
