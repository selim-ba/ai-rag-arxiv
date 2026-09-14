"""Stage 4 step 1 - a graph that changes nothing.

Fake retriever, monkeypatched generation: no index, no API key, no network. What is being
tested is the wiring - that nodes run in order, that partial updates merge correctly, and
that the reducer on `trace` appends rather than overwrites.
"""

import pytest

from arxiv_rag.agent import nodes as agent_nodes
from arxiv_rag.agent.graph import build_graph, run_agent
from arxiv_rag.agent.state import AgentState, initial_state
from arxiv_rag.config import Settings
from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.answer import Answer
from arxiv_rag.retrieval.store import SearchHit

CHUNKS = [
    Chunk(
        chunk_id=f"1811.04551::{i}",
        arxiv_id="1811.04551",
        title="Learning Latent Dynamics",
        section="Method",
        text=f"passage {i}",
        index=i,
        token_count=5,
    )
    for i in range(3)
]


class FakeRetriever:
    def __init__(self, hits=None):
        self.hits = CHUNKS if hits is None else hits
        self.queries: list[str] = []

    def search(self, query, k=5, chunk_filter=None):
        self.queries.append(query)
        return [SearchHit(c, 1.0) for c in self.hits[:k]]


@pytest.fixture
def fake_generate(monkeypatch):
    """Replace the LLM call with something deterministic that records what it saw."""
    seen = {}

    def generate_answer(question, hits, settings):
        seen["question"] = question
        seen["hits"] = hits
        return Answer(
            question=question,
            text="an answer [1811.04551]",
            citations=["1811.04551"],
            refused=False,
            retrieved_ids=[h.chunk.chunk_id for h in hits],
        )

    monkeypatch.setattr(agent_nodes, "generate_answer", generate_answer)
    return seen


# -- state and reducers ---------------------------------------------------------------


def test_initial_state_fills_every_key_a_node_might_read():
    state = initial_state("why do world models collapse?")
    assert state["question"] == "why do world models collapse?"
    assert state["query"] == state["question"]
    assert state["attempts"] == 0
    assert state["hits"] == [] and state["trace"] == []


def test_trace_accumulates_across_nodes(fake_generate):
    """The reducer: two nodes each append one line, and both survive.

    Without `Annotated[list[str], operator.add]` the second node's return would OVERWRITE
    the first's and the trace would have one entry. No error - just a shorter list.
    """
    graph = build_graph(FakeRetriever(), Settings())
    final: AgentState = graph.invoke(initial_state("a question"))
    assert len(final["trace"]) == 2
    assert "retrieve" in final["trace"][0]


# -- wiring ---------------------------------------------------------------------------


def test_nodes_run_in_order_retrieve_then_generate(fake_generate):
    graph = build_graph(FakeRetriever(), Settings())
    graph.invoke(initial_state("a question"))
    assert fake_generate["hits"], "generate ran before retrieve, or got no hits"
    assert [h.chunk.chunk_id for h in fake_generate["hits"]] == [c.chunk_id for c in CHUNKS]


def test_generate_answers_the_original_question_not_the_rewritten_query(fake_generate):
    """A rewritten query is a retrieval device. Answering it instead of what was asked is
    a subtle way to be confidently off-topic - and there will be a rewrite node soon."""
    graph = build_graph(FakeRetriever(), Settings())
    state = initial_state("what did the user actually ask?")
    state["query"] = "some rewritten search string"
    graph.invoke(state)
    assert fake_generate["question"] == "what did the user actually ask?"


def test_the_retriever_is_given_the_query_not_the_question(fake_generate):
    """The mirror image: retrieval searches `query`, which is what a rewrite changes."""
    retriever = FakeRetriever()
    graph = build_graph(retriever, Settings())
    state = initial_state("what did the user actually ask?")
    state["query"] = "some rewritten search string"
    graph.invoke(state)
    assert retriever.queries == ["some rewritten search string"]


def test_run_agent_returns_an_answer_not_a_state(fake_generate):
    """Everything upstream cares about an Answer. The graph is an implementation detail."""
    answer = run_agent(build_graph(FakeRetriever(), Settings()), "a question")
    assert isinstance(answer, Answer)
    assert answer.citations == ["1811.04551"]
    assert answer.retrieved_ids == [c.chunk_id for c in CHUNKS]


def test_empty_retrieval_still_produces_an_answer(fake_generate):
    """No hits is a normal outcome - generate_answer refuses. It is not a graph failure."""
    answer = run_agent(build_graph(FakeRetriever(hits=[]), Settings()), "a question")
    assert isinstance(answer, Answer)


def test_chunk_filter_reaches_the_retriever(fake_generate):
    from arxiv_rag.retrieval.filters import ChunkFilter

    seen = {}

    class Recording(FakeRetriever):
        def search(self, query, k=5, chunk_filter=None):
            seen["filter"] = chunk_filter
            return super().search(query, k, chunk_filter)

    f = ChunkFilter(arxiv_ids=frozenset({"1811.04551"}))
    run_agent(build_graph(Recording(), Settings()), "a question", chunk_filter=f)
    assert seen["filter"] == f


def test_retrieval_timing_survives_onto_the_answer(fake_generate):
    """The agent must report the same fields as the pipeline, or the harness drops rows.

    Measured the hard way: `retrieve_ms` defaulted to 0.0 on the agent path, and
    `if r.get("retrieve_ms")` treated that as absent, so the whole latency row disappeared
    from the eval output without any error. Fourth falsy-zero bug in this project.
    """
    answer = run_agent(build_graph(FakeRetriever(), Settings()), "a question")
    assert answer.retrieve_ms > 0.0


def test_trace_records_how_long_retrieval_took(fake_generate):
    graph = build_graph(FakeRetriever(), Settings())
    final = graph.invoke(initial_state("a question"))
    assert "ms" in final["trace"][0]
