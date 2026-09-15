"""Stage 2 - the HTTP layer. No API key, no index on disk, no network.

``answer_question`` is monkeypatched and the retriever is injected through
``dependency_overrides``. That is the point of routing both through seams: the endpoint
can be tested for what it actually does - translate HTTP to a library call and back -
without an index, an API key, or a network call.
"""

import pytest
from fastapi.testclient import TestClient

from arxiv_rag.api import main as api
from arxiv_rag.api.main import app, get_graph, get_retriever
from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.answer import Answer
from arxiv_rag.retrieval.store import SearchHit

CHUNKS = [
    Chunk(
        chunk_id="1811.04551::4",
        arxiv_id="1811.04551",
        title="Learning Latent Dynamics for Planning from Pixels",
        section="Method",
        text="PlaNet plans with CEM.",
        index=4,
        token_count=10,
    ),
    Chunk(
        chunk_id="1912.01603::7",
        arxiv_id="1912.01603",
        title="Dream to Control",
        section="",
        text="Dreamer learns an actor.",
        index=7,
        token_count=10,
    ),
]


class FakeRetriever:
    """Satisfies the Retriever protocol without an index or a network call."""

    def search(self, query: str, k: int = 5, chunk_filter=None) -> list[SearchHit]:
        return [SearchHit(chunk, 1.0) for chunk in CHUNKS[:k]]


class FakeGraph:
    """Stands in for the compiled LangGraph. Never invoked on the pipeline path.

    The endpoint depends on the graph whichever path it takes, because a dependency is
    resolved before the body runs. Without this override every existing /ask test would
    get a 503 from `get_graph` - the lifespan finds no index, so `app.state.graph` is None.
    """

    def invoke(self, state):  # pragma: no cover - agent tests monkeypatch run_agent
        raise AssertionError("FakeGraph.invoke called; monkeypatch run_agent instead")


@pytest.fixture
def client(monkeypatch):
    app.dependency_overrides[get_retriever] = FakeRetriever
    app.dependency_overrides[get_graph] = FakeGraph
    with TestClient(app) as c:
        c.app.state.chunks_by_id = {ch.chunk_id: ch for ch in CHUNKS}
        yield c
    app.dependency_overrides.clear()


def fake_answer(question, retriever, settings, k=None, chunk_filter=None):
    return Answer(
        question=question,
        text="PlaNet plans with CEM [1811.04551].",
        citations=["1811.04551"],
        refused=False,
        retrieved_ids=["1811.04551::4", "1912.01603::7"],
    )


def test_health_is_up():
    with TestClient(app) as c:
        body = c.get("/health").json()
    assert body["status"] == "ok"
    assert "openai_key_configured" in body


def test_health_never_leaks_the_key():
    with TestClient(app) as c:
        body = c.get("/health").text
    assert "sk-" not in body


def test_ask_returns_the_answer_and_citations(client, monkeypatch):
    monkeypatch.setattr(api, "answer_question", fake_answer)
    body = client.post("/ask", json={"question": "How does PlaNet plan?"}).json()
    assert body["answer"] == "PlaNet plans with CEM [1811.04551]."
    assert body["citations"] == ["1811.04551"]
    assert body["refused"] is False


def test_sources_carry_titles_and_links(client, monkeypatch):
    """retrieved_ids is meaningless to a caller; a title and a url are not."""
    monkeypatch.setattr(api, "answer_question", fake_answer)
    sources = client.post("/ask", json={"question": "How does PlaNet plan?"}).json()["sources"]
    assert len(sources) == 2
    assert sources[0]["title"] == "Learning Latent Dynamics for Planning from Pixels"
    assert sources[0]["url"] == "https://arxiv.org/abs/1811.04551"
    assert sources[0]["chunk_id"] == "1811.04551::4"


def test_sources_preserve_rank_order(client, monkeypatch):
    """Rank order is information. Do not sort, do not use a set."""
    monkeypatch.setattr(api, "answer_question", fake_answer)
    sources = client.post("/ask", json={"question": "How does PlaNet plan?"}).json()["sources"]
    assert [s["arxiv_id"] for s in sources] == ["1811.04551", "1912.01603"]


def test_refusal_is_reported_not_hidden(client, monkeypatch):
    def refusing(question, retriever, settings, k=None, chunk_filter=None):
        return Answer(question=question, text="INSUFFICIENT_CONTEXT no.", refused=True)

    monkeypatch.setattr(api, "answer_question", refusing)
    body = client.post("/ask", json={"question": "What learning rate?"}).json()
    assert body["refused"] is True
    assert body["sources"] == []


def test_empty_question_is_rejected_by_validation(client):
    """422 from pydantic, before any model is ever called. Free input validation."""
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_k_out_of_range_is_rejected(client):
    assert client.post("/ask", json={"question": "a real question", "k": 99}).status_code == 422


def test_ask_returns_503_when_no_index_is_loaded():
    """A container with an unmounted volume should say so, not crash on startup."""
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        c.app.state.retriever = None
        response = c.post("/ask", json={"question": "a real question"})
    assert response.status_code == 503


def test_the_retriever_reaches_answer_question(monkeypatch):
    """The seam: dense, hybrid and hybrid+rerank all satisfy one shape.

    The endpoint is handed a Retriever, never asks what kind, and passes it straight to
    `answer_question`. That is what makes swapping the shipped configuration a constructor
    change in the lifespan handler rather than an edit here.

    `answer_question` is replaced by a stand-in that really calls the retriever, so this
    asserts the wiring rather than a hardcoded return - and still makes no network call.
    """

    class OnlySecondChunk:
        def search(self, query, k=5, chunk_filter=None):
            return [SearchHit(CHUNKS[1], 0.5)]

    def answer_using_the_retriever(question, retriever, settings, k=None, chunk_filter=None):
        hits = retriever.search(question, k or 5)
        return Answer(
            question=question,
            text="answered",
            retrieved_ids=[h.chunk.chunk_id for h in hits],
        )

    monkeypatch.setattr(api, "answer_question", answer_using_the_retriever)
    app.dependency_overrides[get_retriever] = OnlySecondChunk
    try:
        with TestClient(app) as c:
            c.app.state.chunks_by_id = {ch.chunk_id: ch for ch in CHUNKS}
            body = c.post("/ask", json={"question": "anything at all"}).json()
    finally:
        app.dependency_overrides.clear()
    assert [s["chunk_id"] for s in body["sources"]] == ["1912.01603::7"]


# -- Stage 4 step 6: /ask can route through the agent ----------------------------------
#
# Until this existed, every graph, node, cap and measurement in Stage 4 was reachable only
# from scripts/eval.py. An agent nothing can call is a library, not a service.


def fake_run_agent(graph, question, chunk_filter=None, k=None):
    return Answer(
        question=question,
        text="PlaNet plans with CEM [1811.04551].",
        citations=["1811.04551"],
        refused=False,
        retrieved_ids=["1811.04551::4"],
        retrieve_ms=3.5,
        generate_ms=900.0,
        trace=[
            "retrieve('...') -> 5 hits in 3.5ms",
            "grade -> relevant",
            "generate -> 1 citations",
        ],
    )


def test_the_pipeline_is_the_default(client, monkeypatch):
    monkeypatch.setattr(api, "answer_question", fake_answer)
    body = client.post("/ask", json={"question": "does PlaNet plan?"}).json()
    assert body["mode"] == "pipeline"
    assert body["trace"] == []


def test_the_request_can_ask_for_the_agent(client, monkeypatch):
    monkeypatch.setattr(api, "run_agent", fake_run_agent)
    body = client.post("/ask", json={"question": "does PlaNet plan?", "use_agent": True}).json()
    assert body["mode"] == "agent"
    assert body["trace"][0].startswith("retrieve")


def test_use_agent_false_overrides_a_true_setting(client, monkeypatch):
    """`is None`, not `or`. With `or`, an explicit False silently falls through to the
    setting - the same falsy-default bug this project has now shipped five times."""
    monkeypatch.setattr(api, "answer_question", fake_answer)
    settings = api.get_settings()
    monkeypatch.setattr(settings, "use_agent", True)
    body = client.post("/ask", json={"question": "does PlaNet plan?", "use_agent": False}).json()
    assert body["mode"] == "pipeline"


def test_the_setting_alone_switches_the_path(client, monkeypatch):
    monkeypatch.setattr(api, "run_agent", fake_run_agent)
    settings = api.get_settings()
    monkeypatch.setattr(settings, "use_agent", True)
    body = client.post("/ask", json={"question": "does PlaNet plan?"}).json()
    assert body["mode"] == "agent"


def test_k_reaches_the_agent(client, monkeypatch):
    """`AskRequest.k` was honoured by the pipeline and ignored by the agent, so flipping
    the flag silently changed how many passages a caller got."""
    seen = {}

    def recording(graph, question, chunk_filter=None, k=None):
        seen["k"] = k
        return fake_run_agent(graph, question, chunk_filter, k)

    monkeypatch.setattr(api, "run_agent", recording)
    client.post("/ask", json={"question": "does PlaNet plan?", "use_agent": True, "k": 3})
    assert seen["k"] == 3


def test_both_paths_return_the_same_response_shape(client, monkeypatch):
    monkeypatch.setattr(api, "answer_question", fake_answer)
    pipeline = client.post("/ask", json={"question": "does PlaNet plan?"}).json()
    monkeypatch.setattr(api, "run_agent", fake_run_agent)
    agent = client.post("/ask", json={"question": "does PlaNet plan?", "use_agent": True}).json()
    assert set(pipeline) == set(agent)


def test_timings_are_reported(client, monkeypatch):
    monkeypatch.setattr(api, "run_agent", fake_run_agent)
    body = client.post("/ask", json={"question": "does PlaNet plan?", "use_agent": True}).json()
    assert body["retrieve_ms"] == 3.5 and body["generate_ms"] == 900.0
