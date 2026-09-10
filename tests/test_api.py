"""Stage 2 - the HTTP layer. No API key, no index on disk, no network.

``answer_question`` is monkeypatched and the store is injected through
``dependency_overrides``. That is the point of routing both through seams: the endpoint
can be tested for what it actually does - translate HTTP to a library call and back -
without any of the machinery underneath it.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from arxiv_rag.api import main as api
from arxiv_rag.api.main import app, get_store
from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.answer import Answer
from arxiv_rag.retrieval.store import ChunkStore

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


@pytest.fixture
def client(monkeypatch):
    store = ChunkStore(CHUNKS, np.array([[1.0, 0.0], [0.0, 1.0]]))
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as c:
        c.app.state.chunks_by_id = {ch.chunk_id: ch for ch in CHUNKS}
        yield c
    app.dependency_overrides.clear()


def fake_answer(question, store, settings, k=None):
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
    def refusing(question, store, settings, k=None):
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
        c.app.state.store = None
        response = c.post("/ask", json={"question": "a real question"})
    assert response.status_code == 503
