"""Stage 4 step 6 - the agent's tools. No index on disk, no network, no model.

`fetch_arxiv_metadata` is the only one that reaches the network, and its client call is
monkeypatched here. The other two are pure functions over a fake store, which is the
point of keeping them that way.
"""

import numpy as np
import pytest

from arxiv_rag.agent import tools as tools_mod
from arxiv_rag.agent.tools import (
    fetch_arxiv_metadata,
    indexed_ids,
    list_indexed_papers,
    search_papers,
)
from arxiv_rag.ingestion.models import Chunk, Paper
from arxiv_rag.retrieval.store import ChunkStore, SearchHit

CHUNKS = [
    Chunk(
        chunk_id="1912.01603::0",
        arxiv_id="1912.01603",
        title="Dream to Control",
        section="Method",
        text="Dreamer learns an actor.",
        index=0,
        token_count=5,
    ),
    Chunk(
        chunk_id="1912.01603::1",
        arxiv_id="1912.01603",
        title="Dream to Control",
        section="Results",
        text="Dreamer exceeds D4PG.",
        index=1,
        token_count=5,
    ),
    Chunk(
        chunk_id="2404.08471::0",
        arxiv_id="2404.08471",
        title="V-JEPA",
        section="Abstract",
        text="Feature prediction beats pixels.",
        index=0,
        token_count=5,
    ),
]


@pytest.fixture
def store():
    return ChunkStore(CHUNKS, np.ones((len(CHUNKS), 4), dtype=np.float32))


class FakeRetriever:
    """Records the filter it was given, and honours it just enough to be checkable."""

    def __init__(self):
        self.calls: list[dict] = []

    def search(self, query, k=5, chunk_filter=None):
        self.calls.append({"query": query, "k": k, "filter": chunk_filter})
        allowed = [c for c in CHUNKS if chunk_filter is None or chunk_filter.matches(c)]
        return [SearchHit(c, 1.0) for c in allowed[:k]]


# -- list_indexed_papers ---------------------------------------------------------------


def test_papers_are_grouped_by_arxiv_id(store):
    papers = list_indexed_papers(store)
    assert [p.arxiv_id for p in papers] == ["1912.01603", "2404.08471"]
    assert [p.chunk_count for p in papers] == [2, 1]


def test_titles_come_through(store):
    assert list_indexed_papers(store)[0].title == "Dream to Control"


def test_the_order_is_stable(store):
    """An agent prompt containing a list that reorders between calls is a prompt that
    produces different answers for no reason."""
    assert list_indexed_papers(store) == list_indexed_papers(store)


def test_indexed_ids_is_the_id_set(store):
    assert indexed_ids(store) == {"1912.01603", "2404.08471"}


# -- search_papers ---------------------------------------------------------------------


def test_an_unfiltered_search_passes_no_constraint(store):
    """`arxiv_ids=None` means every paper. It must NOT become an empty frozenset."""
    retriever = FakeRetriever()
    hits, note = search_papers(retriever, store, "dreamer")
    f = retriever.calls[0]["filter"]
    assert f is None or f.is_noop
    assert len(hits) == 3 and note == ""


def test_a_known_id_restricts_the_search(store):
    retriever = FakeRetriever()
    hits, note = search_papers(retriever, store, "dreamer", arxiv_ids=["1912.01603"])
    assert {h.chunk.arxiv_id for h in hits} == {"1912.01603"}
    assert note == ""


def test_versioned_and_prefixed_ids_are_normalised(store):
    """A caller may write "2404.08471v2" or "arXiv:2404.08471"; index keys are bare."""
    retriever = FakeRetriever()
    hits, _ = search_papers(retriever, store, "jepa", arxiv_ids=["arXiv:2404.08471v2"])
    assert {h.chunk.arxiv_id for h in hits} == {"2404.08471"}


def test_an_unknown_id_is_dropped_and_reported(store):
    """Silently widening the search is the failure mode: the agent asked for one paper,
    got the whole corpus, and has no way to know its constraint was ignored."""
    retriever = FakeRetriever()
    hits, note = search_papers(retriever, store, "dreamer", arxiv_ids=["1912.01603", "9999.99999"])
    assert {h.chunk.arxiv_id for h in hits} == {"1912.01603"}
    assert "9999.99999" in note


def test_all_ids_unknown_returns_nothing_with_an_explanation(store):
    """The other failure mode: filtering on an absent id yields no passages, and the agent
    reports that the corpus says nothing - when the corpus was never asked."""
    retriever = FakeRetriever()
    hits, note = search_papers(retriever, store, "td-mpc2", arxiv_ids=["9999.99999"])
    assert hits == []
    assert "9999.99999" in note and note != ""


def test_sections_restrict_the_search(store):
    retriever = FakeRetriever()
    hits, _ = search_papers(retriever, store, "dreamer", sections=["Results"])
    assert [h.chunk.section for h in hits] == ["Results"]


def test_k_reaches_the_retriever(store):
    retriever = FakeRetriever()
    search_papers(retriever, store, "dreamer", k=2)
    assert retriever.calls[0]["k"] == 2


# -- fetch_arxiv_metadata --------------------------------------------------------------


def fake_paper(arxiv_id="2203.00001"):
    return Paper(
        arxiv_id=arxiv_id,
        title="TD-MPC2: Scalable Robust World Models",
        authors=["Nicklas Hansen", "Hao Su"],
        abstract="A secret abstract that must never reach the generator.",
        published="2023-10-25",
        categories=["cs.LG"],
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def test_an_unindexed_paper_is_named_as_unindexed(store, monkeypatch):
    monkeypatch.setattr(tools_mod, "fetch_by_ids", lambda ids, delay_seconds=3.0: [fake_paper()])
    out = fetch_arxiv_metadata("2203.00001", store, delay_seconds=0)
    assert "NOT indexed" in out
    assert "TD-MPC2" in out


def test_the_abstract_never_appears(store, monkeypatch):
    """The scoping decision, enforced. An abstract in the context is a passage the
    generator will answer from, and it did not come from the indexed corpus - which is
    what makes six eval questions unanswerable in the first place."""
    monkeypatch.setattr(tools_mod, "fetch_by_ids", lambda ids, delay_seconds=3.0: [fake_paper()])
    out = fetch_arxiv_metadata("2203.00001", store, delay_seconds=0)
    assert "secret abstract" not in out


def test_an_indexed_paper_is_named_as_indexed(store, monkeypatch):
    monkeypatch.setattr(
        tools_mod, "fetch_by_ids", lambda ids, delay_seconds=3.0: [fake_paper("1912.01603")]
    )
    out = fetch_arxiv_metadata("1912.01603", store, delay_seconds=0)
    assert "indexed in this corpus" in out and "NOT indexed" not in out


def test_a_network_failure_still_answers(store, monkeypatch):
    """Fails open with a sentence. A metadata lookup that cannot reach arXiv must not take
    down a request that was already going to refuse."""

    def boom(ids, delay_seconds=3.0):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(tools_mod, "fetch_by_ids", boom)
    out = fetch_arxiv_metadata("2203.00001", store, delay_seconds=0)
    assert "not in this index" in out and "could not be reached" in out


def test_an_id_arxiv_does_not_know_is_reported(store, monkeypatch):
    monkeypatch.setattr(tools_mod, "fetch_by_ids", lambda ids, delay_seconds=3.0: [])
    out = fetch_arxiv_metadata("9999.99999", store, delay_seconds=0)
    assert "not found on arXiv" in out
