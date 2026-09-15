"""What the agent can do besides retrieve.

Up to here the agent has exactly one capability - search the index - and a graph with a
loop is not much of an agent. Tools are where "decides what to do" starts meaning
something, and they are also where an agent quietly acquires powers nobody measured.

**The scoping decision that matters most is in `fetch_arxiv_metadata`.** This corpus is 49
papers, and six eval questions are labelled unanswerable *relative to that corpus* -
q004 asks about TD-MPC2, which is not indexed. A tool that fetches a paper's abstract from
arXiv and hands it to the generator would make those six answerable, silently invalidating
`refusal_rate`, the six gold labels, and every refusal number in `docs/results.md`.

So the tool identifies papers; it does not feed answers. It exists to turn "I cannot answer
that" into "TD-MPC2 is not in this index", which is a materially better refusal and
changes no metric. Extending the corpus is an ingestion decision, not something an agent
should do mid-request.

The other two tools expose capabilities that already exist and are currently reachable
from nothing but `scripts/eval.py`: Stage 3's metadata filtering, and the plain fact of
which papers are indexed.
"""

import logging
from dataclasses import dataclass

from arxiv_rag.agent.aliases import aliases_for
from arxiv_rag.ingestion.arxiv_client import fetch_by_ids, normalise_arxiv_id
from arxiv_rag.retrieval.filters import ChunkFilter
from arxiv_rag.retrieval.hybrid import Retriever
from arxiv_rag.retrieval.store import ChunkStore, SearchHit

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedPaper:
    """One paper in the corpus, as the agent sees it."""

    arxiv_id: str
    title: str
    chunk_count: int
    # The names people actually use. Empty for most papers; see `agent.aliases` for why
    # this is hand-written rather than derived.
    aliases: tuple[str, ...] = ()


def list_indexed_papers(store: ChunkStore) -> list[IndexedPaper]:
    """Every paper in the index, by arXiv id. Deterministic; no model, no network.

    The cheapest tool here and the most useful, because it is the only way the agent can
    know what it does NOT have. Five of the six unanswerable eval questions are
    unanswerable because the paper or the value is absent, and "not in this index" is a
    better answer than "insufficient context".

    Sorted by arXiv id so the output is stable: an agent prompt containing a list that
    reorders between calls is a prompt that produces different answers for no reason.
    """
    titles: dict[str, str] = {}
    counts: dict[str, int] = {}
    for chunk in store.chunks:
        titles.setdefault(chunk.arxiv_id, chunk.title)
        counts[chunk.arxiv_id] = counts.get(chunk.arxiv_id, 0) + 1
    return [
        IndexedPaper(
            arxiv_id=aid,
            title=titles[aid],
            chunk_count=counts[aid],
            aliases=aliases_for(aid),
        )
        for aid in sorted(titles)
    ]


def indexed_ids(store: ChunkStore) -> frozenset[str]:
    """Just the ids. Given to you - used by the two tools below."""
    return frozenset(chunk.arxiv_id for chunk in store.chunks)


def search_papers(
    retriever: Retriever,
    store: ChunkStore,
    query: str,
    arxiv_ids: list[str] | None = None,
    sections: list[str] | None = None,
    k: int = 5,
) -> tuple[list[SearchHit], str]:
    """Search the index, optionally restricted to named papers or sections.

    Returns ``(hits, note)``. The note is for the agent to read: empty when the search ran
    as asked, and otherwise a sentence explaining what was ignored and why.

    Steps:

    1. Normalise every requested id with ``normalise_arxiv_id`` - a caller may write
       "2404.08471v2" or "arXiv:2404.08471", and the index keys are bare ids.
    2. Split the requested ids into those that ARE indexed and those that are not.
    3. **Decide what to do with the unknown ones.** This is the real decision in this
       function, and both obvious answers are wrong:

       - filtering on an id that is not in the index yields an empty result, so the agent
         sees "no passages" and reports that the corpus says nothing on the subject - when
         in fact the corpus was never asked;
       - dropping the unknown ids silently runs a broader search than the caller asked
         for, and the agent has no way to know its constraint was ignored.

       So: filter on the ids that ARE indexed, and say in ``note`` which were dropped. If
       NONE of the requested ids are indexed, return no hits and a note saying so - an
       empty result with an explanation, rather than an unconstrained search.
    4. Build a ``ChunkFilter`` and call ``retriever.search(query, k=k, chunk_filter=...)``.
       Remember that ``arxiv_ids=None`` means "no constraint" while ``frozenset()`` means
       "allow nothing" - conflating them is how a filter silently becomes a no-op.
    """
    requested = [normalise_arxiv_id(raw) for raw in arxiv_ids] if arxiv_ids else []
    known = indexed_ids(store)
    wanted = [aid for aid in requested if aid in known]
    unknown = [aid for aid in requested if aid not in known]

    note = ""
    if unknown and not wanted:
        # Every requested paper is absent. Returning no hits WITH a reason beats both
        # alternatives: an unconstrained search answers a question nobody asked, and a
        # bare empty result reads as "the corpus says nothing about this".
        return [], (
            f"None of the requested papers are in this index: {', '.join(unknown)}. "
            "No search was run."
        )
    if unknown:
        note = f"Searched only the indexed papers; not in this index: {', '.join(unknown)}."

    # `None` for "no constraint", never an empty frozenset, which would allow nothing.
    chunk_filter = ChunkFilter(
        arxiv_ids=frozenset(wanted) if wanted else None,
        sections=frozenset(sections) if sections else None,
    )
    # A no-op filter is passed as None rather than as an object that allows everything.
    # The two behave identically for the caller and not for the retriever: `allowed_indices`
    # returns None for "no constraint" and skips the per-chunk scan entirely.
    applied = None if chunk_filter.is_noop else chunk_filter
    return retriever.search(query, k=k, chunk_filter=applied), note


def fetch_arxiv_metadata(arxiv_id: str, store: ChunkStore, delay_seconds: float = 3.0) -> str:
    """Identify a paper by arXiv id. Deliberately does NOT return text to answer from.

    Returns a one-paragraph description - title, first author, date, and whether the paper
    is in this index. The abstract is **not** included, and that is the whole design:

    - an abstract in the context is a passage the generator will answer from, and it did
      not come from the indexed corpus;
    - the eval set's six unanswerable questions are unanswerable *because* the paper is not
      indexed, so an agent that can answer from arXiv makes `refusal_rate` measure nothing.

    Reaches the network, which the rest of the agent does not, and arXiv asks for a delay
    between requests (`settings.arxiv_delay_seconds`, 3s). That is far too slow for the
    request path, so this belongs on the refusal branch - explaining why an answer is not
    available - and never on the path to producing one.

    Fails open with a plain sentence rather than raising: a metadata lookup that cannot
    reach arXiv must not take down a request that was already going to refuse.
    """
    clean = normalise_arxiv_id(arxiv_id)
    known = clean in indexed_ids(store)
    try:
        papers = fetch_by_ids([clean], delay_seconds=delay_seconds)
    except Exception as exc:  # noqa: BLE001 - any transport error, deliberately
        log.warning("arxiv metadata lookup failed for %s: %s", clean, exc)
        return (
            f"arXiv:{clean} is {'in' if known else 'not in'} this index, and arXiv could "
            "not be reached for its title."
        )
    if not papers:
        return f"arXiv:{clean} was not found on arXiv, and is not in this index."
    paper = papers[0]
    author = paper.authors[0] if paper.authors else "unknown"
    where = "indexed in this corpus" if known else "NOT indexed in this corpus"
    return f'arXiv:{clean} - "{paper.title}" by {author} et al., {paper.published}. It is {where}.'
