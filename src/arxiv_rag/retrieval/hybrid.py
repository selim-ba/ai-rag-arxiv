"""One search interface, several retrievers behind it.

``ChunkStore.search`` takes a **vector**; BM25 needs the **query text**. So the shared
shape cannot be `ChunkStore.search` itself - it has to sit one level up::

    search(query: str, k: int) -> list[SearchHit]

``DenseRetriever`` adapts the vector store to that shape by embedding the query first.
``HybridRetriever`` holds any number of retrievers, runs them all, and fuses their
rankings. Everything above retrieval - `answer_question`, `/ask`, the eval harness, and
Stage 4's agent - depends only on this shape, so swapping dense for hybrid for
hybrid-plus-reranking is a constructor change and nothing else.

**Why over-retrieve before fusing.** Fusing each retriever's top 5 would throw away the
chunk BM25 had at rank 8 - and measured on this corpus, BM25 is the only retriever that
finds q030 and q034 at all. Each retriever goes ``depth`` deep (30 by default) and fusion
picks the final ``k`` from that larger pool. Deeper is not free: more noise enters the
pool, and a reranker downstream pays per candidate. ``depth`` is a knob with its own
results row, not a constant to guess once.
"""

import logging
from typing import Protocol

from arxiv_rag.config import Settings
from arxiv_rag.retrieval.bm25 import BM25Index
from arxiv_rag.retrieval.embeddings import embed_query
from arxiv_rag.retrieval.filters import ChunkFilter, allowed_indices
from arxiv_rag.retrieval.fusion import DEFAULT_K, fuse_hits
from arxiv_rag.retrieval.store import ChunkStore, SearchHit

log = logging.getLogger(__name__)

DEFAULT_DEPTH = 30


class Retriever(Protocol):
    """Anything that turns a question into ranked chunks.

    A ``Protocol`` rather than a base class: ``DenseRetriever`` and ``BM25Index`` are
    unrelated types that happen to share a shape, and nothing is gained by forcing them
    into an inheritance tree. Structural typing says "if it has this method, it fits",
    which is exactly the claim being made.
    """

    def search(
        self, query: str, k: int = 5, chunk_filter: ChunkFilter | None = None
    ) -> list[SearchHit]: ...


class DenseRetriever:
    """Embed the query, then search the vector store.

    ``embed`` is injectable so tests can run without an API key, and so Stage 4 can pass
    a rewritten query through the same path.
    """

    def __init__(self, store: ChunkStore, settings: Settings, embed=embed_query) -> None:
        self.store = store
        self.settings = settings
        self.embed = embed

    def __len__(self) -> int:
        return len(self.store)

    def search(
        self, query: str, k: int = 5, chunk_filter: ChunkFilter | None = None
    ) -> list[SearchHit]:
        allowed = allowed_indices(self.store.chunks, chunk_filter)
        return self.store.search(self.embed(query, self.settings), k=k, allowed=allowed)


class HybridRetriever:
    """Several retrievers, fused by reciprocal rank."""

    def __init__(
        self,
        retrievers: list[Retriever],
        depth: int = DEFAULT_DEPTH,
        rrf_k: int = DEFAULT_K,
        weights: list[float] | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError(f"{len(weights)} weights for {len(retrievers)} retrievers")
        self.retrievers = retrievers
        self.depth = depth
        self.rrf_k = rrf_k
        self.weights = weights

    def search(
        self, query: str, k: int = 5, chunk_filter: ChunkFilter | None = None
    ) -> list[SearchHit]:
        """Run every retriever ``depth`` deep, fuse the rankings, return the top ``k``.

        - **Retrieve ``depth``, return ``k``.** Fusing the top 5 of each defeats the point;
          the whole value is in the candidates ranked 6-30 that one retriever found and the
          other did not.
        - **A retriever returning nothing is normal, not an error.** BM25 returns ``[]``
          when no indexed term appears in the query. Fusion already ignores empty rankings.
        """
        hit_lists = [
            retriever.search(query, self.depth, chunk_filter) for retriever in self.retrievers
        ]
        return fuse_hits(hit_lists, k=self.rrf_k, top_k=k, weights=self.weights)


def build_hybrid(store: ChunkStore, settings: Settings, **overrides) -> HybridRetriever:
    """Dense + BM25 over the same chunks, configured from ``settings``.

    BM25 is built from ``store.chunks``, so both retrievers see exactly the same corpus
    and the comparison is of methods rather than of what got indexed.

    **Every caller goes through here.** The API, three eval scripts and the agent each used
    to spell out `depth=settings.fusion_depth` themselves; two tuning knobs later that is
    three places to forget. `overrides` exists for the grid sweep, which is the one caller
    that legitimately wants to ignore the configured values.
    """
    kwargs: dict = {
        "depth": settings.fusion_depth,
        "rrf_k": settings.rrf_k,
        "weights": settings.fusion_weight_list,
    }
    kwargs.update(overrides)
    return HybridRetriever(
        [DenseRetriever(store, settings), BM25Index(store.chunks)],
        **kwargs,
    )
