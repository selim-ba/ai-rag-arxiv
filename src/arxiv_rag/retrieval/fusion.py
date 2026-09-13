"""Reciprocal rank fusion: combining rankings that have no common scale.

Measured on this corpus, dense and BM25 hit 12 of the same questions, dense wins 9 that
BM25 misses, BM25 wins 4 that dense misses. Fusing them can reach at most 0.735 hit@5
against dense's 0.618. This module is how you try to collect that.

**Fuse ranks, not scores.** BM25 scores are unbounded and depend on corpus statistics -
a score of 12.4 means nothing on its own. Cosine similarities live in [-1, 1] and cluster
tightly, typically 0.3 to 0.7 on real text. Adding them is meaningless, and normalising
them (min-max, z-score) requires distribution assumptions that quietly break on the next
corpus or the next query. Rank is the one thing both retrievers agree on the meaning of:
"this was my best guess, this my second."

    RRF(d) = sum over retrievers of  1 / (k + rank_i(d))        k = 60 by convention

**What k does.** Without it, rank 1 scores 1.0 and rank 2 scores 0.5 - one retriever's
favourite would beat anything the other produced. k = 60 flattens that: rank 1 scores
1/61, rank 2 scores 1/62, a 1.6% difference. So a document *both* retrievers placed
respectably beats a document one retriever loved and the other never returned. That is
precisely the behaviour you want from a consensus method, and it is why RRF needs no
tuning to work on a new corpus.

**Where 60 comes from.** Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms
Condorcet and Individual Rank Learning Methods" (SIGIR 2009). They tuned it on TREC data,
found performance flat across a wide range, and picked 60. It is an empirical default that
proved insensitive, not a derived constant - and RRF needing no tuning is most of why it
is used.

**Alternatives considered.**

- *Linear score combination (CombSUM)*: normalise each retriever's scores and add. Needs
  comparable scales, and min-max normalising makes a chunk's score depend on which other
  chunks happened to be retrieved alongside it - the same chunk scores differently per
  query.
- *CombMNZ*: CombSUM multiplied by how many lists found the document. Same normalisation
  problem, plus an explicit agreement bonus that RRF gets for free.
- *Borda count*: rank-based and scale-free like RRF, but linear - the gap between rank 1
  and 2 counts the same as between 200 and 201. In retrieval the top of the list is where
  everything happens, which is what ``1 / (k + rank)`` encodes and Borda does not.
- *Condorcet fusion*: pairwise voting, O(n^2), and beaten by RRF in the 2009 paper.
- *Learning to rank*: needs labelled training data. With 34 questions that fits the eval
  set, not the task.

**What RRF gives up.** Score magnitude, entirely. A retriever that was certain and one
that barely cleared its threshold contribute identically at the same rank. And it rewards
*agreement*, which is not the same as relevance: measured on q030, the top fused result is
a V-JEPA **2** chunk both retrievers liked, while the correct V-JEPA chunk was found by
BM25 alone. Closing that gap is what a cross-encoder reranker is for.
"""

from collections import defaultdict

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = DEFAULT_K, weights: list[float] | None = None
) -> list[str]:
    """Fuse several ranked id lists into one, best first.

    ``rankings`` is a list of ranked lists - one per retriever, each best first, each
    possibly a different length. BM25 in particular returns fewer results than asked for
    when few documents share a term with the query, so never assume equal lengths.

    Every id that appears anywhere comes back, ranked by summed reciprocal rank. An id
    only one retriever found still scores; it just scores once.

    Ties must break **deterministically**, or the same query returns different orderings
    across runs and your eval numbers wobble for no reason. Break them by the id's best
    (lowest) rank across the input rankings, and then by the id itself.

    Return ``[]`` for no rankings, and ignore empty ones rather than treating them as an
    error - a retriever that found nothing is a normal outcome, not a failure.

    ``weights`` scales each retriever's contribution, one per ranking. Plain RRF (all
    weights equal) treats every retriever as equally trustworthy, which is a real
    assumption rather than a neutral one: here dense hits 0.618 and BM25 0.471. Whether
    weighting helps is an empirical question and belongs in its own results row, not in
    a default.
    """
    scores = _rrf_scores(rankings, k, weights)
    if not scores:
        return []

    # Best (lowest) rank any retriever gave each id, used only to break ties.
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            if rank < best_rank.get(chunk_id, rank + 1):
                best_rank[chunk_id] = rank

    # Sort by score descending, then by best rank ascending, then by id. The last two keys
    # exist so that equal scores always order the same way: without them the result
    # depends on dict insertion order, and identical queries would return different
    # orderings across runs - eval numbers moving for reasons unrelated to retrieval.
    return sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))


def fuse_hits(
    hit_lists: list[list],
    k: int = DEFAULT_K,
    top_k: int = 5,
    weights: list[float] | None = None,
) -> list:
    """``reciprocal_rank_fusion`` over ``SearchHit`` lists, returning ``SearchHit``s.

    Given to you. The returned hits carry the **fused score**, not the original dense or
    BM25 score - those were on incomparable scales, which is the whole reason for fusing
    ranks. Do not read an RRF score as a similarity; it only means something relative to
    other RRF scores from the same fusion.
    """
    from arxiv_rag.retrieval.store import SearchHit

    by_id = {}
    for hits in hit_lists:
        for hit in hits:
            by_id.setdefault(hit.chunk.chunk_id, hit.chunk)

    rankings = [[h.chunk.chunk_id for h in hits] for hits in hit_lists]
    scores = _rrf_scores(rankings, k, weights)
    fused = reciprocal_rank_fusion(rankings, k, weights)[:top_k]
    return [SearchHit(by_id[chunk_id], scores[chunk_id]) for chunk_id in fused]


def _rrf_scores(
    rankings: list[list[str]], k: int = DEFAULT_K, weights: list[float] | None = None
) -> dict[str, float]:
    """The raw scores behind the fused ordering. Given to you — useful for debugging
    why one chunk beat another."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"{len(weights)} weights for {len(rankings)} rankings")

    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += weight / (k + rank)
    return dict(scores)
