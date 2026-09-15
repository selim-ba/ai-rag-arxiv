"""Scoring whole papers from chunk hits, so retrieval can narrow before it ranks.

**Measured motivation.** An oracle filter - each question restricted to the paper its gold
chunk lives in - takes hit@5 from 0.676 to 0.824 and recall@5 from 0.578 to 0.725, losing
nothing. That is larger than every improvement in this project combined, and the agent's
`filtered` route cannot reach it: not one of the five questions it rescues names a paper,
and four of them need two or three papers. The user does not know which paper holds the
answer. The retriever has to work it out.

The coarse pass needs **no new index**. Retrieving deep already returns chunks from many
papers; grouping those hits by paper and scoring each one is an aggregation over results
already in hand. `ChunkFilter` then does the fine pass, exactly as the oracle measurement
did.

**The aggregation is a real choice, not a detail.** Papers here hold between 6 and 43
chunks, a sevenfold spread:

* ``sum`` and ``count`` reward a paper for being long - a 43-chunk paper accumulates score
  from mediocre chunks that a 6-chunk paper cannot;
* ``mean`` punishes a paper with one excellent chunk among thirty irrelevant ones, which
  is what a long paper answering a narrow question looks like;
* ``max`` ignores length entirely and asks only how good the single best chunk is, which
  makes it blind to a paper that is broadly relevant without being sharply so.

Each is defensible and they disagree, so `scripts/retrieval_eval.py --paper-recall` sweeps
them rather than assuming.
"""

from collections import defaultdict
from typing import Literal

from arxiv_rag.retrieval.store import SearchHit

Aggregate = Literal["max", "sum", "mean", "count"]


def score_papers(hits: list[SearchHit], aggregate: Aggregate = "max") -> list[tuple[str, float]]:
    """Score each paper represented in ``hits``, best first. Pure.

    ``hits`` is a deep retrieval - typically 100 chunks spanning perhaps fifteen papers.
    The returned list is every paper that owns at least one hit, with its aggregated score.

    Ties break on arXiv id so the ordering is deterministic. A coarse pass that returns
    different papers for the same query on different runs would make every number
    downstream unreproducible, which is the same reason `reciprocal_rank_fusion` breaks
    ties on id.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for hit in hits:
        grouped[hit.chunk.arxiv_id].append(hit.score)

    scored: list[tuple[str, float]] = []
    for arxiv_id, scores in grouped.items():
        if aggregate == "max":
            value = max(scores)
        elif aggregate == "sum":
            value = sum(scores)
        elif aggregate == "mean":
            value = sum(scores) / len(scores)
        elif aggregate == "count":
            value = float(len(scores))
        else:  # pragma: no cover - Literal makes this unreachable from typed callers
            raise ValueError(f"unknown aggregate: {aggregate}")
        scored.append((arxiv_id, value))

    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))


def top_papers(hits: list[SearchHit], n: int, aggregate: Aggregate = "max") -> list[str]:
    """The ``n`` best papers by aggregated score.

    Returns fewer than ``n`` when the hits span fewer papers - which is normal, and must
    not be padded. Padding with arbitrary papers would widen the fine pass for no reason.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return [arxiv_id for arxiv_id, _ in score_papers(hits, aggregate)[:n]]
