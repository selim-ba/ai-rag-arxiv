"""Reranking: read the query and the passage together, then reorder.

Measured on this corpus: of the 9 questions neither retriever places in the top 5, **6
have a gold chunk inside one retriever's top 30**. The evidence is already being
retrieved; it is being ranked wrong. A perfect reranker over a depth-30 pool would reach
hit@5 = 0.823 against hybrid's 0.647 - several times the headroom fusion ever had.

**Bi-encoder vs cross-encoder.** Dense retrieval embeds query and passage *separately*, so
passage vectors can be precomputed for all 874 chunks - and the two never meet until a dot
product at the end, which cannot model any interaction between them. A cross-encoder
concatenates ``[CLS] query [SEP] passage [SEP]`` and runs attention over the pair, so every
query token can attend to every passage token. Far more accurate, and impossible to
precompute: the score exists only for that exact pair. So it can never be the retriever -
874 forward passes per query - and is only ever run over a shortlist.

    cheap recall (874 chunks, one matrix multiply)  ->  expensive precision (30 pairs)

**Why pointwise.** Learning-to-rank distinguishes pointwise (score each pair
independently), pairwise (which of these two is better) and listwise (order the whole
list). Pointwise is used here because n=30 means 30 forward passes rather than pairwise's
~435 comparisons, because a cross-encoder is deterministic where an LLM reranker is not -
and this project has already been bitten by measurement noise - and because pairwise and
listwise learning-to-rank want labelled preference data that 34 questions cannot supply.

A listwise LLM reranker (RankGPT-style) is often *more accurate*, at the cost of latency,
money, position bias and nondeterminism. The ``Reranker`` protocol below can express one;
it belongs in its own results row, not in the default.
"""

import json
import logging
import re
from typing import Protocol

from arxiv_rag.config import Settings
from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.hybrid import Retriever
from arxiv_rag.retrieval.store import SearchHit

log = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_DEPTH = 30


class Reranker(Protocol):
    """Scores passages against a query. Higher is more relevant.

    Scores are comparable only within one call - like RRF scores, and unlike cosine
    similarity, they carry no absolute meaning. Do not threshold on them without
    calibrating first.
    """

    def score(self, query: str, chunks: list[Chunk]) -> list[float]: ...


class CrossEncoderReranker:
    """A pointwise cross-encoder.

    The model is loaded lazily, on first use rather than at construction, because it is
    roughly 90MB and downloaded on first run - a cost that should be paid when reranking
    is actually used, not when the module is imported by something that only wants BM25.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "reranking needs sentence-transformers: pip install -e '.[rerank]'"
                ) from exc
            log.info("loading cross-encoder %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._load().predict(pairs, batch_size=self.batch_size)
        return [float(s) for s in scores]


class RerankingRetriever:
    """Over-retrieve with a cheap retriever, rescore with an expensive one, keep the best.

    Implements the same ``Retriever`` shape as ``DenseRetriever`` and ``HybridRetriever``,
    so it wraps any of them and everything downstream is unaffected.
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        self.base = base
        self.reranker = reranker
        self.depth = depth

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Retrieve ``self.depth`` candidates, rescore them all, return the best ``k``."""
        candidates = self.base.search(query, self.depth)
        if not candidates:
            return []

        scores = self.reranker.score(query, [hit.chunk for hit in candidates])

        # (score, base rank, hit). Base rank is captured here rather than looked up in
        # the sort key: .index() would be an O(n) scan per comparison, and it compares by
        # value, so two equal candidates would resolve to the same position.
        scored = [
            (score, base_rank, hit)
            for base_rank, (hit, score) in enumerate(zip(candidates, scores, strict=True))
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))

        return [SearchHit(hit.chunk, score) for score, _, hit in scored[:k]]


# -- listwise, via an LLM -------------------------------------------------------------

LISTWISE_SYSTEM = """You rank passages by how well they answer a question about \
machine-learning research papers.

You are given a QUESTION and numbered PASSAGES. Return the passage numbers ordered from \
most to least useful for answering that exact question.

Rank on whether the passage answers THE QUESTION ASKED, not on whether it is about the \
same topic. In particular, this corpus contains closely named systems - V-JEPA and \
V-JEPA 2, I-JEPA, Sub-JEPA, Var-JEPA, Dreamer and DreamerV2 and DreamerV3. A passage \
about a different version of the system is NOT an answer, however similar it reads. Put \
those below any passage about the system actually asked about.

Include only passages that contribute something. Omitting the rest is better than \
padding the list.

Reply with JSON only: {"ranking": [7, 2, 19]}"""

MAX_PASSAGE_CHARS = 2500


def order_to_scores(order: list[int], n: int) -> list[float]:
    """Turn a 1-based ranking into per-candidate scores, best first.

    ``order`` is what the model returned: passage numbers, most relevant first. ``n`` is
    how many candidates it was shown.

    Ranked candidates get descending positive scores. Everything the model left out keeps
    its **base order**, below everything it ranked - a passage the model declined to
    mention is not evidence that it is worse than every other unmentioned passage, so the
    retriever's opinion is the sensible fallback.

    Numbers outside ``1..n`` are ignored, and a repeat keeps its first position. A model
    inventing passage 47 out of 30 must not silently shift the ordering - the same rule as
    ``resolve_citations`` and the gold auditor's ``clean_id``: verify identifiers against
    the set you hold.
    """
    scores = [float(-(i + 1)) for i in range(n)]  # unranked: base order, all negative
    seen: set[int] = set()
    rank = 0
    for position in order:
        if not isinstance(position, int) or not 1 <= position <= n or position in seen:
            continue
        seen.add(position)
        scores[position - 1] = float(n - rank)
        rank += 1
    return scores


class LLMListwiseReranker:
    """Show the model every candidate at once and ask for an ordering.

    Different from a cross-encoder in one important way: it sees the candidates *together*,
    so it can act on "this one is about V-JEPA 2, that one is about V-JEPA" - a comparison
    a pointwise scorer cannot make by construction.

    The costs are real and belong in the results row: money and latency per query, and
    nondeterminism at ``temperature=0``. Listwise LLM rankers also have a documented
    position bias toward passages presented early; candidates are shown in base-retriever
    order here, so that bias reinforces the base ranking rather than scrambling it.
    """

    def __init__(self, settings: Settings, max_chars: int = MAX_PASSAGE_CHARS) -> None:
        self.settings = settings
        self.max_chars = max_chars

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        from openai import OpenAI

        passages = "\n\n".join(
            f"[{i}] {c.title} | {c.section}\n{c.text[: self.max_chars]}"
            for i, c in enumerate(chunks, start=1)
        )
        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": LISTWISE_SYSTEM},
                {"role": "user", "content": f"QUESTION:\n{query}\n\nPASSAGES:\n{passages}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            order = json.loads(raw).get("ranking", [])
        except json.JSONDecodeError:
            log.warning("listwise reranker returned unparseable JSON; keeping base order")
            order = []
        # A model sometimes answers with "[7]" strings rather than integers.
        cleaned = [int(m.group()) for x in order if (m := re.search(r"\d+", str(x)))]
        return order_to_scores(cleaned, len(chunks))
