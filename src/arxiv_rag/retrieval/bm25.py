"""BM25: lexical retrieval, and the half of hybrid search that dense vectors cannot do.

A dense embedding is an average of meaning over 1536 dimensions. A token appearing in one
chunk out of 874 barely moves that average, which is how the V-JEPA paper ended up at
**rank 170** for a question naming V-JEPA. BM25 is built on the opposite instinct: the
rarer the token, the louder it is.

The score of a chunk ``d`` for a query is a sum over the query's terms::

                          f(t,d) * (k1 + 1)
    sum over t:  IDF(t) * ---------------------------------------
                          f(t,d) + k1 * (1 - b + b * |d| / avgdl)

    IDF(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))

Three ideas, each of which is what a careful librarian would do:

- **IDF - rare terms matter more.** ``N`` is the number of chunks, ``n_t`` how many
  contain the term. A term in 1 chunk of 874 scores hugely; one in 700 scores near zero.
  This is the entire reason BM25 is in this project.
- **k1 - repetition saturates.** Ten mentions of "planning" is not ten times better than
  one. As ``f(t,d)`` grows the fraction approaches ``k1 + 1`` and stops rewarding more.
  ``k1 = 1.5`` is the usual default.
- **b - long chunks are discounted.** A long chunk matches more terms by luck alone, so
  its score is divided down by its length relative to the corpus average. ``b = 0`` ignores
  length entirely, ``b = 1`` normalises fully; ``0.75`` is the usual compromise.

Only ``chunk.text`` is indexed, because that is exactly what the dense index embedded
(`scripts/index.py`). Adding the title would probably help queries that name a paper - and
would also make the dense-versus-sparse comparison a comparison of *fields* rather than of
*methods*. Ablate it later, on its own row.

``BM25Index.search`` deliberately has the same signature and return type as
``ChunkStore.search``, so the two are interchangeable and fusion has a uniform thing to
consume.
"""

import re
from collections import Counter, defaultdict
from math import log

from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.store import SearchHit

# Letters and digits, and hyphens *inside* a word. The hyphen rule is the whole ballgame:
# splitting on it turns "V-JEPA" into "v" and "jepa", which makes it indistinguishable
# from I-JEPA on the half that carries meaning and adds a junk token that appears
# everywhere. BM25 earns its place by weighting rare tokens, so a tokeniser that shatters
# rare tokens throws away the reason it is here. Same for MDN-RNN and cross-entropy.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, then pull out word tokens, keeping hyphenated compounds whole.

    ``"V-JEPA predicts (masked) features."`` gives
    ``["v-jepa", "predicts", "masked", "features"]``.

    No stopword list. IDF already drives "the" and "of" to near-zero weight, and a
    hand-written stopword list is one more thing to get wrong on a technical corpus - "a"
    is a stopword, "A*" is an algorithm.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """A sparse index over the same chunks the dense store holds."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        tokens = [tokenize(c.text) for c in chunks]
        self.freqs: list[Counter] = [Counter(t) for t in tokens]
        self.doc_len: list[int] = [len(t) for t in tokens]
        self.avgdl: float = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

        # postings: term -> the documents containing it, so scoring a query touches only
        # the handful of chunks that share a term with it instead of all 874.
        self.postings: dict[str, list[int]] = defaultdict(list)
        for doc_id, counter in enumerate(self.freqs):
            for term in counter:
                self.postings[term].append(doc_id)

        n_docs = len(chunks)
        self.idf: dict[str, float] = {
            term: log(1 + (n_docs - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self.postings.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 5, chunk_filter: object | None = None) -> list[SearchHit]:
        """Return the ``k`` best-scoring chunks, best first.

        Steps:

        1. ``tokenize`` the query.
        2. For each term that exists in ``self.idf``, walk ``self.postings[term]`` and add
           that term's contribution to a running score per document. A term the corpus has
           never seen contributes nothing - skip it rather than treating it as an error.
        3. Sort by score descending, take ``k``, wrap each in a ``SearchHit``.

        Documents scoring nothing are simply absent, so this can return fewer than ``k``
        results - unlike the dense store, which always has 874 things to rank. Callers
        must not assume a full list.

        Note the loop order: outer over query terms, inner over that term's postings. The
        other way round - every document, then every term - is the same arithmetic over
        874 chunks instead of the few dozen that share a word with the query.
        """
        from arxiv_rag.retrieval.filters import allowed_indices

        allowed = allowed_indices(self.chunks, chunk_filter)
        scores: dict[int, float] = defaultdict(float)

        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue  # the corpus has never seen it: no evidence, not an error
            for doc_id in self.postings[term]:
                if allowed is not None and doc_id not in allowed:
                    continue
                frequency = self.freqs[doc_id][term]
                length_penalty = 1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * length_penalty
                scores[doc_id] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [SearchHit(self.chunks[doc_id], float(score)) for doc_id, score in ranked]
