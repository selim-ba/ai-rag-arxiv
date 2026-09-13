"""The vector store: chunks, their vectors, and search over them.

This is exact search — every query is compared against every chunk. At 874 chunks that
is a 874x1536 dot product, roughly a millisecond, and it is *more* accurate than an
approximate index because it never misses a neighbour.

Approximate nearest-neighbour indexes (HNSW, IVF) exist to trade a little accuracy for
speed at millions of vectors. Reaching for one at this size is cargo cult. Stage 6
migrates to pgvector, at which point the database earns its place by holding chunk
metadata and vectors together rather than by being faster.

Keep everything outside this file talking to ``ChunkStore.search`` and nothing else.
That is what makes the pgvector migration a swap rather than a rewrite.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arxiv_rag.ingestion.models import Chunk

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """One retrieved chunk and how well it matched."""

    chunk: Chunk
    score: float


class ChunkStore:
    """Chunks plus a matrix of their embeddings, one row per chunk."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors")
        self.chunks = chunks
        # Pre-normalise once at build time: with unit vectors, cosine similarity is just
        # a dot product, so every query afterwards is one matrix multiply.
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        k: int = 5,
        allowed: set[int] | None = None,
    ) -> list[SearchHit]:
        """Return the ``k`` chunks closest to ``query_vector``, best first.

        ``allowed`` restricts which chunk positions may be returned. It is applied to the
        score vector *before* ranking - a pre-filter - so ``k`` still means ``k``. Ranking
        first and dropping afterwards would return fewer than ``k`` results and could never
        surface a permitted chunk that ranked 40th.
        """
        if k <= 0:
            return []
        query = np.asarray(query_vector, dtype=float)
        query = query / np.linalg.norm(query)  # normalise to unit length, same as the store

        scores = self.vectors @ query  # dot product with every chunk, shape (n_chunks,)
        if allowed is not None:
            if not allowed:
                return []
            mask = np.full(scores.shape, False)
            mask[list(allowed)] = True
            # -inf rather than 0: cosine similarity is legitimately negative, so zeroing
            # would rank a forbidden chunk above a permitted but dissimilar one.
            scores = np.where(mask, scores, -np.inf)
            k = min(k, len(allowed))

        indices = np.argsort(scores)[::-1][:k]  # top k indices, best first

        return [SearchHit(self.chunks[i], float(scores[i])) for i in indices]

    # ---- persistence -----------------------------------------

    def save(self, directory: Path) -> None:
        """Write the store as a .npy matrix plus a .jsonl of chunks, same order."""
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "chunks.jsonl").open("w") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        log.info("saved index: %d chunks -> %s", len(self.chunks), directory)

    @classmethod
    def load(cls, directory: Path) -> "ChunkStore":
        vectors = np.load(directory / "vectors.npy")
        chunks = [
            Chunk.model_validate_json(line)
            for line in (directory / "chunks.jsonl").read_text().splitlines()
        ]
        return cls(chunks, vectors)


def load_chunks(chunks_dir: Path) -> list[Chunk]:
    """Read every chunk from ``data/chunks/*.jsonl``. Given to you."""
    chunks: list[Chunk] = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))
    return chunks


_ = json  # used once you extend the store
