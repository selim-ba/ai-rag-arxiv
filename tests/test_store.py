"""Stage 2 — vector search. Hand-made 2-D vectors, so the right answer is obvious."""

import numpy as np
import pytest

from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.store import ChunkStore


def make_chunk(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"paper::{index}",
        arxiv_id="1234.5678",
        title="A paper",
        section="Method",
        text=f"chunk {index}",
        index=index,
        token_count=10,
    )


def build_store() -> ChunkStore:
    # Ordered by closeness to [1, 0]: identical, near, orthogonal, opposite.
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    return ChunkStore([make_chunk(i) for i in range(4)], vectors)


def test_returns_hits_best_first():
    hits = build_store().search([1.0, 0.0], k=3)
    assert [h.chunk.chunk_id for h in hits] == ["paper::0", "paper::1", "paper::2"]


def test_scores_descend():
    scores = [h.score for h in build_store().search([1.0, 0.0], k=4)]
    assert scores == sorted(scores, reverse=True)


def test_identical_vector_scores_one():
    top = build_store().search([1.0, 0.0], k=1)[0]
    assert top.score == pytest.approx(1.0)


def test_opposite_vector_scores_minus_one():
    hits = build_store().search([1.0, 0.0], k=4)
    assert hits[-1].score == pytest.approx(-1.0)


def test_query_length_does_not_matter():
    """Cosine compares direction, not magnitude — a longer query vector ranks the same."""
    a = [h.chunk.chunk_id for h in build_store().search([1.0, 0.0], k=4)]
    b = [h.chunk.chunk_id for h in build_store().search([50.0, 0.0], k=4)]
    assert a == b


def test_k_larger_than_store_returns_everything():
    assert len(build_store().search([1.0, 0.0], k=99)) == 4


def test_k_zero_returns_nothing():
    assert build_store().search([1.0, 0.0], k=0) == []


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        ChunkStore([make_chunk(0)], np.array([[1.0, 0.0], [0.0, 1.0]]))
