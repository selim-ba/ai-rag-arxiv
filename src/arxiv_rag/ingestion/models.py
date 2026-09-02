"""Domain models. Given to you — this is the shape everything downstream expects.

Read these before writing any Stage 1 code. The ``Chunk`` model in particular is the
contract between ingestion and retrieval: whatever metadata is not on a chunk here is
metadata your retriever cannot filter on and your answers cannot cite.
"""

from datetime import date

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """One arXiv paper's metadata."""

    arxiv_id: str = Field(description="Version-stripped id, e.g. '2005.11401'")
    title: str
    authors: list[str]
    abstract: str
    published: date
    categories: list[str] = Field(default_factory=list)
    pdf_url: str

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


class Chunk(BaseModel):
    """One retrievable piece of a paper.

    ``chunk_id`` must be stable across re-ingestion of the same paper: your evaluation
    set (Stage 2) refers to chunks by id, and you do not want re-running ingestion to
    invalidate it.
    """

    chunk_id: str = Field(description="Stable id: '<arxiv_id>::<index>'")
    arxiv_id: str
    title: str
    section: str = Field(default="", description="Section heading this chunk came from")
    text: str
    index: int = Field(description="Position of this chunk within the paper")
    token_count: int
