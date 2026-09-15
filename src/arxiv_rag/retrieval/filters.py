"""Structured constraints on what a retriever is allowed to return.

Similarity is not a substitute for a constraint. "In the PlaNet paper, how is reward
modelled?" gives dense retrieval no way to *exclude* Dreamer chunks - only to hope PlaNet
ones score higher. Measured here: q030 asks about V-JEPA and gets V-JEPA 2, because the
corpus holds more V-JEPA 2 material and no retriever could be told "only 2404.08471".

**Pre-filter, never post-filter.** Retrieving the top 5 and then dropping non-matching
results leaves fewer than 5, and can never surface a matching chunk that ranked 40th.
Restricting the candidate set *before* scoring means k still means k. Every retriever here
resolves a filter to allowed indices first, then ranks within them.

**A filter is a hard constraint, and that cuts both ways.** A wrong one does not degrade
recall, it destroys it: filter to the wrong paper and the gold chunk is unreachable at any
k. So filters belong where something reliable determines them - an explicit API parameter,
or a Stage 4 agent that has *decided* a question is about one paper - not guessed from
query text by string matching.
"""

from dataclasses import dataclass

from arxiv_rag.ingestion.models import Chunk


@dataclass(frozen=True)
class ChunkFilter:
    """Which chunks a retriever may consider.

    Frozen so a filter can be cached, compared, and safely shared between the retrievers
    inside a ``HybridRetriever`` without one of them mutating it.

    ``None`` means "no constraint on this field", which is deliberately different from an
    empty collection: ``arxiv_ids=None`` allows every paper, ``arxiv_ids=frozenset()``
    allows none. Conflating them is how a filter silently becomes a no-op.
    """

    arxiv_ids: frozenset[str] | None = None
    sections: frozenset[str] | None = None
    exclude_arxiv_ids: frozenset[str] = frozenset()

    def matches(self, chunk: Chunk) -> bool:
        """Is this chunk allowed?

        Rules, in order:

        - ``exclude_arxiv_ids`` wins over everything. An explicit exclusion is a stronger
          statement than an inclusion, and a chunk named in both should be excluded.
        - ``arxiv_ids`` - if set, the chunk's ``arxiv_id`` must be in it.
        - ``sections`` - if set, the chunk's ``section`` must be in it, compared
          **case-insensitively**: section headings come from PDF extraction and arrive as
          "Method", "METHOD" and "method" across papers.
        - ``None`` on a field means no constraint; an empty frozenset means nothing passes.
        """
        if chunk.arxiv_id in self.exclude_arxiv_ids:
            return False
        # `is not None` rather than truthiness: an empty frozenset means "nothing passes",
        # and `if self.arxiv_ids:` would read that as "no constraint" - a filter that
        # silently stops filtering.
        if self.arxiv_ids is not None and chunk.arxiv_id not in self.arxiv_ids:
            return False
        if self.sections is not None:
            wanted = {section.lower() for section in self.sections}
            if chunk.section.lower() not in wanted:
                return False
        return True

    @property
    def is_noop(self) -> bool:
        """True when this filter constrains nothing."""
        return self.arxiv_ids is None and self.sections is None and not self.exclude_arxiv_ids

    def describe(self) -> str:
        """One line for logs and eval rows."""
        if self.is_noop:
            return "no filter"
        parts = []
        if self.arxiv_ids is not None:
            parts.append(f"papers={sorted(self.arxiv_ids)}")
        if self.sections is not None:
            parts.append(f"sections={sorted(self.sections)}")
        if self.exclude_arxiv_ids:
            parts.append(f"not={sorted(self.exclude_arxiv_ids)}")
        return " ".join(parts)


def allowed_indices(chunks: list[Chunk], chunk_filter: ChunkFilter | None) -> set[int] | None:
    """Positions in ``chunks`` the filter permits, or ``None`` for "everything".

    ``None`` rather than "all indices" is the point: it lets each retriever
    skip the masking work entirely on the common unfiltered path, instead of building and
    intersecting an 874-element set on every query that does not need one.
    """
    if chunk_filter is None or chunk_filter.is_noop:
        return None
    return {i for i, chunk in enumerate(chunks) if chunk_filter.matches(chunk)}
