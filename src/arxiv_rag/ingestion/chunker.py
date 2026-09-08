"""Chunking: splitting a paper into retrievable pieces.

Strategy is structure first, size second:

1. Split on the paper's own section headings. Sections are semantic boundaries the
   author already drew; a fixed-size splitter ignores them and produces chunks that
   straddle Results and Limitations, which are about opposite things.
2. Only then split sections that are still too long, preferring paragraph boundaries,
   falling back to sentences, and to raw token windows only for a single sentence
   longer than the whole budget.

Overlap carries the tail of each chunk into the next, so a sentence landing on a
boundary survives intact in at least one of them.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from arxiv_rag.ingestion.models import Chunk, Paper

# Headings we recognise even without a section number.
KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "preliminaries",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "analysis",
    "ablation study",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "acknowledgments",
    "acknowledgements",
    "references",
    "appendix",
}

# A heading line is either numbered ("3.1 Experimental Setup") or a bare known heading
# ("Introduction", "REFERENCES").
_HEADING_RE = re.compile(
    r"^\s*(?:(?P<num>\d+(?:\.\d+)*)\.?\s+)?(?P<title>[A-Za-z][A-Za-z0-9 &:\-]{2,60})\s*$"
)
_JOIN = "\n\n"
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Section:
    """A titled span of a paper. ``title`` is "" for text before the first heading."""

    title: str
    body: str


@lru_cache
def _encoding():
    """cl100k_base is what the OpenAI embedding models use. Cached — it is not cheap
    to construct, and the first call downloads the vocabulary file."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Token count as the embedding model sees it.

    Chunk sizes are measured in tokens, never characters. A character budget means your
    chunks are a different real size for English prose, for code, and for equations.
    """
    return len(_encoding().encode(text))


def heading_of(line: str) -> str | None:
    """Return the cleaned heading title if ``line`` is a section heading, else None"""
    match = _HEADING_RE.match(line)
    if not match:
        return None
    title = " ".join(match.group("title").split())
    if match.group("num"):
        return title
    return title if title.lower() in KNOWN_HEADINGS else None


def split_into_sections(text: str) -> list[Section]:
    """Split cleaned paper text into sections on heading lines."""
    sections: list[Section] = []
    buffer: list[str] = []
    current_title = ""

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(title=current_title, body=body))
        buffer.clear()

    for line in text.splitlines():
        heading = heading_of(line)
        if heading is not None:
            flush()
            current_title = heading
        else:
            buffer.append(line)

    flush()
    return sections


def strip_references(text: str) -> str:
    """Drop the bibliography and everything after it"""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        heading = heading_of(line)
        if heading and heading.lower() == "references":
            return "\n".join(lines[:i]).strip()
    return text.strip()


def _split_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Last-resort splitter: fixed token windows, cutting wherever they land.

    Only reached by a single sentence longer than the whole budget — an unbroken
    equation dump, usually. Every other path ends chunks on a real boundary.
    """
    enc = _encoding()
    ids = enc.encode(text)
    if len(ids) <= chunk_size:
        return [text]
    stride = chunk_size - overlap
    out = []
    for start in range(0, len(ids), stride):
        out.append(enc.decode(ids[start : start + chunk_size]))
        if start + chunk_size >= len(ids):
            break
    return out


def _split_sentences(text: str) -> list[str]:
    """Split after sentence-ending punctuation, keeping the punctuation attached."""
    return [s for s in _SENTENCE_END_RE.split(text) if s.strip()]


def _to_units(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Break text into the largest natural pieces that each fit the budget:
    paragraphs where possible, sentences where not, token windows as a last resort."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) <= chunk_size:
            units.append(para)
            continue
        for sentence in _split_sentences(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if count_tokens(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(_split_by_tokens(sentence, chunk_size, overlap))
    return units


def _overlap_seed(units: list[str], overlap: int) -> list[str]:
    """The trailing units of a finished chunk that carry over into the next one."""
    seed: list[str] = []
    total = 0
    for unit in reversed(units):
        n = count_tokens(unit)
        if total + n > overlap:
            break
        seed.insert(0, unit)
        total += n
    return seed


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into token-bounded chunks that end at natural boundaries."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")
    if not text.strip():
        return []

    units = _to_units(text, chunk_size, overlap)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        if current and count_tokens(_JOIN.join(current + [unit])) > chunk_size:
            chunks.append(_JOIN.join(current))
            seed = _overlap_seed(current, overlap)
            # A seed plus a nearly-budget-sized unit can overflow. Drop the seed
            # rather than the budget — the unit alone always fits, by construction.
            if seed and count_tokens(_JOIN.join(seed + [unit])) > chunk_size:
                seed = []
            current = seed
        current.append(unit)

    if current:
        chunks.append(_JOIN.join(current))
    return chunks


def _merge_short_sections(sections: list[Section], min_tokens: int = 60) -> list[Section]:
    """Fold undersized sections into the one before them.

    A real section is never twenty tokens long, but a spurious heading is: a numbered
    table row ("1 Warmup frozen ...") is indistinguishable from "3.1 Experimental Setup"
    by regex alone. Merging recovers that text into its neighbour instead of leaving it
    as a fragment too small to retrieve and too noisy to be useful.
    """
    merged: list[Section] = []
    for section in sections:
        if merged and count_tokens(section.body) < min_tokens:
            previous = merged[-1]
            merged[-1] = Section(title=previous.title, body=f"{previous.body}\n\n{section.body}")
        else:
            merged.append(section)
    return merged


def chunk_paper(paper: Paper, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Turn one paper's cleaned text into Chunk models."""
    body = strip_references(text)

    chunks: list[Chunk] = []
    index = 0
    for section in _merge_short_sections(split_into_sections(body)):
        for piece in chunk_text(section.body, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.arxiv_id}::{index}",
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    section=section.title,
                    text=piece,
                    index=index,
                    token_count=count_tokens(piece),
                )
            )
            index += 1
    return chunks
