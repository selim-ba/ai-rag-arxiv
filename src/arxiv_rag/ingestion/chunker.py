"""Chunking: splitting a paper into retrievable pieces.

This is the highest-leverage code in the whole project and it looks like the most
boring. Everything downstream is limited by it: if the sentence that answers a question
is split across two chunks, no reranker, no agent, and no amount of prompt engineering
will recover it.

The strategy here is **structure first, size second**:

1. Split the paper on its section headings. Sections are semantic boundaries the author
   already drew for you; a fixed-size splitter ignores them and produces chunks that
   straddle "Results" and "Limitations", which are about opposite things.
2. Only then, split any section that is still too long, with overlap so a sentence at a
   boundary survives in at least one chunk intact.

Read that twice. "Why did you chunk it that way?" is the single most common follow-up
question when someone with real RAG experience reads a RAG project.
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
# ("Introduction", "REFERENCES"). Given to you — regexes are fiddly and not the lesson.
_HEADING_RE = re.compile(
    r"^\s*(?:(?P<num>\d+(?:\.\d+)*)\.?\s+)?(?P<title>[A-Za-z][A-Za-z0-9 &:\-]{2,60})\s*$"
)


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
    """Token count as the embedding model sees it. Given to you.

    Chunk sizes are measured in tokens, never characters. A character budget means your
    chunks are a different real size for English prose, for code, and for equations.
    """
    return len(_encoding().encode(text))


def heading_of(line: str) -> str | None:
    """Return the cleaned heading title if ``line`` is a section heading, else None.

    Given to you. Note it returns the title *without* the number: "3.1 Related Work"
    comes back as "Related Work".
    """
    match = _HEADING_RE.match(line)
    if not match:
        return None
    title = " ".join(match.group("title").split())
    if match.group("num"):
        return title
    return title if title.lower() in KNOWN_HEADINGS else None


def split_into_sections(text: str) -> list[Section]:
    """Split cleaned paper text into sections on heading lines.

    TODO(you): implement this. ``tests/test_chunker.py`` is the exact spec.

    Contract:

    - Walk the text line by line. ``heading_of(line)`` tells you whether a line starts a
      new section.
    - Text appearing before the first heading is a ``Section`` with ``title=""``.
    - A section's ``body`` is everything up to the next heading, stripped, and does
      **not** include the heading line itself.
    - Sections whose body is empty after stripping are dropped entirely.

    Hint: accumulate lines into a buffer, and flush the buffer into a Section whenever
    you hit a heading (and once more at the end — forgetting the final flush is the
    classic bug here, and it silently loses your Conclusion).
    """
    raise NotImplementedError("Stage 1: implement split_into_sections")


def strip_references(text: str) -> str:
    """Drop the bibliography and everything after it.

    TODO(you): implement this.

    Why bother: a references section is a hundred citation strings, dense with title
    keywords and no actual content. They match queries beautifully and answer nothing.
    Left in, they will crowd real content out of your top-k. Removing them is one of
    the cheapest retrieval wins available.

    Contract: find the first line whose ``heading_of`` is "References" (compare
    case-insensitively) and return everything before it, stripped. If there is no such
    line, return the text unchanged.
    """
    raise NotImplementedError("Stage 1: implement strip_references")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into token-bounded chunks with overlap.

    TODO(you): implement this. ``tests/test_chunker.py`` is the exact spec.

    Contract:

    - Empty or whitespace-only text returns ``[]``.
    - Text that fits in ``chunk_size`` tokens returns exactly one chunk.
    - No chunk exceeds ``chunk_size`` tokens.
    - Consecutive chunks overlap: the end of chunk *i* reappears at the start of chunk
      *i+1* (when ``overlap > 0``).
    - No content is lost: every word of the input appears somewhere in the output.

    Two ways to do it, and the choice is worth thinking about:

    (a) Encode the whole text to token ids, slide a window of ``chunk_size`` with stride
        ``chunk_size - overlap``, decode each window. Exact sizes, trivially correct —
        but windows cut mid-sentence and even mid-word.

    (b) Split into sentences or paragraphs first, then greedily pack them into chunks
        until adding the next one would exceed the budget. Chunks end at real
        boundaries; sizes vary a bit.

    (b) retrieves better. Start with whichever you can get green, then try the other and
    compare them on the eval set in Stage 2 — that comparison is a genuinely good story
    to be able to tell.

    Guard against the infinite loop: if ``overlap >= chunk_size`` your stride is zero or
    negative and the loop never terminates. Raise ``ValueError`` instead.
    """
    raise NotImplementedError("Stage 1: implement chunk_text")


def chunk_paper(paper: Paper, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Turn one paper's cleaned text into ``Chunk`` models.

    TODO(you): implement this. It is mostly gluing together what you just wrote:

    1. ``strip_references`` the text.
    2. ``split_into_sections``.
    3. For each section, ``chunk_text`` its body.
    4. Build a ``Chunk`` per piece, with ``index`` counting continuously across the
       whole paper (not restarting per section) and
       ``chunk_id = f"{paper.arxiv_id}::{index}"``.
    5. Carry ``paper.title`` and the section title onto every chunk.

    One design decision to make deliberately: consider prefixing each chunk's embedded
    text with the paper title and section, e.g.
    ``"[LoRA: Low-Rank Adaptation] Results\\n\\n<chunk text>"``. It costs a few tokens
    and it gives an otherwise context-free chunk something to anchor on. Try it both
    ways in Stage 2 and measure. If you do it, keep the raw text on the model too, so
    the generator does not have to read the prefix back out.
    """
    raise NotImplementedError("Stage 1: implement chunk_paper")
