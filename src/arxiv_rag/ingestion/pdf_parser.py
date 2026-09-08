"""PDF download and text extraction.

Fair warning: this is the ugliest part of any RAG pipeline. Academic PDFs are typeset
for print, not for parsing. You will see two-column text interleaved into nonsense,
ligatures turned into mojibake, page numbers glued onto sentences, and equations
reduced to rubble.

You are not aiming for perfection. You are aiming to *look at the output*, notice the
specific ways it is broken, and clean up the ones that would hurt retrieval.
"""

import logging
import re
from collections import Counter
from pathlib import Path

import httpx
from pypdf import PdfReader

from arxiv_rag.ingestion.chunker import heading_of

log = logging.getLogger(__name__)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ARXIV_STAMP_RE = re.compile(r"^\s*arXiv:\d{4}\.\d{4,5}v\d+\s+\[[\w.\-]+\].*$")
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")


def download_pdf(pdf_url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Download a PDF to ``dest`` unless it is already there. Given to you."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with httpx.stream("GET", pdf_url, timeout=timeout, follow_redirects=True) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for block in r.iter_bytes():
                fh.write(block)
    return dest


def extract_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF, page by page.

    TODO(you): implement this.

    Hints:

    - ``reader = PdfReader(pdf_path)``, then ``page.extract_text()`` for each
      ``reader.pages``.
    - Join pages with ``"\\n\\n"``.
    - Some pages return ``None`` rather than a string. Handle it.
    - Wrap the whole thing so one corrupt PDF does not kill an ingestion run of fifty
      papers — log it and return ``""``.

    When it runs, open the output of a real paper and read it. Find three things that
    are wrong. Those three things are what ``clean_text`` is for.
    """
    try:
        reader = PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text is not None:
                text.append(page_text)
        return "\n\n".join(text)
    except Exception as e:
        log.warning("Failed to extract text from %s: %s", pdf_path.name, e)
        return ""


def _strip_boilerplate(lines: list[str]) -> list[str]:
    """Drop page numbers, the arXiv margin stamp, and repeated running headers."""
    counts = Counter(ln.strip() for ln in lines if ln.strip())
    kept = []
    for ln in lines:
        s = ln.strip()
        if _PAGE_NUM_RE.match(ln) or _ARXIV_STAMP_RE.match(ln):
            continue
        if s and len(s) < 60 and counts[s] >= 4 and not s.endswith((".", ":", ";", ",")):
            continue  # short line repeated on many pages == running header
        kept.append(ln)
    return kept


def clean_text(raw: str) -> str:
    """Normalise text extracted from a PDF into readable paragraphs."""
    if not raw:
        return ""

    # Must come first: NUL and friends would otherwise collide with the sentinel below.
    text = _CONTROL_RE.sub("", raw)

    lines = _strip_boilerplate(text.split("\n"))

    # Section headings sit on their own line with no blank line around them. Promote
    # them to their own paragraph, or the line-wrap collapse below swallows them into
    # the surrounding prose and split_into_sections finds nothing.
    spaced: list[str] = []
    for ln in lines:
        if heading_of(ln):
            spaced.extend(["", ln, ""])
        else:
            spaced.append(ln)
    text = "\n".join(spaced)

    text = re.sub(r"-\n(?=\w)", "", text)  # de-hyphenate across line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\x00", text)  # mark real paragraph breaks
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)  # line wraps become spaces
    text = text.replace("\x00", "\n\n")  # restore paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _unused_import_guard() -> None:  # pragma: no cover
    """Keeps the linter quiet about an import you need once extract_text exists."""
    _ = PdfReader
