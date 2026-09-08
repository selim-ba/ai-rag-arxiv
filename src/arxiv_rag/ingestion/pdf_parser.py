"""PDF download and text extraction.

Academic PDFs are typeset for print, not for parsing: equations become rubble, running
headers land mid-sentence, and words break across lines. ``clean_text`` repairs what is
repairable; the rest is documented as a known limitation.

Extraction uses PyMuPDF rather than pypdf. Measured on six papers, pypdf dropped the
spaces between words in one of them (0.77% of its words ran together, the worst being a
92-character run) and pdfplumber was far worse (up to 10.7%). PyMuPDF produced zero on
all six and recovered ~19% more words from the affected paper. See docs/results.md.
"""

import logging
import re
from collections import Counter
from pathlib import Path

import httpx
import pymupdf

from arxiv_rag.ingestion.chunker import heading_of

log = logging.getLogger(__name__)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ARXIV_STAMP_RE = re.compile(r"^\s*arXiv:\d{4}\.\d{4,5}v\d+\s+\[[\w.\-]+\].*$")
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")


def download_pdf(pdf_url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Download a PDF to ``dest`` unless it is already there."""
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

    Broad ``except`` on purpose: this runs over a batch of untrusted files, and one
    unreadable PDF should cost you that paper, not the other thirty-nine.
    """
    try:
        with pymupdf.open(pdf_path) as doc:
            pages = [page.get_text() for page in doc]
        return "\n\n".join(page for page in pages if page)
    except Exception as exc:
        log.warning("could not extract text from %s: %s", pdf_path.name, exc)
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

    # De-hyphenate before anything else touches the line structure. A heading that wraps
    # ("... Predictor Ge-\nneralization") is only recognisable once its two halves are
    # rejoined into a single line.
    text = re.sub(r"-\n(?=\w)", "", text)

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

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\x00", text)  # mark real paragraph breaks
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)  # line wraps become spaces
    text = text.replace("\x00", "\n\n")  # restore paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
