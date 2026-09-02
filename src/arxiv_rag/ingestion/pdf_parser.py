"""PDF download and text extraction.

Fair warning: this is the ugliest part of any RAG pipeline. Academic PDFs are typeset
for print, not for parsing. You will see two-column text interleaved into nonsense,
ligatures turned into mojibake, page numbers glued onto sentences, and equations
reduced to rubble.

You are not aiming for perfection. You are aiming to *look at the output*, notice the
specific ways it is broken, and clean up the ones that would hurt retrieval.
"""

from pathlib import Path

import httpx
from pypdf import PdfReader


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
    raise NotImplementedError("Stage 1: implement extract_text")


def clean_text(raw: str) -> str:
    """Clean extracted PDF text.

    TODO(you): implement this. ``tests/test_pdf_parser.py`` is the spec — read it first.

    You must handle, at minimum:

    1. **De-hyphenation.** PDFs break words across lines: ``"retrie-\\nval"`` must
       become ``"retrieval"``. If you skip this, your embeddings contain hundreds of
       words that do not exist, and keyword search in Stage 3 will miss them entirely.
    2. **Single newlines are line wraps, not paragraph breaks.** A newline inside a
       paragraph should become a space. Two or more newlines is a real break and should
       stay one blank line.
    3. **Collapse runs of spaces and tabs** into a single space.
    4. Strip leading/trailing whitespace from the result.

    Order matters: de-hyphenate before you collapse newlines, or the hyphen and the
    newline stop being adjacent and rule 1 can no longer fire.

    Optional, once the tests pass: drop lines that repeat on nearly every page (running
    headers and footers), and strip standalone page numbers.
    """
    raise NotImplementedError("Stage 1: implement clean_text")


def _unused_import_guard() -> None:  # pragma: no cover
    """Keeps the linter quiet about an import you need once extract_text exists."""
    _ = PdfReader
