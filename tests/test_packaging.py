"""Stage 6 - what the runtime image is allowed to import. No network, no container.

The first image shipped 63 MB of `pymupdf` and 4 MB of `tiktoken` to a service that never
parses a PDF and never counts a token: the API reads a prebuilt index. Both moved to the
`[ingest]` extra, which only helps for as long as nothing on the request path imports them
again - and a single convenience import in a future module would silently put 67 MB back,
visible only as a bigger image nobody looks at.

So this is a structural test, in the same family as the one that greps for `OpenAI(`:
it does not check that the code works, it checks that the code stays the shape the
packaging assumes.

It runs in a subprocess deliberately. `sys.modules` inside the test session is polluted by
every other test file - `test_chunker.py` imports `tiktoken` - so asking the current
interpreter what is loaded would answer a question about pytest, not about the API.
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# `fitz` is pymupdf's legacy import name; a module using it would look innocent in a diff.
INGEST_ONLY = ("pymupdf", "fitz", "tiktoken")


def loaded_after_importing(module: str) -> set[str]:
    probe = (
        f"import sys; import {module}; "
        f"print(' '.join(m for m in {INGEST_ONLY!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    return set(result.stdout.split())


def test_the_api_imports_nothing_the_container_leaves_out():
    """The one that matters: `uvicorn arxiv_rag.api.main:app` must work in an image built
    without the `[ingest]` extra."""
    leaked = loaded_after_importing("arxiv_rag.api.main")
    assert leaked == set(), f"ingestion-only dependencies on the request path: {leaked}"


def test_the_agent_and_retrieval_stay_clean_too():
    """Both are reachable from the API, and both import `Chunk` from `ingestion.models` -
    which is safe only because `ingestion/__init__.py` is empty. That empty file is
    load-bearing, and this is what says so."""
    for module in ("arxiv_rag.agent.graph", "arxiv_rag.retrieval.hybrid"):
        assert loaded_after_importing(module) == set(), module


def test_the_ingestion_path_really_does_need_the_extra():
    """The negative control. If this ever passes empty, the imports moved somewhere else
    and the two tests above stopped proving anything."""
    assert loaded_after_importing("arxiv_rag.ingestion.pdf_parser") >= {"pymupdf", "tiktoken"}
