"""FastAPI application.

Stage 0: a health endpoint and nothing else. You will add ``/ask`` in Stage 2 and make
it stream in Stage 5.

Run it with ``make dev``, then open http://localhost:8000/docs — FastAPI generates that
page from your type hints, which is a large part of why it is worth using.
"""

from fastapi import Depends, FastAPI

from arxiv_rag import __version__
from arxiv_rag.config import Settings, get_settings

app = FastAPI(
    title="Agentic RAG over arXiv",
    description="Agentic RAG over arXiv papers",
    version=__version__,
)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    """Liveness probe.

    Deliberately reports whether a key is configured without ever returning the key
    itself. Docker and your cloud platform will poll this in Stage 6.
    """
    return {
        "status": "ok",
        "version": __version__,
        "llm_model": settings.llm_model,
        "openai_key_configured": bool(settings.openai_api_key),
    }
