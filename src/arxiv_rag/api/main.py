"""FastAPI application.

Stage 0 gave this a health endpoint. Stage 2 adds ``POST /ask``, which is the moment
the project stops being a library and becomes a service.

Two things here are worth understanding rather than copying.

**The index is loaded once, in the lifespan handler, not per request.** ``ChunkStore``
holds an 874x1536 matrix; reading it off disk on every request would add latency and
churn memory for no reason. ``lifespan`` runs once on startup and once on shutdown, and
what it puts on ``app.state`` is shared by every request.

**``ask`` is declared ``def``, not ``async def``.** ``answer_question`` makes blocking
network calls. In an ``async def`` endpoint those block the event loop, and the server
serves one request at a time no matter how many workers you give it. A plain ``def``
endpoint is run by FastAPI in a threadpool, which is the correct home for blocking work.
The rule: ``async def`` only when everything inside it is awaited.

Run it with ``make dev``, then open http://localhost:8000/docs - FastAPI generates that
page from the type hints below, which is a large part of why it is worth using.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from arxiv_rag import __version__
from arxiv_rag.agent.graph import build_graph, run_agent
from arxiv_rag.config import Settings, get_settings
from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.answer import answer_question
from arxiv_rag.retrieval.hybrid import Retriever, build_hybrid
from arxiv_rag.retrieval.store import ChunkStore

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index once at startup. Given to you.

    A missing index is not a crash: the app still starts and ``/health`` still answers,
    but ``/ask`` returns 503. A container that dies on startup because a volume was not
    mounted tells you nothing; one that starts and reports itself unready tells you
    exactly what is wrong.
    """
    settings = get_settings()
    try:
        store = ChunkStore.load(settings.index_dir)
        log.info("index loaded: %d chunks", len(store))
    except (FileNotFoundError, OSError) as exc:
        store = None
        log.warning("no index at %s (%s) - /ask will return 503", settings.index_dir, exc)

    # Hybrid is the shipped default: dense + BM25 fused by reciprocal rank. Measured on the
    # eval set, hit@1 0.324 -> 0.382 and MRR 0.457 -> 0.498 against dense alone, for about
    # 1ms. Reranking is deliberately NOT wired in here: the listwise reranker scores better
    # still (hit@1 0.529) but costs 7.8s at p95, which is the wrong default for a request
    # whose generation step is already over a second. See docs/results.md.
    retriever = None
    if store is not None:
        retriever = build_hybrid(store, settings)
        log.info(
            "retriever: hybrid dense+bm25, depth %d, rrf_k %d, weights %s",
            settings.fusion_depth,
            settings.rrf_k,
            settings.fusion_weight_list or "equal",
        )

    # Compiled once, for the same reason the index is loaded once. Compilation is cheap;
    # what matters is that the Grader and the retriever are bound at construction time, so
    # a request never builds its own dependencies.
    # `store=` is what turns the router on. The API gets it; `scripts/eval.py` does not,
    # because all 40 eval questions are corpus-content questions and routing them would add
    # a model call per question to a harness whose numbers are compared across three weeks
    # of runs.
    app.state.graph = (
        build_graph(retriever, settings, store=store) if retriever is not None else None
    )
    if app.state.graph is not None:
        log.info("agent graph compiled; POST /ask uses it: %s", settings.use_agent)

    app.state.store = store
    app.state.retriever = retriever
    app.state.chunks_by_id = {c.chunk_id: c for c in store.chunks} if store else {}
    yield
    app.state.store = None
    app.state.retriever = None
    app.state.graph = None


app = FastAPI(
    title="Agentic RAG over arXiv",
    description="Agentic RAG over arXiv papers",
    version=__version__,
    lifespan=lifespan,
)


def get_retriever(request: Request) -> Retriever:
    """Dependency: the configured retriever, or 503. Given to you.

    Going through a dependency rather than reading ``app.state`` inline is what lets the
    tests swap in a fake retriever with ``app.dependency_overrides``, without an index,
    without a network call, and without touching the endpoint.
    """
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    return retriever


def get_graph(request: Request):
    """Dependency: the compiled agent graph, or 503. Same shape as ``get_retriever``."""
    graph = request.app.state.graph
    if graph is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    return graph


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int | None = Field(default=None, ge=1, le=20, description="passages to retrieve")
    # Per-request override of the configured default, so the two paths can be compared
    # against each other on a live server without a restart. `None` means "use the
    # setting" - a third state, which is why this is `bool | None` and not `bool`.
    use_agent: bool | None = Field(
        default=None, description="override PT_USE_AGENT for this request"
    )


class Source(BaseModel):
    """One retrieved passage, in a form a caller can actually use."""

    chunk_id: str
    arxiv_id: str
    title: str
    section: str
    url: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(description="arXiv ids cited in the answer, in order")
    refused: bool
    sources: list[Source] = Field(description="passages retrieved, best first")
    # Which path served this request, and what it did. Reported rather than inferred:
    # the agent and the pipeline return identical answers at `max_retries=0`, so without
    # this a caller cannot tell them apart, and neither could anyone debugging a latency
    # complaint.
    mode: str = Field(description='"agent" or "pipeline"')
    trace: list[str] = Field(default_factory=list, description="agent steps; empty for pipeline")
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    """Liveness probe.

    Reports whether a key is configured without ever returning the key itself. Docker
    and your cloud platform will poll this in Stage 6.
    """
    return {
        "status": "ok",
        "version": __version__,
        "llm_model": settings.llm_model,
        "openai_key_configured": bool(settings.openai_api_key),
        "chunks_indexed": len(app.state.store) if app.state.store else 0,
    }


def to_source(chunk: Chunk) -> Source:
    """Turn a retrieved chunk into an API source.

    ``retrieved_ids`` is a list like ``["1811.04551::4"]``, which is meaningful inside
    this codebase and meaningless to anyone calling the API. A caller wants a title to
    display and a link to open.

    The url is the abs page - ``https://arxiv.org/abs/<arxiv_id>`` - not the PDF. Send a
    reader to the landing page and they can choose the PDF, the HTML version or the
    citation; send them straight to a PDF and you have made that choice for them.
    """
    return Source(
        chunk_id=chunk.chunk_id,
        arxiv_id=chunk.arxiv_id,
        title=chunk.title,
        section=chunk.section,
        url=f"https://arxiv.org/abs/{chunk.arxiv_id}",
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever),
    graph=Depends(get_graph),
) -> AskResponse:
    """Answer one question against the indexed corpus.

    Two paths, one response shape. ``use_agent`` on the request overrides the setting for
    this call, so the two can be compared against a running server without a restart - and
    ``mode`` in the response says which one actually ran, because at ``max_retries=0`` they
    are measured to produce identical answers and a caller could not otherwise tell.
    """
    # `is None` and not `or`: `use_agent=False` on the request must override a `True`
    # setting, and `or` would silently ignore it. Sixth place this distinction matters.
    use_agent = settings.use_agent if payload.use_agent is None else payload.use_agent

    if use_agent:
        answer = run_agent(graph, payload.question, k=payload.k)
    else:
        answer = answer_question(payload.question, retriever, settings, k=payload.k)

    by_id = app.state.chunks_by_id
    sources = [to_source(by_id[cid]) for cid in answer.retrieved_ids if cid in by_id]
    return AskResponse(
        question=answer.question,
        answer=answer.text,
        citations=answer.citations,
        refused=answer.refused,
        sources=sources,
        mode="agent" if use_agent else "pipeline",
        trace=answer.trace,
        retrieve_ms=round(answer.retrieve_ms, 1),
        generate_ms=round(answer.generate_ms, 1),
    )
