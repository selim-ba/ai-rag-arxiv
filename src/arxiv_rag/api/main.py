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

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from arxiv_rag import __version__
from arxiv_rag.agent.followup import Turn, resolve_followup
from arxiv_rag.agent.graph import build_graph, run_agent
from arxiv_rag.agent.session import SessionStore, new_conversation_id
from arxiv_rag.api.errors import classify, failure_payload, install_error_handlers
from arxiv_rag.config import Settings, get_settings
from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.observability import (
    RequestLogMiddleware,
    configure_logging,
    current_request_id,
    elapsed_ms,
    note,
)
from arxiv_rag.retrieval.answer import answer_question, assemble_answer, stream_generate
from arxiv_rag.retrieval.hybrid import Retriever, build_hybrid
from arxiv_rag.retrieval.store import ChunkStore

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index once at startup.

    A missing index is not a crash: the app still starts and ``/health`` still answers,
    but ``/ask`` returns 503. A container that dies on startup because a volume was not
    mounted tells you nothing; one that starts and reports itself unready tells you
    exactly what is wrong.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
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

    # One store for the process. See `agent.session` for why this is a dictionary and not
    # a LangGraph checkpointer, and for what breaks under horizontal scaling.
    app.state.sessions = SessionStore()

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

# Outermost: every response gets an `X-Request-ID` header and every request gets one JSON
# summary line, including the ones that fail validation before an endpoint is reached. A
# 422 that no endpoint ever saw is exactly the kind of failure a caller reports as "it
# returned an error" with nothing to grep for.
app.add_middleware(RequestLogMiddleware)

# Provider failures become defined statuses instead of an untyped 500. Registered
# below the middleware, so these responses still carry an id and still log one line.
install_error_handlers(app)


def get_retriever(request: Request) -> Retriever:
    """Dependency: the configured retriever, or 503.

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
    # Returned by a previous call. An unknown id starts a new conversation rather than
    # failing: ids are server-generated, so an unrecognised one is a stale client, not an
    # attack worth a 4xx.
    conversation_id: str | None = Field(
        default=None, max_length=64, description="continue a conversation"
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
    conversation_id: str = Field(description="pass this back to ask a follow-up")
    # Only set when the follow-up was rewritten. Reported for the same reason as `mode`:
    # a question answered differently from the one that was typed is something the caller
    # must be able to see, and over-resolution is silent by nature.
    resolved_question: str | None = Field(
        default=None, description="the standalone question actually answered, if rewritten"
    )
    # Also on the `X-Request-ID` header. In the body as well because the header is the
    # first thing a client library drops, and an id nobody can find is not an id.
    request_id: str | None = Field(
        default=None, description="quote this when reporting a problem with this answer"
    )


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

    # A first turn has no history, so `resolve_followup` returns immediately without a
    # model call. The cost of sessions is paid only from the second turn on.
    sessions: SessionStore = app.state.sessions
    conversation_id = payload.conversation_id or new_conversation_id()
    history = sessions.history(payload.conversation_id)
    resolution = resolve_followup(history, payload.question, settings)
    question = resolution.resolved

    if use_agent:
        answer = run_agent(graph, question, k=payload.k)
    else:
        answer = answer_question(question, retriever, settings, k=payload.k)

    # On the summary line rather than in a second log call: what makes a line worth
    # keeping is that one row answers "which request, which path, what came out, and what
    # did it cost" without a join. `resolved` is here because a question answered
    # differently from the one typed is the first thing to check when an answer looks wrong.
    note(
        mode="agent" if use_agent else "pipeline",
        conversation_id=conversation_id,
        turn=len(history) + 1,
        resolved=resolution.changed,
        refused=answer.refused,
        citations=len(answer.citations),
        retrieve_ms=round(answer.retrieve_ms, 1),
        generate_ms=round(answer.generate_ms, 1),
    )

    # Record what was ASKED, not the resolution: the next turn's resolver reads this as a
    # transcript, and a transcript of rewritten questions drifts further from the
    # conversation with every turn.
    sessions.append(conversation_id, Turn(question=payload.question, answer=answer.text))

    by_id = app.state.chunks_by_id
    sources = [to_source(by_id[cid]) for cid in answer.retrieved_ids if cid in by_id]
    return AskResponse(
        question=payload.question,
        answer=answer.text,
        citations=answer.citations,
        refused=answer.refused,
        sources=sources,
        mode="agent" if use_agent else "pipeline",
        trace=answer.trace,
        retrieve_ms=round(answer.retrieve_ms, 1),
        generate_ms=round(answer.generate_ms, 1),
        conversation_id=conversation_id,
        resolved_question=question if resolution.changed else None,
        request_id=current_request_id(),
    )


def sse(event: str, payload: dict) -> str:
    """One server-sent event.

    The wire format is two lines and a blank one. Written out rather than pulled from a
    library because the whole protocol is visible here, and a stray newline inside `data:`
    would silently split one event into two.
    """
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.post("/ask/stream")
def ask_stream(
    payload: AskRequest,
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever),
) -> StreamingResponse:
    """Answer one question, streaming the text as it is produced.

    **Three event types.** `token` carries a piece of the answer; `done` carries everything
    that is only knowable once the answer is complete; `error` carries a failure that
    happened after the response had already begun.

    **The answer streams with `[P1]` markers in it, not arXiv ids, and that is deliberate.**
    `resolve_citations` maps a marker to the id of the passage it points at, and it needs
    the finished text. Three ways to handle that were considered:

    1. stream the markers, resolve in the `done` event - this one. The guarantee is
       untouched and the client substitutes;
    2. buffer output at marker boundaries. Also correct, and the buffering breaks in ways
       that only show up when an answer ends mid-marker;
    3. give the model real arXiv ids so no resolution is needed. Out on evidence: an early
       version invented `2606.09985` for a paper numbered `2506.09985`, which is why
       markers exist.

    **Retrieval happens before the first token**, so a retrieval failure is still a normal
    HTTP error rather than an `error` frame in a 200 response. Once bytes are on the wire
    the status code is already sent, which is why the failure path splits here.

    Not routed through the agent. Streaming the pipeline covers the part a user waits for;
    the router and grader run before generation and would arrive as one silent pause, so
    `event: step` frames for them are worth building only alongside `graph.stream`.
    """
    sessions: SessionStore = app.state.sessions
    conversation_id = payload.conversation_id or new_conversation_id()
    history = sessions.history(payload.conversation_id)
    resolution = resolve_followup(history, payload.question, settings)
    question = resolution.resolved

    # Before the stream opens: a failure here can still be a 5xx.
    hits = retriever.search(question, k=payload.k or settings.top_k)

    note(
        mode="stream",
        conversation_id=conversation_id,
        turn=len(history) + 1,
        resolved=resolution.changed,
    )
    request_id = current_request_id()

    def events():
        pieces: list[str] = []
        try:
            for piece in stream_generate(question, hits, settings):
                # Time to first token, measured from when the request arrived rather than
                # from when generation began: retrieval and follow-up resolution happen
                # first, and the user is waiting through those too. This is the number
                # streaming exists to lower, and the only one they feel.
                if not pieces:
                    note(ttft_ms=elapsed_ms())
                pieces.append(piece)
                yield sse("token", {"text": piece})
        except Exception as exc:  # noqa: BLE001 - the response has already started
            # A stream cannot change its status code: 200 went out with the first byte.
            # So the same taxonomy arrives in the frame instead, and `code` means exactly
            # what it means on the non-streaming path - `rate_limited` is wait and repeat,
            # `misconfigured` is stop. A client can branch on one field either way.
            failure = classify(exc)
            log.warning("stream failed after %d pieces: %s", len(pieces), exc)
            note(stream_error=type(exc).__name__, code=failure.code, pieces=len(pieces))
            yield sse("error", failure_payload(failure) | {"partial": "".join(pieces)})
            return

        # The same pure function the non-streaming path uses, on the same complete text.
        answer = assemble_answer(question, hits, "".join(pieces))
        sessions.append(conversation_id, Turn(question=payload.question, answer=answer.text))
        note(refused=answer.refused, citations=len(answer.citations))
        by_id = app.state.chunks_by_id
        yield sse(
            "done",
            {
                "question": payload.question,
                "answer": answer.text,
                "citations": answer.citations,
                "refused": answer.refused,
                "sources": [
                    to_source(by_id[cid]).model_dump()
                    for cid in answer.retrieved_ids
                    if cid in by_id
                ],
                "conversation_id": conversation_id,
                "resolved_question": question if resolution.changed else None,
                "request_id": request_id,
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Without this an intermediary may buffer the whole response and deliver it at
        # once, which looks exactly like the endpoint not streaming at all.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
