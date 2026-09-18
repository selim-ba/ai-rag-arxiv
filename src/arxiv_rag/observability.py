"""Structured logging, a request id, and a count of what the SDK did quietly.

**One JSON line per request, and no second format.** A human-readable mode would be
pleasanter on one terminal and would be a second code path, with its own escaping bugs, for
something no measurement asks for. To read them:

    make dev 2>&1 | jq -Rc --unbuffered 'fromjson? | select(.request_id)'

`jq .` alone does not work, and the reason is worth keeping: uvicorn's own startup lines
share the stream and are not JSON, so jq dies on the first one and takes the server with it
through the broken pipe. `-R` reads raw lines and `fromjson?` drops the ones that are not
objects. A log line whose documented command fails on the first run is the same defect as a
metric nobody checked.

**Why counting retries is the point.** The OpenAI SDK retries twice with exponential
backoff and honours `Retry-After` for up to two minutes. A request that succeeded first try
and one that succeeded after two retries and a ninety-second sleep look identical from
outside, and the second is a system in trouble. `scripts/eval.py` learned this the
expensive way: its failure banner is what caught forty straight 403s from a model
permission that had not propagated, and a connection drop that killed a paid run. The API
had no equivalent.

An earlier version of this docstring called those retries *silent*. They are not: the SDK
logs `Retrying request in 0.49 seconds` at INFO. They were invisible for a different
reason - **nothing in the API ever configured logging**, so no INFO record from any library
was ever printed. What the line below adds over the SDK's is attribution: the SDK's record
carries no request id, no route and no total, so it cannot tell you which request paid for
the wait or how much of the wait it was.

The count comes from an httpx event hook rather than from wrapping every call site, because
the SDK's retries happen *below* the call site - by the time `chat.completions.create`
returns, the retries are over and invisible. The hook sees each attempt.
"""

import json
import logging
import re
import sys
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from uuid import uuid4

from arxiv_rag.pricing import cost_usd

log = logging.getLogger("arxiv_rag.request")

# Every request the SDK sends carries its own retry number in this header (0 on the first
# attempt, 1 on the first retry, and so on). It is what makes a retry distinguishable from
# a second logical call at the transport layer, where the two otherwise look identical.
RETRY_COUNT_HEADER = "x-stainless-retry-count"


def configure_logging(level: str = "INFO") -> None:
    """Make the lines visible, and keep the JSON ones parseable.

    Two facts that only meet in a function like this. Uvicorn configures its own loggers
    and leaves the root logger alone, so an application `log.info` has nowhere to go and
    disappears - `index loaded: 874 chunks` had been silently dropping since Stage 0. And
    the default format prefixes `INFO arxiv_rag.request:`, which would make every summary
    line invalid JSON and `| jq .` useless. So the request logger gets a bare formatter of
    its own and stops propagating; everything else keeps the decorated format.
    """
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.handlers = [handler]  # assignment, not append: this runs once per process, in
    log.propagate = False  # theory, and twice under `--reload` in practice
    log.setLevel(logging.INFO)


@dataclass
class RequestStats:
    """What happened while serving one request."""

    request_id: str
    attempts: int = 0  # HTTP requests sent to the model provider, retries included
    rate_limited: int = 0  # 429s seen, each of which the SDK slept on
    server_errors: int = 0  # 5xx seen
    upstream_ms: float = 0.0  # time inside provider calls
    tokens_in: int = 0  # prompt tokens, summed over every call this request made
    tokens_out: int = 0  # completion tokens
    cost_usd: float = 0.0  # what those tokens cost, at the prices in `pricing.py`
    extra: dict = field(default_factory=dict)
    # Not reported as a field: a perf_counter origin is meaningless on its own. It is here
    # so that anything during the request can ask how long the *caller* has been waiting,
    # which is the only clock that matters for time-to-first-token.
    started: float = field(default_factory=time.perf_counter)

    @property
    def retries(self) -> int:
        """Attempts beyond the first for each logical call.

        Not derivable from `attempts` alone - a request making three logical calls with no
        retries also shows three attempts. `calls` counts the attempts that announced
        themselves as first tries; without it this reports 0 rather than guessing, because
        a guessed retry count is worse than none.
        """
        calls = self.extra.get("calls", 0)
        return max(0, self.attempts - calls) if calls else 0


_current: ContextVar[RequestStats | None] = ContextVar("request_stats", default=None)

# Deliberately narrow: an id that fails this is replaced, not sanitised into shape.
_SAFE_ID = re.compile(r"\A[A-Za-z0-9._-]{1,64}\Z")


def new_request_id() -> str:
    return uuid4().hex[:12]


def safe_request_id(value: str | None) -> str:
    """A caller-supplied id if it is safe to repeat, a fresh one otherwise.

    An id is accepted from the client so that a trace survives a proxy hop, and accepted
    only as a short run of plain characters. The value ends up in two places that punish
    trust: a JSON log line, where an attacker-chosen string is how a forged line gets into
    a log file, and a response header, where a newline is a response-splitting bug.
    """
    return value if value and _SAFE_ID.match(value) else new_request_id()


def start_request(request_id: str | None = None) -> RequestStats:
    """Begin collecting for this request and make it the current one."""
    stats = RequestStats(request_id=request_id or new_request_id())
    _current.set(stats)
    return stats


def current_stats() -> RequestStats | None:
    """The in-flight request's stats, or None outside a request.

    A ContextVar rather than a global: FastAPI runs sync endpoints in a threadpool, and a
    module-level counter would attribute one request's retries to another under any
    concurrency at all.
    """
    return _current.get()


def elapsed_ms() -> float | None:
    """Milliseconds since this request arrived, or None outside a request."""
    stats = _current.get()
    return round((time.perf_counter() - stats.started) * 1000, 1) if stats else None


def current_request_id() -> str | None:
    """The in-flight request's id, for echoing back to the caller."""
    stats = _current.get()
    return stats.request_id if stats else None


def record_attempt(retry_count: int) -> None:
    """One provider HTTP request leaving the process. Safe outside a request."""
    stats = _current.get()
    if stats is None:
        return
    stats.attempts += 1
    if retry_count == 0:
        stats.extra["calls"] = stats.extra.get("calls", 0) + 1


def record_response(status_code: int, elapsed_ms: float) -> None:
    """One provider HTTP response. Called from the httpx hook; safe outside a request.

    Counted separately from the attempt, and not by incrementing `attempts` here, because
    a retry caused by a dropped connection never produces a response at all. Counting on
    the way out would make the most interesting failure the one that leaves no trace.
    """
    stats = _current.get()
    if stats is None:
        return
    stats.upstream_ms += elapsed_ms
    if status_code == 429:
        stats.rate_limited += 1
    elif status_code >= 500:
        stats.server_errors += 1


def record_usage(model: str, tokens_in: int, tokens_out: int) -> None:
    """One call's token usage. Safe outside a request, like the other recorders.

    Counted here rather than estimated from chunk sizes. The estimate said a request costs
    about $0.00054; this is what says whether that was right, and it is the difference
    between a daily budget denominated in dollars and one denominated in guesses.
    """
    stats = _current.get()
    if stats is None:
        return
    stats.tokens_in += tokens_in
    stats.tokens_out += tokens_out
    stats.cost_usd += cost_usd(model, tokens_in, tokens_out)


def note(**fields) -> None:
    """Attach request-scoped facts - route, mode, conversation - to the summary line."""
    stats = _current.get()
    if stats is not None:
        stats.extra.update(fields)


def emit(stats: RequestStats, **fields) -> None:
    """The one line. Everything about the request, as JSON, on a single row."""
    payload = {k: v for k, v in asdict(stats).items() if k not in ("extra", "started")}
    payload["retries"] = stats.retries
    # Six decimals: a request costs about $0.0005, and rounding to four would report most
    # of them as $0.0005 or $0.0000 - a field that cannot distinguish a cheap request from
    # a free one is not worth the width.
    payload["cost_usd"] = round(stats.cost_usd, 6)
    payload |= stats.extra
    payload |= fields
    payload["upstream_ms"] = round(stats.upstream_ms, 1)
    log.info(json.dumps(payload, default=str))


def timing_hook(response) -> None:
    """httpx response hook: time the attempt and record its status.

    Registered on the shared client so it sees every attempt the SDK makes, including the
    retries it performs internally and never reports.
    """
    started = getattr(response.request, "_arxiv_rag_started", None)
    elapsed = (time.perf_counter() - started) * 1000 if started else 0.0
    record_response(response.status_code, elapsed)


def request_hook(request) -> None:
    """httpx request hook: stamp the start time, and count the attempt as call or retry."""
    request._arxiv_rag_started = time.perf_counter()
    raw = request.headers.get(RETRY_COUNT_HEADER)
    try:
        retry_count = int(raw) if raw is not None else 0
    except ValueError:  # a header we do not control; a bad value must not break the call
        retry_count = 0
    record_attempt(retry_count)


class RequestLogMiddleware:
    """ASGI middleware: one request id in, one summary line out.

    Written as raw ASGI rather than `@app.middleware("http")`, for two reasons that each
    cost a rewrite to find. Starlette's `BaseHTTPMiddleware` runs the application in a
    child task, so a `ContextVar` set here reaches the endpoint but nothing the endpoint
    records comes back. And it regains control when the response *starts*, which for
    `/ask/stream` is before a single token exists: the line would report time-to-first-byte
    as the duration of a request that had barely begun. A raw wrapper runs in the same task
    and sees the final body message, so the duration covers the whole stream and `note()`
    from inside the endpoint lands on the same object this emits.
    """

    HEADER = b"x-request-id"

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":  # lifespan, websockets: nothing to log
            await self.app(scope, receive, send)
            return

        incoming = next(
            (v.decode("latin-1") for k, v in scope["headers"] if k == self.HEADER), None
        )
        stats = start_request(safe_request_id(incoming))
        state = {"status": 0, "emitted": False}

        def finish(outcome: str) -> None:
            if state["emitted"]:  # first ending wins; the rest are unwinding
                return
            state["emitted"] = True
            emit(
                stats,
                method=scope.get("method", ""),
                path=scope.get("path", ""),
                status=state["status"],
                duration_ms=round((time.perf_counter() - stats.started) * 1000, 1),
                outcome=outcome,
            )

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                # Echoed on every response, success or failure, so a caller reporting a
                # problem can quote something greppable without reading the body.
                message["headers"] = [
                    *message.get("headers", []),
                    (self.HEADER, stats.request_id.encode()),
                ]
            await send(message)
            done = message["type"] == "http.response.body" and not message.get("more_body")
            if done:
                # Three outcomes, not two: a 422 is the service working correctly and a
                # 503 is not, and `outcome == "ok"` has to mean something for a success
                # rate grepped out of these lines to be worth reading.
                status = state["status"]
                finish("ok" if status < 400 else "client_error" if status < 500 else "error")

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            # The 500 is produced *above* this middleware, by Starlette's error handler,
            # and sent straight to the server - `send_wrapper` never sees it. So the
            # status is recorded here or not at all, and that response carries no id
            # header. The line is the only record, which is the case that most needs one.
            state["status"] = 500
            note(exception=type(exc).__name__)
            finish("exception")
            raise
        # Reached without a final body message: the client hung up mid-stream, which is
        # invisible in the access log and worth a name of its own.
        finish("disconnected")
