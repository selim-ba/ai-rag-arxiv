"""What the API returns when the model provider fails.

The SDK already retries twice with exponential backoff and honours `Retry-After`; this is
what happens when that is exhausted, which until now was an untyped 500 with a stack trace
in the log and nothing in the response a caller could act on.

**The upstream status is not the status returned.** That is the whole of the design, and
it is easy to get backwards by proxying the number through:

| upstream | returned | code |
|---|---|---|
| 429 rate limit | 503 | `rate_limited` |
| timeout | 504 | `upstream_timeout` |
| connection error | 502 | `upstream_unreachable` |
| 500-599 | 502 | `upstream_error` |
| 401, 403, 400 | 500 | `misconfigured` |

The two rows that are not obvious. **429 becomes 503** because the quota is this
service's, not the caller's: a 429 tells a client it is sending too fast and a
well-behaved one slows down, for a limit it had no part in. 503 says the service is
temporarily unable, which is what is true. **401, 403 and 400 become 500** because a bad
key, a model the account cannot use and a malformed prompt are all ours; returning the
upstream 4xx would tell a caller to fix a request that was never the problem. That exact
failure - a model permission that had not propagated - once cost `scripts/eval.py` forty
straight calls.

**The provider's message never reaches the response.** Each code carries a fixed sentence.
Upstream error bodies quote organisation ids, model names and quota details, and an error
path is the least-watched place in a service for those to start appearing.

**`Retry-After` is echoed only when upstream sent one.** A default would be a guess
presented to the client as a fact, and a client that trusts it retries into the same wall.

**Known bound, not fixed here.** `openai_timeout_seconds=30` and `openai_max_retries=2`
put a ceiling of about 91s on a single logical call, and an agent request makes up to
three - roughly 275s worst case. A per-request deadline that shrinks each call's timeout
to the time left is the real answer; it needs the budget threaded through every call site,
and the shared client's `with_options(timeout=...)` is where it would go. Bounded and
measured beats unbounded and silent, which is where this started.
"""

import logging
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from arxiv_rag.llm import MissingAPIKey
from arxiv_rag.observability import current_request_id, note

log = logging.getLogger(__name__)


class Refused(Exception):
    """A request this service declines for its own reasons rather than the provider's.

    Carries the same `UpstreamFailure` shape so a caller sees one response format for
    every failure: a `code` to branch on, a sentence for a human, a request id to quote.
    A demo that ran out of budget and a provider that rate-limited us are different
    events, and a client should be able to tell them apart without reading prose.
    """

    def __init__(self, failure: "UpstreamFailure") -> None:
        super().__init__(failure.code)
        self.failure = failure


@dataclass(frozen=True)
class UpstreamFailure:
    """One row of the table above: what the caller is told, and how long to wait."""

    status: int
    code: str
    detail: str
    retry_after: int | None = None


def _retry_after(exc: Exception) -> int | None:
    """The provider's own hint, in seconds, or None. Never a substitute of our own."""
    response = getattr(exc, "response", None)
    raw = response.headers.get("retry-after") if response is not None else None
    try:
        return max(0, int(float(raw))) if raw else None
    except (TypeError, ValueError):
        return None  # the header may also be an HTTP date; unparsed is better than wrong


def classify(exc: Exception) -> UpstreamFailure:
    """Map a provider exception onto the response the caller gets.

    Order matters once: `APITimeoutError` is a subclass of `APIConnectionError`, so the
    general case would swallow the specific one and a timeout would be reported as an
    unreachable host - the same symptom, a different cause, and the wrong thing to chase.
    """
    if isinstance(exc, APITimeoutError):
        return UpstreamFailure(
            504, "upstream_timeout", "the model provider did not respond in time"
        )
    if isinstance(exc, APIConnectionError):
        return UpstreamFailure(502, "upstream_unreachable", "could not reach the model provider")
    if isinstance(exc, RateLimitError):
        return UpstreamFailure(
            503,
            "rate_limited",
            "the service is rate limited by the model provider; try again shortly",
            retry_after=_retry_after(exc),
        )
    ours = MissingAPIKey | AuthenticationError | PermissionDeniedError | BadRequestError
    if isinstance(exc, ours):
        return UpstreamFailure(500, "misconfigured", "the service is misconfigured")
    if isinstance(exc, APIStatusError):
        return UpstreamFailure(502, "upstream_error", "the model provider returned an error")
    return UpstreamFailure(500, "internal_error", "internal error")


def failure_payload(failure: UpstreamFailure) -> dict:
    """The body, in the one shape every failure uses.

    `code` is the field a client branches on - `rate_limited` means wait and repeat,
    `misconfigured` means stop - and `detail` is for the human reading it. `request_id`
    is here as well as on the header because a failure is exactly when someone copies the
    body into a message to you.
    """
    payload = {
        "detail": failure.detail,
        "code": failure.code,
        "request_id": current_request_id(),
    }
    if failure.retry_after is not None:
        payload["retry_after"] = failure.retry_after
    return payload


def install_error_handlers(app: FastAPI) -> None:
    """Register the provider-failure handler.

    One handler on the base class rather than seven: Starlette looks a handler up along
    the exception's MRO, so `OpenAIError` catches every subclass, including ones added by
    a future SDK version - which then land on the `internal_error` row instead of escaping
    as a 500 with a stack trace.

    `MissingAPIKey` is registered alongside it because it is ours, not the SDK's: a key
    that was never configured is a failure this service can name before asking the
    provider anything. The container checks found that gap - an unset key surfaced as
    `internal_error`, which tells an operator nothing.

    It runs *inside* the request-logging middleware, so these responses still carry an
    `X-Request-ID` and still produce exactly one summary line, now with the failure on it.
    """

    @app.exception_handler(Refused)
    def handle_refused(request: Request, exc: Refused) -> JSONResponse:
        note(refused_by=exc.failure.code)
        headers = {}
        if exc.failure.retry_after is not None:
            headers["Retry-After"] = str(exc.failure.retry_after)
        return JSONResponse(
            status_code=exc.failure.status,
            content=failure_payload(exc.failure),
            headers=headers,
        )

    @app.exception_handler(MissingAPIKey)
    @app.exception_handler(OpenAIError)
    def handle_provider_error(request: Request, exc: Exception) -> JSONResponse:
        failure = classify(exc)
        note(
            upstream_error=type(exc).__name__,
            code=failure.code,
            # The provider's own request id, when there was a response to carry one. It is
            # what their support asks for first, and it is unrecoverable after this point.
            upstream_request_id=getattr(exc, "request_id", None),
        )
        log.warning(
            "provider failure: %s -> %d %s", type(exc).__name__, failure.status, failure.code
        )
        headers = {}
        if failure.retry_after is not None:
            headers["Retry-After"] = str(failure.retry_after)
        return JSONResponse(
            status_code=failure.status, content=failure_payload(failure), headers=headers
        )
