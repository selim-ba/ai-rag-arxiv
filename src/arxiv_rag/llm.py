"""One OpenAI client for the process, with bounds the defaults do not give it.

**Why this exists, from reading the SDK rather than guessing.** `openai._constants`:

    DEFAULT_TIMEOUT       = Timeout(timeout=600, connect=5.0)   # ten minutes
    DEFAULT_MAX_RETRIES   = 2
    INITIAL_RETRY_DELAY   = 0.5     MAX_RETRY_DELAY = 8.0
    MAX_RETRY_AFTER_DELAY = 120

Nothing in this codebase overrode either. So a single hung call could block a request for
ten minutes, and with two retries for thirty; an agent request makes up to three calls. A
429 carrying `Retry-After: 60` sleeps a minute inside the SDK, invisibly, twice.

**The fix was not another retry layer.** The SDK already retries twice with exponential
backoff and honours `Retry-After`; wrapping that in a loop of three would give nine
attempts per call and solve nothing. The defect is that a slow call has no bound, so the
bound is what gets set here.

**And eight modules each built their own client.** `OpenAI()` constructs an `httpx2.Client`
with its own connection pool, so a new client per call meant a new pool and a fresh TLS
handshake on every model call - paid and discarded, eight places over. One cached client
keeps the pool alive across calls and leaves one place to configure.

Cached on the settings values that matter rather than on the Settings object, which is not
hashable. Tests that build a Settings with a different timeout get a different client.
"""

import logging
from functools import lru_cache

import httpx2 as httpx
from openai import OpenAI

from arxiv_rag.config import Settings
from arxiv_rag.limits import budget
from arxiv_rag.observability import record_usage, request_hook, timing_hook

log = logging.getLogger(__name__)


class MissingAPIKey(RuntimeError):
    """No key configured, raised before the SDK is asked to do anything.

    Without this the failure is an `OpenAIError` from the client CONSTRUCTOR - no request
    attempted, nothing in the SDK's typed hierarchy - which classified as `internal_error`
    and told a caller nothing. Worse, it is indistinguishable from a bare `OpenAIError`
    raised for an unrelated reason, such as a content filter.

    Detected from configuration instead. `/health` has always reported whether a key is
    set; this is the same question asked on the path that needs the answer.
    """


@lru_cache(maxsize=8)
def _build(api_key: str, timeout: float, max_retries: int) -> OpenAI:
    log.info("openai client: timeout=%gs max_retries=%d", timeout, max_retries)
    # The hooks go on the HTTP client, below the SDK's retry loop, which is the only place
    # they can see it. By the time `chat.completions.create` returns, the retries are over
    # and were never reported; the hook sees each attempt as it happens.
    http = httpx.Client(
        timeout=timeout,
        event_hooks={"request": [request_hook], "response": [timing_hook]},
    )
    return OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries, http_client=http)


def get_client(settings: Settings) -> OpenAI:
    """The shared client. Every model call in this codebase goes through here."""
    if not settings.openai_api_key:
        raise MissingAPIKey("no OpenAI API key configured (set OPENAI_API_KEY)")
    return _build(
        settings.openai_api_key,
        settings.openai_timeout_seconds,
        settings.openai_max_retries,
    )


def chat(settings: Settings, **kwargs):
    """Every chat completion in this codebase goes through here, so every one is counted.

    The alternative was recording usage at each of the eight call sites, which works until
    the ninth is written. `tests/test_packaging.py` already keeps ingestion-only imports
    off the request path by grepping the source; the same trick keeps this seam honest.

    **Streaming needs asking.** A streamed completion reports no usage unless
    `stream_options={"include_usage": True}` is sent, and then it arrives in a final chunk
    with no choices in it. Without this the streaming path - the one a user actually hits
    - would report zero tokens and the budget would undercount exactly the traffic it
    exists to cap.
    """
    client = get_client(settings)
    model = kwargs.get("model", "")
    if kwargs.get("stream"):
        kwargs.setdefault("stream_options", {"include_usage": True})
        return _counted_stream(client.chat.completions.create(**kwargs), model)
    response = client.chat.completions.create(**kwargs)
    _record(response, model)
    return response


def embed(settings: Settings, **kwargs):
    """The embedding seam. Cheap - about $0.0000004 a query - and counted anyway, because
    a budget with a hole in it is a budget you have to reason about."""
    response = get_client(settings).embeddings.create(**kwargs)
    _record(response, kwargs.get("model", ""))
    return response


def _record(response, model: str) -> None:
    """Charge the request's log line and the day's budget, from the one place that knows.

    Both consumers hang off this seam rather than off each other: `observability` reports,
    `limits` enforces, and neither imports the other. The alternative - having the request
    logger tell the budget - would make a monitoring module load-bearing for spending.
    """
    usage = getattr(response, "usage", None)
    if usage is None:  # some endpoints omit it; counting nothing beats counting wrong
        return
    tokens_out = getattr(usage, "completion_tokens", 0) or 0
    record_usage(model, usage.prompt_tokens, tokens_out)
    budget().record(model, usage.prompt_tokens, tokens_out)


def _counted_stream(stream, model: str):
    """Pass the chunks through untouched, and take the usage from the last one."""
    for chunk in stream:
        _record(chunk, model)
        yield chunk
