"""Stage 5 - the request id, the retry count, and the one JSON line. No network.

Three claims are worth testing rather than trusting, because each was wrong in an earlier
draft:

1. **a retry is not a second call.** At the transport layer they are the same event; only
   the SDK's `x-stainless-retry-count` header tells them apart;
2. **the summary line is emitted when the response *ends*, not when it starts.** For
   `/ask/stream` those are seconds apart, and the wrong one reports time-to-first-byte as
   the duration of the request;
3. **`note()` from inside a streaming generator reaches the line.** It does only because
   the middleware is raw ASGI and runs in the same task; under `BaseHTTPMiddleware` it
   silently does not, which is a test that would have caught a rewrite.
"""

import json
import logging
import time
from contextvars import copy_context

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from arxiv_rag import observability as obs
from arxiv_rag.observability import (
    RequestLogMiddleware,
    RequestStats,
    configure_logging,
    current_request_id,
    emit,
    new_request_id,
    note,
    record_attempt,
    record_response,
    request_hook,
    safe_request_id,
    start_request,
    timing_hook,
)


class ListHandler(logging.Handler):
    """Collects the emitted lines. `caplog` cannot: `configure_logging` sets
    `propagate = False` on the request logger, so its records never reach the root
    handler pytest installs - which is the whole point of that flag."""

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def lines():
    handler = ListHandler()
    obs.log.addHandler(handler)
    obs.log.setLevel(logging.INFO)
    yield handler.lines
    obs.log.removeHandler(handler)


class FakeRequest:
    """The two attributes the httpx hooks touch."""

    def __init__(self, retry_count=None):
        self.headers = {} if retry_count is None else {obs.RETRY_COUNT_HEADER: retry_count}


class FakeResponse:
    def __init__(self, status_code, request=None):
        self.status_code = status_code
        self.request = request or FakeRequest()


# --- counting ---------------------------------------------------------------------


def test_retries_are_zero_when_the_number_of_calls_is_unknown():
    """Three attempts could be three calls or one call retried twice. Without `calls` the
    honest answer is 0, not a guess: a fabricated retry count would be read as a system in
    trouble and send someone looking for a problem that is not there."""
    stats = RequestStats(request_id="x", attempts=3)
    assert stats.retries == 0


def test_retries_are_attempts_beyond_the_first_of_each_call():
    stats = RequestStats(request_id="x", attempts=4, extra={"calls": 3})
    assert stats.retries == 1


def test_a_first_attempt_is_a_call_and_a_later_one_is_a_retry():
    stats = start_request("r1")
    record_attempt(0)
    record_attempt(1)
    record_attempt(2)
    record_attempt(0)
    assert stats.attempts == 4
    assert stats.extra["calls"] == 2
    assert stats.retries == 2


def test_the_retry_header_is_what_separates_them():
    """The header is set by the SDK on every request it sends, including the retries it
    performs internally and reports nowhere else."""
    stats = start_request("r2")
    request_hook(FakeRequest("0"))
    request_hook(FakeRequest("1"))
    assert (stats.attempts, stats.extra["calls"], stats.retries) == (2, 1, 1)


def test_a_missing_or_unparseable_header_counts_as_a_first_attempt():
    """A header this code does not control must never raise inside a hook: the exception
    would surface as a failed model call, which is a monitoring bug causing an outage."""
    stats = start_request("r3")
    request_hook(FakeRequest())
    request_hook(FakeRequest("not-a-number"))
    assert stats.attempts == 2
    assert stats.extra["calls"] == 2


def test_statuses_are_counted_and_attempts_are_not_counted_twice():
    """Responses are counted separately from requests because a retry caused by a dropped
    connection produces no response at all - counting on the way out would lose exactly
    the failure worth seeing."""
    stats = start_request("r4")
    record_attempt(0)
    record_response(429, 12.0)
    record_attempt(1)
    record_response(500, 8.0)
    record_attempt(2)
    record_response(200, 30.0)
    assert stats.attempts == 3
    assert (stats.rate_limited, stats.server_errors) == (1, 1)
    assert stats.upstream_ms == pytest.approx(50.0)


def test_the_hooks_are_silent_outside_a_request():
    """`scripts/eval.py` and the ingestion scripts share the client and have no request
    context. The hooks run there on every call and must do nothing."""
    obs._current.set(None)
    request_hook(FakeRequest("0"))
    timing_hook(FakeResponse(200))
    assert obs.current_stats() is None


def test_timing_hook_uses_the_stamp_the_request_hook_left():
    stats = start_request("r5")
    request = FakeRequest("0")
    request_hook(request)
    time.sleep(0.01)
    timing_hook(FakeResponse(200, request))
    assert stats.upstream_ms >= 10.0


def test_one_requests_stats_do_not_leak_into_another():
    """A module-level counter would pass every test above and attribute one request's
    retries to another the moment two arrive at once. FastAPI runs sync endpoints in a
    threadpool, so this is the normal case, not the edge case."""

    def one(name):
        stats = start_request(name)
        record_attempt(0)
        return stats

    a = copy_context().run(one, "a")
    b = copy_context().run(one, "b")
    assert (a.attempts, b.attempts) == (1, 1)
    assert a is not b


# --- the id -----------------------------------------------------------------------


def test_a_clean_caller_supplied_id_is_kept():
    """Accepted so that a trace survives a proxy hop."""
    assert safe_request_id("abc-123_XY.4") == "abc-123_XY.4"


@pytest.mark.parametrize(
    "hostile",
    [
        "",
        "has space",
        'evil" injected',
        "a" * 65,
        '{"level":"forged"}',
        "line\nbreak",
    ],
)
def test_a_hostile_id_is_replaced_rather_than_cleaned(hostile):
    """The value lands in a JSON log line and in a response header. A newline in either is
    a real defect - a forged log record in the first, response splitting in the second -
    and repairing an id is a worse answer than issuing one."""
    issued = safe_request_id(hostile)
    assert issued != hostile
    assert len(issued) == 12


def test_ids_are_unique():
    assert len({new_request_id() for _ in range(500)}) == 500


# --- the line ---------------------------------------------------------------------


def test_the_line_is_one_json_object_with_the_notes_merged_in(lines):
    stats = start_request("abc123")
    record_attempt(0)
    record_response(200, 25.4)
    note(mode="agent", refused=False)
    emit(stats, path="/ask", status=200, duration_ms=1234.5)

    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["request_id"] == "abc123"
    assert row["mode"] == "agent"
    assert row["refused"] is False
    assert row["path"] == "/ask"
    assert row["status"] == 200
    assert row["upstream_ms"] == 25.4
    assert row["retries"] == 0
    assert "extra" not in row  # flattened, not nested: `jq .mode`, not `jq .extra.mode`


def test_a_value_json_cannot_serialise_does_not_lose_the_line(lines):
    """A line that raises inside the logger takes the request with it. `default=str` is
    there so an unexpected object costs a readable field, not a 500."""
    stats = start_request("abc")
    note(settings=object())
    emit(stats)
    assert json.loads(lines[0])["settings"].startswith("<object")


def test_note_outside_a_request_is_a_no_op():
    obs._current.set(None)
    note(mode="agent")  # must not raise
    assert current_request_id() is None


# --- the middleware ---------------------------------------------------------------


class Payload(BaseModel):
    question: str


def build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestLogMiddleware)

    @app.get("/ping")
    def ping():
        note(mode="pipeline")
        return {"request_id": current_request_id()}

    @app.get("/boom")
    def boom():
        raise RuntimeError("nope")

    @app.post("/validated")
    def validated(payload: Payload):
        return {"ok": True}

    @app.get("/stream")
    def stream():
        def pieces():
            yield "a"
            time.sleep(0.05)
            note(streamed=True)
            yield "b"

        return StreamingResponse(pieces(), media_type="text/plain")

    return app


@pytest.fixture
def client():
    with TestClient(build_app(), raise_server_exceptions=False) as c:
        yield c


def test_every_response_carries_the_id_and_the_body_agrees_with_the_header(client, lines):
    response = client.get("/ping")
    assert response.status_code == 200
    header = response.headers["x-request-id"]
    assert len(header) == 12
    # The endpoint read the same id the middleware issued: `current_request_id()` inside a
    # sync endpoint is answered from the threadpool's copied context, and if that copy did
    # not happen this would be None.
    assert response.json()["request_id"] == header
    assert json.loads(lines[0])["request_id"] == header


def test_the_caller_can_supply_the_id_and_a_hostile_one_is_replaced(client):
    kept = client.get("/ping", headers={"X-Request-ID": "upstream-42"})
    assert kept.headers["x-request-id"] == "upstream-42"

    replaced = client.get("/ping", headers={"X-Request-ID": "a b c"})
    assert replaced.headers["x-request-id"] != "a b c"


def test_one_line_per_request_with_route_status_and_duration(client, lines):
    client.get("/ping")
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert (row["method"], row["path"], row["status"]) == ("GET", "/ping", 200)
    assert row["outcome"] == "ok"
    assert row["mode"] == "pipeline"
    assert row["duration_ms"] >= 0


def test_a_request_rejected_before_the_endpoint_is_still_logged(client, lines):
    """A 422 never reaches the endpoint, so a decorator on the handler would miss it -
    and "it just returns an error" with no id is precisely the report that needs one."""
    response = client.post("/validated", json={})
    assert response.status_code == 422
    assert "x-request-id" in response.headers
    assert json.loads(lines[0])["status"] == 422


def test_an_unhandled_exception_is_logged_with_its_type(client, lines):
    response = client.get("/boom")
    assert response.status_code == 500
    row = json.loads(lines[0])
    assert row["outcome"] == "exception"
    assert row["exception"] == "RuntimeError"
    assert row["status"] == 500


def test_the_line_for_a_stream_covers_the_whole_stream(client, lines):
    """The claim the middleware exists for. `BaseHTTPMiddleware` returns when the response
    *starts*: this line would be written before the sleep, `duration_ms` would be a few
    milliseconds, and `streamed` would be missing entirely."""
    response = client.get("/stream")
    assert response.text == "ab"
    row = json.loads(lines[0])
    assert row["streamed"] is True
    assert row["duration_ms"] >= 50


# --- logging configuration --------------------------------------------------------


def test_the_summary_logger_writes_bare_json_and_does_not_propagate():
    """Uvicorn's format prefixes `INFO arxiv_rag.request:`, which would make every line
    invalid JSON; propagation would then print each line twice, once decorated."""
    configure_logging("INFO")
    assert obs.log.propagate is False
    assert len(obs.log.handlers) == 1
    assert obs.log.handlers[0].formatter._fmt == "%(message)s"


def test_configuring_twice_does_not_double_every_line():
    """`uvicorn --reload` and the test suite both do it."""
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(obs.log.handlers) == 1
