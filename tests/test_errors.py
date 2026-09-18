"""Stage 5 step 5 - what a caller is told when the model provider fails. No network.

The SDK's two retries with backoff happen underneath all of this. These tests are about
the moment after they are exhausted, which used to be an untyped 500.

The claim under test is one sentence: **the upstream status is not the status returned.**
Proxying the number through is the easy version and it is wrong in both directions - a 429
would blame the caller for a quota it has no part in, and a 401 would invite it to fix a
request that was never the problem.
"""

import json

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from arxiv_rag.api import main as api
from arxiv_rag.api.errors import classify, failure_payload
from arxiv_rag.api.main import app, get_graph, get_retriever
from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.llm import MissingAPIKey
from arxiv_rag.retrieval.store import SearchHit

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

# A message shaped like the ones providers actually send: account ids, model names, quota.
LEAKY = "Rate limit reached for gpt-4o-mini in organization org-4f2b91 on tokens per min"

CHUNKS = [
    Chunk(
        chunk_id="1811.04551::4",
        arxiv_id="1811.04551",
        title="Learning Latent Dynamics for Planning from Pixels",
        section="Method",
        text="PlaNet plans with CEM.",
        index=4,
        token_count=10,
    )
]


def status_error(cls, status: int, message: str = "boom", **headers):
    """An SDK exception built the way the SDK builds one, from a real response."""
    response = httpx.Response(status, request=REQUEST, headers=headers)
    return cls(message, response=response, body=None)


class FakeRetriever:
    def search(self, query: str, k: int = 5, chunk_filter=None) -> list[SearchHit]:
        return [SearchHit(chunk, 1.0) for chunk in CHUNKS]


class FakeGraph:
    def invoke(self, state):  # pragma: no cover - the pipeline path is the one under test
        raise AssertionError("FakeGraph.invoke called")


@pytest.fixture
def client():
    app.dependency_overrides[get_retriever] = FakeRetriever
    app.dependency_overrides[get_graph] = FakeGraph
    with TestClient(app) as c:
        c.app.state.chunks_by_id = {ch.chunk_id: ch for ch in CHUNKS}
        yield c
    app.dependency_overrides.clear()


def fails_with(monkeypatch, exc: Exception):
    """Generation raises, the way it does when the retries are spent."""

    def explode(question, retriever, settings, k=None, chunk_filter=None):
        raise exc

    monkeypatch.setattr(api, "answer_question", explode)


# --- the mapping ------------------------------------------------------------------


def test_a_timeout_is_not_reported_as_an_unreachable_host():
    """`APITimeoutError` subclasses `APIConnectionError`, so an `isinstance` chain in the
    wrong order reports every timeout as a connection failure: same symptom, different
    cause, and a day spent looking at the wrong layer."""
    assert classify(APITimeoutError(request=REQUEST)).status == 504
    assert classify(APIConnectionError(request=REQUEST)).status == 502


def test_a_provider_rate_limit_is_a_503_and_not_a_429():
    """The quota belongs to this service. A 429 tells the caller to send less, which it
    cannot fix and did not cause."""
    failure = classify(status_error(RateLimitError, 429))
    assert failure.status == 503
    assert failure.code == "rate_limited"


def test_retry_after_is_echoed_when_the_provider_sends_one():
    failure = classify(status_error(RateLimitError, 429, **{"retry-after": "12"}))
    assert failure.retry_after == 12


@pytest.mark.parametrize("header", [{}, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}])
def test_retry_after_is_omitted_rather_than_guessed(header):
    """The header may be absent or an HTTP date. A default would be a guess handed to the
    client as a fact, and every client that trusted it would retry into the same wall at
    the same moment."""
    failure = classify(status_error(RateLimitError, 429, **header))
    assert failure.retry_after is None
    assert "retry_after" not in failure_payload(failure)


@pytest.mark.parametrize(
    "cls,status",
    [(AuthenticationError, 401), (PermissionDeniedError, 403), (BadRequestError, 400)],
)
def test_our_own_misconfiguration_is_a_500_and_not_the_upstream_4xx(cls, status):
    """A bad key, a model the account cannot use, a malformed prompt. Passing the 4xx on
    would tell a caller its request was wrong. `scripts/eval.py` burned forty calls on
    exactly this - a model permission that had not propagated."""
    failure = classify(status_error(cls, status))
    assert failure.status == 500
    assert failure.code == "misconfigured"


def test_an_unset_key_is_misconfigured_rather_than_a_mystery():
    """The container checks caught this one. A key that was never configured is a failure
    the service can name before asking the provider anything, and `internal_error` is what
    you report when you do not know - so reporting it here was a lie of omission."""
    failure = classify(MissingAPIKey("no OpenAI API key configured"))
    assert (failure.status, failure.code) == (500, "misconfigured")


def test_an_unset_key_reaches_the_caller_as_a_defined_failure(client, monkeypatch):
    """Registered as its own handler: it is ours, not the SDK's, so it is not an
    `OpenAIError` and would otherwise escape as a 500 with a stack trace."""
    fails_with(monkeypatch, MissingAPIKey("no OpenAI API key configured"))
    response = client.post("/ask", json={"question": "does PlaNet plan?"})
    assert response.status_code == 500
    assert response.json()["code"] == "misconfigured"


def test_a_provider_outage_is_a_bad_gateway():
    assert classify(status_error(InternalServerError, 503)).code == "upstream_error"


def test_anything_unrecognised_still_gets_a_defined_shape():
    """Including an exception class a future SDK version adds."""
    failure = classify(RuntimeError("something new"))
    assert (failure.status, failure.code) == (500, "internal_error")


def test_the_providers_message_never_reaches_the_caller():
    """Upstream bodies quote organisation ids, model names and quota. An error path is the
    least-watched place in a service for those to start appearing."""
    payload = failure_payload(classify(status_error(RateLimitError, 429, LEAKY)))
    assert "org-4f2b91" not in json.dumps(payload)
    assert "gpt-4o-mini" not in json.dumps(payload)


# --- the response -----------------------------------------------------------------


def test_a_rate_limit_becomes_a_503_with_a_retry_after_header(client, monkeypatch):
    fails_with(monkeypatch, status_error(RateLimitError, 429, LEAKY, **{"retry-after": "9"}))
    response = client.post("/ask", json={"question": "does PlaNet plan?"})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "9"
    body = response.json()
    assert body["code"] == "rate_limited"
    assert body["retry_after"] == 9
    # The id a caller quotes, in the body and on the header, and the same one both times.
    assert body["request_id"] == response.headers["x-request-id"]


def test_a_timeout_becomes_a_504(client, monkeypatch):
    fails_with(monkeypatch, APITimeoutError(request=REQUEST))
    response = client.post("/ask", json={"question": "does PlaNet plan?"})
    assert response.status_code == 504
    assert response.json()["code"] == "upstream_timeout"
    assert "retry-after" not in response.headers


def test_a_bad_key_is_our_500_and_says_nothing_about_the_key(client, monkeypatch):
    fails_with(monkeypatch, status_error(AuthenticationError, 401, "Incorrect API key sk-abc"))
    response = client.post("/ask", json={"question": "does PlaNet plan?"})
    assert response.status_code == 500
    assert response.json()["code"] == "misconfigured"
    assert "sk-abc" not in response.text


def test_a_failure_is_one_log_line_with_the_cause_on_it(client, monkeypatch, lines):
    """The point of doing this in a handler rather than a try/except in the endpoint: it
    runs inside the logging middleware, so a failed request still produces exactly one
    line, and that line now says why."""
    fails_with(
        monkeypatch,
        status_error(RateLimitError, 429, LEAKY, **{"x-request-id": "req_upstream_77"}),
    )
    client.post("/ask", json={"question": "does PlaNet plan?"})
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["status"] == 503
    assert row["outcome"] == "error"
    assert row["code"] == "rate_limited"
    assert row["upstream_error"] == "RateLimitError"
    # Their id, for their support desk. Unrecoverable once the exception is gone.
    assert row["upstream_request_id"] == "req_upstream_77"


def test_a_404_is_logged_as_a_client_error_not_a_server_one(client, lines):
    """`outcome` has to distinguish them or a success rate grepped from these lines counts
    every typo'd URL as an outage."""
    client.get("/no-such-route")
    assert json.loads(lines[0])["outcome"] == "client_error"


# --- mid-stream -------------------------------------------------------------------


def test_a_rate_limit_mid_stream_carries_the_same_code_in_the_frame(client, monkeypatch):
    """A stream cannot change its status: 200 went out with the first byte. So the
    taxonomy moves into the frame, and `code` means the same thing on both paths."""

    def explode(question, hits, settings):
        yield "PlaNet plans "
        raise status_error(RateLimitError, 429, LEAKY, **{"retry-after": "5"})

    monkeypatch.setattr(api, "stream_generate", explode)
    response = client.post("/ask/stream", json={"question": "does PlaNet plan?"})
    assert response.status_code == 200

    frames = [block.split("\n") for block in response.text.split("\n\n") if block.strip()]
    events = [(line[0][7:], json.loads(line[1][6:])) for line in frames]
    assert [event for event, _ in events] == ["token", "error"]

    error = events[-1][1]
    assert error["code"] == "rate_limited"
    assert error["retry_after"] == 5
    assert error["partial"] == "PlaNet plans "
    assert error["request_id"]
    assert "org-4f2b91" not in json.dumps(error)
