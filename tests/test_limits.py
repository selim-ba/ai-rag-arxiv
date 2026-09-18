"""Stage 7 - the two limits that stand between a demo and a bill. No network.

The rate limit is the row a definition of done usually loses, and it is the one that
matters: everything else in this stage fails by costing time, this one fails by costing
money while nobody is looking.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from arxiv_rag import limits
from arxiv_rag.api import main as api
from arxiv_rag.api.main import app, get_graph, get_retriever
from arxiv_rag.config import get_settings
from arxiv_rag.limits import DailyBudget, TokenBucket, client_key
from arxiv_rag.retrieval.answer import Answer
from tests.test_errors import CHUNKS, FakeGraph, FakeRetriever

# -- the budget --------------------------------------------------------------------


def test_spending_accumulates_in_dollars_not_requests():
    """A pipeline answer costs about $0.0005 and an agent answer about twice that, so a
    request count is a budget that means a different amount every day."""
    budget = DailyBudget(limit_usd=1.0)
    budget.record("gpt-4o-mini", 3000, 150)
    budget.record("gpt-4o-mini", 3000, 150)
    assert budget.spent_usd == pytest.approx(2 * 0.00054, abs=1e-5)


def test_the_cap_trips_at_the_limit():
    budget = DailyBudget(limit_usd=0.001)
    assert not budget.exhausted
    budget.record("gpt-4o-mini", 3000, 150)  # ~0.00054
    assert not budget.exhausted
    budget.record("gpt-4o-mini", 3000, 150)  # ~0.00108, over
    assert budget.exhausted


def test_a_zero_limit_means_unlimited_for_local_runs():
    budget = DailyBudget(limit_usd=0)
    budget.record("gpt-4o", 1_000_000, 1_000_000)
    assert not budget.exhausted


def test_the_budget_rolls_over_at_utc_midnight(monkeypatch):
    """UTC so the reset does not move twice a year with daylight saving."""
    budget = DailyBudget(limit_usd=0.0001)
    budget.record("gpt-4o-mini", 3000, 150)
    assert budget.exhausted

    tomorrow = datetime.now(UTC) + timedelta(days=1)
    monkeypatch.setattr(DailyBudget, "_today", staticmethod(lambda: tomorrow.date()))
    assert not budget.exhausted
    assert budget.spent_usd == 0.0


def test_retry_after_is_the_time_until_that_rollover():
    """Exact, not guessed. The service knows when its own day ends, which is more than it
    can say about a provider's rate limit."""
    seconds = DailyBudget(limit_usd=1.0).seconds_until_reset()
    assert 0 < seconds <= 86400


# -- the bucket --------------------------------------------------------------------


def test_a_burst_of_clicks_is_allowed():
    """Three example questions in a row is a visitor reading the page, not an attack."""
    bucket = TokenBucket(per_minute=10, capacity=5)
    assert all(bucket.allow("1.2.3.4") for _ in range(5))


def test_and_then_it_says_no():
    bucket = TokenBucket(per_minute=10, capacity=5)
    for _ in range(5):
        bucket.allow("1.2.3.4")
    assert not bucket.allow("1.2.3.4")


def test_tokens_come_back_over_time(monkeypatch):
    """A fixed window would let someone spend a whole allowance either side of the reset,
    which is the shape of the traffic being stopped."""
    clock = {"now": 1000.0}
    monkeypatch.setattr("arxiv_rag.limits.monotonic", lambda: clock["now"])
    bucket = TokenBucket(per_minute=60, capacity=2)  # one token a second

    assert bucket.allow("ip") and bucket.allow("ip")
    assert not bucket.allow("ip")
    clock["now"] += 1.0
    assert bucket.allow("ip")


def test_one_client_cannot_exhaust_another():
    bucket = TokenBucket(per_minute=10, capacity=2)
    bucket.allow("noisy")
    bucket.allow("noisy")
    assert not bucket.allow("noisy")
    assert bucket.allow("quiet")


def test_the_bucket_table_is_bounded():
    """A dictionary keyed by something a stranger chooses is a memory leak with extra
    steps - the same argument `SessionStore` makes for conversations."""
    bucket = TokenBucket(per_minute=60, capacity=1, max_keys=10)
    for i in range(50):
        bucket.allow(f"ip-{i}")
    assert len(bucket._buckets) <= 10


def test_the_client_behind_a_proxy_is_the_first_forwarded_address():
    """Cloud Run and Fly put the real client first and append themselves."""

    class Request:
        headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}
        client = type("C", (), {"host": "10.0.0.1"})()

    assert client_key(Request()) == "203.0.113.7"


def test_without_a_proxy_the_socket_address_is_used():
    class Request:
        headers = {}
        client = type("C", (), {"host": "127.0.0.1"})()

    assert client_key(Request()) == "127.0.0.1"


# -- through the API ---------------------------------------------------------------


def reset_limits() -> None:
    """The budget and the buckets are process singletons by design - the charging happens
    three layers below any request. A test that inherits another test's spending is a test
    that passes in isolation only, so each one starts from an empty cache."""
    get_settings.cache_clear()
    limits.budget.cache_clear()
    limits.buckets.cache_clear()


@pytest.fixture
def client():
    app.dependency_overrides[get_retriever] = FakeRetriever
    app.dependency_overrides[get_graph] = FakeGraph
    reset_limits()
    with TestClient(app) as c:
        c.app.state.chunks_by_id = {ch.chunk_id: ch for ch in CHUNKS}
        yield c
    app.dependency_overrides.clear()
    reset_limits()


def answered(question, retriever, settings, k=None, chunk_filter=None):
    return Answer(
        question=question,
        text="PlaNet plans with CEM [1811.04551].",
        citations=["1811.04551"],
        refused=False,
        retrieved_ids=[CHUNKS[0].chunk_id],
    )


def test_an_exhausted_budget_refuses_before_spending_anything(client, monkeypatch):
    """The point of checking first: a refusal costs no retrieval, no model call, no
    tokens. So the cap can only be exceeded by the one request that crossed it."""
    called = []

    def must_not_run(*args, **kwargs):
        called.append(1)
        raise AssertionError("the model was called after the budget was exhausted")

    monkeypatch.setattr(api, "answer_question", must_not_run)
    limits.budget().record("gpt-4o", 1_000_000, 1_000_000)  # far over any sane cap

    response = client.post("/ask", json={"question": "does PlaNet plan?"})
    assert response.status_code == 503
    assert response.json()["code"] == "budget_exhausted"
    assert int(response.headers["retry-after"]) > 0
    assert called == []


def test_too_many_requests_from_one_client_is_a_429(client, monkeypatch):
    """The opposite answer to the provider's 429, and deliberately so: this caller really
    is sending too fast, and slowing down really does fix it."""
    monkeypatch.setattr(api, "answer_question", answered)
    # Set through the real setting rather than by patching the object: this also proves
    # the knob is wired, which is the half of a limiter people forget to check.
    monkeypatch.setenv("PT_RATE_LIMIT_BURST", "2")
    reset_limits()

    codes = [
        client.post("/ask", json={"question": "does PlaNet plan?"}).status_code for _ in range(4)
    ]
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:]


def test_health_reports_the_budget_so_a_page_can_ask_first(client):
    """A demo that lets you type, waits six seconds and then says "no budget" is worse
    than one that tells you up front and shows the recorded answers instead."""
    body = client.get("/health").json()
    assert "budget_spent_usd" in body
    assert body["budget_exhausted"] is False
