import logging
from pathlib import Path

import pytest

from arxiv_rag import limits
from arxiv_rag import observability as obs

FIXTURES = Path(__file__).parent / "fixtures"


class ListHandler(logging.Handler):
    """Collects the summary lines.

    `caplog` cannot: `configure_logging` sets `propagate = False` on the request logger,
    so its records never reach the root handler pytest installs - which is the whole point
    of that flag, since propagation would print every line twice, once decorated and once
    as the JSON something downstream is parsing.
    """

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def lines():
    """The JSON summary lines emitted during a test, in order.

    Ask for it *after* any fixture that starts the app: the lifespan calls
    `configure_logging`, which replaces the logger's handlers and would drop this one.
    """
    handler = ListHandler()
    obs.log.addHandler(handler)
    obs.log.setLevel(logging.INFO)
    yield handler.lines
    obs.log.removeHandler(handler)


@pytest.fixture
def arxiv_atom_xml() -> str:
    return (FIXTURES / "arxiv_response.xml").read_text()


@pytest.fixture(autouse=True)
def fresh_limits():
    """Every test starts with an empty budget and a full token bucket.

    Autouse, and worth the bluntness. The limiter is a process singleton keyed by client
    address - correct in production, where addresses differ - and `TestClient` reports the
    same client for every request in the session. Without this, the sixth API call
    anywhere in the suite gets a 429 and thirty tests fail for a reason that has nothing
    to do with what they assert.

    That is not only a test problem. It is the same shape as the ContextVar argument in
    `observability`: process-wide state is invisible until two things share it, and the
    suite is the first place that happens.
    """
    limits.budget.cache_clear()
    limits.buckets.cache_clear()
    yield
    limits.budget.cache_clear()
    limits.buckets.cache_clear()
