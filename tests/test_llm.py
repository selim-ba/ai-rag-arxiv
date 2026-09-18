"""Stage 5 - the shared OpenAI client. No network: nothing here makes a call.

Built after reading `openai._constants` rather than after guessing:

    DEFAULT_TIMEOUT     = Timeout(timeout=600, connect=5.0)
    DEFAULT_MAX_RETRIES = 2

Nothing overrode either, so one hung call could block a request for ten minutes and an
agent request makes up to three calls. The fix is a bound, not another retry layer - the
SDK already retries twice with exponential backoff and honours Retry-After.
"""

import pathlib
import re

import pytest

from arxiv_rag.config import Settings
from arxiv_rag.llm import MissingAPIKey, chat, get_client
from arxiv_rag.observability import start_request


def test_the_same_settings_give_the_same_client():
    """Eight modules used to build one client each, per call. Every `OpenAI()` builds an
    httpx client with its own connection pool, so that was a fresh pool and TLS handshake
    on every model call."""
    settings = Settings(openai_api_key="k")
    assert get_client(settings) is get_client(settings)


def test_a_different_timeout_gives_a_different_client():
    a = get_client(Settings(openai_api_key="k", openai_timeout_seconds=10))
    b = get_client(Settings(openai_api_key="k", openai_timeout_seconds=20))
    assert a is not b


def test_a_different_key_gives_a_different_client():
    """Caching on the key matters: two keys sharing a client would send one tenant's
    requests with another's credentials."""
    a = get_client(Settings(openai_api_key="one"))
    b = get_client(Settings(openai_api_key="two"))
    assert a is not b


def test_the_timeout_is_ours_and_not_the_sdk_default():
    """600 seconds is the SDK default and the defect this replaces."""
    client = get_client(Settings(openai_api_key="k", openai_timeout_seconds=30))
    assert client.timeout == 30
    assert client.timeout != 600


def test_retries_are_the_sdks_own_count_not_a_second_layer():
    client = get_client(Settings(openai_api_key="k"))
    assert client.max_retries == 2


def test_retries_can_be_switched_off():
    client = get_client(Settings(openai_api_key="k", openai_max_retries=0))
    assert client.max_retries == 0


ROOT = pathlib.Path(__file__).resolve().parents[1]


def sources() -> list[pathlib.Path]:
    """Every module in the project, `llm.py` excepted - it is the seam, not a user of it.

    `scripts/` is included deliberately. It was not, and `scripts/gold_audit.py` was
    quietly building its own client: no timeout, no pinned retry count, and no token
    accounting. A structural test that only looks where you remember to look finds what
    you already knew."""
    files = list((ROOT / "src" / "arxiv_rag").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    return [f for f in files if f.name != "llm.py"]


def test_nothing_outside_llm_py_constructs_a_client():
    """One place to configure, one place to forget a timeout. A module reaching for
    `OpenAI(...)` directly would silently get the ten-minute default back."""
    offenders = [
        f.relative_to(ROOT).as_posix()
        for f in sources()
        if re.search(r"\bOpenAI\s*\(", f.read_text())
    ]
    assert offenders == [], f"construct clients via llm.get_client: {offenders}"


def test_nothing_outside_llm_py_calls_the_provider_directly():
    """The same argument, for the token accounting. A call that bypasses `llm.chat` costs
    money nobody counted, which is the one thing a daily budget cannot survive."""
    offenders = [
        f.relative_to(ROOT).as_posix()
        for f in sources()
        if re.search(r"\.(chat\.completions|embeddings)\.create\s*\(", f.read_text())
    ]
    assert offenders == [], f"call the provider via llm.chat / llm.embed: {offenders}"


def test_no_key_is_our_error_raised_before_the_sdk_is_touched():
    """Found by the container checks, not by a unit test: with no key the SDK raises a
    bare `OpenAIError` from its CONSTRUCTOR - no request attempted, nothing in its typed
    hierarchy - which classified as `internal_error` and told an operator nothing.

    Asked as a configuration question instead. `/health` has always reported whether a key
    is set; this is the same question, on the path that needs the answer."""
    with pytest.raises(MissingAPIKey):
        get_client(Settings(openai_api_key=""))


class FakeCompletions:
    """Records what `chat()` sent, and hands back something with a usage object."""

    def __init__(self, streamed=False):
        self.kwargs = None
        self.streamed = streamed

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.streamed:
            return iter([Chunk(None), Chunk(Usage(1000, 50))])
        return Response(Usage(1000, 50))


class Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class Response:
    def __init__(self, usage):
        self.usage = usage


class Chunk:
    def __init__(self, usage):
        self.usage = usage


class FakeClient:
    def __init__(self, streamed=False):
        self.completions = FakeCompletions(streamed)
        self.chat = type("Chat", (), {"completions": self.completions})()
        self.embeddings = self.completions


def test_chat_counts_the_tokens_it_used(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr("arxiv_rag.llm.get_client", lambda settings: fake)
    stats = start_request("t1")
    chat(Settings(openai_api_key="k"), model="gpt-4o-mini", messages=[])
    assert (stats.tokens_in, stats.tokens_out) == (1000, 50)
    assert stats.cost_usd == pytest.approx(1000 * 0.15 / 1e6 + 50 * 0.60 / 1e6)


def test_a_streamed_call_asks_for_its_usage(monkeypatch):
    """A streamed completion reports nothing unless `stream_options` asks, and the
    streaming path is the one a user actually hits - so without this the budget would
    undercount exactly the traffic it exists to cap."""
    fake = FakeClient(streamed=True)
    monkeypatch.setattr("arxiv_rag.llm.get_client", lambda settings: fake)
    stats = start_request("t2")

    chunks = list(chat(Settings(openai_api_key="k"), model="gpt-4o-mini", messages=[], stream=True))

    assert fake.completions.kwargs["stream_options"] == {"include_usage": True}
    assert len(chunks) == 2  # the usage chunk is passed through, not swallowed
    assert (stats.tokens_in, stats.tokens_out) == (1000, 50)


def test_usage_is_counted_once_per_call_not_once_per_chunk(monkeypatch):
    fake = FakeClient(streamed=True)
    monkeypatch.setattr("arxiv_rag.llm.get_client", lambda settings: fake)
    stats = start_request("t3")
    list(chat(Settings(openai_api_key="k"), model="gpt-4o-mini", messages=[], stream=True))
    assert stats.tokens_in == 1000
