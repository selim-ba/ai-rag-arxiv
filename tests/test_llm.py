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

from arxiv_rag.config import Settings
from arxiv_rag.llm import get_client


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


def test_nothing_outside_llm_py_constructs_a_client():
    """Structural, and the point of the refactor: one place to configure, one place to
    forget a timeout. A new module reaching for `OpenAI(...)` directly would silently get
    the ten-minute default back."""
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "arxiv_rag"
    offenders = [
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if path.name != "llm.py" and re.search(r"\bOpenAI\s*\(", path.read_text())
    ]
    assert offenders == [], f"construct clients via llm.get_client: {offenders}"
