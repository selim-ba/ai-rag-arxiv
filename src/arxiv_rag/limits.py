"""What stops a public demo costing money: a daily spend cap and a per-IP rate limit.

Built before the URL exists, not after. Every other failure in this project cost time; a
public endpoint with a key behind it costs money, and the bill arrives whether or not
anyone is watching the logs.

**Two different limits, two different meanings, two different status codes.**

`api/errors.py` argues at length that a provider rate limit must NOT be passed to the
caller as 429: the quota is the service's and the caller had no part in it. The per-IP
limit here is the opposite case and gets the opposite answer - **429 is exactly right**,
because this caller really is sending too fast and slowing down really does fix it. The
daily budget is a 503: the service is temporarily unable, and nothing the caller does
changes that before midnight.

**A token bucket, not a fixed window.** A window that resets on the hour lets someone
spend a whole allowance in the two seconds either side of the boundary, which is the
shape of the traffic you are trying to stop.

**Both live in this process.** They do not survive horizontal scaling: two instances mean
two budgets and twice the cap. That is the same limitation `agent/session.py` documents
for conversations, it is the reason to pin the deployment to one instance, and it is why
the provider's own spend cap remains the backstop rather than the ornament.
"""

import logging
import threading
from collections import OrderedDict
from datetime import UTC, date, datetime
from functools import lru_cache
from time import monotonic

from arxiv_rag.config import get_settings
from arxiv_rag.pricing import cost_usd

log = logging.getLogger(__name__)


class DailyBudget:
    """Dollars spent today, and whether that is now too many.

    Denominated in dollars rather than requests because requests are not interchangeable:
    a pipeline answer costs about $0.0005 and an agent answer about twice that, so a
    request count is a budget that means a different amount every day.
    """

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self._spent = 0.0
        self._day = self._today()
        # Requests are served from a threadpool, so two can record at once. A lost update
        # here is a budget that quietly permits more than it says.
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> date:
        # UTC, so the reset does not move twice a year with daylight saving.
        return datetime.now(UTC).date()

    def _roll(self) -> None:
        today = self._today()
        if today != self._day:
            log.info("daily budget: new day, %.4f USD spent yesterday", self._spent)
            self._day, self._spent = today, 0.0

    def record(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Charge one call. Returns what it cost."""
        cost = cost_usd(model, tokens_in, tokens_out)
        with self._lock:
            self._roll()
            self._spent += cost
        return cost

    @property
    def spent_usd(self) -> float:
        with self._lock:
            self._roll()
            return self._spent

    @property
    def exhausted(self) -> bool:
        """Checked BEFORE a request, so the limit is never exceeded by more than the cost
        of the request that crossed it - about half a tenth of a cent."""
        if self.limit_usd <= 0:  # 0 or negative disables the cap, for local runs
            return False
        return self.spent_usd >= self.limit_usd

    def seconds_until_reset(self) -> int:
        """For `Retry-After`. The answer is honest here in a way it cannot be for a
        provider rate limit: this service knows exactly when its own budget rolls over."""
        now = datetime.now(UTC)
        midnight = datetime.combine(now.date(), datetime.min.time(), tzinfo=UTC)
        return int(86400 - (now - midnight).total_seconds())


class TokenBucket:
    """Per-key rate limiting, with a burst allowance.

    `capacity` tokens, refilled at `per_minute / 60` a second. A visitor clicking three
    example questions in a row is not abuse and should not be punished for it; a script
    asking a hundred times a minute is, and runs dry within seconds.
    """

    def __init__(self, per_minute: float, capacity: int, max_keys: int = 4096) -> None:
        self.rate = per_minute / 60.0
        self.capacity = capacity
        self.max_keys = max_keys
        # Bounded and LRU-ordered for the same reason `SessionStore` is: a dictionary
        # keyed by something a stranger chooses is a memory leak with extra steps.
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.capacity <= 0:  # disabled
            return True
        now = monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            permitted = tokens >= 1.0
            if permitted:
                tokens -= 1.0
            self._buckets[key] = (tokens, now)
            self._buckets.move_to_end(key)
            while len(self._buckets) > self.max_keys:
                # Evicting the least recently seen gives them a full bucket back. That is
                # the safe direction: the alternative is evicting someone mid-spend and
                # handing them a fresh allowance, which is what an attacker would want.
                self._buckets.popitem(last=False)
            return permitted

    @property
    def retry_after_seconds(self) -> int:
        """How long until one token exists again. Sent rather than guessed at, which is
        the same standard `api/errors.py` holds the provider's `Retry-After` to."""
        return max(1, int(60.0 / max(self.rate * 60.0, 1e-9)))


def client_key(request) -> str:
    """Who to rate-limit, behind a proxy that rewrites the socket address.

    Cloud Run and Fly both put the real client first in `X-Forwarded-For` and append
    themselves. Taking the leftmost entry is therefore right in production and forgeable
    in principle - a client can send its own header - so this is a demo-grade answer, and
    the daily budget is what actually bounds the damage if someone bothers to rotate it.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@lru_cache(maxsize=1)
def budget() -> DailyBudget:
    """The process's budget.

    A module-level singleton rather than something on `app.state`, because the charging
    happens in `llm._record` - three layers below any request - and threading an object
    down through every call site to reach it would be a worse trade than this.
    """
    return DailyBudget(get_settings().daily_budget_usd)


@lru_cache(maxsize=1)
def buckets() -> TokenBucket:
    settings = get_settings()
    return TokenBucket(settings.rate_limit_per_minute, settings.rate_limit_burst)
