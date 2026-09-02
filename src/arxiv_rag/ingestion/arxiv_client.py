"""arXiv API client.

The arXiv API is free and needs no key. It returns Atom XML, which ``feedparser``
turns into nested dicts.

Notice the shape of this module: ``parse_atom_feed`` is pure (string in, models out)
and ``search`` does the network call. That split is deliberate — it means you can test
all the messy parsing logic against a saved fixture with no network, which is exactly
what ``tests/test_arxiv_client.py`` does. Do this everywhere: keep I/O at the edges.

Docs: https://info.arxiv.org/help/api/user-manual.html
"""

import re
import time
from datetime import datetime

import feedparser
import httpx

from arxiv_rag.ingestion.models import Paper

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# arXiv ids look like '2005.11401v4' or (older) 'cs/0501001v1'. We drop the version
# suffix so re-ingesting a revised paper does not create a duplicate.
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")

_last_request_at: float = 0.0


def _throttle(delay_seconds: float) -> None:
    """Block until at least ``delay_seconds`` have passed since the last request.

    arXiv asks for one request every 3 seconds. Ignoring this gets you rate-limited,
    and being a good API citizen is a habit worth having.
    """
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < delay_seconds:
        time.sleep(delay_seconds - elapsed)
    _last_request_at = time.monotonic()


def normalise_arxiv_id(raw_id: str) -> str:
    """'http://arxiv.org/abs/2005.11401v4' -> '2005.11401'."""
    tail = raw_id.rstrip("/").split("/abs/")[-1]
    return _VERSION_SUFFIX_RE.sub("", tail)


def parse_atom_feed(xml: str) -> list[Paper]:
    """Turn an arXiv Atom response into ``Paper`` models.

    TODO(you): implement this.

    Hints, in the order you will hit the problems:

    1. ``feed = feedparser.parse(xml)`` then iterate ``feed.entries``.
    2. Inspect one entry first. Genuinely do this — open a REPL, print
       ``feed.entries[0].keys()``. Guessing at an unfamiliar data structure is how you
       lose an hour.
    3. ``entry.id`` is a full URL. Use ``normalise_arxiv_id``.
    4. **Titles and summaries contain embedded newlines and runs of spaces** from the
       PDF-era formatting. Collapse whitespace: ``" ".join(value.split())``. If you skip
       this, every title in your citations looks broken.
    5. Authors are ``entry.authors``, a list of objects with ``.name``.
    6. ``entry.published`` is an ISO-8601 string with a 'Z'. Python's
       ``datetime.fromisoformat`` handles 'Z' from 3.11 onward. You want a ``date``.
    7. The PDF link is the one in ``entry.links`` whose ``type`` is
       ``application/pdf``. Do not build the URL by string-mangling the abs URL — the
       API tells you what it is, so use what it tells you.
    8. Categories are ``entry.tags``, each with a ``.term``.

    Return an empty list rather than raising if the feed has no entries.
    """
    raise NotImplementedError("Stage 1: implement parse_atom_feed")


def search(query: str, limit: int = 10, delay_seconds: float = 3.0) -> list[Paper]:
    """Search arXiv and return parsed papers.

    TODO(you): implement this.

    Hints:

    - Query params: ``search_query``, ``start``, ``max_results``, and
      ``sortBy=submittedDate`` with ``sortOrder=descending`` so you get recent work.
    - The ``search_query`` syntax is field-prefixed: ``all:"retrieval augmented"``,
      or ``cat:cs.CL AND abs:agent``. Read the user manual linked above; this syntax
      is the whole reason this API is more useful than scraping.
    - Call ``_throttle(delay_seconds)`` before the request.
    - Use ``httpx.get(..., timeout=30)`` and ``response.raise_for_status()``.
    - Then hand ``response.text`` to ``parse_atom_feed``. This function should be about
      eight lines; all the real work is in the parser you already wrote.
    """
    raise NotImplementedError("Stage 1: implement search")


def _unused_import_guard() -> None:  # pragma: no cover
    """Keeps linters quiet about imports you will need once you implement the TODOs."""
    _ = (feedparser, httpx, datetime)
