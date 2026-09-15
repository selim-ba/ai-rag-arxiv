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

ARXIV_API_URL = "https://export.arxiv.org/api/query"

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
    """'http://arxiv.org/abs/2005.11401v4' -> '2005.11401'.

    Also strips a leading ``arXiv:`` prefix, because that is the spelling this codebase
    itself emits: ``format_context`` renders every passage header as
    ``[P1] arXiv:2404.08471 | title | Section: ...``. A model reading that context and
    then calling a tool with a paper id will write it back the way it just saw it, and an
    id that fails to normalise silently becomes an "unknown paper".
    """
    tail = raw_id.strip().rstrip("/").split("/abs/")[-1]
    if tail.lower().startswith("arxiv:"):
        tail = tail[len("arxiv:") :]
    return _VERSION_SUFFIX_RE.sub("", tail)


def parse_atom_feed(xml: str) -> list[Paper]:
    """Parse an arXiv Atom response into Paper models. An empty feed gives []."""
    result = []
    feed = feedparser.parse(xml)
    entries = feed.entries

    for entry in entries:
        paper = Paper(
            arxiv_id=normalise_arxiv_id(entry.id),
            title=" ".join(entry.title.split()),
            authors=[author.name for author in entry.authors],
            abstract=" ".join(entry.summary.split()),
            published=datetime.fromisoformat(entry.published.replace("Z", "+00:00")).date(),
            categories=[tag.term for tag in entry.tags],
            pdf_url=next(link.href for link in entry.links if link.type == "application/pdf"),
        )

        result.append(paper)

    return result


def search(query: str, limit: int = 10, delay_seconds: float = 3.0) -> list[Paper]:
    """Search arXiv and return parsed papers."""

    _throttle(delay_seconds)

    params = {
        "search_query": query,
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    response = httpx.get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()  # 4xx/5xx becomes an exception, not a silently empty feed

    return parse_atom_feed(response.text)


_ID_BATCH = 50


def fetch_by_ids(ids: list[str], delay_seconds: float = 3.0) -> list[Paper]:
    """Fetch specific papers by arXiv id.

    Same endpoint as ``search``, addressed by ``id_list`` instead of ``search_query``.
    This is what makes a curated corpus reproducible: the index is defined by a list of
    ids in a file rather than by whatever a query happened to rank highly that day.

    Requests are batched because a URL has a length limit and arXiv has a rate limit.
    """
    papers: list[Paper] = []
    for start in range(0, len(ids), _ID_BATCH):
        batch = ids[start : start + _ID_BATCH]
        _throttle(delay_seconds)
        params = {"id_list": ",".join(batch), "max_results": len(batch)}
        response = httpx.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()
        papers.extend(parse_atom_feed(response.text))
    return papers


def _unused_import_guard() -> None:  # pragma: no cover
    """Keeps linters quiet about imports you will need once you implement the TODOs."""
    _ = (feedparser, httpx, datetime)
