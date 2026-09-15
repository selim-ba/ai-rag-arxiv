"""Deciding what kind of question this is, before doing any work.

Three routes, and the reason there are exactly three is that each one has a tool behind it
and a different failure if it is chosen wrongly:

* ``retrieve``  - a question about the papers, unscoped. The default.
* ``filtered``  - the question is scoped to named papers; search only those.
* ``catalog``   - the question is about the corpus itself, or about a paper that is not in
  it. Answered from `tools.list_indexed_papers` / `tools.fetch_arxiv_metadata`, never from
  passages.

There is deliberately no "answer directly without retrieving" route. The generator is
instructed to use only the passages, so such a route would either never fire or would
break the grounding rule the whole project rests on.

**The router needs the catalog in its prompt, and that is not an optimisation.** Consider
the two questions the eval set pairs on purpose:

    "What learning rate was used to train Dreamer?"   -> filtered (Dreamer IS indexed;
                                                        the value is not, so: refuse)
    "What did TD-MPC2 change relative to TD-MPC?"     -> catalog  (the paper is absent)

Identical surface shape. The only thing separating them is corpus membership, so a router
that cannot see the corpus cannot distinguish them even in principle. An absent value is
not an absent paper, and a system that conflates them tells users it lacks papers it has.

Seventh place in this codebase where a model's output is checked rather than trusted:
`verify_route` turns claims the index contradicts into the route the index supports.
"""

import json
import logging
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from arxiv_rag.agent.tools import IndexedPaper
from arxiv_rag.config import Settings

log = logging.getLogger(__name__)

Route = Literal["retrieve", "filtered", "catalog"]

ROUTER_SYSTEM = """You route a question about a fixed corpus of machine-learning papers \
to one of three handlers. Reply with JSON only.

INDEXED PAPERS - this is the entire corpus. Nothing else is available:
{catalog}

Routes:

- "retrieve": a question about the content of the papers, not scoped to particular ones. \
The default. Choose it when the user has not named which papers to look in - the search \
should discover that, not you.
- "filtered": the question is scoped to one or more papers the user named, by title or by \
acronym. Put their arXiv ids in "arxiv_ids". If the question compares two papers, give \
BOTH ids; dropping one answers half the question.
- "catalog": the question is about the corpus itself - which papers are in it, how many \
there are - or names a paper that is NOT in the list above. No passage answers these.

"catalog" is about MEMBERSHIP, never about ideas. A question about a general concept - \
what a stochastic latent state is, why training collapses, how imagination works - is \
"retrieve", however unspecific it sounds. If the question could be answered by quoting a \
passage, it is not a catalog question.

The distinction that matters most:

    "What learning rate was used to train Dreamer?"  -> filtered. Dreamer IS in the list. \
Whether the passages happen to state the learning rate is not your problem; the search \
will find out and say so.
    "What did TD-MPC2 change relative to TD-MPC?"    -> catalog. TD-MPC2 is NOT in the \
list, so there is nothing to search.

An absent VALUE is not an absent PAPER. If the paper is in the list, route to the papers \
and let the search report what it finds. Only route to catalog when the paper itself is \
missing, or when the question asks about the collection rather than about its content.

Reply with JSON only:
{{"route": "<retrieve or filtered or catalog>", "arxiv_ids": ["<bare ids, filtered only>"], \
"reason": "<one clause>"}}

The angle brackets mark where you substitute your own decision. They are placeholders, \
not default values."""


class RouteDecision(BaseModel):
    """Where a question should go, and why."""

    route: Route = "retrieve"
    arxiv_ids: list[str] = Field(default_factory=list)
    reason: str = ""


def build_router_prompt(papers: list[IndexedPaper]) -> str:
    """Inject the corpus into the prompt. Given to you.

    Roughly a thousand tokens for 49 papers, paid on every routed request. That is the
    price of being able to tell "the value is missing" from "the paper is missing", which
    is not something a model can know about *your* index from training data.
    """
    lines = []
    for paper in papers:
        # Alias first, because it is what a question will say. Measured: with titles
        # alone the router scored 0 of 4 on picking the right paper.
        name = f"{' / '.join(paper.aliases)} - " if paper.aliases else ""
        lines.append(f"  {paper.arxiv_id}  {name}{paper.title}")
    return ROUTER_SYSTEM.format(catalog="\n".join(lines))


def verify_route(decision: RouteDecision, indexed: frozenset[str]) -> RouteDecision:
    """Reconcile the model's route with what the index actually contains. Pure."""
    # retrieve/catalog make no membership claim that needs verification.
    if decision.route != "filtered":
        return RouteDecision(
            route=decision.route,
            arxiv_ids=list(decision.arxiv_ids),
            reason=decision.reason,
        )

    # 1. A filtered search without a filter is just retrieval.
    if not decision.arxiv_ids:
        return RouteDecision(
            route="retrieve",
            arxiv_ids=[],
            reason=f"{decision.reason}; filtered route had no paper ids",
        )

    known = [arxiv_id for arxiv_id in decision.arxiv_ids if arxiv_id in indexed]
    unknown = [arxiv_id for arxiv_id in decision.arxiv_ids if arxiv_id not in indexed]

    # 2. None of the requested papers exist in the index.
    if not known:
        return RouteDecision(
            route="catalog",
            arxiv_ids=list(decision.arxiv_ids),
            reason=f"{decision.reason}; unindexed paper ids: {', '.join(unknown)}",
        )

    # 3. Some requested papers exist: search only those and record what was dropped.
    if unknown:
        return RouteDecision(
            route="filtered",
            arxiv_ids=known,
            reason=f"{decision.reason}; dropped unindexed ids: {', '.join(unknown)}",
        )

    # All ids are valid.
    return RouteDecision(
        route="filtered",
        arxiv_ids=known,
        reason=decision.reason,
    )


def route_question(
    question: str,
    papers: list[IndexedPaper],
    settings: Settings,
) -> RouteDecision:
    """Classify one question. Given to you.

    Fails OPEN to ``retrieve``: a router that cannot answer must not stop the request, and
    ``retrieve`` is the behaviour the system had before a router existed. Same reasoning as
    the grader - a failure in a component that only *chooses* work should degrade to doing
    the usual work, not to doing none.
    """
    indexed = frozenset(p.arxiv_id for p in papers)
    try:
        response = _client(settings).chat.completions.create(
            model=settings.router_model,
            messages=[
                {"role": "system", "content": build_router_prompt(papers)},
                {"role": "user", "content": question},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content or "")
        decision = RouteDecision.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - any transport or parse error
        log.warning("router unavailable, defaulting to retrieve: %s", exc)
        return RouteDecision(route="retrieve", reason=f"router failed: {exc}")
    return verify_route(decision, indexed)


def _client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)
