"""Rewriting a query after the grader says retrieval failed.

Deferred from Stage 3 on purpose. As an unconditional preprocessing step it is an
answer-quality intervention measured on an eval set that cannot resolve answer quality.
Here it fires **only when the grader flags a failure**, and its success is measurable
directly: did the retry surface the gold chunk the first attempt missed?

**The trap this module exists to avoid.** Hybrid retrieval works because BM25 weights rare
tokens - "V-JEPA" is the only reason q030's gold chunk is findable at all (dense ranked it
170th). A rewrite that smooths "What data is V-JEPA trained on?" into "what training data
was used for the video joint-embedding predictive architecture" is *more fluent and much
worse*: it has deleted the one token carrying the signal. So the rewrite is checked, and a
rewrite that drops a key term is rejected in favour of a safe fallback.

Sixth place in this codebase where a model's output is verified rather than trusted.
"""

import json
import logging
import re

from arxiv_rag.config import Settings
from arxiv_rag.llm import get_client as _client
from arxiv_rag.retrieval.bm25 import tokenize

log = logging.getLogger(__name__)

REWRITE_SYSTEM = """You rewrite a search query that failed to find the right passages.

You are given the QUESTION a user asked and what the retrieved passages were MISSING. \
Write one better search query.

Rules:
- Keep every technical term, model name, acronym and hyphenated name from the question \
EXACTLY as written. V-JEPA, I-JEPA, DreamerV3, MDN-RNN, RSSM, CEM. These are rare tokens \
and they are what makes the search work; paraphrasing them away is the single worst thing \
you can do here.
- Add words describing the missing information, taken from MISSING.
- Do not make the query more general. The first attempt already failed at general.
- No question mark, no filler - this is a search query, not a sentence.

Reply with JSON only: {"query": "..."}"""

# Tokens a rewrite must not lose: hyphenated compounds, anything with a digit, and
# capitalised words that are not sentence-initial. Deliberately crude - it only has to
# catch the names that carry retrieval signal, not parse English.
_KEY_TERM_RE = re.compile(r"\b(?:[A-Za-z]+-[A-Za-z0-9-]+|[A-Za-z]*\d[A-Za-z0-9]*|[A-Z]{2,})\b")


def key_terms(text: str) -> set[str]:
    """Rare tokens whose loss would break lexical retrieval. Lowercased.

    ``"What data is V-JEPA trained on?"`` -> ``{"v-jepa"}``
    ``"how does DreamerV3 handle discrete actions"`` -> ``{"dreamerv3"}``
    ``"what is an RSSM"`` -> ``{"rssm"}``

    Ordinary words are not key terms - losing "data" or "trained" costs nothing, because
    dense retrieval handles meaning. Only the tokens BM25 leans on matter.
    """
    return {m.group().lower() for m in _KEY_TERM_RE.finditer(text)}


def preserves_key_terms(original: str, rewritten: str) -> bool:
    """Did the rewrite keep every key term from the original?

    A rewrite that drops one is rejected: measured in Stage 3, dense retrieval put V-JEPA's
    own paper at rank 170 and BM25 put it at 2, entirely on the strength of that one rare
    token.

    **Asymmetric on purpose.** Terms are *detected* in the original and then *searched for*
    in the rewrite, using `bm25.tokenize` - the same tokenizer that will do the lexical
    retrieval this guard exists to protect. Comparing `key_terms(a) <= key_terms(b)` looks
    equivalent and is not: the detector recognises an acronym by its capitals, so
    "RSSM" -> "rssm" reads as a dropped term even though BM25 lowercases both and cannot
    tell them apart. That rejected good rewrites for a difference with no effect on
    retrieval - caught by `test_case_does_not_count_as_dropping_a_term`.

    Using the retriever's own tokenizer is what makes the check mean what it claims: a term
    "survives" exactly when BM25 will still match on it.
    """
    surviving = set(tokenize(rewritten))
    return all(term in surviving for term in key_terms(original))


def rewrite_query(question: str, missing: str, settings: Settings) -> str:
    """Produce a better search query, or fall back to something safe"""
    try:
        response = _client(settings).chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": f"QUESTION: {question}\nMISSING: {missing}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content or "")
        query = str(payload.get("query", "")).strip()
    except Exception as exc:
        # Fails open, like the grader: a degraded retry beats a failed request.
        log.warning("rewrite call failed, falling back: %s", exc)
        return f"{question} {missing}"

    if not query:
        log.warning("rewrite returned an empty query, falling back")
        return f"{question} {missing}"

    # The guard. Counted separately from a failed call, because the rate at which the
    # model paraphrases away a rare token is a number worth knowing.
    if not preserves_key_terms(question, query):
        log.warning("rewrite dropped key term(s), falling back: %r", query)
        return f"{question} {missing}"

    return query
