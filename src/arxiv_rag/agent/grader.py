"""Deciding whether retrieved passages are good enough to answer from.

This is **control flow, not a metric**. A conditional edge reads the verdict: false means
rewrite the query and retrieve again, true means generate. A lenient grader never triggers
a retry and the agent is just the pipeline with extra latency; a strict one retries
constantly and burns money reaching the same answer.

**Measure it before wiring it.** `scripts/grade_eval.py` runs it over the 34 answerable
questions, where it is already known whether the gold chunk was retrieved, and reports how
often the grader agrees. Stage 2's judge looked fine until a known-answer test showed it
was grading one axis twice; a grader that drives a loop deserves the same suspicion.

**The verdict must be structured, not prose.** A branch is reading it. Same discipline as
`Verdict`: a required prefix on the reason makes the contract checkable, and the model has
to name what is missing rather than gesture at it.
"""

import json
import logging

from openai import OpenAI
from pydantic import BaseModel, Field

from arxiv_rag.config import Settings
from arxiv_rag.retrieval.answer import format_context
from arxiv_rag.retrieval.store import SearchHit

log = logging.getLogger(__name__)

GRADER_SYSTEM = """You decide whether a set of retrieved passages is sufficient to answer a \
question about machine-learning research papers.

Answer one question only: could someone holding ONLY these passages give a correct, \
specific answer to what was asked?

Say relevant = true when the passages contain the facts the question asks for, even if \
they also contain unrelated material. Extra noise is fine; the generator can ignore it.

Say relevant = false when:
- the passages are about the topic but do not contain the specific fact asked for;
- they describe a DIFFERENT system than the one asked about. This corpus contains V-JEPA \
and V-JEPA 2, I-JEPA, Sub-JEPA, Var-JEPA, Dreamer and DreamerV2 and DreamerV3. Passages \
about the wrong version are not an answer, however similar they read;
- they would force a guess.

Do NOT reward passages for looking authoritative or for being on-topic. The question is \
narrow: is the answer IN here.

Reply with JSON only:
{"relevant": <true or false>, "missing": "<what a correct answer needs that is absent, or \
empty when relevant is true>"}

The angle brackets mark where you substitute your own verdict. They are placeholders, not \
default values."""


class Grade(BaseModel):
    """One grading verdict. ``missing`` is what a rewrite node would act on."""

    relevant: bool
    missing: str = Field(default="", description="what the passages lack; empty when relevant")


class Grader:
    """Grades retrieved passages against a question."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def grade(self, question: str, hits: list[SearchHit]) -> Grade:
        """Return a verdict for these passages.

        Steps:

        1. No hits at all is trivially irrelevant - return
           ``Grade(relevant=False, missing="nothing was retrieved")`` without calling the
           model. Paying for a judgement on an empty list is pure latency.
        2. ``format_context(hits)`` - the same rendering the generator sees, so the grader
           judges exactly what would be answered from.
        3. ``chat.completions.create`` with ``GRADER_SYSTEM``, ``temperature=0``,
           ``response_format={"type": "json_object"}``, and ``settings.grader_model``.
        4. Parse into ``Grade``.

        On a parse failure, return ``Grade(relevant=True, missing="grader failed: ...")``
        and log it. Note the direction: **failing open**. A grader that cannot be parsed
        should not silently send every question into the retry loop - that would turn one
        broken call into 3x the cost on every question. Failing open degrades to the
        pipeline, which is a known-good baseline.
        """
        # TODO(you): the four steps above.
        raise NotImplementedError


def _client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


_ = json  # used once you implement the TODO
