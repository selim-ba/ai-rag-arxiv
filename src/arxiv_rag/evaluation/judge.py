"""LLM-as-judge: scoring answers that have no single correct string.

``metrics.py`` scores retrieval, and it can be exact because retrieval has a right
answer: the gold chunk is either in the top k or it is not. Generation has no such
luxury. "The RNN outputs a mixture of Gaussians" and "the memory model predicts a
probability density over the next latent, parameterised as a Gaussian mixture" are the
same answer, and no string comparison will ever agree.

So you grade with a model, on the two axes that actually matter for RAG:

- **Faithfulness** - is every claim in the answer supported by the passages that were
  retrieved? This is the hallucination metric. Note that it is judged against the
  *context*, not against the reference answer: an answer can be true about the world
  and still unfaithful, and unfaithful is the failure mode you can actually fix.
- **Correctness** - does the answer say what the reference answer says? This is judged
  against your hand-written ``reference_answer`` field, and it catches the opposite
  failure: faithfully reporting the wrong passage.

Refusal behaviour is deliberately *not* judged by a model. Whether the system refused
is a boolean you already have from ``Answer.refused``, and whether it should have is a
boolean you already have from ``questions.jsonl``. Comparing two booleans with a
language model would be an expensive way to compute ``==``.

A word of caution you should be able to say out loud in an interview: the judge is a
model, so the judge is fallible, and a judge built from the same family as the
generator tends to like its own style. Treat these numbers as a way to compare *your
own* runs against each other, not as absolute truth. Spot-check the disagreements by
hand — that is what ``reason`` is for.
"""

import logging

from openai import OpenAI
from pydantic import BaseModel, Field

from arxiv_rag.config import Settings

log = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """You are a strict grader for a retrieval-augmented QA system.

You are given a QUESTION, the PASSAGES that were retrieved, the ANSWER the system \
produced, and a REFERENCE ANSWER written by a human.

Grade two things independently:

- faithful: true only if every factual claim in the ANSWER is supported by the \
PASSAGES. An answer that is true in the world but not present in the passages is NOT \
faithful. An answer that adds a plausible detail nobody wrote down is NOT faithful.
- correct: true only if the ANSWER conveys the substance of the REFERENCE ANSWER. \
Different wording is fine. Missing the main point, or contradicting it, is not.

Reply with JSON only, no prose, in exactly this shape:
{"faithful": true, "correct": false, "reason": "one sentence"}"""


class Verdict(BaseModel):
    """One graded answer."""

    faithful: bool
    correct: bool
    reason: str = ""


class RefusalRecord(BaseModel):
    """What the system did on one question, versus what it should have done."""

    question_id: str
    should_answer: bool = Field(description="the 'answerable' field from questions.jsonl")
    did_answer: bool = Field(description="not Answer.refused")


# -- pure function -------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float:
    """Fraction, or 0.0 when the denominator is empty. Never raises."""
    return numerator / denominator if denominator else 0.0


def refusal_scores(records: list[RefusalRecord]) -> dict[str, float]:
    """Score refusal behaviour. No network — this is a 2x2 contingency table.

    - ``answer_rate``: of the questions that *should* be answered, the fraction that
      were. This is recall on the answerable set.
    - ``refusal_rate``: of the questions that should *not* be answered, the fraction
      that were correctly refused. Hallucination avoidance.
    - ``false_refusal_rate``: of the questions that should be answered, the fraction
      wrongly refused. The cost of being cautious, and the number that goes up when you
      over-tighten the prompt.
    - ``accuracy``: fraction of all records where behaviour matched intent.
    """
    answerable = [r for r in records if r.should_answer]
    unanswerable = [r for r in records if not r.should_answer]

    return {
        "answer_rate": _rate(sum(r.did_answer for r in answerable), len(answerable)),
        "refusal_rate": _rate(sum(not r.did_answer for r in unanswerable), len(unanswerable)),
        "false_refusal_rate": _rate(sum(not r.did_answer for r in answerable), len(answerable)),
        "accuracy": _rate(sum(r.did_answer == r.should_answer for r in records), len(records)),
    }


# -- the thin network layer ----------------------------------------------------------


def judge_answer(
    question: str,
    context: str,
    answer: str,
    reference_answer: str,
    settings: Settings,
) -> Verdict:
    """Ask the model to grade one answer. Returns a ``Verdict``.

    The parse is wrapped, the API call is not: a malformed reply costs one question,
    but a rate-limit still ends the run. That is why ``scripts/eval.py`` writes each
    verdict to disk as it arrives instead of accumulating them in a list.
    """
    user_message = (
        f"QUESTION:\n{question}\n\n"
        f"PASSAGES:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"REFERENCE ANSWER:\n{reference_answer}"
    )

    response = _client(settings).chat.completions.create(
        model=settings.judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        return Verdict.model_validate_json(response.choices[0].message.content or "")
    except Exception as exc:
        log.error("judge failed to parse: %s", exc)
        return Verdict(faithful=False, correct=False, reason=f"judge failed: {exc}")


def _client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)
