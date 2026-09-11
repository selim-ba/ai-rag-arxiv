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

Grade two things that are INDEPENDENT of each other. Do them in this order.

STEP 1 - faithful. Look ONLY at the PASSAGES and the ANSWER. Ignore the REFERENCE \
ANSWER completely; it is not evidence about faithfulness. For each factual claim in the \
ANSWER, find the span of a passage that states it. faithful is true only if every claim \
has such a span. An answer that is true about the world but absent from the passages is \
NOT faithful. In faithful_reason, quote a few words from the passage that settled it, \
or name the claim you could not find.

STEP 2 - correct. Decide whether the ANSWER answers THE QUESTION. The REFERENCE ANSWER \
is ground truth for what a right answer contains, but it is deliberately denser than \
the reply the system is asked for, and it often carries supporting detail the question \
did not ask for.

Apply this test, in order:
1. Does the ANSWER contradict the REFERENCE ANSWER on a point of fact? If yes, correct \
is false. Name the contradiction: quote the ANSWER's claim and the REFERENCE's claim.
2. Is the ANSWER about a different system than the question asked about? If yes, \
correct is false. Name both systems.
3. Otherwise correct is TRUE.

Your correct_reason must begin with one of exactly three prefixes, and the prefix must \
match the verdict:
- "OK: " when correct is true, followed by one sentence saying what the answer got right.
- "CONTRADICTION: " when correct is false by test 1, followed by both quoted claims.
- "WRONG SYSTEM: " when correct is false by test 2, followed by both system names.
There is no other way to fail. If you find yourself wanting to write correct_reason \
without one of these prefixes, the verdict is OK.

If you cannot complete sentence 1 or 2 with specific quoted text, you have not found a \
failure, and correct is true. "Does not mention X" is not a failure. "Omits a key \
detail" is not a failure. "Less complete than the reference" is not a failure. The \
system is instructed to answer in three or four sentences; brevity is the specification, \
not a defect.

The two verdicts often disagree, and that disagreement is the useful signal:
- faithful=true, correct=false: the system reported its passages accurately, but they \
were the wrong passages. A retrieval failure, not a generation failure.
- faithful=false, correct=true: the system answered from memory and got lucky. The most \
dangerous case.

Never justify faithful by referring to the REFERENCE ANSWER. If your faithful_reason \
mentions the reference answer, you have graded the wrong thing.

Reply with JSON only, no prose, with exactly these four keys. The angle brackets below \
mark where you substitute your own verdict - they are placeholders, NOT default values, \
and both booleans are genuinely independent:
{"faithful": <true or false>, "faithful_reason": "<quote from a passage, or the claim \
you could not find - never a reference to the REFERENCE ANSWER>", "correct": <true or \
false>, "correct_reason": "<OK: ... | CONTRADICTION: ... | WRONG SYSTEM: ...>"}"""


class Verdict(BaseModel):
    """One graded answer.

    Two verdicts, two separate reasons. The separate reason fields are not decoration:
    asking for one combined ``reason`` let the judge write a single sentence about the
    reference answer and staple it to both verdicts, which collapsed the two axes into
    one. Measured: faithful and correct agreed on every question until this was split.
    """

    faithful: bool
    correct: bool
    faithful_reason: str = Field(default="", description="evidence from the PASSAGES only")
    correct_reason: str = Field(default="", description="comparison with the reference")
    inconsistent: bool = Field(
        default=False,
        description="judge broke its own output contract twice; treat this verdict as unreliable",
    )


class RefusalRecord(BaseModel):
    """What the system did on one question, versus what it should have done."""

    question_id: str
    should_answer: bool = Field(description="the 'answerable' field from questions.jsonl")
    did_answer: bool = Field(description="not Answer.refused")


OK_PREFIX = "OK:"
FAILURE_PREFIXES = ("CONTRADICTION:", "WRONG SYSTEM:")


def verdict_is_consistent(verdict: Verdict) -> bool:
    """Did the judge follow its own rules?

    The prompt sets three invariants. Prose cannot be checked; a required prefix can, so
    the prompt now demands one and this function verifies it:

    - ``correct`` is true  => ``correct_reason`` starts with ``OK:``
    - ``correct`` is false => ``correct_reason`` starts with ``CONTRADICTION:`` or
      ``WRONG SYSTEM:``
    - ``faithful_reason`` never mentions the reference answer, because faithfulness is
      judged against the passages alone

    Measured before this existed: q018 and q023 came back with ``correct: false`` and
    ``correct_reason: "no contradiction found"`` - a verdict that contradicts the rule
    that produced it. About 7% of graded answers.

    What this function must NOT do is *fix* anything. An inconsistent verdict is a signal
    that the judge is unreliable on that question, and quietly flipping the boolean would
    delete the signal and inflate the score. ``judge_answer`` retries once; a verdict that
    fails twice is kept as-is and counted.
    """
    reason = verdict.correct_reason.strip()
    if verdict.correct and not reason.startswith(OK_PREFIX):
        return False
    if not verdict.correct and not reason.startswith(FAILURE_PREFIXES):
        return False
    if "reference answer" in verdict.faithful_reason.lower():
        return False
    return True


RETRY_NUDGE = (
    "Your previous reply broke the output contract: when correct is false, "
    "correct_reason must start with CONTRADICTION: or WRONG SYSTEM:; when correct is "
    "true it must start with OK:; and faithful_reason must argue from the PASSAGES "
    "only, never from the REFERENCE ANSWER. Re-grade and reply again in the same JSON "
    "shape."
)


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

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    verdict = _grade(messages, settings)
    if verdict is not None and verdict_is_consistent(verdict):
        return verdict

    # One retry, with the contract restated. Not a loop: a judge that breaks the contract
    # twice is telling you something about this question, and the useful response is to
    # record that rather than to keep paying for rerolls until it agrees with itself.
    if verdict is not None:
        log.warning("judge broke its output contract, retrying: %r", verdict.correct_reason[:80])
        messages += [
            {"role": "assistant", "content": verdict.model_dump_json()},
            {"role": "user", "content": RETRY_NUDGE},
        ]

    retried = _grade(messages, settings)
    if retried is None:
        return Verdict(
            faithful=False,
            correct=False,
            faithful_reason="judge failed to parse twice",
            correct_reason="judge failed to parse twice",
            inconsistent=True,
        )
    retried.inconsistent = not verdict_is_consistent(retried)
    return retried


def _grade(messages: list[dict], settings: Settings) -> Verdict | None:
    """One judge call. ``None`` when the reply could not be parsed."""
    response = _client(settings).chat.completions.create(
        model=settings.judge_model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        return Verdict.model_validate_json(response.choices[0].message.content or "")
    except Exception as exc:
        log.error("judge failed to parse: %s", exc)
        return None


def _client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)
