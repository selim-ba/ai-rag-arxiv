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
import re

from openai import OpenAI
from pydantic import BaseModel, Field

from arxiv_rag.config import Settings
from arxiv_rag.evaluation.quotes import normalise, quote_supported

log = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """You are a strict grader for a retrieval-augmented QA system.

You are given a QUESTION, the PASSAGES that were retrieved, the ANSWER the system \
produced, and a REFERENCE ANSWER written by a human.

Grade two things that are INDEPENDENT of each other. Do them in this order.

STEP 1 - faithful. Look ONLY at the PASSAGES and the ANSWER. Ignore the REFERENCE \
ANSWER completely; it is not evidence about faithfulness.

Work through this procedure in order. Do not stop early.

1. List every factual claim the ANSWER makes. If it makes none - it declined to answer, \
or said the passages were insufficient - then faithful is TRUE and faithful_reason is \
"NO CLAIMS: " plus a few words saying so. An answer that asserts nothing cannot assert \
something unsupported.
2. Take EACH claim in turn, including the ones after the first. Find the span of a \
passage that states it. Most wrong verdicts here come from checking the first claim, \
finding it supported, and never reading the rest of the answer.
3. Check which SYSTEM each passage is about before using it as evidence. A passage \
describing DreamerV2's reward predictor is not evidence about Ha and Schmidhuber's World \
Models, and a passage about V-JEPA 2 is not evidence about V-JEPA. This corpus contains \
World Models, PlaNet, Dreamer/V2/V3, TD-MPC/TD-MPC2, IRIS, I-JEPA, V-JEPA/V-JEPA 2, \
Sub-JEPA, Var-JEPA and LeJEPA. Every passage is retrieved by similarity, so passages \
about a neighbouring system are always present.
4. If a passage CONTRADICTS a claim - states the opposite, or reverses the direction of \
a result - faithful is false. Check this before checking for missing support: a reversed \
claim is worse than an absent one and is easy to miss because its vocabulary matches.
5. If a claim has no span anywhere, faithful is false. An answer that is true about the \
world but absent from the passages is NOT faithful.
6. Otherwise faithful is true.

**The burden of proof is on the failure, and it is discharged with quotes.** To fail an \
answer you must produce two things: the words from the ANSWER that make the claim, and \
either the words from a PASSAGE that contradict them or a statement that no passage \
contains them. If you cannot quote the answer's own words, you have not found a claim - \
you have found something the answer failed to say, which is not a faithfulness failure. \
If you cannot quote the passage words that contradict it, you have not found a \
contradiction. In both cases faithful is TRUE.

These are not failures and must never appear behind a failure prefix: the answer not \
mentioning something; the answer being less complete than the reference; a claim the \
QUESTION raised that the answer never asserted; anything the REFERENCE ANSWER says. If \
the text you are about to write restates what the answer said and a passage agrees with \
it, you have found SUPPORT, not a contradiction - write SUPPORTED and set faithful true.

Your faithful_reason must begin with one of exactly four prefixes, matching the verdict:
- "SUPPORTED: " when faithful is true, followed by the exact words from the passage that \
settled the last claim you checked.
- "NO CLAIMS: " when faithful is true because the answer asserted nothing.
- "CONTRADICTED: " when faithful is false by step 4, formatted exactly as: the ANSWER's \
claim in quotes, then " but passage says ", then the contradicting passage words in \
quotes. Both quotes must be copied text, not paraphrase.
- "UNSUPPORTED: " when faithful is false by step 5, formatted exactly as: the ANSWER's \
claim in quotes, then " - no passage states this". The quote must be words the answer \
actually wrote.

These are NOT faithfulness failures, and none of them may appear in faithful_reason: \
disagreeing with the reference answer; being incomplete; answering about a paper the \
question did not ask about. The last one is a CORRECTNESS failure - an answer that \
reports the wrong passages accurately is faithful and incorrect, and separating those two \
is the entire purpose of having two axes.

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
{"faithful": <true or false>, "faithful_reason": "<SUPPORTED: ... | NO CLAIMS: ... | \
CONTRADICTED: ... | UNSUPPORTED: ...>", "correct": <true or false>, "correct_reason": \
"<OK: ... | CONTRADICTION: ... | WRONG SYSTEM: ...>"}"""


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

# The same forcing function, applied to the axis that never had one. Measured: against a
# blind re-labelling of ten answers, `correct` - which has ordered tests, mandatory
# prefixes and this check - agreed 10/10 (kappa +1.00), while `faithful` - one paragraph,
# no contract - agreed 7/10 at kappa -0.15, worse than guessing at its own base rate. The
# axis with the enforced contract is the one that agrees.
FAITHFUL_PREFIXES = ("SUPPORTED:", "NO CLAIMS:")
UNFAITHFUL_PREFIXES = ("CONTRADICTED:", "UNSUPPORTED:")


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

    faithful_reason = verdict.faithful_reason.strip()
    if verdict.faithful and not faithful_reason.startswith(FAITHFUL_PREFIXES):
        return False
    if not verdict.faithful and not faithful_reason.startswith(UNFAITHFUL_PREFIXES):
        return False
    if "reference answer" in faithful_reason.lower():
        return False
    return True


_QUOTED = re.compile(
    r"[\"'\u2018\u2019\u201c\u201d]([^\"'\u2018\u2019\u201c\u201d]{6,})"
    r"[\"'\u2018\u2019\u201c\u201d]"
)


def quoted_spans(reason: str) -> list[str]:
    """Every quoted run of six or more characters in a reason. Straight and curly quotes."""
    return [m.group(1).strip() for m in _QUOTED.finditer(reason)]


def spans_overlap(spans: list[str]) -> bool:
    """Do two of these quoted spans say the same thing? Pure, and definitional.

    A CONTRADICTED verdict quotes the answer's claim and the passage words that contradict
    it. If one of those quotes contains the other, they are not two statements in conflict;
    they are one statement quoted twice. Nothing contradicts itself.

    Measured, in run after run, both verdicts fully substantiated:

        q013  "PlaNet does not train an explicit policy network"
              but passage says "no explicit policy or value function network is used"

        q017  "the agent does not learn without it"
              but passage says "the stochastic component is even more important -
                                the agent does not learn without it"

    q017's passage span contains the answer span verbatim. Roughly a fifth of the
    UNFAITHFUL list - the list a human is asked to read by hand - was this.

    **Why this is not a fourth round of tuning.** The rule was declined once, on the
    grounds that a fourth pass at the rubric would be fitting to ten labelled answers. The
    distinction that changed the decision: "raise a threshold until agreement improves"
    depends on the sample, while "two overlapping quotes do not contradict each other" is
    true on any dataset, needs no labels, and would have been right before q013 was ever
    seen. It can be stated without reference to the data - which is the test for whether a
    rule is a definition or a fit.

    Comparison is on `normalise`d text, so ligatures, wrapping and punctuation do not let
    the same sentence pass as two different ones.
    """
    cleaned = [normalise(span) for span in spans]
    cleaned = [c for c in cleaned if c]
    for i, a in enumerate(cleaned):
        for b in cleaned[i + 1 :]:
            if a in b or b in a:
                return True
    return False


def failure_is_substantiated(verdict: Verdict, answer: str, context: str) -> bool:
    """Does a ``faithful=false`` verdict quote text that actually exists?

    The prefix contract made the judge WORSE: agreement with a blind re-labelling fell
    from 7/10 to 5/10 while contract violations stayed at 0/31. It learned the format and
    kept the reasoning. Measured examples, all prefix-compliant:

    - q013 wrote ``CONTRADICTED: PlaNet does not train an explicit policy network...``,
      which restates the ANSWER and which passage 2107.08241::17 states almost verbatim;
    - q005 wrote ``UNSUPPORTED: The claim that the world model is updated...`` about a
      refusal that made no claim at all - the claim came from the QUESTION;
    - q001 quoted the REFERENCE answer as the contradiction.

    All three share one shape: **the failure names a claim the answer never made, or a
    contradiction no passage states.** That is checkable. A failing verdict must quote the
    answer's own words, and a CONTRADICTED verdict must also quote the passage words that
    contradict them; `quote_supported` (Stage 2, written when the gold auditor invented
    evidence) checks both against the real text.

    **This does not flip the verdict.** The judge's verdict IS the measurement, and
    silently repairing it would launder a failure into a score - the distinction drawn in
    ``verify_evidence``, where a grader's verdict is a *claim* driving control flow and
    flipping it costs only a retry. Here an unsubstantiated failure is counted as a
    contract violation, retried once, and then kept and reported.
    """
    if verdict.faithful:
        return True
    spans = quoted_spans(verdict.faithful_reason)
    if not spans:
        return False
    # At least one quote must be the answer's own words: you cannot fail an answer for a
    # claim it did not make.
    if not any(quote_supported(span, answer) for span in spans):
        return False
    if verdict.faithful_reason.strip().startswith("CONTRADICTED:"):
        # ...and the contradicting words must be in the passages, not invented or lifted
        # from the reference answer.
        if not any(quote_supported(span, context) for span in spans):
            return False
        # ...and the two quotes must not be the same statement. Nothing contradicts itself.
        return not spans_overlap(spans)
    return True


RETRY_NUDGE = (
    "Your previous reply broke the output contract. correct_reason must start with "
    "CONTRADICTION: or WRONG SYSTEM: when correct is false, and OK: when it is true. "
    "faithful_reason must start with CONTRADICTED: or UNSUPPORTED: when faithful is "
    "false, and SUPPORTED: or NO CLAIMS: when it is true. faithful_reason must argue "
    "from the PASSAGES only, never from the REFERENCE ANSWER. Re-grade and reply again "
    "in the same JSON shape. If faithful is false you must QUOTE the answer's own words "
    "that make the claim, and for CONTRADICTED also quote the passage words that "
    "contradict them. If you cannot quote both, you have not found a failure and "
    "faithful is true."
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

    def acceptable(v: Verdict | None) -> bool:
        return (
            v is not None
            and verdict_is_consistent(v)
            and failure_is_substantiated(v, answer, context)
        )

    verdict = _grade(messages, settings)
    if acceptable(verdict):
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
    retried.inconsistent = not acceptable(retried)
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
