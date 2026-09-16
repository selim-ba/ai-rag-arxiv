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

import logging

from pydantic import BaseModel, Field

from arxiv_rag.config import Settings
from arxiv_rag.evaluation.quotes import quote_supported
from arxiv_rag.llm import get_client as _client
from arxiv_rag.retrieval.answer import format_context
from arxiv_rag.retrieval.store import SearchHit

log = logging.getLogger(__name__)

GRADER_SYSTEM = """You decide whether a set of retrieved passages is sufficient to answer a \
question about machine-learning research papers.

Answer one question only: could someone holding ONLY these passages give a correct, \
specific answer to what was asked?

**You must point at the answer, not assert that it is there.** When you say relevant, quote \
the words from the passages that contain the answer - copied exactly, at least six \
consecutive words. The quote is checked against the passages. If you cannot find words to \
quote, the answer is not in the passages and relevant is false.

Being on-topic is not being an answer. These passages are all retrieved by similarity, so \
they are ALWAYS about roughly the right subject. The question is narrower: is the specific \
fact asked for actually stated here.

Say relevant = false when:
- the passages discuss the topic but never state the fact asked for. This is the common \
case and the one graders get wrong;
- they describe a DIFFERENT system than the one asked about. This corpus contains V-JEPA \
and V-JEPA 2, I-JEPA, Sub-JEPA, Var-JEPA, Dreamer and DreamerV2 and DreamerV3. Passages \
about the wrong version are not an answer, however similar they read;
- answering would need a guess, an inference across papers, or a number that is implied \
but never written.

Extra irrelevant material alongside a real answer is fine - the generator can ignore noise.

Reply with JSON only:
{"relevant": <true or false>, "evidence": "<exact quote containing the answer, or empty>", \
"missing": "<what a correct answer needs that is absent, or empty>"}

"missing" and relevant=true are contradictory. If you can name something a correct answer \
needs and the passages do not state it, then relevant is false. Leave "missing" empty only \
when nothing is missing.

The angle brackets mark where you substitute your own verdict. They are placeholders, not \
default values."""


class Grade(BaseModel):
    """One grading verdict. ``missing`` is what a rewrite node would act on."""

    relevant: bool
    evidence: str = Field(default="", description="quote the grader says contains the answer")
    missing: str = Field(default="", description="what the passages lack; empty when relevant")


def verify_evidence(grade: Grade, context: str) -> Grade:
    """Downgrade a "relevant" verdict whose quote is not in the passages. Pure.

    "Relevant" is an opinion until the grader points at the words. Measured before this
    existed: the grader passed 10 of 12 retrievals that demonstrably could not produce a
    correct answer - every one of those questions also scored ``correct=False`` in the full
    eval run. It was not lying, it was pattern-matching on topic.

    So the contract becomes: claim relevance, show the span. ``quote_supported`` - written
    in Stage 2 when the gold auditor invented its evidence - checks the quote against the
    text with ligature, whitespace and case normalisation, requiring a contiguous run
    covering 75% of it.

    **This is not the same as repairing a verdict.** ``verdict_is_consistent`` in the judge
    deliberately refuses to flip a boolean, because there the verdict IS the measurement and
    quietly correcting it would launder a failure into a score. Here the verdict is a
    *claim* - "the answer is in these passages" - and an unsubstantiated claim is not a
    claim. Flipping it costs a retry; trusting it costs a wrong answer.
    """
    if not grade.relevant:
        return grade

    # Self-contradiction, checked mechanically. Measured: the grader returned
    # relevant=true alongside missing="The specific loss function ... is not stated" -
    # naming what a correct answer needs and passing the retrieval anyway. If something
    # required is absent, the passages are not sufficient; that is the definition, not a
    # judgement call. Same shape as the Stage 2 judge failing an answer while reporting it
    # had found no contradiction.
    if grade.missing.strip():
        return Grade(
            relevant=False,
            evidence=grade.evidence,
            missing=grade.missing,
        )

    if quote_supported(grade.evidence, context):
        return grade
    return Grade(
        relevant=False,
        evidence=grade.evidence,
        missing=(
            f"claimed evidence is not in the passages: {grade.evidence[:80]!r}"
            if grade.evidence.strip()
            else "claimed relevant but quoted nothing"
        ),
    )


class Grader:
    """Grades retrieved passages against a question."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def grade(self, question: str, hits: list[SearchHit]) -> Grade:
        """Return a verdict for these passages."""
        if not hits:
            return Grade(relevant=False, missing="nothing was retrieved")

        context = format_context(hits)
        try:
            response = _client(self.settings).chat.completions.create(
                model=self.settings.grader_model,
                messages=[
                    {"role": "system", "content": GRADER_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nPassages:\n{context}",
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            # content is a JSON *string*. Parse it; `**` on a string raises TypeError.
            grade = Grade.model_validate_json(response.choices[0].message.content or "")
            # "Relevant" is a claim until the quote is found in the passages.
            return verify_evidence(grade, context)
        except Exception as exc:
            # Fail OPEN: a grader that cannot answer must not send every question into the
            # retry loop. This catches the call as well as the parse, so a broken key would
            # make every verdict `relevant` - `missing` carries the marker so a run can
            # count them instead of reading a systemic failure as a lenient grader.
            log.warning("grader unavailable, defaulting to relevant: %s", exc)
            return Grade(relevant=True, missing=f"grader failed: {exc}")
