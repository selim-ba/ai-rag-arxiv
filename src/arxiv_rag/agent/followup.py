"""Turning a follow-up question into one that stands on its own.

"How does it avoid collapse?" is not a query. It is a query plus a pointer into the
conversation, and retrieval has no way to follow a pointer. So the follow-up is resolved
into a standalone question first, and everything downstream - router, retriever, grader,
generator - keeps seeing exactly what it saw in Stage 4: one self-contained question. That
is deliberate. Threading history through every node would invalidate every Stage 4
measurement; resolving before the graph leaves all of them intact.

**Over-resolution is the dangerous direction, and the asymmetry drives the design.**

    under-resolved:  "How does it avoid collapse?" retrieves badly and the failure is
                     visible in the answer.
    over-resolved:   "Why does JEPA training collapse?" rewritten to be about V-JEPA
                     because the previous turn was, retrieves well, reads well, and
                     answers a question nobody asked.

One is a bad answer, the other is a confident answer to the wrong question. So every
failure path here returns the follow-up **unchanged**: a model that errors, a resolution
that loses the follow-up's own terms, or a `changed=false` verdict all leave the text
alone. The system degrades toward the behaviour it had before this module existed.

Eighth place in this codebase where a model's output is checked rather than trusted.
"""

import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel

from arxiv_rag.agent.rewrite import preserves_key_terms
from arxiv_rag.config import Settings
from arxiv_rag.llm import get_client as _client

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Turn:
    """One exchange. The answer is carried because a third of the references need it.

    Measured while writing `eval/followups.jsonl`: "why are they frozen?" points at random
    projection matrices that appear only in the previous ANSWER, and "what does the second
    one change?" is an ordinal into a list the answer produced. A resolver given only
    question history cannot serve either.
    """

    question: str
    answer: str = ""


RESOLVE_SYSTEM = """You rewrite a follow-up question so that it stands alone, using the \
conversation before it.

Reply with JSON only:
{"changed": <true or false>, "resolved": "<the standalone question>", "reason": "<one clause>"}

Rewrite ONLY what the follow-up cannot supply for itself:

- a pronoun or a bare reference - "it", "they", "that paper", "the second one" - gets \
replaced with what it points at;
- an omitted question gets restored. "What about Dreamer?" after "Does PlaNet train a \
policy network?" becomes "Does Dreamer train a policy network?" - the SUBJECT changed and \
the question carried over. Replacing only the name produces "Dreamer", which asks nothing.

**If the follow-up already stands alone, set changed to false and return it verbatim.** \
This is the common case and the important one. A question that names its own subject and \
asks its own question needs nothing from the conversation, even when it follows a related \
turn. Pulling the previous subject into it produces a fluent question about the wrong \
thing, which is worse than leaving it alone.

Specifically, do NOT narrow a question that is deliberately broad. "Why does JEPA training \
collapse to trivial solutions?" is about the whole body of work; if the previous turn was \
about V-JEPA, it is still about the whole body of work. "Which papers do you have?" is \
about the collection and has no subject to inherit.

Keep the follow-up's own wording wherever it already works. You are filling gaps, not \
rephrasing."""


class Resolution(BaseModel):
    """A follow-up, possibly rewritten."""

    changed: bool = False
    resolved: str = ""
    reason: str = ""


def verify_resolution(resolution: Resolution, followup: str) -> Resolution:
    """Reconcile the model's rewrite with the follow-up it was given. Pure.

    Three rules, each failing toward *unchanged*:

    1. ``changed=false`` must mean the text is untouched. A model that says it changed
       nothing and returns different words has done something nobody asked for and nobody
       will notice.
    2. An empty resolution is no resolution.
    3. The rewrite must keep the follow-up's own key terms. `preserves_key_terms` - written
       in Stage 4 to stop a query rewrite paraphrasing "V-JEPA" away - applies unchanged
       here: a resolution that drops a name the USER wrote has replaced their question
       rather than completed it.

    Note which direction rule 3 runs. It checks the follow-up's terms survive, not that the
    referent was added: adding is the job, and whether it happened correctly is what
    `scripts/followup_eval.py` measures against the specification.
    """
    text = resolution.resolved.strip()
    if not text:
        return Resolution(changed=False, resolved=followup, reason="empty resolution")
    if not resolution.changed and text != followup.strip():
        return Resolution(
            changed=False,
            resolved=followup,
            reason=f"claimed unchanged but rewrote it: {resolution.reason}",
        )
    if resolution.changed and not preserves_key_terms(followup, text):
        return Resolution(
            changed=False,
            resolved=followup,
            reason=f"resolution dropped the follow-up's own terms: {text[:60]!r}",
        )
    return Resolution(changed=resolution.changed, resolved=text, reason=resolution.reason)


def resolve_followup(history: list[Turn], followup: str, settings: Settings) -> Resolution:
    """Make a follow-up standalone, or leave it alone.

    Fails open to the follow-up verbatim: a resolver that cannot run must not stop the
    request, and the unresolved question is what the system answered before this existed.
    """
    if not history:
        return Resolution(changed=False, resolved=followup, reason="no history")

    transcript = "\n\n".join(
        f"Q: {turn.question}\nA: {turn.answer}" if turn.answer else f"Q: {turn.question}"
        for turn in history[-3:]
    )
    try:
        response = _client(settings).chat.completions.create(
            model=settings.followup_model,
            messages=[
                {"role": "system", "content": RESOLVE_SYSTEM},
                {
                    "role": "user",
                    "content": f"CONVERSATION SO FAR:\n{transcript}\n\nFOLLOW-UP: {followup}",
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content or "")
        resolution = Resolution.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - any transport or parse error
        log.warning("follow-up resolution failed, using the question as asked: %s", exc)
        return Resolution(changed=False, resolved=followup, reason=f"resolver failed: {exc}")
    return verify_resolution(resolution, followup)
