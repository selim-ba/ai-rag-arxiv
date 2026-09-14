"""Generation: turning retrieved chunks into a grounded, cited answer.

Retrieval is only half of RAG. This file is the other half, and it is where the
project stops being a search engine and starts being an assistant.

Three things this module has to get right, in order of how much they matter:

1. **Grounding.** The model answers *from the supplied passages*, not from what it
   remembers about DreamerV3. A model that answers correctly from memory is
   indistinguishable from one that got lucky, and it will confabulate the moment you
   ask about a paper it never saw.
2. **Citation.** Every claim carries the arXiv id it came from. This is what makes the
   answer checkable by a human in ten seconds instead of ten minutes, and it is the
   single feature that makes people trust a RAG system.
3. **Refusal.** When the passages do not contain the answer, say so. Your eval set has
   six unanswerable questions for exactly this reason. A system that answers all forty
   is worse than one that answers thirty-four and refuses six, and no retrieval metric
   will ever tell you that.

``format_context`` and ``extract_citations`` are unit-testable without an API key;
``answer_question`` is not, so it is kept as small as possible.
"""

import logging
import re
from time import perf_counter

from openai import OpenAI
from pydantic import BaseModel, Field

from arxiv_rag.config import Settings
from arxiv_rag.retrieval.filters import ChunkFilter
from arxiv_rag.retrieval.hybrid import Retriever
from arxiv_rag.retrieval.store import SearchHit

log = logging.getLogger(__name__)

# The exact string the model must emit when the passages are insufficient. A sentinel
# beats fuzzy matching on "I'm sorry" or "I don't know": those phrases also appear
# inside perfectly good answers ("the paper does not say what learning rate...").

REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You answer questions about machine-learning research papers using \
ONLY the numbered passages provided.

Rules:
- Use only what is in the passages. Do not use prior knowledge about these papers, even \
if you are confident it is correct.
- Cite by passage marker. Each passage begins with a marker like [P1] or [P4]. Put that \
marker inline after every claim, exactly as written. A sentence drawing on two passages \
carries both markers.
- Never write an arXiv id yourself, and never copy a bracketed reference marker out of a \
passage body - the passages contain the papers' own markers such as [6] or [15], which \
mean nothing here. The only brackets you write are [P1], [P2], and so on.
- Check that the passage is about the system the question asks about. Several papers \
here are versioned or closely named (V-JEPA and V-JEPA 2, I-JEPA, Sub-JEPA, Var-JEPA; \
Dreamer, DreamerV2, DreamerV3). If the passages describe a related but different system, \
say so instead of answering as though they were the same.
- If the passages do not contain enough information to answer, reply with exactly \
{REFUSAL_TOKEN} followed by one sentence saying what is missing. Do not guess, and do \
not answer from memory.
- Be concise. Three or four sentences is usually enough.
- If the question's premise is wrong (it asks about something the papers show does not \
exist), say so rather than inventing an answer that fits the premise."""


class Answer(BaseModel):
    """A generated answer plus everything needed to audit it."""

    question: str
    text: str
    citations: list[str] = Field(default_factory=list, description="arXiv ids cited, in order")
    refused: bool = False
    retrieved_ids: list[str] = Field(
        default_factory=list, description="chunk_ids fed to the model, best first"
    )
    # One retrieval number, not a breakdown. Once retrieval is pluggable, "embedding
    # time" is an implementation detail of one retriever: BM25 has no embedding step and a
    # reranker's cost is a model forward pass. What the caller can compare across
    # configurations is total time to candidates.
    retrieve_ms: float = Field(default=0.0, description="whatever the retriever did")
    generate_ms: float = Field(default=0.0, description="the LLM call")


# -- pure functions ------------------------------------------------------------------


def format_context(hits: list[SearchHit]) -> str:
    """Render retrieved chunks as the numbered passage block the prompt refers to.

    Target format, one blank line between passages::

        [P1] arXiv:1803.10122 | World Models | Section: Introduction
        We explore building generative neural network models...

        [P2] arXiv:1811.04551 | Learning Latent Dynamics... | Section: Method
        ...

    Three decisions made for auditability and model performance:

    - **The arXiv id goes in the header of every passage.** The model can only cite an
      id it can see. Leave it out and you get invented citations.
    - **Passages are marked ``[P1]``, ``[P2]``, not ``[1]``, ``[2]``.** The chunk text is
      full of the papers' own reference markers like ``[6]`` and ``[15]``. A bare
      bracketed integer is therefore ambiguous; the ``P`` prefix is not. Measured: with
      bare ``[1]`` numbering the model cited passage numbers on every answer.
    - **The model cites the marker, never the id.** Asked to transcribe an arXiv id it
      fabricated two out of three - once recalling a famous id that was never retrieved,
      once corrupting a digit (2506 -> 2606). Small integers it copies reliably; ten
      character digit strings it does not. ``resolve_citations`` does the mapping in
      code, which makes a fabricated citation impossible rather than merely rare.
    - **The title and section are included.** "Section: Ablations" is a strong signal
      that a number is from an ablation table and not the headline result.

    """
    passages: list[str] = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = f"[P{i}] arXiv:{chunk.arxiv_id} | {chunk.title}"
        if chunk.section:
            header += f" | Section: {chunk.section}"
        passages.append(f"{header}\n{chunk.text}")
    return "\n\n".join(passages)


def extract_citations(text: str) -> list[str]:
    """Pull the arXiv ids out of an answer, in order of first appearance, deduplicated.

    ``"Dreamer [1912.01603] extends PlaNet [1811.04551], and [1912.01603] adds..."``
    gives ``["1912.01603", "1811.04551"]``.

    """
    pattern = r"\[(\d{4}\.\d{4,5})\]"
    seen: set[str] = set()
    citations: list[str] = []
    for match in re.finditer(pattern, text):
        arxiv_id = match.group(1)
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            citations.append(arxiv_id)
    return citations


def resolve_citations(text: str, hits: list[SearchHit]) -> str:
    """Rewrite the model's ``[P1]`` markers into real ``[arxiv_id]`` citations.

    ``"MPC planning [P1], not a policy network [P2]."`` with hits from ``2107.08241``
    and ``1811.04551`` becomes ``"MPC planning [2107.08241], not a policy network
    [1811.04551]."``

    This function is the reason fabricated citations cannot happen: the only ids it can
    ever emit are ids taken from ``hits``. The model chooses *which passage*, which it
    does reliably; the id lookup is arithmetic, which it does not.

    Rules:

    - ``[P1]`` is the first hit, ``[P2]`` the second. One-based, matching
      ``format_context``'s ``enumerate(hits, start=1)``.
    - A marker pointing past the end of ``hits`` - ``[P9]`` when five were retrieved -
      is dropped from the text entirely. It refers to nothing, so it cannot be resolved,
      and leaving it in would put a meaningless token in front of the reader.
    - Everything else in the text is left exactly as it is, including the papers' own
      ``[6]``-style reference markers. Those are noise, but they are the passages' noise,
      and silently deleting things from a generated answer is worse than leaving them.

    ``re.sub`` takes a function as its replacement argument: it is called with each
    match and returns the replacement string. Cleaner than looping over ``finditer``
    and splicing the string by index.
    """

    def replace(match: re.Match[str]) -> str:
        position = int(match.group(1))
        if 1 <= position <= len(hits):
            return f"[{hits[position - 1].chunk.arxiv_id}]"
        return ""  # marker points past the end of the hits - it resolves to nothing

    return re.sub(r"\[P(\d+)\]", replace, text)


def is_refusal(text: str) -> bool:
    """Did the model decline to answer? Given to you — one line, but a real decision.

    ``startswith`` rather than ``in``: the token appearing mid-answer means the model
    is talking *about* refusing while still answering, which is not a refusal.
    """
    return text.strip().startswith(REFUSAL_TOKEN)


# -- the thin network layer ----------------------------------------------------------


def answer_question(
    question: str,
    retriever: Retriever,
    settings: Settings,
    k: int | None = None,
    chunk_filter: ChunkFilter | None = None,
) -> Answer:
    """Retrieve, then generate. The whole RAG loop in one function."""
    started = perf_counter()
    hits = retriever.search(
        question, k=settings.top_k if k is None else k, chunk_filter=chunk_filter
    )
    retrieve_ms = (perf_counter() - started) * 1000
    answer = generate_answer(question, hits, settings)
    answer.retrieve_ms = retrieve_ms
    return answer


def generate_answer(question: str, hits: list[SearchHit], settings: Settings) -> Answer:
    """Turn retrieved passages into a grounded answer. No retrieval of its own.

    Split out of ``answer_question`` so Stage 4's graph can own retrieval and generation as
    separate nodes while calling exactly this code - the agent has to be measurably the same
    pipeline before it is allowed to be a different one.
    """
    started = perf_counter()

    if not hits:
        return Answer(
            question=question,
            text=f"{REFUSAL_TOKEN} nothing was retrieved for this question.",
            refused=True,
        )

    context = format_context(hits)
    response = _client(settings).chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    generate_ms = (perf_counter() - started) * 1000

    raw = response.choices[0].message.content or ""
    text = resolve_citations(raw, hits)
    return Answer(
        question=question,
        text=text,
        citations=extract_citations(text),
        refused=is_refusal(text),
        retrieved_ids=[hit.chunk.chunk_id for hit in hits],
        generate_ms=generate_ms,
    )


def _client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)
