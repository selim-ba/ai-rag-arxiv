# Agentic RAG over arXiv

Ask a research question about world models and JEPA, get an answer grounded in real
papers with citations — from an agent that decides how to search, checks what it found,
and reports what it did.

Built and measured stage by stage. Every claim below has a number and a method behind it
in [`docs/results.md`](docs/results.md), including the things that did **not** work.

## Stack

Python 3.11 · FastAPI · LangGraph · OpenAI · NumPy · Docker

## Quickstart

```bash
cp .env.example .env      # add your OpenAI key
make install
make test

make ingest               # 49 papers from corpus.txt -> 874 chunks
make index                # embed and build the vector index
make dev                  # http://localhost:8000/docs
```

Or in a container, with no local Python at all:

```bash
make docker-build          # 472MB, ~20s cold / 3s after a source change
make docker-run            # reads OPENAI_API_KEY from .env at RUNTIME, never from a layer
make docker-checks         # asserts the three defined failure modes still hold
```

Ask it something:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How does PlaNet search for a good action sequence?","use_agent":true}'
```

```json
{
  "answer": "PlaNet searches using the Cross Entropy Method (CEM) ... [1811.04551].",
  "citations": ["1811.04551"],
  "refused": false,
  "mode": "agent",
  "trace": [
    "route -> retrieve (unscoped question about the papers)",
    "retrieve('How does PlaNet search...') -> 5 hits in 12.4ms",
    "grade -> relevant",
    "generate -> 1 citations"
  ],
  "sources": [{"chunk_id": "1811.04551::4", "url": "https://arxiv.org/abs/1811.04551"}]
}
```

`mode` and `trace` are reported rather than inferred: the agent and the plain pipeline
return the same answers on this corpus, so without them a caller cannot tell which ran.

Stream it instead, and ask a follow-up:

```bash
curl -N -s localhost:8000/ask/stream -H 'content-type: application/json' \
  -d '{"question":"How does PlaNet search for a good action sequence?"}'
```

```
event: token
data: {"text":"PlaNet searches using"}
...
event: done
data: {"answer":"...[1811.04551].","citations":["1811.04551"],"conversation_id":"7a0e...",
       "sources":[...],"request_id":"cf01560e69a8"}
```

Pass `conversation_id` back and the next question may refer to the last one - "does it
need action labels?" is resolved into a standalone question **before** retrieval sees it,
so every component behaves exactly as it was measured. The resolver leaves a question that
is already self-contained alone: 0 over-resolutions across 18 specified cases.

**Citations cannot be fabricated.** The model cites passage markers (`[P1]`), and the
arXiv id is substituted in code from the retrieved set. This was not a precaution — an
early version invented `2606.09985` for a paper numbered `2506.09985`.

## How it works

```
POST /ask
   |
   v
route ----+-- catalog ------------------------------> catalog ---> answer
          |   (what is indexed? do you cover X?)      no model involved
          |
          +-- retrieve / filtered --> retrieve -> grade -+-> generate -> answer
                                         ^               |
                                         +--- rewrite <--+   (off by default)
```

* **route** classifies the question against the indexed corpus: unscoped, scoped to named
  papers, or about the corpus itself. `filtered` is not a separate path — it is retrieval
  plus a chunk filter.
* **retrieve** is dense + BM25 fused by reciprocal rank, with per-retriever weights.
* **grade** asks whether the passages can answer the question, and must quote the span
  that proves it.
* **generate** answers from the passages only, or refuses.
* **catalog** answers from the index's metadata, with no model in the loop — because
  "which papers do you have" is a fact, and a generator asked for it would invent a list.

## Results

A hand-built evaluation set of 40 questions (20 factual, 10 comparison, 4 definitional,
6 deliberately unanswerable) over 874 chunks from 49 papers, plus an 18-question routing
specification. Full methodology and limitations in [`docs/results.md`](docs/results.md).

### Retrieval

| Configuration | hit@1 | hit@5 | recall@5 | MRR@5 | p95 |
|---|---|---|---|---|---|
| Dense only (Stage 2) | 0.324 | 0.618 | 0.520 | 0.457 | 1.1 ms |
| BM25 only | 0.324 | 0.471 | 0.402 | 0.406 | 1.9 ms |
| Hybrid, equal weights (Stage 3) | 0.382 | 0.647 | 0.534 | 0.498 | 2.5 ms |
| **Hybrid, BM25 weighted 0.3** | 0.353 | **0.676** | **0.578** | 0.473 | 2.4 ms |

Equal-weight fusion netted nothing at k=5: it won four questions and lost five, because
reciprocal rank fusion rewards agreement and demotes a chunk only one retriever found.
Down-weighting the weaker retriever recovered them — chosen on the dev split from a
30-configuration grid, where the same result held across five values of the RRF constant.

### Routing

| Split | route accuracy | correct paper ids |
|---|---|---|
| dev (12) | 0.917 | 3/4 |
| **test (6), prompt frozen** | **1.000** | 1/2 |

Scored by exact match against a specification written before the router existed — the one
metric here that needs no evaluator. Route accuracy alone was misleading: the first
version scored 0.833 while picking the **wrong paper every time**, because the corpus
titles do not contain the names anyone uses ("V-JEPA" is filed as *Revisiting Feature
Prediction for Learning Visual Representations from Video*). Adding an alias table took
paper ids from 0/4 to 3/4.

### Answer quality

| | |
|---|---|
| correctness | 0.387 — validated, Cohen's κ **+0.80** against blind re-labelling |
| faithfulness | 0.710 — **not validated**, κ ranged −0.19 to +0.29 across identical runs |
| refusal accuracy | 0.900, answer rate 0.912 |

**The system is retrieval-bound**, and faithfulness is reported as a contract-compliant,
human-auditable signal rather than a measurement — two identical runs disagreed by enough
to flip the sign of its agreement statistic, so no change to it is falsifiable at this
sample size. Correctness carries the answer-quality claims.

### Latency, and what happens when the provider fails

| | |
|---|---|
| time to first token (streamed) | **657 ms** median, against 1764 ms for the whole answer |
| retrieval | 2.4 ms p95 |
| generation | 1250 ms p50, 1850 ms p95 |

Every request emits one JSON line carrying a request id, the route, the timings, and the
number of provider calls **and retries** - counted from an httpx hook on the shared
client, because the SDK's two retries happen below the call site and are over by the time
it returns. A request that succeeded first try and one that survived two retries and a
`Retry-After` sleep used to be indistinguishable from outside.

Provider failures map to defined statuses rather than an untyped 500: a rate limit becomes
**503** with `Retry-After` (the quota is the service's, not the caller's), a timeout
**504**, an outage **502**, and a bad key or unavailable model **500** — because that one
is ours, and returning the upstream 401 would send a caller to fix a request that was
never the problem.

### The container

A 472MB image on `python:3.12-slim`, running as uid 10001 under
`--read-only --cap-drop ALL`, healthy 1.67s after start, with no secret in any layer
(verified with `docker history`, not assumed). 102MB of it was dependencies the API never
imports — `pymupdf` and `tiktoken` belong to ingestion, and a structural test now fails if
anything on the request path imports them again.

`make docker-checks` asserts the three ways it is allowed to fail: no index → 503, no key
→ 500 `misconfigured`, wrong key → 500 `misconfigured` with the key absent from the body.
Those checks found two defects that 456 unit tests could not, because **unit tests cannot
see configuration, and configuration is most of what a container changes**.

### Things that were built, measured, and switched off

* **A rewrite-and-retry loop.** Across two runs it rescued 1 question and lost 2 — both
  losses were the grader's false alarms on retrievals that already held the gold chunk.
  Its false-alarm rate *doubled* when retrieval improved. Wired, tested, `max_retries=0`.
* **A verify node.** Its target — `refusal_rate` 0.833 → 1.000 — turned out to be a
  measurement artefact: the one "failure" was a correct refusal written in prose without
  the expected token. Two prompt fixes closed the gap and cost 18 and 13 correct answers.
* **Three faithfulness rubrics.** Each made the output more rigorous and the verdicts no
  better.

## Layout

```
src/arxiv_rag/
  config.py           typed settings, validated at startup
  api/main.py         FastAPI app; POST /ask and /ask/stream, sessions      (Stage 5)
  api/errors.py       provider failures -> defined statuses and codes      (Stage 5)
  observability.py    request id, retry counting, one JSON line per request (Stage 5)
Dockerfile            python:3.12-slim, non-root, read-only-safe          (Stage 6)
scripts/container_checks.sh   the three failure modes, against the image  (Stage 6)
  ingestion/          arXiv -> PDF -> clean text -> chunks       (Stage 1)
  retrieval/          embeddings, BM25, fusion, filters, rerank  (Stages 2-3)
  evaluation/         metrics, LLM judge, quote verification     (Stage 2)
  agent/              LangGraph agent: router, nodes, tools      (Stage 4)
eval/                 40 questions, 18 routing cases, judge labels
scripts/              ingest, index, and one harness per measurement
docs/results.md       every number above, with its method
```

## Evaluation harnesses

```bash
make retrieval-eval            # free: retrieval metrics, no LLM
make retrieval-eval ARGS="--grid"   # sweep RRF constant and weights
make route-eval                # router vs the routing specification
make followup-eval             # follow-up resolution vs the follow-up specification
make grade-eval                # the grader against known retrieval outcomes
make eval ARGS="--agent"       # end to end, with the judge
make score-judge               # judge agreement against blind labels
```

Each one exists because a number was once wrong in a way nothing could see. The recurring
lesson of this project is in `docs/results.md`: **every problem found so far was in the
instrument before it was in the system.**
