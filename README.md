# Agentic RAG for World Models & JEPA Research

Ask a research question about world models and get a cited answer drawn from one or
several of 49 arXiv papers — or an honest refusal when those papers do not contain the
answer.

Built and measured stage by stage. **Every claim below has a number and a method behind
it** in [`docs/results.md`](docs/results.md), including the things that did *not* work:
three components were measured and rejected, one "improvement" turned out to be a
measurement artefact, and a metric is reported as unmeasurable because it could not be
validated.

Python 3.11+ · FastAPI · LangGraph · OpenAI · NumPy · Docker · GitHub Actions

## Quickstart

```bash
cp .env.example .env      # add your OpenAI key
make install
make test                 # 490 tests, no network

make dev                  # http://localhost:8000 — the demo page and /docs
```

The corpus and its index are committed, so nothing has to be ingested to run it. To
rebuild them from scratch (needs a key, downloads 49 PDFs):

```bash
make ingest               # 49 papers from corpus.txt -> 874 chunks
make index                # embed and build the vector index
```

In a container, with no local Python at all:

```bash
make docker-build         # 472MB, ~20s cold / 3s after a source change
make docker-run           # reads OPENAI_API_KEY from .env at RUNTIME, never from a layer
make docker-checks        # asserts the five defined failure modes still hold
```

Ask it something:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How does PlaNet search for a good action sequence?"}'
```

```json
{
  "answer": "PlaNet searches using the Cross Entropy Method (CEM) ... [1811.04551].",
  "citations": ["1811.04551"],
  "refused": false,
  "mode": "pipeline",
  "sources": [{"chunk_id": "1811.04551::4", "url": "https://arxiv.org/abs/1811.04551"}],
  "request_id": "05eb5a993227"
}
```

Or stream it, and ask a follow-up:

```bash
curl -N -s localhost:8000/ask/stream -H 'content-type: application/json' \
  -d '{"question":"How does PlaNet search for a good action sequence?"}'
```

Pass `conversation_id` back and the next question may refer to the last one — *"does it
need action labels?"* is resolved into a standalone question **before** retrieval sees it,
so every component behaves exactly as it was measured. The resolver leaves an
already-complete question alone: **0 over-resolutions across 18 specified cases**.

**Citations cannot be fabricated.** The model never sees an arXiv id — it points at a
retrieved chunk and the id is substituted in code. This was not a precaution: an early
version invented `2606.09985` for a paper numbered `2506.09985`.

## How it works

Two routes through the same system. The **pipeline** is the default and the configuration
every published number describes; the **agent path** adds a router and a grader.

```
                                   pipeline (default)
question ─────────────► retrieve ─────────────────────────► generate ──► answer
                        dense + BM25                        or refuse
                        fused, weighted


                                   agent path
question ──► route ──┬── catalog ─────────────────────────────────────► answer
                     │   "which papers do you have?"  (no model call)
                     │
                     └── retrieve ──► grade ──► generate ──► answer
                         scoped to      must     or refuse
                         named papers   quote
                         if any         the span
```

* **retrieve** — 874 chunks, each stored as a 1,536-dimension embedding and in a BM25
  index. Both are searched and the rankings fused by weighted reciprocal rank: each
  contributes `1/(5 + rank)`, and BM25's side is multiplied by **0.3**.
* **grade** — asks whether the retrieved chunks can answer the question, and must quote
  the span that proves it. The quote is verified against the chunk in code, so the
  grader is trusted only where it can be checked.
* **generate** — answers from the chunks only, or refuses.
* **catalog** — answers "which papers do you have" from the index's metadata with no model
  in the loop, because that is a fact and a generator asked for a list invents one.

**Why the pipeline is the default**, measured rather than assumed: routing wins two
comparison questions on hit@5 (0.676 → 0.735) and makes the system answer two questions it
should have refused (refusal accuracy 0.925 → 0.875). For a system whose main claim is
that it declines what it cannot support, that is not a good trade — so the agent path is
used where routing *is* the point, and not as the default.

## Results

A hand-built evaluation set of 40 questions (20 factual, 10 comparison, 4 definitional,
**6 deliberately unanswerable**) over 874 chunks from 49 papers, plus an 18-question
routing specification and an 18-case follow-up specification. Full methodology and
limitations in [`docs/results.md`](docs/results.md).

### Retrieval (34 questions with gold chunks, k=5)

| Configuration | hit@1 | hit@5 | recall@5 | MRR@5 | p95 |
|---|---|---|---|---|---|
| Dense only (Stage 2) | 0.324 | 0.618 | 0.520 | 0.457 | 1.1 ms |
| BM25 only | 0.324 | 0.471 | 0.402 | 0.406 | 1.9 ms |
| Hybrid, equal weights (Stage 3) | 0.382 | 0.647 | 0.534 | 0.498 | 2.5 ms |
| **Hybrid, BM25 weighted 0.3** | 0.353 | **0.676** | **0.578** | 0.473 | 2.4 ms |
| + the router's paper filter | 0.324 | 0.735 | 0.593 | 0.457 | — |

Equal-weight fusion netted nothing at k=5: it won four questions and lost five, because
reciprocal rank fusion rewards agreement and demotes a chunk only one retriever found.
Down-weighting the weaker retriever recovered them — chosen on the dev split from a
30-configuration grid, where the same result held across five values of the RRF constant.

### Routing

| Split | route accuracy | correct paper ids |
|---|---|---|
| dev (12) | 0.917 | 3/4 |
| **test (6), prompt frozen** | **1.000** | 1/2 |

Scored by exact match against a specification written before the router existed. Route
accuracy alone was misleading: the first version scored 0.833 while picking the **wrong
paper every time**, because the corpus titles do not contain the names anyone uses
("V-JEPA" is filed as *Revisiting Feature Prediction for Learning Visual Representations
from Video*). An alias table took paper ids from 0/4 to 3/4.

### Answer quality

| | |
|---|---|
| correctness | 0.375 — validated, Cohen's κ **+0.80** against blind re-labelling |
| the same configuration, two days earlier | 0.323 — **the noise floor is about 5 points** |
| faithfulness | **not validated** — κ ranged −0.19 to +0.29 across identical runs |
| refusal accuracy (the 2×2, not refusal rate) | 0.925 |
| false refusal rate | 0.059 |
| follow-up resolution | 13/18, with **0 over-resolutions** |

**The system is retrieval-bound.** Two runs of the identical configuration, two days
apart, produced byte-identical retrieval numbers and correctness five points apart — so
any improvement smaller than that gap is indistinguishable from luck. Faithfulness is
reported as a contract-compliant, human-auditable signal rather than a measurement.

### Latency, cost, and failure

| | |
|---|---|
| time to first streamed word | **657 ms**, against 1.76 s for the whole answer |
| retrieval | 2.4 ms p95 |
| generation | 1250 ms p50, 1850 ms p95 |
| **cost per question** | **$0.000517**, measured from real token counts |
| container image · cold start | 472 MB · 1.67 s |

Every request emits one JSON line with a request id, the route, the timings, the token
counts, the cost, and the number of provider calls **and retries** — counted from an httpx
hook on the shared client, because the SDK's two retries happen below the call site and
are over by the time it returns.

Provider failures map to defined statuses rather than an untyped 500: a rate limit becomes
**503** with `Retry-After` (the quota is the service's, not the caller's), a timeout
**504**, an outage **502**, and a bad key or unavailable model **500** — that one is ours,
and returning the upstream 401 would send a caller to fix a request that was never the
problem.

A public demo also needs limits that bind before the bill does: a **daily budget in
dollars** (checked before the model is called, so a refusal costs nothing) and a per-IP
token bucket. Both are tested against the running container.

### Built, measured, and switched off

* **A rewrite-and-retry loop.** Across two runs it rescued 1 question and lost 2 — both
  losses were the grader's false alarms on retrievals that already held the gold chunk.
  Its false-alarm rate *doubled* when retrieval improved. Wired, tested, `max_retries=0`.
* **A verification node.** Never built: its target — `refusal_rate` 0.833 → 1.000 — turned
  out to be a measurement artefact. The one "failure" was a correct refusal written in
  prose without the expected token. Two prompt fixes closed the gap and cost 18 and 13
  correct answers.
* **Two-stage retrieval.** Both coarse passes measured, both failed: 49 papers curated on
  one subject are not distinguishable at document level.
* **A multi-stage Docker build.** Saved 1 MB, because deleting a file in a later layer
  does not shrink an image — the bytes stay in the layer that added them.

## How it is tested

`make test` runs 490 tests with no network. On every push, CI runs:

```
lint → tests → the retrieval evaluation against a committed floor
     → build the image → the container's five failure modes → push to the registry
```

**The retrieval evaluation gates the build.** `eval/thresholds.json` holds the floor, in
the repository rather than in the workflow, because a threshold in a YAML comment is a
number nobody reviews. It can gate a build because this harness needs no API key — the
index and the eval questions' embeddings are both committed — and because retrieval here
is deterministic, so a drop of any size is a regression rather than noise.

**The paid harnesses are deliberately not in CI.** `make eval`, `make route-eval` and
`make followup-eval` cost money per run and their numbers move about five points between
runs of the same code. A green badge implying they were checked on every push would be
exactly the kind of claim this project exists to avoid.

## Layout

```
src/arxiv_rag/
  config.py           typed settings, validated at startup
  llm.py              one shared client; every model call goes through it   (Stage 5)
  observability.py    request id, retry and token counting, one JSON line   (Stage 5)
  limits.py           daily budget in dollars, per-IP token bucket          (Stage 7)
  pricing.py          what a call costs, dated and sourced                  (Stage 7)
  api/main.py         FastAPI app: /ask, /ask/stream, the demo pages        (Stage 5-7)
  api/errors.py       provider failures -> defined statuses and codes       (Stage 5)
  ingestion/          arXiv -> PDF -> clean text -> chunks                  (Stage 1)
  retrieval/          embeddings, BM25, fusion, filters, rerank             (Stage 2-3)
  evaluation/         metrics, LLM judge, quote verification                (Stage 2)
  agent/              LangGraph agent: router, grader, sessions, tools      (Stage 4-5)
web/                  the demo page, the corpus, the paths, the metrics     (Stage 7)
eval/                 40 questions, routing and follow-up specs, CI floor
data/index/           874 chunks and their embeddings — committed, so the image builds
scripts/              ingest, index, and one harness per measurement
Dockerfile            python:3.12-slim, non-root, survives --read-only      (Stage 6)
.github/workflows/    lint, tests, the retrieval gate, the image            (Stage 7)
docs/results.md       every number above, with its method
```

## Evaluation harnesses

```bash
make retrieval-eval                  # free, no model calls — this is the one CI gates on
make retrieval-eval ARGS="--check"   # ...against eval/thresholds.json
make retrieval-eval ARGS="--grid"    # sweep the RRF constant and the weights
make route-eval                      # router vs the routing specification      (paid)
make followup-eval                   # follow-up resolution vs its specification (paid)
make grade-eval                      # the grader against known retrieval outcomes (paid)
make eval ARGS="--agent"             # end to end, with the judge               (paid)
make eval ARGS="--agent --router"    # ...with routing on: the configuration compared
make score-judge                     # judge agreement against blind labels     (paid)
make docker-checks                   # the container's failure modes: no index, no key,
                                     # wrong key, rate limit, exhausted budget
```

Each one exists because a number was once wrong in a way nothing could see. The recurring
lesson, documented case by case in `docs/results.md`: **every problem found so far was in
the instrument before it was in the system** — including the claim, made in this
repository's own CI configuration, that the retrieval harness made no model calls.
