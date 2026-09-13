# Results

All numbers are reproducible: `make ingest && make index && make eval`.
Per-question records for every run are written to `data/eval/run-<timestamp>.jsonl`.

---

## Stage 1 — PDF text extraction

Three parsers on the same three papers. "Glued words" counts alphabetic tokens of 20+
characters, which is what happens when a parser drops the spaces between words.

| Parser | Chars | Glued words | Licence | Verdict |
|---|---|---|---|---|
| pypdf | 100% (baseline) | 412 | BSD | unusable — spacing collapses in two-column layouts |
| pdfplumber | ~99% | 39 | MIT | acceptable, ~6x slower |
| **PyMuPDF** | ~102% | **3** | AGPL | **chosen** |

PyMuPDF is AGPL. That is fine for a portfolio project and for anything self-hosted, but
it would need replacing or licensing in a closed-source commercial product. Recorded
here so the choice is visible rather than accidental.

Corpus: 49 papers → **874 chunks**. Chunk tokens: min 61, mean 519, max 800.

---

## Stage 2 — Dense retrieval baseline

`text-embedding-3-small` (1536 dims), exact cosine search over all 874 chunks, `k = 5`.
No BM25, no reranking, no agent. **This is the number Stage 3 has to beat.**

Evaluation set: 40 hand-written questions — 20 factual, 10 comparison, 4 definitional,
6 unanswerable; 30 dev / 10 test; 18 requiring more than one gold chunk.

### Retrieval (n = 34 questions with gold chunks)

| Metric | Hand labels | **Audited labels** |
|---|---|---|
| hit@1 | 0.088 | **0.324** |
| hit@5 | 0.441 | **0.618** |
| recall@5 | 0.348 | **0.520** |
| MRR@5 | 0.213 | **0.434** |
| search latency p95 | — | 5.1 ms |

**The jump is a correction, not an improvement.** Retrieval is byte-for-byte identical
between the two columns; only the labels changed. The original `gold_chunk_ids` were
written by hand from a handful of chunks, and the index holds 874 — so a question
answered correctly from an unlabelled but valid chunk scored as a miss. 18 alternative
sources across 13 questions were found and added (see *Auditing the gold labels* below).
Everything before this correction should be read as a floor.

The check that the correction did not corrupt anything: **correctness barely moved**
(0.400 to 0.367, one question, within judge noise). It should not move at all — answer
quality is graded against the hand-written `reference_answer`, which the audit never
touched, and the generated answers are unchanged. A correctness score that had jumped
alongside hit@5 would have meant the pipeline was leaking labels into grading.

Precision@k is deliberately absent. Gold labels are incomplete — a retrieved chunk that
is not on the gold list is often still a valid source — so precision would measure the
labelling, not the retriever.

### Answer quality (n = 30 answered questions, LLM judge)

| Metric | Value |
|---|---|
| Faithfulness — every claim traceable to a retrieved passage | **0.933** |
| Correctness — answers the question without contradicting the reference | 0.400 |
| Judge output-contract violations | **0 / 30** |

The judge must justify every failing verdict with a `CONTRADICTION:` or `WRONG SYSTEM:`
prefix and quoted text, and every pass with `OK:`. A verdict whose prefix disagrees with
its boolean, or whose faithfulness reason appeals to the reference answer, is a contract
violation: `judge_answer` retries it once and then records it rather than repairing it.
Quote this count next to the scores — it is the error bar on them.

### Refusal (n = 40)

| Metric | Value |
|---|---|
| answer_rate — answerable questions that were answered | 0.882 |
| refusal_rate — unanswerable questions correctly refused | **1.000** |
| false_refusal_rate — answerable questions wrongly refused | 0.118 |
| accuracy — behaviour matched intent | 0.900 |

All six unanswerable questions were refused, including the false-premise one ("which RL
algorithm does I-JEPA use for planning?" — I-JEPA has no planner). Zero fabricated
citations across all 40, by construction: the model cites passage markers and the
arXiv id is substituted in code by `resolve_citations`.

---

## The finding: this system is retrieval-bound

Splitting answer quality by whether retrieval succeeded:

| | n | correctness | faithfulness |
|---|---|---|---|
| Gold chunk in top-5 | 20 | **0.550** | 1.000 |
| Gold chunk not in top-5 | 10 | **0.000** | 1.000 |

Every single correct answer came from a successful retrieval. Faithfulness is 1.000 either way, which is the
diagnostic: the generator reports its passages accurately whether or not they are the
right passages. When the evidence is not retrieved, the answer is faithful to whatever
was retrieved instead, and wrong.

So the ceiling is hit@5 = 0.441, and Stage 3 (BM25 + reciprocal rank fusion + metadata
filtering + cross-encoder reranking) is aimed at the right thing.

Two named cases carried forward as before/after tests:

| Query | Gold chunk | Dense rank | Why dense fails |
|---|---|---|---|
| "What data is V-JEPA trained on...?" | `2404.08471::0` | **170** | "V-JEPA" is a rare token; in a 1536-dim average it carries almost no weight. Dense retrieval returned V-JEPA **2** instead, and the answer was faithful to the wrong paper. |
| "Why does JEPA training collapse to trivial solutions?" | `2605.09241::1` | 9 | Outside `k = 5` despite near-verbatim phrasing overlap — exactly what BM25 is for. |

---

## Stage 3 — retrieval techniques, one row at a time

Same 34 answerable questions, same generator, k=5. Retrieval metrics need no LLM, so each
configuration is measured with `make retrieval-eval` in about a second.

| Configuration | hit@1 | hit@5 | recall@5 | MRR@5 | p95 latency |
|---|---|---|---|---|---|
| Dense only (Stage 2 baseline) | 0.324 | 0.618 | 0.520 | 0.457 | 1.2 ms |
| BM25 only | 0.324 | 0.471 | 0.402 | 0.406 | 2.1 ms |
| **Dense + BM25, reciprocal rank fusion** | 0.382 | **0.647** | 0.534 | 0.498 | 2.6 ms |
| + cross-encoder rerank (ms-marco-MiniLM) | 0.324 | 0.588 | 0.490 | 0.450 | 120 ms |
| **+ listwise LLM rerank (gpt-4o-mini)** | **0.529** | 0.647 | **0.549** | **0.600** | 7784 ms |

**BM25 alone is worse than dense and still belongs in the table.** It wins four questions
dense misses entirely (q003, q007, q030, q034), which is the whole premise for fusing them:
the union ceiling of the two is 0.735 against dense's 0.618.

**Fusion bought ordering, not coverage.** hit@1 +0.058 and MRR +0.041, but hit@5 moved by a
single question — about 25% of the available headroom. RRF promotes chunks *both*
retrievers found, which pushes single-retriever finds down; the four "BM25 only" questions
have to survive that crowding.

**The cross-encoder lost on every metric, and the diagnosis matters more than the number.**
58% of chunks exceed its 512 word-piece context (median chunk: 551), because chunk size was
chosen in Stage 1 for the generator's context window and no one asked what a reranker
wanted. But truncation is not the whole story: on q030 the gold chunk fits comfortably at
373 word-pieces and still scored −0.702 while four wrong-paper chunks scored up to +3.4.
That chunk is a title + author list + abstract block, and `ms-marco-MiniLM` was trained on
clean web passages. Across all questions the gold moved **worse on 13, unchanged on 10,
better on 6**.

**The listwise LLM reranker won on ordering and lost on latency.** Five more questions
answered from the very top result, MRR up 0.10, and both deliberately planted hard cases
(q030, q034) landed at rank 1–2. hit@5 did not move at all, because reranking reorders a
pool and cannot add to it. Cost is about $0.002 per query — but p95 is **7.8 seconds**
against 2.6 ms, roughly 3000×, on top of ~1.3 s of generation.

**Metadata filtering has no row, deliberately.** `ChunkFilter` restricts retrieval by
`arxiv_id` or `section`, applied before ranking so `k` still means `k`. The eval questions
carry no paper constraints, so it cannot move hit@5 — and measured on q030 it actually
pushed the gold chunk from rank 3 to rank 5, because a filter narrows the candidate pool
without improving the ranking inside it. What it does buy is a guarantee: with the filter
set, all five passages come from the paper asked about, so the "faithful answer about
V-JEPA 2 instead of V-JEPA" failure becomes impossible rather than unlikely. It is also the
mechanism a Stage 4 agent needs to act on "this question names one paper".

**Shipped default: hybrid without reranking.** The listwise reranker is kept behind a flag.
A 9-second request is the wrong default for a system whose generation step is already the
slow part, and Stage 4's agent is the right place to spend it — escalating to reranking
only when the retrieval grader says the passages are weak, rather than on every query.
---

## Auditing the gold labels

The labels were written by hand while drafting each question, from chunks chosen by
searching the index. But facts repeat: papers restate their predecessors and surveys
describe everything. One paragraph of the model-based RL survey (`2107.08241::17`) turned
out to answer three separate questions about PlaNet that were labelled only against the
PlaNet paper.

`scripts/gold_audit.py` retrieves the top 10 per question, asks the model which unlabelled
candidates support a claim that an existing gold chunk also supports, and writes proposals
to a review file. **Nothing is written to the eval set without a human decision** — the
gold set is the ground truth every number here rests on, and generating it with the same
model family being evaluated would be circular.

Gold ids are stored as **groups**, meaning *one chunk from each group*:

```jsonc
[["1912.01603::7"], ["2010.02193::8"], ["2301.04104::5"]]   // 3 facts, all needed
[["1811.04551::4", "2107.08241::17"]]                       // 1 fact, 2 valid sources
```

`recall@k` counts groups satisfied, not chunks matched. Without that distinction, adding
a second valid source to a question would have *halved* its recall — improving the labels
would have degraded the metric.

**The auditor fabricated evidence.** Asked to quote the words supporting each proposal, it
returned fluent, on-topic quotes that did not appear in the chunks they named — 11 of 54
across a full run. Two were about to be accepted precisely because the quote read like
confirmation. `evaluation/quotes.py` now verifies every quote against the chunk text
(normalising ligatures, line breaks, case and punctuation, requiring a contiguous run
covering 75% of the quote) and demotes any proposal whose evidence cannot be found. This
is the third place in the codebase where a model's own report is verified in code rather
than trusted: arXiv ids in citations, chunk ids in the audit, and now quoted spans.

Result: 43 verified proposals, 18 accepted, 25 rejected. The three recurring rejection
patterns were abstract boilerplate, performance results offered for mechanism questions,
and — most dangerous — **wrong-system chunks**, such as DreamerV2 describing itself being
offered as evidence for a question about IRIS.
---

## Known limitations

1. **Gold labels are incomplete.** Two of five spot-checked questions were answered
   correctly from chunks not on their gold list — including a survey paper that
   describes PlaNet, and the DreamerV2 paper describing what DreamerV1 did. Papers cite
   their predecessors, so a fact often lives in several places. hit@5 therefore
   understates real performance. A completeness pass is outstanding.
2. **The judge is a model, and it was iterated.** Three revisions were needed before it
   graded the two axes independently: v1 justified faithfulness by citing the reference
   answer (both axes agreed on every question); v2 copied the literal booleans out of
   the JSON format example (faithfulness pinned at exactly 1.000); v3 uses a
   default-to-true forcing function and placeholder-only examples. Each revision was
   validated against four questions with known real failures (q012, q017, q020, q030),
   which had to stay incorrect.
3. **Self-contradiction is fixed; weak reasoning is not.** The judge previously
   returned `correct_reason: "no contradiction found"` alongside `correct: false` on
   about 7% of graded answers. Requiring a machine-checkable prefix, verified by
   `verdict_is_consistent` and retried once, took that to 0/30 and raised correctness
   from 0.333 to 0.400.

   The contract checks **form, not substance.** It catches a reason that disagrees with
   its own verdict; it cannot catch a well-formed reason that is simply wrong. Two
   verdicts now quote a "contradiction" that is really an omission — q018 cites
   `'multi-step predictions of all distances'` against the reference's longer
   `'...purely in latent space, without decoding extra images'`. Same claim, more
   detail. What the prefix bought is auditability: a human can now check a verdict in
   one line instead of re-reading five passages. The remaining fix is hand-labelled
   verdicts to measure judge-human agreement, which is outstanding.
4. **Judge and generator share a model** (`gpt-4o-mini`). `judge_model` is a separate
   setting so this can be changed without touching generation, but as it stands the
   grader may favour its own phrasing.
5. **Tuning happened on dev.** Test split (n=8 judged) came out at correctness 0.250,
   faithfulness 1.000 against dev's 0.455 / 0.909 — the gap is within what n=8 can
   resolve, but the split is small and the dev numbers should be read as the optimistic
   end.
6. **The judge is not run-to-run stable.** Across runs at `temperature=0` with no change
   to the generator, faithfulness moved between 0.933 and 0.967 — a single flipped
   verdict is 3.3 points at n=30. Differences smaller than about 5 points between
   configurations are noise, not signal.
7. **`false_refusal_rate` conflates two causes.** An over-cautious prompt and a genuine
   retrieval failure that the generator handled honestly both land in the same bucket.
   q002 refused with its gold chunk at rank 102, which is arguably correct behaviour.
