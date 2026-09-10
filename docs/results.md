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

| Metric | Value |
|---|---|
| hit@1 | 0.088 |
| hit@5 | 0.441 |
| recall@5 | 0.348 |
| MRR@5 | 0.213 |

Precision@k is deliberately absent. Gold labels are incomplete — a retrieved chunk that
is not on the gold list is often still a valid source — so precision would measure the
labelling, not the retriever.

### Answer quality (n = 30 answered questions, LLM judge)

| Metric | Value |
|---|---|
| Faithfulness — every claim traceable to a retrieved passage | **0.967** |
| Correctness — answers the question without contradicting the reference | 0.333 |

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
| Gold chunk in top-5 | 15 | **0.600** | 1.000 |
| Gold chunk not in top-5 | 15 | **0.067** | 0.933 |

Correctness is **9x higher** when retrieval works. Faithfulness is near-perfect either
way, which is the diagnostic: the generator reports its passages accurately whether or
not they are the right passages. 14 of the 20 incorrect answers had `recall@5 = 0.00` —
the evidence was never in front of the model.

So the ceiling is hit@5 = 0.441, and Stage 3 (BM25 + reciprocal rank fusion + metadata
filtering + cross-encoder reranking) is aimed at the right thing.

Two named cases carried forward as before/after tests:

| Query | Gold chunk | Dense rank | Why dense fails |
|---|---|---|---|
| "What data is V-JEPA trained on...?" | `2404.08471::0` | **170** | "V-JEPA" is a rare token; in a 1536-dim average it carries almost no weight. Dense retrieval returned V-JEPA **2** instead, and the answer was faithful to the wrong paper. |
| "Why does JEPA training collapse to trivial solutions?" | `2605.09241::1` | 9 | Outside `k = 5` despite near-verbatim phrasing overlap — exactly what BM25 is for. |

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
3. **The judge still self-contradicts about 7% of the time.** q018 and q023 returned
   `correct_reason: "no contradiction found"` alongside `correct: false`, which its own
   step 3 forbids. Correctness would be 0.400 rather than 0.333 without them. The
   principled fix is a small hand-labelled set of verdicts to measure judge agreement
   against a human, rather than further prompt iteration.
4. **Judge and generator share a model** (`gpt-4o-mini`). `judge_model` is a separate
   setting so this can be changed without touching generation, but as it stands the
   grader may favour its own phrasing.
5. **Tuning happened on dev.** Test split (n=8 judged) came out at correctness 0.250,
   faithfulness 1.000, hit@5 0.500 — close enough to dev that the dev numbers are not
   obviously inflated, but the split is small.
6. **`false_refusal_rate` conflates two causes.** An over-cautious prompt and a genuine
   retrieval failure that the generator handled honestly both land in the same bucket.
   q002 refused with its gold chunk at rank 102, which is arguably correct behaviour.
