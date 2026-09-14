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

### End to end, with the generator behind each retriever

`make eval` on all 40 questions, `answer_question` fed by each retriever in turn:

| | dense | hybrid |
|---|---|---|
| hit@1 / hit@5 | 0.324 / 0.618 | **0.382 / 0.647** |
| faithfulness | 0.933 | 1.000 |
| correctness | 0.433 | 0.406 |
| answer_rate (answerable) | 0.882 | **0.941** |
| refusal_rate (unanswerable) | **1.000** | 0.833 |
| refusal accuracy | 0.900 | **0.925** |
| retrieve p95 | 8.2 ms | 11.3 ms |

**Answer quality did not move, and this eval set cannot tell whether it should have.**
Correctness 0.433 vs 0.406 is a single question. Judge-scored metrics on ~31 answered
questions have a resolution of about 3 points per flipped verdict, and run-to-run variation
with no code change at all has been measured at 0.367-0.452. A real 5-point effect is below
the noise floor here; resolving one would need several hundred questions. Retrieval metrics
are deterministic and do not have this problem, which is why they are the headline for this
stage.

**Better retrieval caused a hallucination, and that is the interesting result.** q040 is
deliberately unanswerable - "how many GPU hours does V-JEPA need compared with I-JEPA?",
where I-JEPA reports GPU hours and V-JEPA reports iterations, so the comparison exists in
neither paper. Under dense retrieval the context was poor enough that the system refused.
Under hybrid, BM25 pulled in genuinely relevant V-JEPA and I-JEPA passages, and better
looking context talked the generator into answering.

Net refusal accuracy still improved (0.900 to 0.925): two false refusals traded for one
hallucination. Whether that is a good trade is a product question, not a metrics question -
for a cited research assistant, probably yes; where a confident wrong answer is expensive,
no. It is only visible at all because the eval set contains six unanswerable questions and
refusal is measured separately from correctness.

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

## Stage 4 — the agent loop, and why the retry rescues nothing

`retrieve → grade → (rewrite → retrieve | generate)`, with the retry capped by a counter
in the graph state rather than by the grader's agreement. `max_retries = 1`.

### The loop fires, and earns nothing

| | n = 40 |
|---|---|
| retried | **12 (0.300)** |
| — unanswerable, nothing to rescue | 5 |
| — gold was **already retrieved** (grader false alarm) | 2 |
| — genuine miss, a rescue was possible | **5** |
| rescued | **0** |
| lost | 0 |
| same chunks returned | 0 |

hit@5 was 0.647 before the loop and 0.647 after it.

**That equality is why the loop had to be instrumented before it could be read.** An
unchanged aggregate is equally consistent with "the loop never fired", "it fired and
changed nothing", and "three rescues cancelled three losses" — three different systems.
Recording `attempts` and the first retrieval's chunk ids per question is what separates
them. The first agent run reported none of this and was uninterpretable.

The rewrites themselves are not the problem. None fell back to the safe
`question + missing` string, none returned the same chunks, and they read reasonably:

```
q011  -> optimization differences between original World Models paper and Dreamer
q024  -> IRIS authors actor-critic objectives design borrow
q002  -> World Models agent memory component stochastic environments specific details
```

(The trailing `specific details` and `clear statement` in several rewrites come from the
grader's `missing` field, which is often vague boilerplate. Real, and not what costs the
five.)

### Why: the misses are not a phrasing problem

Gold rank for the five questions where a retry was possible, out of 100 retrieved:

| question | dense | bm25 | hybrid d=10 | hybrid d=30 | reachable by |
|---|---|---|---|---|---|
| q002 | **4** | 40 | 7 | 11 | better fusion — dense already had it inside k=5 |
| q015 | 12 | 9 | 12 | **8** | reranking at depth ≥ 10 |
| q028 | 16 | 42 | – | **29** | reranking at depth ≥ 30 |
| q024 | 44 | 31 | – | – | neither |
| q011 | 66 | 54 | – | – | neither |

A query rewrite reshuffles which five of the same candidate pool surface. It cannot reach
rank 29, and it cannot reach rank 8 reliably. **Hybrid retrieval had already eaten the
failure mode a rewrite is for** — vocabulary mismatch, the q030 case where dense ranks the
V-JEPA paper 170th and BM25 ranks it 2nd. The rewrite node was built to fix a problem
Stage 3 had already fixed.

### Fusion's net gain at k = 5 is about zero

The rank table shows something the Stage 3 summary row hides. Questions where dense had
gold inside k = 5 and fusion pushed it out:

| question | dense rank | hybrid d=30 rank |
|---|---|---|
| q008 | 2 | 6 |
| q021 | 2 | 9 |
| q029 | 4 | 7 |
| q002 | 4 | 11 |
| q006 | 4 | 12 |

Five demotions, against the four questions BM25 wins outright (q003, q007, q030, q034).
`hit@5`: dense 0.618, hybrid d=30 0.618, hybrid d=20 0.647 — a one-question spread across
34, which is noise.

So the earlier reading, "fusion bought ordering, not coverage", was too kind. **Fusion
bought ordering and paid for it in coverage**, roughly one-for-one. RRF promotes chunks
both retrievers found, and a chunk only one retriever found is demoted by exactly the
mechanism that makes RRF robust. The 0.735 union ceiling is unchanged, and the gap to it
is now attributable: it is the demotions.

### Fusion weighting: the fix the loop's diagnostics found

The rank table showed something the Stage 3 summary hid. Questions where dense had gold
inside k = 5 and equal-weight fusion pushed it out:

| question | dense rank | hybrid d=30, `w=1,1` |
|---|---|---|
| q008 | 2 | 6 |
| q021 | 2 | 9 |
| q029 | 4 | 7 |
| q002 | 4 | 11 |
| q006 | 4 | 12 |

Five demotions against the four questions BM25 wins outright — which is why `hit@5` read
0.647 for hybrid and 0.618 for dense alone. RRF rewards *agreement*: a chunk both
retrievers ranked 8th scores `2/(k+8)` while one dense ranked 2nd scores `1/(k+2)`, and at
k = 60 the flat denominator makes agreement win. BM25 is the weaker retriever here
(hit@5 0.471 vs 0.618) and was voting at full strength.

A 30-configuration grid over `rrf_k × weights`, run on the **dev split only** because 30
comparisons on 34 questions will always produce a winner:

| config | hit@1 | hit@5 | recall@5 | +dense | −dense |
|---|---|---|---|---|---|
| `k=60 w=1,1` (Stage 3) | 0.346 | 0.538 | 0.468 | 4 | 4 |
| `k=0…10 w=1,0.3` | 0.308 | **0.654** | **0.564** | **3** | **0** |

Four values of k give the identical result, and the three questions gained are three of
the four BM25-only questions. The consistency across k is the evidence; a single winning
cell would not be. `w=1,0` reproduces the dense row exactly on both splits — the
known-answer check on the grid itself.

Adopted: **`rrf_k = 5`, `fusion_weights = "1,0.3"`**. On the full set, hit@5 0.647 → 0.676,
recall@5 0.534 → 0.578.

**The price, recorded because it is a named test in this document:** q030 — "What data is
V-JEPA trained on?", dense rank >100, BM25 rank 2 — falls from fused rank **3 to 20**.
Down-weighting BM25 divides the only evidence for that chunk by three. The question that
justified adding BM25 is the one most exposed by trusting it less. Net across the set this
is +1 question and +0.044 recall@5; on q030 alone it is a regression.

### The loop is net-negative, and switched off

Re-run end to end on the new retrieval:

| | retriever alone | through the agent |
|---|---|---|
| hit@5 | **0.676** (23/34) | **0.647** (22/34) |

23 − 2 + 1 = 22. The gap *is* the retry loop: rescued 1 (q029), lost 2 (q002, q034).

The two losses are the grader's own false alarms, traceable across two files:

```
grade-eval   q002  said IRRELEVANT but gold was there
             q034  said IRRELEVANT but gold was there
eval --agent LOOP  lost 2  ['q002', 'q034']
```

It flagged retrievals that already contained the gold chunk, the rewrite fired, and the
retry came back worse. **The grader's false-alarm rate doubled when retrieval improved**
(0.091 → 0.174, catch 0.417 → 0.455): more good retrievals means more opportunities to
wrongly flag one, so a retry loop gets *more* dangerous as the system underneath it gets
better.

`max_retries` is therefore **0** by default. The loop stays wired, tested and measured; it
is re-enabled only alongside a retry action shown to help.

Confirmed end to end with the loop off — the agent now reports exactly what the retriever
does, which is the point:

| | Stage 3 baseline | loop on | **loop off** |
|---|---|---|---|
| hit@5 | 0.647 | 0.647 | **0.676** |
| recall@5 | 0.534 | 0.534 | **0.578** |
| correctness | 0.419 | 0.290 | 0.355 |
| retrieve p95 | — | 946 ms | **17 ms** |

The latency line was not something anyone was tracking. Every retry was a rewritten query
missing the embedding cache and paying a live embeddings call; removing the loop removed a
50× tail.

**Better retrieval did not produce better answers.** hit@5 +0.029 and recall@5 +0.044, and
correctness moved from 0.419 to 0.355 — two questions at n = 31, inside the noise band this
document already establishes for the judge (a single flipped verdict is 3.3 points). Two
readings survive it: the eval set cannot resolve answer-quality differences this small, and
correctness is bound by more than retrieval. Both were already known; neither is refuted.

Two per-question results worth reading rather than counting:

- **q002 refused with its gold chunk retrieved.** Retrieval succeeded and the generator
  declined anyway — a false refusal that no retrieval work can fix.
- **q024 was scored correct with gold missed.** Either the answer is right for reasons the
  gold labels do not capture, or the label is incomplete. The second gold-audit pass is
  still outstanding and this is the kind of case it exists to find.

One detail worth keeping: `preserves_key_terms` did reject a real rewrite (q029,
"masking differences for images in I-JEPA and video in V-JEPA comparison"), and the crude
`question + missing` fallback then rescued that question. The single success the loop had
came from the fallback, not the rewrite.

### Validating the judge: correctness holds, faithfulness does not

Deferred since Stage 2 and finally done. Ten answers from the latest run were re-labelled
against a **blind** worksheet — question, passages, answer, reference, and deliberately not
the judge's verdict or its reason, which would turn the exercise into checking whether the
judge's reasoning sounds plausible. It always does. The sample is stratified rather than
random: faithfulness runs ~0.94, so ten random rows would be nine easy agreements.

**Provenance, stated plainly: the labels are Claude Opus 5's, not a human's**
(`eval/judge_labels_claude.jsonl`). This measures *judge vs a stronger model*, which is a
weaker claim than judge-vs-human and shares some blind spots. A human pass would be
`judge_labels_<name>.jsonl`; `score` compares every labeller it finds.

| axis | agreement | Cohen's kappa |
|---|---|---|
| correct | **10/10** | **+1.00** |
| faithful | 7/10 | **−0.15** |

Raw agreement is the wrong statistic here and kappa is why. Faithfulness is ~94% True, so
a judge answering True unconditionally would score ~0.94 agreement while contributing
nothing. Kappa subtracts that floor; negative means *below* what two raters with these base
rates reach by guessing. At n = 10 this is directional, not conclusive — but every
disagreement landed on one axis, and each has a named mechanism rather than being a
coin-flip:

1. **q001 — attributed the passages to the wrong system.** The answer said Ha and
   Schmidhuber's world model is not described as using reward. The judge objected that
   "the reward predictor outputs a univariate Gaussian", which is in the *DreamerV2*
   passage. Right corpus, wrong paper — the same confusion the grader prompt already warns
   about and the judge prompt does not.
2. **q005 — graded a claim the answer never made.** The answer refused. The judge's reason
   was "the passages do not explicitly state that the world model is held fixed while
   behaviors are learned" — which is the *reference answer's* claim. It imported
   correctness into a faithfulness check, collapsing the distinction the two axes exist to
   maintain.
3. **q031 — stopped at the first supported claim.** The answer reverses I-JEPA's result
   (`2301.08243::13` says representations degrade when the loss is computed in *pixel*
   space; the answer says the opposite, then contradicts itself one sentence later). The
   judge quoted the V-JEPA sentence, which is genuinely supported, and never reached the
   contradicted one.

#### Three rubrics later, the axis still does not track a careful reader

The correctness axis has ordered tests, mandatory prefixes, an explicit list of
non-failures and a machine check. Faithfulness had a paragraph of prose. That difference
looked like the whole explanation, so it was closed in two steps and measured after each.

| rubric | faithfulness | agreement vs the frozen labels | kappa |
|---|---|---|---|
| v1 — prose | 0.935 | 7/10 | −0.154 |
| v2 — four mandatory prefixes | 0.806 | 5/10 | −0.190 |
| v3 — + quotes verified against answer and passages | 0.688 | 5/10 | −0.190 |

**Each iteration made the output more rigorous and the verdict no better.** v2 reached
0/31 contract violations while agreement *fell*: the judge learned the format and kept the
reasoning. Third demonstration in this project that a contract constrains form, not
thought.

v3 added `failure_is_substantiated`, which requires a failing verdict to quote the
answer's own words, and a `CONTRADICTED:` verdict to also quote passage words that
contradict them (`quote_supported` again, its third use). The quotes duly became real, and
produced this:

```
q013  CONTRADICTED: "PlaNet does not train an explicit policy network"
            but passage says "no explicit policy or value function network is used"

q017  CONTRADICTED: "the agent does not learn without it"
            but passage says "the stochastic component is even more important -
                              the agent does not learn without it"
```

Both quotes verified; both pairs **say the same thing**. q017's passage span contains the
answer span verbatim. The check can confirm that a quote exists and cannot tell agreement
from contradiction.

**Kept at v3 rather than reverted, and faithfulness is marked as not a metric.** Reverting
restores 0.935 — a healthy-looking number this measurement says is untrustworthy, and a
number that looks fine invites being believed. v3 reports 0.688 with failure reasons a
human can audit in one line, which is the same thing the prefix contract bought the
correctness axis: not a better verdict, a checkable one.

A fourth fix is visible — reject a `CONTRADICTED:` whose two quotes overlap — and was
deliberately **not** made. Three rubrics tuned against ten labelled answers is the limit at
which an improvement can still be distinguished from a coincidence.

**Report faithfulness as: contract-compliant, human-auditable, and not validated.** Use
correctness (kappa +0.80) for answer-quality claims.

**Consequence for Stage 4.** Step 4 is a verify node — "is every claim in this answer
supported by the passages?" — which is exactly the axis that just failed validation.
Building it on this rubric would reproduce all three failures *inside the control flow*,
where unlike the judge it would act on its verdict. Mechanism 2 is the worst of them there:
at request time no reference answer exists, so a check that silently reaches for one has no
defined behaviour in production. The rubric is repaired before the node is built.

**And a separate defect found while reading q005.** The answer ends *"Thus, the answer is
INSUFFICIENT_CONTEXT."* and was scored as an answered question. `is_refusal` uses
`startswith`, deliberately — the token mid-paragraph means the model is talking *about*
refusing while still answering. That reasoning does not hold for a final sentence that is
itself the refusal, so **`refusal_rate = 0.833` is a lower bound**, on the one metric Stage
4 is meant to move.

### What this rules in and out

- **Rewrite-and-retry**: measured at 0 rescues from 5 chances. Kept in the codebase,
  wired, tested and off the critical path — the null result is the finding.
- **Escalate to the listwise reranker on a failed grade**: the obvious next move, and
  weaker than it looks. The Stage 3 row shows listwise reranking at depth 20 taking hit@1
  from 0.382 to 0.529 and leaving **hit@5 at 0.647, unchanged**. It reorders within the
  top five; it does not pull rank 8 into them. Would plausibly reach q015 and q028 only if
  run at greater depth than it was measured at.
- **Fusion weighting or a guaranteed slot per retriever**: costs nothing, targets five
  measured demotions, and is the only intervention here aimed at the actual failure.

The honest summary of Stage 4 so far: the agent's control flow works, its instrumentation
works, and the action it takes when it detects a failure is aimed at the wrong failure.

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
