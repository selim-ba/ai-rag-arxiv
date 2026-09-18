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

### The second pass: a better candidate pool, and nothing to change

The first pass drew candidates from **dense retrieval's top 10 alone**, while every number
in this document comes from the hybrid. A chunk BM25 ranks 2nd and dense does not rank at
all was therefore never a candidate for labelling — the audit was auditing a retriever
nobody ships. The pool is now the union of dense, BM25 and the fused hybrid, each at
depth 10, which is the set of chunks any measured configuration can surface. Seven
proposals came from chunks the hybrid never ranked, four of them found only by BM25.

Auditing *deeper* was considered and rejected: the metrics are @1 and @5, so a chunk
nothing ranks in its top 10 cannot turn a measured miss into a hit, and every extra
candidate is another chance for the auditor to fabricate one.

| | first pass | second pass |
|---|---|---|
| candidate pool | dense top 10 | dense ∪ BM25 ∪ hybrid, top 10 each |
| rows written | 54 | 244 (41 proposed as alternatives, 203 pre-marked drop) |
| fabricated quotes caught | 11 | **15**, about 27% of everything called an alternative |
| **accepted** | 18 | **0** |

**Only 11 of the 41 proposals could have changed a published number**, and all 11 were
rejected on inspection:

- **eight** would have flipped a question from miss to hit at k=5 — q011, q024 (×3),
  q026, q028 (×2), q033 — worth **hit@5 0.676 → 0.824** if accepted;
- **three** sat at rank 1 and could have moved hit@1 — q017, q022, q029.

Every one was a topic match rather than a claim match. q024's five were all chunks
describing *Dreamer's* actor-critic offered as evidence that *IRIS borrowed it* — the
attribution is the entire claim and none of them make it. q011 and q026 were handed the
same generic survey paragraph with two different sentences cherry-picked from it; one
paragraph that is evidence for two unrelated claims is evidence for neither. q022's was
the DreamerV2 abstract, which alludes to "discrete representations" without stating the
architecture the question asks about.

**The bias in this tool runs one way, and it is worth naming.** The auditor only ever sees
chunks the retriever *returned*, so every accepted proposal raises the score and a rejected
one costs nothing. A false accept inflates; a false reject is invisible and harmless. That
asymmetry is the argument for reviewing the consequential proposals hardest — they are
exactly the ones the bias points at.

The 30 proposals that could move no metric were dropped without individual review, and
that is recorded here rather than hidden: dropping is the conservative direction, and a
label that changes no number is not worth a review budget.

**The result is a negative one, and it strengthens the numbers rather than changing them.**
The nine questions that neither retriever answers at k=5 are genuine retrieval failures,
not labelling artefacts. Diminishing returns are themselves a finding: the first audit
found 18 real omissions, the second — with a strictly better pool — found none.

A separate finding fell out of it. q024 had been recorded as "answered correctly while its
retrieval scored a miss", which is what reopened this audit. In the current run it scores
`correct: false`, with the judge calling it a CONTRADICTION between *"did not design their
own actor-critic objectives; instead built upon existing frameworks"* and the reference's
*"they borrowed them, adopting the objectives of DreamerV2"* — which agree. The answer is
vaguer, not contradictory. So the note was stale, and the judge is wrong here in the harsh
direction: another well-formed reason that is substantively wrong, which is the failure
mode the contract was already known not to catch.
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

### refusal_rate 0.833 was already correct, and chasing it cost 18 answers

Stage 4's definition of done asks for `refusal_rate` 0.833 -> 1.000, on the belief that
the missing 0.167 was a hallucination on an unanswerable question. It was not.

**q027** — *"What imagination horizon did the original Dreamer use?"* The first Dreamer
paper lists H as a hyperparameter symbol and puts the value in an appendix that was not
ingested; DreamerV2 says 15 and DreamerV3 says 16, which are tempting and belong to
different agents. The model answered:

> The original Dreamer does not specify a fixed imagination horizon in the provided
> passages... does not provide a specific value for H itself [1912.01603]. Thus, the exact
> imagination horizon used by the original Dreamer is not mentioned.

That is a **correct refusal**, and everything it says about the passages is supported. It
simply never emits `INSUFFICIENT_CONTEXT`, and `is_refusal` is `startswith`, so it scores
as a failure to refuse. The entire justification for a verify node was this artefact — the
fifth time in this project that the first thing to check was the instrument.

Two prompt edits tried to close the gap. Both hit the target and broke the system:

| generator prompt | refusal_rate | false_refusal | answer_rate | **accuracy** |
|---|---|---|---|---|
| one refusal bullet (shipped) | 0.833 | **0.059** | **0.941** | **0.925** |
| + "being on-topic is not being an answer" | **1.000** | 0.559 | 0.441 | 0.525 |
| + "decline only for the MAIN thing asked" | **1.000** | 0.382 | 0.618 | 0.675 |

19 and then 13 of 34 answerable questions refused, many with the gold chunk retrieved.
The second edit was written specifically to *reduce* refusing and still over-refused,
which points at salience rather than content: a prompt carrying two bullets about
declining produces more declining than one carrying a single bullet, whatever the second
says. Untested — confirming it would have meant a third prompt tuned against 40 questions.

**The selection effect is the part worth keeping.** While this happened, answer quality
*improved*:

| | shipped | Goodharted |
|---|---|---|
| faithfulness | 0.806 | **0.867** |
| correctness | 0.406 | **0.733** |
| judged questions | 32 | **15** |

Both metrics are computed over answered questions only, so refusing the hard half raised
them. Read alone, that run looks like the best answer quality the project has recorded.
The only number that caught it is `accuracy`, the 2x2 of should-answer against did-answer,
which fell 0.925 -> 0.525. That is the argument for scoring refusal as a confusion matrix
instead of a rate, and it is the second time a metric in this project has been defended by
a companion metric rather than by inspection.

Reverted to the shipped prompt. Kept: `refusal_token_misplaced`, which counts answers
carrying the token somewhere other than the start (q005's shape), and a regression test
whose docstring holds the table above so the next person to have this idea meets the
evidence first.

**The target is restated: `refusal_rate = 0.833` is correct behaviour on this eval set.**
The residual 0.167 is one question whose correct refusal is written in prose. Any future
work on it is formatting, not reasoning, and must be scored on `accuracy`.

### The faithfulness axis cannot be validated on this eval set

The three-rubric arc above ended with faithfulness at kappa −0.19 and the axis marked
unvalidated. A fourth change was then made — `spans_overlap`, rejecting a `CONTRADICTED:`
verdict whose two quoted spans contain one another, because nothing contradicts itself.
Unlike the rubric edits this is **definitional**: it is true on any dataset, needs no
labels, and would have been right before the failing cases were ever seen.

It appeared to work. Then the same run was repeated with no code change at all:

| run | agreement | kappa |
|---|---|---|
| first | 7/10 | **+0.286** |
| repeat, identical config | 6/10 | **−0.176** |

**One question moved and the sign of kappa flipped.** Every faithfulness result in this
document — v1's −0.154, v2 and v3's −0.190, v4's +0.286 — sits inside that band. The
judge is not run-to-run stable (Stage 2 measured faithfulness moving 0.933–0.967 at
`temperature=0` with no change), and ten labels cannot resolve a one-question swing.

**So the axis is closed, not merely unvalidated.** Further work on it is unfalsifiable
here: no change can be distinguished from noise without roughly 40 labels and a judge that
returns the same verdict twice. Neither is worth building for this project. Faithfulness
is reported as contract-compliant, human-auditable, and **not a measurement**; correctness
(kappa +0.80, and 9/10 or 10/10 on both repeat runs) carries the answer-quality claims.

`spans_overlap` is kept anyway. Its one confirmed effect is that q017 — whose "passage
says" span contains its "answer claims" span verbatim — now fails substantiation, is
retried, and is *counted* in `judge broke its contract` instead of appearing in the
hand-read UNFAITHFUL list as a finding. Making a bad verdict countable is worth eight
deterministic lines even when it moves no aggregate.

### The verify node: deliberately not built

Stage 4 step 4 is a verify node — re-read the generated answer against the passages and
refuse or retry if a claim is unsupported. It is not built, and the reasons are
measurements rather than scheduling:

1. **Its stated target was an artefact.** The brief asks for `refusal_rate` 0.833 → 1.000.
   That 0.167 is q027, whose *correct* prose refusal simply lacks the token. Two prompt
   edits closed it and cost 18 and then 13 correct answers.
2. **It cannot be scored.** A verify node is a faithfulness check, and faithfulness has a
   variance band wider than any effect it could plausibly have. The alternative,
   `accuracy`, moves in steps of 0.025 on 40 questions — one flipped question.
3. **The failure it would target is a retrieval failure.** The clearest case, q030,
   attributes V-JEPA 2's "1M hours of internet-scale video" to V-JEPA. Its gold chunk sits
   at fused rank 20 (`hit@5` false) because Stage 4's fusion reweighting demoted it from
   rank 3. The generator is reporting the passages it was given. Fixing retrieval for that
   question is a smaller change with a metric that can see it.

Building an unmeasurable component, in a project whose recurring finding is that
unmeasured components are how it gets into trouble, would contradict the whole record
above. Reasons to revisit: an eval set large enough to resolve answer-quality differences
(~100 questions), or a use for verification that is not scored by a judge — refusing to
emit an arXiv id that was not retrieved, say, which `resolve_citations` already does in
code without a model.

### The router: the first component with a metric that needs no evaluator

Three routes - `retrieve` (unscoped question), `filtered` (scoped to named papers),
`catalog` (about the corpus itself, or a paper not in it). No "answer directly" route: the
generator may only use retrieved passages, so such a route would either never fire or break
the grounding rule.

`eval/routes.jsonl` - 18 questions, 6 per route, 12 dev / 6 test - was written **before**
the router existed. It is a *specification*, not ground truth discovered by observation,
which is why the project can author it: the failure it guards against is a router tuned
until its own output looks reasonable. Scoring is exact match. **No rubric, no judge, no
kappa, no noise band** - the first metric here that needs no evaluator, after a week in
which the faithfulness axis turned out to be unmeasurable at this sample size.

| configuration | route accuracy | filtered with the right ids |
|---|---|---|
| titles only | 0.833 | **0 / 4** |
| + paper aliases, + "catalog is membership, not ideas" | 0.917 | 3 / 4 |
| + "copy the id character for character" | 0.917 | 1 / 4 |
| reverted to the previous line | 0.917 | 3 / 4 |
| **held-out test split, prompt frozen** | **1.000 (6/6)** | 1 / 2 |

**Route accuracy alone would have called the first configuration an 83% success.** It got
the right route and the wrong paper every single time. Searching the wrong paper is worse
than not filtering at all - the system answers confidently from a neighbouring model - so
the ids column is reported separately and is the number that matters.

**The fix was data, not prompting.** The router matched questions against a catalog of
arXiv ids and titles, and the titles do not contain the names anyone uses:

    question says     title in the index
    V-JEPA            Revisiting Feature Prediction for Learning Visual ... Video
    I-JEPA            Self-Supervised Learning from Images with a Joint-Embedding ...
    IRIS              Transformers are Sample-Efficient World Models

"V-JEPA" went to the one title containing that string - *V-JEPA 2* - reproducing the exact
confusion behind q030's fabricated answer, by a different component. "IRIS" matched nothing
and the router reported, honestly, that IRIS is not indexed. It is.

`agent/aliases.py` is hand-written because it cannot be derived, and both failed attempts
are instructive: from abstracts, V-JEPA's top candidates are CNRS, PSL and LIGM (author
affiliations); by frequency, "V-JEPA" appears 181 times in the **V-JEPA 2** paper and
"DreamerV3" appears **once** in its own, which calls itself Dreamer. The names nest, so
counting cannot tell an introduction from a citation.

**A prompt edit made it worse, reproducibly.** Adding "copy their arXiv ids character for
character" did not fix its target and pushed two other cases onto the V-JEPA 2 row -
1/4, twice, byte-identical output. Reverting restored 3/4. Four runs, unambiguous
attribution, and the lesson is that telling a model to match more literally is the wrong
instruction when the failure is surface-matching in the first place.

**Known residual, deliberately unfixed.** On the held-out split, "Sub-JEPA" routed to JEDI
(`2605.13013`), whose catalog line reads `JEDI - JEDI: Joint Embedding Diffusion World
Model...` - the alias duplicated by the title, beside the words JEPA expands to. The same
doubling affects the V-JEPA 2 row. The fix is obvious (do not render an alias the title
already carries) and was **not applied**, because the case appeared on the test split and
acting on it would spend the only held-out measurement available.

`verify_route` reconciles the model's route with the index: `filtered` with no ids becomes
`retrieve`, `filtered` on an unindexed paper becomes `catalog`, and a mixed list keeps what
exists. It earned its place on the first live run - `f04` recognised IRIS, omitted the id,
and was demoted to `retrieve` rather than becoming a filter that silently constrains
nothing. Seventh place in this codebase where a model's output is checked rather than
trusted.

### The filter ceiling: the biggest headroom in the system, and the router cannot reach it

The `filtered` route rests on an assumption - that restricting retrieval to named papers
improves it. Measured directly, with an oracle: restrict each question to the paper its
own gold chunk lives in. A router that is always right, which no real router can beat. No
model, no cost, fully deterministic.

| | unfiltered | oracle filter |
|---|---|---|
| hit@1 | 0.353 | **0.529** |
| hit@5 | 0.676 | **0.824** |
| recall@5 | 0.578 | **0.725** |
| MRR@5 | 0.473 | **0.663** |

Nothing is lost (0 questions), as construction implies: narrowing the pool can only remove
competitors. **+0.148 hit@5 is larger than every improvement this project has achieved so
far combined**, and 0.824 sits above the 0.735 union ceiling, because that ceiling measures
fusing two retrievers *unfiltered* - narrowing the candidate pool is a different axis.

The five rescued questions - q026, q028, q029, q033, q037 - are all drawn from the nine
that both dense and BM25 miss at k=5. Filtering rescues five of the nine hardest.

**And the router captures none of it.** Not one of the five names a paper:

    q026  "Earlier world models needed their regulariser retuned per environment..."
    q028  "the image and video JEPA models"
    q029  "the two JEPA papers"
    q033  "Two recent JEPA papers attack latent collapse..."
    q037  "In the ablation, which loss terms..."          - names nothing at all

Four of the five need two or three papers, not one. The `filtered` route fires only when a
user scopes a question explicitly, and the questions that benefit from scoping are exactly
the ones where the user does not know which paper holds the answer. The oracle knows; the
person asking does not.

**So the `filtered` route is a UX feature, not a retrieval improvement.** It honours an
explicit constraint correctly and buys no measurable quality. The router's value is
`catalog` - questions no passage can answer - and the measured discipline of
`verify_route`.

**The obvious way to reach it is two-stage retrieval** - a coarse pass that discovers the
two or three relevant papers, then a fine pass within them. Both mechanisms for that coarse
pass were measured before either was built. Both fail.

#### Two-stage retrieval: measured, and closed

*Coarse pass A - aggregate chunk scores by paper.* Retrieve 100 chunks, group by paper,
keep the top N. No new index; the fine pass is the same `ChunkFilter` the oracle used.

| aggregate | N=1 | N=2 | **N=3** | N=5 | N=10 |
|---|---|---|---|---|---|
| max | 0.294 | 0.500 | 0.676 | 0.765 | 0.882 |
| sum | 0.441 | 0.559 | 0.706 | 0.765 | 0.882 |
| mean | 0.294 | 0.500 | 0.559 | 0.647 | 0.824 |
| count | 0.441 | 0.676 | **0.735** | 0.824 | 0.912 |

Paper recall@3 is 0.735 against a pre-set threshold of 0.90. Worse, the per-question
picture is decisive: **all five questions the oracle rescues are stranded by the coarse
pass**, and three that currently work (q012, q020, q021) would be destroyed. Rescued 0,
lost 3.

The diagnosis is *derivation*. A coarse pass built from chunk scores inherits the chunk
retrieval's blindness: on exactly the questions where the gold chunk ranks poorly, the gold
paper collects one weak deep vote (q037's sits at fused rank 53 of 100) while papers with
three strong chunks in the top 20 win the aggregation. **Hierarchical retrieval over
aggregated leaf scores helps only when the leaf retrieval is already nearly right - that
is, when it is least needed.**

*Coarse pass B - an independent signal.* One embedding per paper from its title and opening
text, matched against the question directly. It cannot inherit the chunk ranking.

| | N=1 | N=2 | **N=3** | N=5 | N=10 |
|---|---|---|---|---|---|
| paper embedding | 0.382 | 0.471 | **0.529** | 0.588 | 0.706 |

Worse than aggregation, and the five that matter need N between 12 and 39 - out of 49
papers. Keeping 39 of 49 is not narrowing.

**The reason is a property of the corpus, not of the technique.** These 49 papers are
curated around one subject; at abstract level they are nearly indistinguishable, since
every one of them is about world models, latents, prediction and representation. Document
level embeddings have almost no discriminative power here. *Two-stage retrieval needs
documents that differ from each other, and a corpus curated around a single topic has that
difference only in its chunks - which is where the retrieval already looks.*

**So the ceiling is an upper bound, not available headroom.** The oracle reaches 0.824 by
using gold labels - information about where the answer is. What it demonstrates is that
ranking *within* the right paper is good; the hard part is identifying the paper, and that
needs precisely the discrimination the chunk retrieval already lacks. Revisit only for a
heterogeneous corpus, where papers differ at document level.

Total cost of reaching this conclusion: two harness flags, one pure module, 14 tests, and
about a tenth of a cent. No retriever was written.

## Stage 5 - follow-up resolution

A follow-up is a question plus a pointer into the conversation, and retrieval cannot follow
a pointer. Resolving it into a standalone question **before** the graph means the router,
retriever, grader and generator keep seeing exactly what they saw in Stage 4 - one
self-contained question - so every Stage 4 measurement stays valid.

`eval/followups.jsonl`: 18 cases, 13 dev / 5 test, written before the resolver existed.
It does not assert the resolved TEXT - no resolver produces a given wording, and exact
match on free text is unmeasurable. It asserts what the resolution must and must not
CONTAIN. Deterministic, no evaluator, and it tests the only thing that matters: was the
referent named, and was a self-contained question left alone.

| prompt | accuracy | **over-resolved** | under-resolved |
|---|---|---|---|
| **as shipped** | **10/13** | **0** | 3 |
| + an explicit three-test decision procedure | 7/13 | **2** | 4 |
| reverted | 9/13 | 0 | 4 |

Per kind: pronoun 2/4, ellipsis 2/2, answer-reference 2/3, **self-contained 4/4**.

**The two failure directions are not symmetric, and the spec is built around that.** An
under-resolved question retrieves badly and the failure is visible in the answer. An
over-resolved one retrieves well, reads well, and answers a question nobody asked. So
every failure path in `verify_resolution` returns the follow-up unchanged, and the six
negatives in the spec exist to catch the silent direction.

The second prompt added a decision procedure - "a question with no subject at all is not
standalone, even when it reads like a complete sentence" - and manufactured exactly the
failure the spec was written to catch:

    "Which papers do you have indexed?"   ->   "Which papers does IRIS have indexed?"
    "What is a recurrent state-space model?"  ->  "...used for in I-JEPA?"

The first is both nonsense and a catalog question dragged toward retrieval. Reverted.

**A spec that depends on the system's own output is only as valid as that output.**
`eval/routes.jsonl` is immune - a route depends only on the question. Four follow-up cases
depend on the previous ANSWER, and one of them (u10, an ordinal into a list the answer
produced) turned out to be unmeasurable: the turn-1 answer names PhyLatent and "No Gaussian
Required" rather than the reference answer's Sub-JEPA and Var-JEPA, because that question
is q033, a known retrieval miss. The resolver's output was correct for the conversation it
saw. The case measures retrieval wearing a resolution label.

u10 was replaced with a turn-1 that retrieves reliably (eval q010, whose answer lists
three components in a fixed order) and now passes - same prompt, same resolver, valid
input. The diagnosis held.

Residuals, both under-resolution and both visible: u04 (subject absent rather than
pronominal) and u09 (resolved "they" to "the parameters", vaguer than the pronoun it
replaced - a rewrite can fail by losing specificity, not only by not happening).

**The held-out split, spent once, with the prompt frozen: 3/5.**

| split | accuracy | over-resolved | under-resolved | Wilson 95% |
|---|---|---|---|---|
| dev (13) | 10/13 = 0.769 | 0 | 3 | [0.50, 0.92] |
| test (5) | 3/5 = 0.600 | **0** | 2 | [0.23, 0.88] |
| combined (18) | 13/18 = 0.722 | **0** | 5 | [0.49, 0.88] |

Five cases cannot distinguish 0.60 from 0.77 - the intervals overlap across almost their
whole range - so the drop is not evidence of degradation, and 0.722 is the figure to
quote. What the split does establish is the property the specification was built to
expose: **0 over-resolutions, now across 6 self-contained cases and both splits.** The
asymmetry held out of sample.

Both test misses are the two hardest kinds. u08 ("Which of those works best?") was not
rewritten at all - "those" points at a set named in the turn-1 *question*, not at anything
the answer said. u12 was rewritten to "What is the frozen evaluation protocol?": it found
"frozen" in the previous answer and missed "attentive probing". Containment scoring calls
that a miss, and the rule was fixed before the resolver existed, so it stays a miss - but
it is a near-miss, and a resolver that names half the referent is not the same failure as
one that names none.

One precision about "0 over-resolved": the test is **forbidden terms**, i.e. contamination
from the previous turn. The harness also reports that 3 of 5 questions were rewritten,
which means one of the two self-contained questions was reworded and still passed, because
nothing forbidden came in. The claim the number supports is "no contamination", not "left
untouched".

### Streaming: markers on the wire, ids in the final event

`resolve_citations` maps `[P1]` to the arXiv id of the passage it points at, and it needs
the finished text. Three ways to stream around that:

1. **stream the markers, resolve in a terminal event** - chosen;
2. buffer output at marker boundaries;
3. give the model real arXiv ids so no resolution is needed.

Option 3 is out on evidence: an early version invented `2606.09985` for a paper numbered
`2506.09985`, which is why markers exist at all.

Option 2 looked reasonable until the first live stream. A single citation arrives as four
frames:

    " ["   "P"   "3"   "]."

and a double citation as seven, including `"]["` - one marker closing and the next opening
inside a single token. Boundary buffering would have to hold output across four-frame
spans, detect a boundary mid-token, and treat `][` as both a close and an open. Option 1
needs none of it: `assemble_answer` runs once on the joined text, so the streamed and
non-streamed paths execute the *same* pure function on the *same* string. "Streaming is a
delivery change, not a different answer" is therefore a tested claim rather than an
intention.

Retrieval runs before the stream opens, so a retrieval failure is still an HTTP error;
once bytes are on the wire the status code has been sent and a generation failure has to
be reported in-band, as an `error` frame carrying the partial text.

Not routed through the agent. The router and grader run before generation, so on the agent
path they would arrive as one silent pause before the first token - worse than not
streaming. `event: step` frames driven by `graph.stream` are the way to do it, and are not
built.

Known and unfixed: roughly 130 frames for a 90-word answer, each carrying about five bytes
of text inside forty bytes of protocol. Batching into ~50 ms windows would cut that by an
order of magnitude with no visible change in smoothness - a performance change with no
measurement behind it, so it is written down rather than built.

### One client, a real timeout, and a prediction that was wrong

Eight modules each constructed their own `OpenAI()` per call, and none set a timeout or a
retry count. Reading `openai._constants` rather than guessing:

    DEFAULT_TIMEOUT       = Timeout(timeout=600, connect=5.0)   # ten minutes
    DEFAULT_MAX_RETRIES   = 2
    INITIAL_RETRY_DELAY   = 0.5     MAX_RETRY_DELAY = 8.0
    MAX_RETRY_AFTER_DELAY = 120

So a single hung call could block a request for ten minutes; with two retries, thirty; and
an agent request makes up to three calls. A 429 carrying `Retry-After: 60` sleeps a minute
inside the SDK, invisibly, twice.

**The defect was the absent bound, not an absent retry.** The SDK already retries twice
with exponential backoff and honours `Retry-After`, so a hand-written loop of three on top
would have produced nine attempts per call and fixed nothing. `openai_timeout_seconds` is
set to 30 - comfortably above the measured generate p95 (1.9 s) and the listwise reranker's
p95 (7.8 s), far below anything a caller waits for - and `openai_max_retries` is pinned at
the SDK's own 2 so it is visible rather than inherited.

**The connection-pooling gain was predicted and did not happen.** Every `OpenAI()` builds an
httpx client with its own pool, so a client per call meant a fresh TLS handshake each time;
the estimate was 100-200 ms per call.

| | eight clients | one shared client |
|---|---|---|
| generate p50 | 1254 ms | **1250 ms** |
| generate p95 | 1969 ms | 1850 ms |
| hit@1 / hit@5 / recall@5 | 0.353 / 0.676 / 0.578 | **identical** |

p50 moved 0.3%. The p95 and max look better and are one run at n = 40, which this document
has spent a week declining to read as signal. **Third performance prediction of the stage,
third settled against the guess by the harness.**

What the refactor is worth is therefore what it was designed for and not what was hoped
for: a bound on a call that had none, and one place to configure instead of eight. A
structural test greps the source for `OpenAI(` so a new module cannot silently inherit the
ten-minute default again.

The retrieval numbers being byte-identical is the result that mattered for a change
touching eight modules.

### Time to first token: 657 ms against a 1764 ms answer

Streaming was built on an argument - the user feels time-to-first-token, not total
latency - and an argument is not a measurement. `ttft_ms` is recorded from **request
arrival**, not from the start of generation: retrieval and follow-up resolution run first
and the user waits through those too. Five consecutive requests, same question, warm
embedding cache:

| run | ttft_ms | duration_ms | upstream_ms |
|---|---|---|---|
| 1 (first after restart) | 1351.1 | 2186.2 | 975.6 |
| 2 | 656.9 | 1784.7 | 651.2 |
| 3 | 1115.4 | 1764.1 | 1107.3 |
| 4 | 656.2 | 1363.5 | 651.4 |
| 5 | 614.7 | 1284.5 | 610.1 |

Median 657 ms to the first word against 1764 ms for the whole answer: **the reader waits
37% of the request before something appears**, and the definition-of-done row ("well under
the 1.2 s generate p50") is met at n = 5.

Two readings inside the table are worth more than the median.

**Run 1 spends 375 ms that runs 2-5 spend in 5.** Subtract `upstream_ms` from `ttft_ms`:
the first request after a restart does 375 ms of work outside the provider call, the rest
do about five. That is in-process first-request warm-up - imports, first numpy touch,
tokenizer - and it is the same effect measured separately on `/ask`, where a cold,
cache-missing request took 5486 ms against 1828 ms warm.

**Run 3 is the provider, not us.** Its `ttft_ms` of 1115 ms tracks its `upstream_ms` of
1107 ms exactly. Our own code contributes single-digit milliseconds to time-to-first-token;
everything else is the model's first chunk, and it varies by a factor of two between
consecutive identical requests. A latency budget written against the median here would be
wrong most of the time.

Caveat, and it is a large one: five samples of one question over loopback, with no network
between client and server. This measures the server, not the experience.

### Counting what the SDK does quietly

The API had no equivalent of `scripts/eval.py`'s failure banner - the thing that caught
forty straight 403s from an unpropagated model permission, and a connection drop that
killed a paid run. A request that succeeded first try and one that succeeded after two
retries and a ninety-second `Retry-After` sleep were indistinguishable from outside.

Every request now emits one JSON line. The retry count comes from an **httpx event hook on
the shared client**, not from wrapping the call sites, because the SDK's retries happen
*below* `chat.completions.create` - by the time it returns they are over. The hook reads
`x-stainless-retry-count`, which the SDK stamps on every attempt, and that header is the
only thing at the transport layer separating a retry from a second logical call. So
`calls` and `retries` are exact with no change to any of the eight call sites.

Forcing the failure with `PT_OPENAI_TIMEOUT_SECONDS=0.001`:

    {"request_id":"c49b7480437c","attempts":3,"calls":1,"retries":2,"upstream_ms":0.0,
     "code":"upstream_timeout","status":504,"duration_ms":1808.0,"outcome":"error"}

One logical call, two silent retries, 1808 ms of wall clock the caller paid for and
nothing previously recorded.

`calls` also turns out to be what makes every latency number in this document
interpretable. A cold request embeds its query (2 calls); a repeated one reads the
embedding from cache (1 call) and its retrieval step drops from 3301 ms to 2.5 ms. Every
figure here was measured with a warm cache, which makes them **warm-cache latency** - and
`calls` is the field that says which regime a line belongs to.

A claim in the first version of this work was wrong and is corrected here: those retries
were called *silent*. They are not - the SDK logs `Retrying request in 0.49 seconds` at
INFO. They were invisible because **nothing in the API ever configured logging**, so no
INFO record from any library was printed; `index loaded: 874 chunks` had been dropping
since Stage 0. What the JSON line adds over the SDK's own is attribution: the SDK's record
carries no request id, no route and no total, so it cannot say which request paid the wait
or how much of that request the wait was.

### When the provider fails

The SDK retries twice and then raises. What happened next used to be an untyped 500.

| upstream | returned | code |
|---|---|---|
| 429 rate limit | 503 | `rate_limited` |
| timeout | 504 | `upstream_timeout` |
| connection error | 502 | `upstream_unreachable` |
| 500-599 | 502 | `upstream_error` |
| 401, 403, 400 | 500 | `misconfigured` |

**The upstream status is not the status returned**, and proxying the number through is
wrong in both directions. A 429 says *the caller* is sending too fast; the quota here is
the service's, and a well-behaved client would back off for a limit it had no part in.
A 401 or 403 says the caller's credentials are bad; they are ours - that is exactly the
unpropagated model permission that cost 40 calls in Stage 2, and telling a caller to fix
its request would have sent it looking in the wrong place.

Three details that each cost a test. The provider's message never reaches the response -
upstream bodies quote organisation ids, model names and quota, and an error path is the
least-watched place for those to start appearing. `Retry-After` is echoed only when
upstream sends a parseable one, because a default is a guess presented as a fact and every
client that trusted it would retry into the same wall at the same moment. And
`APITimeoutError` subclasses `APIConnectionError`, so an `isinstance` chain in the wrong
order reports every timeout as an unreachable host: the same symptom, a different cause,
and a day spent looking at the wrong layer.

A stream cannot change its status code - 200 left with the first byte - so the same
taxonomy arrives in the `error` frame, where `code` means exactly what it means on the
non-streaming path.

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

## Stage 6 - the container

| change | image | rebuild after a source edit |
|---|---|---|
| first working image | 574 MB | 15.5 s |
| dependencies the API never imports moved to an `[ingest]` extra | **472 MB** | 15.5 s |
| dependency layer keyed on `pyproject.toml` alone | 472 MB | **3.3 s** |
| multi-stage build | 471 MB | - |

**The build context is 11 MB, from 1.6 GB.** `.dockerignore` excludes the 1.3 GB
virtualenv (macOS arm64 wheels, useless in a Linux image), 272 MB of source PDFs (the
index is the artefact; the PDFs are how it was made) and `.env`. The key never enters the
build at all - it arrives at runtime through the environment, because `docker history`
shows every layer to anyone who pulls the image.

**102 MB of the image was dependencies the running service never imports.** `pymupdf`
(63 MB) appears only in `ingestion/pdf_parser.py` and `tiktoken` (4 MB) only in
`ingestion/chunker.py`; the API reads a prebuilt index and parses nothing. Both moved to
an `[ingest]` extra that the container does not install. The predicted saving was 67 MB
and the measured one was 102 MB - the gap is transitive dependencies (`tiktoken` pulls
`regex` and `requests`, which pull four more), none individually large enough to appear in
a top-12 listing. `tests/test_packaging.py` imports `arxiv_rag.api.main` **in a
subprocess** and asserts none of them is in `sys.modules`, because a single convenience
import in a future module would silently put 67 MB back, visible only as a bigger image
nobody looks at.

**Layer order is worth 12 seconds on every rebuild.** Docker invalidates every layer after
the first changed one, so copying the source before installing dependencies re-ran the
whole install on every edit. Reordered, the dependency layer is keyed on `pyproject.toml`
alone and a source change costs 2.5 s to reinstall the package. The dependency list is
read out of `pyproject.toml` at build time with `tomllib` rather than duplicated into a
`requirements.txt`: a second list is a second thing to forget, and the image would keep
building happily while installing something the project no longer declares.

### Multi-stage: predicted to save nothing, measured at 1 MB, deleted

Written as an experiment with the decision rule fixed in advance - under 20 MB and it goes
- and with the expectation recorded before the build: multi-stage pays when the build
needs a toolchain the runtime does not, and every dependency here installs from a prebuilt
wheel. 471 MB against 472 MB.

**The instructive part is why the one lever that should have worked did not.** The runtime
stage ran `pip uninstall -y pip setuptools wheel` against the base image's own installer,
about 15 MB. It saved nothing, because **deleting a file in a later layer does not shrink
an image**: the bytes stay in the layer that added them and the deletion is a whiteout
marker in a new layer on top. The image carries both. Meanwhile the venv in the build
stage installed a *second* pip, which the builder then removed - earning back exactly the
weight the venv had introduced. Two moves that cancel, and 1 MB of noise.

The container did answer `/health` with `pip`, `setuptools` and `wheel` gone, so nothing
in the dependency tree needs `pkg_resources` at runtime. That is now known rather than
assumed, which is the other thing a rejected experiment is good for.

This is the second performance prediction in the project to be written down before the
measurement, and the first to be right. The one before it - that sharing one HTTP client
would save 100-200 ms per call through connection reuse - moved p50 by 4 ms.

### A write that had to become optional, and the rule it drew

Running the container revealed something reading the code had only predicted: on every
question whose query had not been embedded before, the service **wrote to disk** -
`embedding cache: saved 1 entries`, into a layer that dies with the container. Read-only
or as a non-root user, that write is an `OSError` in the middle of a request.

Three options: bake the 30 MB cache into the image (stale the moment anyone asks something
new), mount a volume (awkward on platforms with ephemeral filesystems), or stop
persisting. Stopping won, and the measurement afterwards made it look better than the
argument for it did.

**The rule it draws is worth keeping: a write the system depends on must fail loudly, a
write that only makes it cheaper must degrade quietly and say so once.** Returning 500
because an optimisation could not persist trades a real failure for an imaginary one. So
there are two mechanisms, not one - a setting (`PT_EMBEDDING_CACHE_WRITES=false`, set in
the image) and an `OSError` guard that disables writes for the life of the process rather
than retrying on every miss, because a read-only filesystem does not become writable while
you watch it.

**A prediction here was wrong.** "Every question pays an embedding call, forever" - it does
not. The cache is still a cache **in memory**; only persistence across restarts is lost.
Measured in the container: an unseen question costs 5.64 s, the same question again costs
1.87 s, against 5.49 s and 1.83 s on the host. The real price is one embedding call per
distinct question per container lifetime, which for a ten-question demo is ten calls per
deploy.

### The three failure modes, checked against the container

`make docker-checks` starts a container per case and asserts the behaviour Stage 5 defined
against fakes:

| case | how | expected |
|---|---|---|
| no index | an empty tmpfs over `/app/data/index` | `/health` 200 with `chunks_indexed: 0`, `/ask` **503** |
| no API key | no environment at all | `/health` 200 with `openai_key_configured: false`, `/ask` **500 `misconfigured`** |
| a wrong API key | `OPENAI_API_KEY=sk-not-a-real-key-000` | `/ask` **500 `misconfigured`**, and the key absent from the body |

**Two defects, neither of which 456 unit tests could see.**

The first: with no key configured, the SDK raises a bare `OpenAIError` from its
*constructor* - `attempts: 0`, no request attempted, outside its typed hierarchy - which
classified as `internal_error`. That is what you report when you do not know, and the
service did know: `/health` has reported whether a key is set since Stage 0. Now
`get_client` raises `MissingAPIKey` from configuration, before the SDK is asked anything,
and it classifies as `misconfigured`.

The second was in the check itself. The assertion read `body_has '"code"'` - that a `code`
field EXISTS - which passed while the code said `internal_error`. An assertion that asks
less than the question it stands in for is the same defect as a metric measuring the wrong
thing, and this document has a chapter of those.

A third was a trap rather than a defect: the check set `PT_OPENAI_API_KEY`, but the
setting carries `validation_alias="OPENAI_API_KEY"` so the `PT_` prefix does not apply.
The "wrong key" case was silently testing a *missing* key for the second time.

**The lesson for the stage: unit tests cannot see configuration, and configuration is most
of what a container changes.**

### What the image is

| | |
|---|---|
| size | **472 MB** (`python:3.12-slim`) |
| build context | 11 MB, from 1.6 GB |
| cold start to a healthy `/health` | **1.67 s** |
| runs as | uid **10001**, non-root |
| survives | `--read-only --cap-drop ALL --security-opt no-new-privileges` |
| secrets in layers | none - `docker history` greps to 0, and the image's environment carries no key |
| healthcheck | `python -c urllib`, no curl installed to answer a question the interpreter can; `attempts: 0`, so it never calls the provider |

The healthcheck detail matters more than it looks: one that reached OpenAI would bill on
every probe and report the container unhealthy during someone else's outage.

**pgvector was cut**, and the reason is a number rather than a preference: the index is
7.2 MB and retrieval runs at 2.4 ms p95, so a network hop per query would be slower than
the thing it replaced. If it is ever done, the bar is that `make retrieval-eval` produces
byte-identical metrics - the same bar the shared-client refactor was held to.

---

## Stage 7 - the router, measured at last

Stage 4 shipped a router and never evaluated it. `scripts/eval.py` deliberately did not
pass `store=`, on the argument that all 40 eval questions are corpus-content questions, so
routing could not change an answer. Stage 7 produced a counterexample by accident - while
recording answers for the landing page, the router scoped a two-paper comparison to the
wrong pair of ids and turned a correct answer into a refusal - so `--router` now exists and
the shipped configuration is measurable.

Three runs. The first two are the SAME configuration two days apart, which is the control
that makes the third readable:

| | agent, 16 Sep | agent, 18 Sep | **agent + router, 18 Sep** |
|---|---|---|---|
| hit@1 | 0.353 | 0.353 | 0.324 |
| hit@5 | 0.676 | 0.676 | **0.735** |
| recall@5 | 0.578 | 0.578 | 0.593 |
| correctness | 0.323 | **0.375** | 0.500 |
| faithfulness | 0.677 | 0.688 | 0.750 |
| refusal accuracy | 0.900 | **0.925** | 0.875 |

**Retrieval is deterministic; the judge is not.** Two runs of the identical configuration
produced byte-identical retrieval numbers and correctness 5 points apart. That 5-point
band is the noise floor for this eval set, measured rather than assumed, and it is the
number every comparison below has to clear.

**What the router demonstrably buys: +2 questions on hit@5**, q029 and q033, both
comparison questions spanning two named papers. The paper filter scopes retrieval to those
papers and the gold chunk comes into the top 5. Deterministic, attributable, and exactly
what the component was built for.

**What it might buy: correctness.** 0.500 against the same-day control's 0.375 is +4
questions, against a noise floor of ~1.7 questions. Suggestive; not established at n=32
with an instrument this unstable. Settling it needs three runs of each configuration -
about $0.60 - and has not been done.

**What it costs, and the cost is hidden in the wrong column.** Refusal accuracy fell 0.925
-> 0.875: **q004 and q038, both deliberately unanswerable, were answered** under routing
when the same questions were refused without it. Refusal rate went 0.833 -> 0.500 - three
of six unanswerable questions answered. Scoping an unanswerable question to a plausible
paper hands the generator something that looks like evidence, and it takes it.

That is the trade in one line: **routing finds two more comparison answers and invents two
more answers that should not exist.** For a system whose headline claim is that it refuses
what it cannot support, that is not a favourable exchange, and it is why the public demo
runs the pipeline and uses the agent only where routing IS the feature - catalog questions,
which take no model call at all, and questions naming a paper by an alias the corpus titles
do not contain.

**The near-miss worth recording.** The first comparison available was the routed run
against a control from two days earlier: correctness 0.323 -> 0.500, an 18-point gain that
would have been written up as the largest quality improvement since fusion weighting. A
third of it was the calendar. The control run cost five cents.

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
7. **Every latency figure here is warm-cache latency.** A first-time question also
   pays an embedding call: measured at 5486 ms against 1828 ms for the same question
   warm, and retrieval alone at 3301 ms against 2.5 ms. The `calls` field on each log
   line is what distinguishes the two regimes.
8. **`upstream_ms` undercounts, in two opposite situations.** A timeout produces no
   response, so the httpx response hook never fires and the field reads 0.0 for the
   failure that cost the most wall clock (1808 ms in the example above). And for a
   streamed completion the hook fires when the response *headers* arrive, so the time
   spent reading the body is not counted. Both are visible as a gap between
   `upstream_ms` and `duration_ms`; neither is fixed.
9. **Time to first token is 5 samples of one question over loopback.** No network
   between client and server, one corpus, one model. Treat 657 ms as an order of
   magnitude, not a p50.
10. **`false_refusal_rate` conflates two causes.** An over-cautious prompt and a genuine
   retrieval failure that the generator handled honestly both land in the same bucket.
   q002 refused with its gold chunk at rank 102, which is arguably correct behaviour.
