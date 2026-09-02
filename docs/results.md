# Evaluation results

All numbers come from `make eval`, run against a hand-written evaluation set of
questions over the indexed arXiv corpus. The set includes deliberately unanswerable
questions to measure hallucination.

**Metrics**

- **Recall@5** — share of questions where at least one gold chunk is in the top 5.
- **MRR** — mean reciprocal rank of the first gold chunk.
- **Faithfulness** — share of answers whose every claim is supported by the retrieved
  context (LLM-judged).
- **Refusal accuracy** — share of unanswerable questions correctly refused.

## Retrieval

| Configuration | Recall@5 | MRR | p95 latency |
|---|---|---|---|
| _baseline: dense top-k — pending Stage 2_ | | | |

## Answer quality

| Configuration | Faithfulness | Refusal accuracy |
|---|---|---|
| _pending Stage 2_ | | |

## What did not work

<!-- Keep this section. Techniques you tried and dropped, with the numbers that made
     you drop them, are stronger evidence of judgement than a longer list of wins. -->
