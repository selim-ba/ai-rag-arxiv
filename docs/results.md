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

## PDF text extraction

Extraction quality bounds everything downstream, so the parser was chosen by
measurement rather than by default. Metric: share of alphabetic words 20+ characters
long — a proxy for lost inter-word spaces, which tokenize into noise and are invisible
to keyword search.

Six papers, spanning 2018 to 2026 typesetting:

| Parser | Words recovered | Glued runs | Worst paper | Licence |
|---|---|---|---|---|
| **PyMuPDF** | **41,054** | **0 (0.00%)** | 0.00% | AGPL-3.0 |
| pypdf | 40,726 | 49 (0.12%) | 0.77% | BSD |
| pdfplumber | 18,152 | 950 (5.23%) | 10.73% | MIT |

PyMuPDF was the only parser with no failures, and it recovered ~19% more words from the
worst-affected paper — pypdf had been fusing whole clauses into single tokens
(`latenthasnointernalstructureforcarryingthebeliefoverhiddencontinuationsthroughblindrollout`).
pdfplumber's low word count is the same defect at greater scale.

PyMuPDF is AGPL-3.0. That is acceptable here because this repository is public and the
deployed service's source is available; a proprietary product would need the commercial
licence or pypdf.

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
