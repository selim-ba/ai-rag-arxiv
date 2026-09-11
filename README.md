# Agentic RAG over arXiv

Agentic RAG over arXiv papers: ask a research question, get an answer grounded in real
papers with citations — from an agent that decides how to search, judges what it found,
and retries when the retrieval was bad.

> **Status: under construction** — built and measured stage by stage.

## Stack

Python 3.11 · FastAPI · LangChain / LangGraph · OpenAI · pgvector · Docker

## Quickstart

```bash
cp .env.example .env      # add your OpenAI key
make install
make test

make ingest               # 49 papers from corpus.txt -> 874 chunks
make index                # embed and build the vector index
make eval                 # run the 40-question evaluation set

make dev                  # http://localhost:8000/docs
```

Ask it something:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How does PlaNet search for a good action sequence?"}'
```

```json
{
  "answer": "PlaNet searches using the Cross Entropy Method (CEM) ... re-fitting the
             belief to the top action sequences over several iterations [1811.04551].",
  "citations": ["1811.04551", "2107.08241"],
  "refused": false,
  "sources": [{"chunk_id": "1811.04551::4", "title": "Learning Latent Dynamics for
                Planning from Pixels", "url": "https://arxiv.org/abs/1811.04551"}]
}
```

Citations cannot be fabricated: the model cites passage markers and the arXiv id is
substituted in code from the retrieved set.

## Layout

```
src/arxiv_rag/
  config.py           typed settings, validated at startup
  api/main.py         FastAPI app
  ingestion/          arXiv -> PDF -> clean text -> chunks       (Stage 1)
  retrieval/          embeddings, hybrid search, reranking       (Stages 2-3)
  agent/              LangGraph research agent                   (Stage 4)
scripts/ingest.py     ingestion CLI
tests/                pytest; the Stage 1 tests are the spec
docs/results.md       retrieval and answer-quality benchmarks
```

## Results

Measured on a hand-built evaluation set of 40 questions (20 factual, 10 comparison,
4 definitional, 6 deliberately unanswerable) over 874 chunks from 49 papers.
Full methodology, judge design and limitations in [`docs/results.md`](docs/results.md).

| Configuration | hit@5 | Recall@5 | MRR@5 | Faithfulness | Refusal acc. |
|---|---|---|---|---|---|
| Dense only (Stage 2 baseline) | 0.618 | 0.520 | 0.434 | 1.000 | 0.900 |
| Hybrid + rerank (Stage 3) | _pending_ | | | | |

**The system is retrieval-bound.** Splitting answer quality by whether the gold chunk
was retrieved:

| | n | Correctness | Faithfulness |
|---|---|---|---|
| Gold chunk in top-5 | 20 | **0.550** | 1.000 |
| Gold chunk not in top-5 | 10 | **0.000** | 1.000 |

Every correct answer came from a successful retrieval. Faithfulness is 1.000 either way —
the generator reports its passages accurately whether or not they are the right passages. All six unanswerable questions were refused, including a false-premise
one. That is what Stage 3 is aimed at.
