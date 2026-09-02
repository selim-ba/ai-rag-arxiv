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
make dev                  # http://localhost:8000/docs
```

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

Measured on a hand-built evaluation set of 40 questions over the indexed corpus.
Full methodology and per-technique breakdown in [`docs/results.md`](docs/results.md).

| Configuration | Recall@5 | MRR | Faithfulness | p95 latency |
|---------------|----------|-----|--------------|-------------|
| _pending Stage 2_ | | | | |
