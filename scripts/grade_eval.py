"""Measure the grader on its own, before it controls anything.

    python -m scripts.grade_eval

The grader decides whether the agent retries. Wiring it to an edge before knowing how
often it is right means an agent doing the wrong thing faster - and this project has
already shipped two graders that looked fine and were not (the Stage 2 judge grading one
axis twice, the gold auditor fabricating its own evidence).

**What is being compared.** For each of the 34 answerable questions, retrieval runs and it
is already known whether a gold chunk landed in the top k. That is the reference:

    gold retrieved  ->  the passages really can answer the question
    gold missed     ->  they probably cannot

**Why "probably".** Gold labels are incomplete - measured in Stage 2, a valid source is
sometimes unlabelled, and the audit only checked the top 10. So a grader saying "relevant"
when gold was missed is not automatically wrong: it may have found a real answer nobody
labelled. Those cases are worth reading, not counting as errors. The number that is not
ambiguous is the other direction.

**The number that matters for the agent** is *catch rate*: when gold was genuinely missed,
how often does the grader notice? That is what decides whether the retry loop ever fires on
the questions that need it. Its cost twin is *false alarm rate*: how often it demands a
retry on retrieval that was already fine.
"""

import json
from pathlib import Path
from time import perf_counter

from arxiv_rag.agent.grader import Grader
from arxiv_rag.config import get_settings
from arxiv_rag.evaluation.metrics import hit_at_k
from arxiv_rag.evaluation.timing import summarise
from arxiv_rag.retrieval.bm25 import BM25Index
from arxiv_rag.retrieval.hybrid import DenseRetriever, HybridRetriever
from arxiv_rag.retrieval.store import ChunkStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = get_settings()
    store = ChunkStore.load(settings.index_dir)
    retriever = HybridRetriever(
        [DenseRetriever(store, settings), BM25Index(store.chunks)],
        depth=settings.fusion_depth,
    )
    grader = Grader(settings)

    path = ROOT / "eval" / "questions.jsonl"
    questions = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    rows, timings = [], []
    print(f"grading {len(questions)} questions with {settings.grader_model} ...\n")
    for q in questions:
        hits = retriever.search(q["question"], k=settings.top_k)
        started = perf_counter()
        grade = grader.grade(q["question"], hits)
        timings.append((perf_counter() - started) * 1000)
        rows.append(
            {
                "id": q["id"],
                "answerable": q["answerable"],
                "gold_retrieved": hit_at_k(
                    [h.chunk.chunk_id for h in hits], q["gold_chunk_ids"], settings.top_k
                )
                if q["gold_chunk_ids"]
                else None,
                "relevant": grade.relevant,
                "missing": grade.missing,
            }
        )

    answerable = [r for r in rows if r["answerable"]]
    good = [r for r in answerable if r["gold_retrieved"]]
    bad = [r for r in answerable if not r["gold_retrieved"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    catch = sum(1 for r in bad if not r["relevant"]) / len(bad) if bad else 0.0
    false_alarm = sum(1 for r in good if not r["relevant"]) / len(good) if good else 0.0
    agree = sum(1 for r in answerable if r["relevant"] == bool(r["gold_retrieved"]))

    print("=" * 62)
    print(f"{'':34}{'n':>4}  grader says relevant")
    print(f"{'gold retrieved (retrieval worked)':34}{len(good):>4}  "
          f"{sum(r['relevant'] for r in good) / len(good):.3f}   <- want high")
    print(f"{'gold missed (retrieval failed)':34}{len(bad):>4}  "
          f"{sum(r['relevant'] for r in bad) / len(bad):.3f}   <- want low")
    if unanswerable:
        rate = sum(r["relevant"] for r in unanswerable) / len(unanswerable)
        print(
            f"{'unanswerable (no gold exists)':34}{len(unanswerable):>4}  "
            f"{rate:.3f}   <- want low"
        )

    print(f"\ncatch rate      {catch:.3f}   bad retrieval the grader flags  (drives the retry)")
    print(f"false alarm     {false_alarm:.3f}   good retrieval it flags anyway  (wasted retries)")
    print(f"agreement       {agree}/{len(answerable)} = {agree / len(answerable):.3f}")
    t = summarise(timings)
    print(f"latency         p50 {t['p50']:.0f} ms   p95 {t['p95']:.0f} ms   max {t['max']:.0f} ms")

    print("\ndisagreements - read these, do not just count them:")
    for r in answerable:
        if r["relevant"] != bool(r["gold_retrieved"]):
            kind = (
                "said RELEVANT but gold was missed"
                if r["relevant"]
                else "said IRRELEVANT but gold was there"
            )
            print(f"  {r['id']:<6} {kind}")
            if r["missing"]:
                print(f"         missing: {r['missing'][:110]}")


if __name__ == "__main__":
    main()
