"""Run the evaluation set end to end and write the numbers down.

    python -m scripts.eval                  # everything, with the judge
    python -m scripts.eval --split dev      # dev questions only
    python -m scripts.eval --no-judge       # retrieval + refusal only, no judge calls
    python -m scripts.eval --limit 5        # smoke test before spending money

Every question's full record is appended to ``data/eval/run-<timestamp>.jsonl`` **as it
completes**, not at the end. A rate-limit thirty questions in then costs you one
question, not thirty - which is the only protection you have, since the API call inside
``judge_answer`` is deliberately unguarded.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from arxiv_rag.config import get_settings
from arxiv_rag.evaluation.judge import RefusalRecord, judge_answer, refusal_scores
from arxiv_rag.evaluation.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank
from arxiv_rag.evaluation.timing import summarise
from arxiv_rag.retrieval.answer import answer_question, format_context, refusal_token_misplaced
from arxiv_rag.retrieval.hybrid import DenseRetriever, build_hybrid
from arxiv_rag.retrieval.store import ChunkStore, SearchHit

ROOT = Path(__file__).resolve().parents[1]


def load_questions(split: str) -> list[dict]:
    path = ROOT / "eval" / "questions.jsonl"
    questions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if split != "all":
        questions = [q for q in questions if q["split"] == split]
    return questions


def rebuild_context(chunk_ids: list[str], by_id: dict) -> str:
    """Re-render the passage block the model saw, for the judge to grade against.

    ``format_context`` only reads ``hit.chunk``, so a zero score is fine here - we are
    reconstructing what was shown, not re-scoring it.
    """
    return format_context([SearchHit(by_id[cid], 0.0) for cid in chunk_ids if cid in by_id])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=["all", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--retriever",
        default="hybrid",
        choices=["dense", "hybrid", "hybrid+rerank"],
        help="which retriever the generator is fed by (default: the shipped one)",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="route through the LangGraph agent instead of calling answer_question directly",
    )
    parser.add_argument(
        "--router",
        action="store_true",
        help="with --agent, also turn the router on: the configuration the API ships",
    )
    args = parser.parse_args()

    settings = get_settings()
    store = ChunkStore.load(settings.index_dir)
    by_id = {c.chunk_id: c for c in store.chunks}

    dense = DenseRetriever(store, settings)
    if args.retriever == "dense":
        retriever = dense
    else:
        retriever = build_hybrid(store, settings)
        if args.retriever == "hybrid+rerank":
            from arxiv_rag.retrieval.rerank import LLMListwiseReranker, RerankingRetriever

            retriever = RerankingRetriever(
                retriever, LLMListwiseReranker(settings), depth=settings.fusion_depth
            )
    questions = load_questions(args.split)[: args.limit]

    # One callable either way, so nothing below this line knows which path it is on. At
    # step 1 of Stage 4 the agent is retrieve -> generate and MUST produce identical
    # numbers: it calls the same retriever and the same generate_answer. A difference here
    # is a wiring bug, not a smarter agent.
    if args.agent:
        from arxiv_rag.agent.graph import build_graph, run_agent

        # `store=` is what turns the router on. It was left out on the argument that all 40
        # questions are corpus-content questions, so routing could not change an answer.
        # Stage 7 produced a counterexample while recording the demo: on a comparison
        # question spanning two papers the router scoped retrieval to the wrong pair of
        # ids and turned a correct answer into a refusal. So the configuration the API
        # actually ships is now measurable, at the cost of one extra model call a question.
        graph = build_graph(retriever, settings, store=store if args.router else None)

        def answer_for(question: str):
            return run_agent(graph, question)
    else:

        def answer_for(question: str):
            return answer_question(question, retriever, settings)

    out_dir = ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"run-{stamp}.jsonl"

    records: list[dict] = []
    mode = "agent+router" if args.agent and args.router else ("agent" if args.agent else "pipeline")
    print(f"{len(questions)} questions, {mode}, retriever={args.retriever} -> {out_path.name}\n")

    failures: list[tuple[str, str]] = []
    for i, q in enumerate(questions, start=1):
        # A run is 40 paid generations. A single dropped connection at question 30 used to
        # raise out of the loop and throw away the other 29 - the jsonl survived, but no
        # summary was ever printed and the run had to be paid for again. The grader has
        # counted its own failures since Stage 4 for exactly this reason; the harness that
        # spends the most money had no such thing.
        #
        # Skipped questions are EXCLUDED from every metric rather than scored as failures.
        # A network error is not evidence about the retriever, and quietly recording it as
        # a miss would make an outage look like a regression.
        try:
            answer = answer_for(q["question"])
        except Exception as exc:  # noqa: BLE001 - any transport error, deliberately
            failures.append((q["id"], f"{type(exc).__name__}: {exc}"))
            print(f"{i:3}/{len(questions)}  {q['id']}  FAILED  {type(exc).__name__}")
            continue
        gold = q["gold_chunk_ids"]
        retrieved = answer.retrieved_ids

        record = {
            "id": q["id"],
            "kind": q["kind"],
            "split": q["split"],
            "answerable": q["answerable"],
            "question": q["question"],
            "answer": answer.text,
            "citations": answer.citations,
            "refused": answer.refused,
            # The refusal contract, counted rather than assumed. See `refusal_token_misplaced`.
            "refusal_misplaced": refusal_token_misplaced(answer.text),
            "retrieved_ids": retrieved,
            "gold_chunk_ids": gold,
            "retrieve_ms": round(answer.retrieve_ms, 1),
            "generate_ms": round(answer.generate_ms, 1),
        }

        # Only on the agent path. `report` reads the run's shape off the records rather
        # than being told it: a pipeline record has no `attempts` key at all, which is
        # different from having one that reads 0. The pipeline cannot retry, so "it
        # retried zero times" is a category error, not a measurement.
        if args.agent:
            record |= {
                "attempts": answer.attempts,
                "first_retrieved_ids": answer.first_retrieved_ids,
                "final_query": answer.final_query,
                # Recorded so the report can tell "the loop was off" from "the grader
                # approved everything". Those have identical attempt counts and opposite
                # meanings, and the report guessed wrong the first time it had to say.
                "max_retries": settings.max_retries,
            }

        if gold:
            record |= {
                "gold_groups_satisfied": sum(1 for group in gold if set(retrieved) & set(group)),
                "gold_groups_total": len(gold),
                "hit@1": hit_at_k(retrieved, gold, 1),
                "hit@5": hit_at_k(retrieved, gold, 5),
                "recall@5": recall_at_k(retrieved, gold, 5),
                "rr": reciprocal_rank(retrieved, gold),
            }
            # The only thing that answers "did the retry earn anything?". Comparing the
            # FIRST retrieval against the final one, per question, because the aggregate
            # cannot: a rescue and a loss cancel out and hit@5 does not move.
            if answer.attempts:
                record["hit@5_first"] = hit_at_k(answer.first_retrieved_ids, gold, 5)

        # Judging a refusal for correctness is meaningless: whether it *should* have
        # refused is already measured, exactly, by refusal_scores.
        if q["answerable"] and not answer.refused and not args.no_judge:
            verdict = judge_answer(
                q["question"],
                rebuild_context(retrieved, by_id),
                answer.text,
                q["reference_answer"],
                settings,
            )
            record |= {
                "faithful": verdict.faithful,
                "correct": verdict.correct,
                "faithful_reason": verdict.faithful_reason,
                "correct_reason": verdict.correct_reason,
                "judge_inconsistent": verdict.inconsistent,
            }

        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        records.append(record)

        mark = "REF" if answer.refused else "ans"
        judged = ""
        if "faithful" in record:
            judged = f"  faithful={str(record['faithful']):5} correct={str(record['correct']):5}"
        hit = record.get("hit@5", "-")
        print(f"{i:3}/{len(questions)}  {q['id']}  {mark}  hit@5={hit}{judged}")

    if failures:
        # Loud, and above the numbers rather than below them. Quote this next to any
        # figure from this run: n is smaller than it looks.
        print(f"\n!! {len(failures)}/{len(questions)} questions FAILED and were skipped")
        for qid, why in failures[:5]:
            print(f"   {qid}  {why}")
        print(
            f"   every number below is computed over {len(records)} questions, not "
            f"{len(questions)} - do not compare it with a complete run."
        )

    report(records)


def report(records: list[dict]) -> None:
    scored = [r for r in records if r.get("gold_chunk_ids")]
    judged = [r for r in records if "faithful" in r]
    refusal = refusal_scores(
        [
            RefusalRecord(
                question_id=r["id"],
                should_answer=r["answerable"],
                did_answer=not r["refused"],
            )
            for r in records
        ]
    )

    print("\n" + "=" * 60)
    print(f"RETRIEVAL   (n={len(scored)}, k=5)")
    print(f"  hit@1     {mean([float(r['hit@1']) for r in scored]):.3f}")
    print(f"  hit@5     {mean([float(r['hit@5']) for r in scored]):.3f}")
    print(f"  recall@5  {mean([r['recall@5'] for r in scored]):.3f}")
    print(f"  MRR@5     {mean([r['rr'] for r in scored]):.3f}")

    if judged:
        bad = [r for r in judged if r.get("judge_inconsistent")]
        print(f"\nANSWER QUALITY   (n={len(judged)}, answered questions only)")
        print(f"  faithfulness  {mean([float(r['faithful']) for r in judged]):.3f}")
        print(f"  correctness   {mean([float(r['correct']) for r in judged]):.3f}")
        # Verdicts the judge produced in violation of its own output contract, twice.
        # Quote this next to the scores: it is the error bar on them.
        print(
            f"  judge broke its contract: {len(bad)}/{len(judged)}"
            f"  {[r['id'] for r in bad] if bad else ''}"
        )

    print(f"\nLATENCY   (n={len(records)}, milliseconds)")
    for label, key in (("retrieve", "retrieve_ms"), ("generate", "generate_ms")):
        values = [r[key] for r in records if r.get(key)]
        if values:
            t = summarise(values)
            print(f"  {label:20} p50 {t['p50']:8.1f}  p95 {t['p95']:8.1f}  max {t['max']:8.1f}")

    retried = [r for r in records if r.get("attempts", 0)]
    if retried:
        # Three outcomes, not one. A loop is only earning its latency if rescued > lost,
        # and an aggregate hit@5 that did not move is equally consistent with "nothing
        # happened" and "three rescues cancelled three losses".
        with_gold = [r for r in retried if "hit@5_first" in r]
        rescued = [r for r in with_gold if r["hit@5"] and not r["hit@5_first"]]
        lost = [r for r in with_gold if r["hit@5_first"] and not r["hit@5"]]
        unchanged = len(with_gold) - len(rescued) - len(lost)
        # A retry that returns the same ids is pure waste - a rewrite call, an embedding
        # call and a second retrieval for a list you already had. Distinct from a retry
        # that found different passages and still missed: that one says the rewrite works
        # and retrieval has a ceiling, which is a different problem with a different fix.
        # Compare as SETS, not lists. Measured: q013's retry returned the same five chunks
        # in a different order, which an `==` on lists calls a change and which is, for
        # every metric in this file and for the generator, not one.
        inert = [r for r in retried if set(r["first_retrieved_ids"]) == set(r["retrieved_ids"])]
        print(f"\nLOOP   (n={len(records)})")
        rate = len(retried) / len(records)
        print(
            f"  retried              {len(retried)}/{len(records)} ({rate:.3f})  "
            f"{[r['id'] for r in retried]}"
        )
        print(f"  no-gold to compare   {len(retried) - len(with_gold)}")
        print(f"  rescued              {len(rescued)}  {[r['id'] for r in rescued]}")
        print(f"  lost                 {len(lost)}  {[r['id'] for r in lost]}")
        print(f"  no change            {unchanged}")
        print(f"  same chunks back     {len(inert)}  {[r['id'] for r in inert]}")
        print("\n  rewrites - read these by hand:")
        for r in retried:
            print(f"    {r['id']}  {r['question'][:58]}")
            print(f"          -> {r.get('final_query', '')[:80]}")
    elif any("attempts" in r for r in records):
        cap = next((r["max_retries"] for r in records if "max_retries" in r), None)
        print(f"\nLOOP   (n={len(records)})")
        if cap == 0:
            # The grade node still ran and still formed verdicts; the edge ignored them.
            print(f"  retried              0/{len(records)} - loop disabled (max_retries=0)")
        else:
            print(
                f"  retried              0/{len(records)} - the grader approved "
                f"every retrieval (max_retries={cap})"
            )

    print(f"\nREFUSAL   (n={len(records)})")
    for key, value in refusal.items():
        print(f"  {key:20} {value:.3f}")
    misplaced = [r["id"] for r in records if r.get("refusal_misplaced")]
    # Quote this next to refusal_rate: each one is an answer that declined in prose and
    # was scored as an answer, then judged on axes that do not apply to a refusal.
    print(f"  {'token misplaced':20} {len(misplaced)}  {misplaced}")

    unfaithful = [r for r in judged if not r["faithful"]]
    if unfaithful:
        print(f"\nUNFAITHFUL ({len(unfaithful)}) - read these by hand:")
        for r in unfaithful:
            print(f"  {r['id']}  {r['faithful_reason']}")


if __name__ == "__main__":
    main()
