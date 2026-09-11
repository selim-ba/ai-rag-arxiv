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
from arxiv_rag.retrieval.answer import answer_question, format_context
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
    args = parser.parse_args()

    settings = get_settings()
    store = ChunkStore.load(settings.index_dir)
    by_id = {c.chunk_id: c for c in store.chunks}
    questions = load_questions(args.split)[: args.limit]

    out_dir = ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"run-{stamp}.jsonl"

    records: list[dict] = []
    print(f"{len(questions)} questions -> {out_path.name}\n")

    for i, q in enumerate(questions, start=1):
        answer = answer_question(q["question"], store, settings)
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
            "retrieved_ids": retrieved,
            "gold_chunk_ids": gold,
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

    print(f"\nREFUSAL   (n={len(records)})")
    for key, value in refusal.items():
        print(f"  {key:20} {value:.3f}")

    unfaithful = [r for r in judged if not r["faithful"]]
    if unfaithful:
        print(f"\nUNFAITHFUL ({len(unfaithful)}) - read these by hand:")
        for r in unfaithful:
            print(f"  {r['id']}  {r['faithful_reason']}")


if __name__ == "__main__":
    main()
