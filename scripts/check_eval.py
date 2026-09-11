"""Validate the evaluation set, and optionally show where each gold chunk ranks.

    python -m scripts.check_eval                 # schema + gold ids only, no API calls
    python -m scripts.check_eval --ranks         # also embed each question and probe

A typo in a ``gold_chunk_ids`` entry is silent and permanent: the chunk never matches,
the question always scores zero, and your baseline is quietly wrong. Run this before
trusting any number that comes out of the eval set.
"""

import argparse
import json
import sys
from pathlib import Path

from arxiv_rag.config import get_settings
from arxiv_rag.evaluation.metrics import flatten

REQUIRED = {"id", "question", "answerable", "gold_chunk_ids", "reference_answer"}
KINDS = {"factual", "comparison", "definitional", "unanswerable"}
SPLITS = {"dev", "test"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate eval/questions.jsonl")
    parser.add_argument("--path", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument(
        "--ranks",
        action="store_true",
        help="embed each question and report where the gold chunk ranks",
    )
    args = parser.parse_args()

    settings = get_settings()

    if not args.path.exists() or not args.path.read_text().strip():
        print(f"{args.path} is empty — nothing to check yet")
        return 0

    questions, problems = [], []
    seen_ids = set()
    for lineno, line in enumerate(args.path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {lineno}: invalid JSON — {exc}")
            continue
        missing = REQUIRED - q.keys()
        if missing:
            problems.append(f"line {lineno}: missing {sorted(missing)}")
            continue
        if q["id"] in seen_ids:
            problems.append(f"line {lineno}: duplicate id {q['id']}")
        seen_ids.add(q["id"])
        if q.get("kind") and q["kind"] not in KINDS:
            problems.append(
                f"{q['id']}: unknown kind {q['kind']!r}, expected one of {sorted(KINDS)}"
            )
        if q.get("split") and q["split"] not in SPLITS:
            problems.append(f"{q['id']}: unknown split {q['split']!r}")
        if q["answerable"] and not q["gold_chunk_ids"]:
            problems.append(f"{q['id']}: answerable but no gold_chunk_ids")
        if not q["answerable"] and q["gold_chunk_ids"]:
            problems.append(f"{q['id']}: unanswerable but has gold_chunk_ids")
        # gold_chunk_ids is a list of GROUPS: one chunk from each group is needed.
        # A flat list of strings is the old schema and would silently score wrong.
        for group in q["gold_chunk_ids"]:
            if not isinstance(group, list):
                problems.append(
                    f"{q['id']}: gold_chunk_ids must be a list of lists "
                    f"(groups), found a bare {type(group).__name__}"
                )
            elif not group:
                problems.append(f"{q['id']}: empty gold group")
        questions.append(q)

    # every gold id must exist in the index
    from arxiv_rag.retrieval.store import ChunkStore

    store = ChunkStore.load(settings.index_dir)
    known = {c.chunk_id for c in store.chunks}
    for q in questions:
        for gold in flatten(q["gold_chunk_ids"]):
            if gold not in known:
                problems.append(f"{q['id']}: gold chunk {gold!r} is not in the index")

    # --- summary ---
    kinds: dict[str, int] = {}
    splits: dict[str, int] = {}
    for q in questions:
        kinds[q.get("kind", "?")] = kinds.get(q.get("kind", "?"), 0) + 1
        splits[q.get("split", "?")] = splits.get(q.get("split", "?"), 0) + 1

    print(f"{len(questions)} questions")
    print(f"  by kind:  {kinds}")
    print(f"  by split: {splits}")
    print(f"  unanswerable: {sum(1 for q in questions if not q['answerable'])}")
    multi = sum(1 for q in questions if len(q["gold_chunk_ids"]) > 1)
    alts = sum(1 for q in questions if any(len(g) > 1 for g in q["gold_chunk_ids"]))
    print(f"  multi-group:  {multi}  (need a chunk from each group)")
    print(f"  with alternatives: {alts}  (a group with more than one acceptable source)")

    if args.ranks:
        from arxiv_rag.retrieval.embeddings import embed_query

        print(f"\ngold chunk ranks (out of {len(store)}):")
        buckets = {"1": 0, "2-5": 0, "6-20": 0, "21+": 0, "missed": 0}
        for q in questions:
            if not q["answerable"]:
                continue
            hits = store.search(embed_query(q["question"], settings), k=len(store))
            ranked = {h.chunk.chunk_id: i for i, h in enumerate(hits, 1)}
            best = min(
                (ranked[g] for g in flatten(q["gold_chunk_ids"]) if g in ranked), default=None
            )
            label = (
                "missed"
                if best is None
                else ("1" if best == 1 else "2-5" if best <= 5 else "6-20" if best <= 20 else "21+")
            )
            buckets[label] += 1
            print(f"  {q['id']}  rank {best if best else '-':<5} {q['question'][:60]}")
        print(f"\ndistribution: {buckets}")
        if buckets["1"] > len(questions) * 0.6:
            print("\n  ⚠ most gold chunks rank first — questions may be borrowing the")
            print("    chunk's own wording. Rephrase as a reader who had not seen it.")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  ✗ {p}")
        return 1

    print("\nno problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
