"""Score the router against hand-specified routes. ~18 calls, no judge.

    python -m scripts.route_eval                 # dev only, the default
    python -m scripts.route_eval --split test    # once, after the prompt is frozen

**The first metric in this project that needs no model to score it.** The judge and the
grader both required validating an evaluator before their numbers meant anything;
faithfulness turned out to be unmeasurable at this sample size. A route either matches the
specification or it does not. Exact match, no rubric, no kappa.

`eval/routes.jsonl` is a SPECIFICATION, not ground truth discovered by observation: it
says what the router should do, and it was written before the router existed. That is why
it can be authored by the project rather than by a blind labeller - the failure mode it
guards against is a router tuned until its own output looks reasonable.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from arxiv_rag.agent.router import route_question
from arxiv_rag.agent.tools import list_indexed_papers
from arxiv_rag.config import get_settings
from arxiv_rag.retrieval.store import ChunkStore

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("retrieve", "filtered", "catalog")


def load_cases(split: str) -> list[dict]:
    path = ROOT / "eval" / "routes.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return rows if split == "all" else [r for r in rows if r["split"] == split]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = parser.parse_args()

    settings = get_settings()
    store = ChunkStore.load(settings.index_dir)
    papers = list_indexed_papers(store)
    cases = load_cases(args.split)
    print(
        f"{len(cases)} cases (split={args.split}), {len(papers)} papers, {settings.router_model}\n"
    )

    rows = []
    for case in cases:
        decision = route_question(case["question"], papers, settings)
        ok = decision.route == case["route"]
        # For `filtered`, the route alone is not the job: the wrong ids search the wrong
        # papers. f03 names two and dropping one answers half the question.
        ids_ok = sorted(decision.arxiv_ids) == sorted(case["arxiv_ids"])
        rows.append(
            {
                **case,
                "got": decision.route,
                "got_ids": decision.arxiv_ids,
                "ok": ok,
                "ids_ok": ids_ok,
                "reason": decision.reason,
            }
        )
        mark = "ok " if ok else "MISS"
        detail = "" if ids_ok else f"ids={decision.arxiv_ids} want={case['arxiv_ids']}"
        print(f"  {case['id']}  {mark}  want={case['route']:9} got={decision.route:9} {detail}")

    print("\n" + "=" * 62)
    correct = sum(r["ok"] for r in rows)
    print(f"ROUTE ACCURACY   {correct}/{len(rows)} = {correct / len(rows):.3f}\n")

    for route in ROUTES:
        group = [r for r in rows if r["route"] == route]
        if group:
            hit = sum(r["ok"] for r in group)
            print(f"  {route:10} {hit}/{len(group)}")

    filtered = [r for r in rows if r["route"] == "filtered"]
    if filtered:
        ids_hit = sum(r["ids_ok"] for r in filtered)
        print(f"\n  filtered with the RIGHT ids   {ids_hit}/{len(filtered)}")

    print("\nconfusion (want -> got)")
    pairs = Counter((r["route"], r["got"]) for r in rows if not r["ok"])
    if not pairs:
        print("  none")
    for (want, got), n in pairs.most_common():
        print(f"  {want:10} -> {got:10}  {n}")

    misses = [r for r in rows if not r["ok"] or not r["ids_ok"]]
    if misses:
        print("\nmisses - read these, the near-misses are the point:")
        for r in misses:
            print(f"  {r['id']}  {r['question'][:64]}")
            print(
                f"        want {r['route']}{r['arxiv_ids'] or ''} "
                f"got {r['got']}{r['got_ids'] or ''}"
            )
            print(f"        router said: {r['reason'][:70]}")
            print(f"        spec says:   {r['why'][:70]}")


if __name__ == "__main__":
    main()
