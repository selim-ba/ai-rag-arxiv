"""Score follow-up resolution against the specification. ~18 calls, no judge.

    python -m scripts.followup_eval              # dev, the default
    python -m scripts.followup_eval --split test # once, after the prompt is frozen

`eval/followups.jsonl` does not assert the resolved TEXT - no resolver would produce a
given wording, and exact match on free text is unmeasurable. It asserts what the resolution
must and must not CONTAIN, which is deterministic, needs no evaluator, and tests the only
thing that matters: was the referent named, and was a self-contained question left alone.

Turn-1 answers are generated once and cached, because four of the cases point at something
only the ANSWER mentions. Later runs read the cache and cost nothing but the resolutions.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from arxiv_rag.agent.followup import Turn, resolve_followup
from arxiv_rag.config import get_settings
from arxiv_rag.retrieval.answer import answer_question
from arxiv_rag.retrieval.hybrid import build_hybrid
from arxiv_rag.retrieval.store import ChunkStore

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "eval" / "followup_history.json"


def load_cases(split: str) -> list[dict]:
    path = ROOT / "eval" / "followups.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return rows if split == "all" else [r for r in rows if r["split"] == split]


def turn1_answers(cases: list[dict], settings) -> dict[str, str]:
    """Answer each distinct turn-1 question once, cached on disk."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = sorted({c["turn1"] for c in cases} - set(cache))
    if missing:
        print(f"answering {len(missing)} turn-1 question(s) to build the history cache ...")
        store = ChunkStore.load(settings.index_dir)
        retriever = build_hybrid(store, settings)
        for question in missing:
            cache[question] = answer_question(question, retriever, settings).text
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=2))
    return cache


def check(resolved: str, case: dict) -> tuple[bool, list[str], list[str]]:
    """Substring containment, case-insensitive. Returns (ok, missing, forbidden)."""
    low = resolved.lower()
    missing = [t for t in case["must_contain"] if t.lower() not in low]
    forbidden = [t for t in case["must_not_contain"] if t.lower() in low]
    return (not missing and not forbidden), missing, forbidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases(args.split)
    history = turn1_answers(cases, settings)
    print(f"\n{len(cases)} cases (split={args.split}), {settings.followup_model}\n")

    rows = []
    for case in cases:
        turn = Turn(question=case["turn1"], answer=history.get(case["turn1"], ""))
        resolution = resolve_followup([turn], case["followup"], settings)
        ok, missing, forbidden = check(resolution.resolved, case)
        rows.append(
            {
                **case,
                "ok": ok,
                "resolved": resolution.resolved,
                "changed": resolution.changed,
                "missing": missing,
                "forbidden": forbidden,
            }
        )
        mark = "ok  " if ok else "MISS"
        print(f"  {case['id']}  {mark}  {case['kind']:15} {resolution.resolved[:58]}")

    print("\n" + "=" * 66)
    correct = sum(r["ok"] for r in rows)
    print(f"RESOLUTION ACCURACY   {correct}/{len(rows)} = {correct / len(rows):.3f}\n")
    for kind in ("pronoun", "ellipsis", "answer_reference", "self_contained"):
        group = [r for r in rows if r["kind"] == kind]
        if group:
            print(f"  {kind:18} {sum(r['ok'] for r in group)}/{len(group)}")

    # The asymmetry the specification exists to expose. Over-resolution answers a question
    # nobody asked and looks fine doing it; under-resolution retrieves badly and shows it.
    over = [r for r in rows if r["kind"] == "self_contained" and not r["ok"]]
    under = [r for r in rows if r["kind"] != "self_contained" and not r["ok"]]
    print(f"\n  OVER-resolved (silent failures)   {len(over)}  {[r['id'] for r in over]}")
    print(f"  under-resolved (visible failures) {len(under)}  {[r['id'] for r in under]}")
    print(f"\n  rewrote the question: {Counter(r['changed'] for r in rows)[True]}/{len(rows)}")

    misses = [r for r in rows if not r["ok"]]
    if misses:
        print("\nmisses:")
        for r in misses:
            print(f"  {r['id']}  ({r['kind']}, refers to {r['refers_to']})")
            print(f"        turn1:    {r['turn1'][:62]}")
            print(f"        followup: {r['followup'][:62]}")
            print(f"        resolved: {r['resolved'][:62]}")
            if r["missing"]:
                print(f"        MISSING:   {r['missing']}")
            if r["forbidden"]:
                print(f"        FORBIDDEN: {r['forbidden']}")
            print(f"        spec:     {r['why'][:62]}")


if __name__ == "__main__":
    main()
