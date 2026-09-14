"""Hand-label judge verdicts, then measure how often the judge agrees with you.

    python -m scripts.label_judge sample          # build a BLIND worksheet + a labels file
    python -m scripts.label_judge score           # compare your labels with the judge

Deferred since Stage 2 and overdue. Every answer-quality number in `docs/results.md` rests
on a judge nobody has ever checked against a human, and the grader - same design, same
model - was measured at 0.174 false alarm and observed destroying two good retrievals. A
judge you have not validated is not a measurement, it is a second opinion with a decimal
point.

**The worksheet is blind on purpose.** It shows the question, the passages, the answer and
the reference, and NOT the judge's verdict or its reason. Seeing them first is anchoring:
you would be checking whether the judge's reasoning sounds plausible, which it always does,
instead of forming your own verdict. The join happens at scoring time, by question id.

**Who labelled matters, so it is in the filename.** `judge_labels_claude.jsonl` was
produced by Claude Opus 5 reading the same blind worksheet, and the resulting number is
*judge vs a stronger model*, not *judge vs human* - a weaker claim that shares some of the
judge's blind spots, and the only one this file supports. A human pass would be
`judge_labels_<name>.jsonl` and `score` compares every labeller it finds, including against
each other.

**Why the sample is stratified.** Faithfulness ran 0.935 in the last run - 29 True out of
31. Ten rows drawn at random would contain roughly nine easy agreements and tell you
almost nothing. Both unfaithful rows are included deliberately, and correctness is split
evenly, because agreement is only informative where the judge actually has to decide.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "eval"
LABEL_GLOB = "judge_labels_*.jsonl"
# One file per labeller, named after the labeller: judge_labels_claude.jsonl,
# judge_labels_selim.jsonl. The provenance is in the filename rather than in a comment,
# because "the judge agrees with a stronger model" and "the judge agrees with a human"
# are different claims and the difference has to survive being read six months later.
# Where two labellers disagree, the TASK is ambiguous - a finding about the rubric, not
# about either rater.
WORKSHEET = ROOT / "scratch" / "judge_worksheet.md"
SEED = 20260914


def latest_run() -> Path:
    runs = sorted((ROOT / "data" / "eval").glob("run-*.jsonl"))
    if not runs:
        raise SystemExit("no runs in data/eval - run `make eval` first")
    return runs[-1]


def load_questions() -> dict[str, dict]:
    path = ROOT / "eval" / "questions.jsonl"
    return {q["id"]: q for q in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def choose(judged: list[dict], n: int) -> list[dict]:
    """Stratified: every rare-class row, then a balanced fill. Deterministic."""
    rng = random.Random(SEED)
    unfaithful = [r for r in judged if not r["faithful"]]
    rest = [r for r in judged if r["faithful"]]
    correct = [r for r in rest if r["correct"]]
    wrong = [r for r in rest if not r["correct"]]
    rng.shuffle(correct)
    rng.shuffle(wrong)

    picked = list(unfaithful[: n // 2])
    half = (n - len(picked)) // 2
    picked += correct[:half]
    picked += wrong[: n - len(picked)]
    return sorted(picked, key=lambda r: r["id"])


def cmd_sample(args) -> None:
    run = latest_run()
    rows = [json.loads(line) for line in run.read_text().splitlines() if line.strip()]
    judged = [r for r in rows if "faithful" in r]
    if not judged:
        raise SystemExit(f"{run.name} has no judged rows")

    questions = load_questions()
    from arxiv_rag.config import get_settings
    from arxiv_rag.retrieval.store import ChunkStore

    store = ChunkStore.load(get_settings().index_dir)
    by_id = {c.chunk_id: c for c in store.chunks}

    picked = choose(judged, args.n)

    out = ["# Judge labelling worksheet", ""]
    out.append(f"Source run: `{run.name}`  -  {len(picked)} of {len(judged)} judged answers.")
    out.append("")
    out.append("**The judge's verdicts are deliberately not shown.** Decide for yourself,")
    out.append(f"write your answers into `eval/judge_labels_{args.labeller}.jsonl`, then run")
    out.append("`python -m scripts.label_judge score`.")
    out.append("")
    out.append("Two independent questions per answer:")
    out.append("")
    out.append("- **faithful** - is every claim in the answer supported by the PASSAGES below?")
    out.append("  Not 'is it true'. A true statement that the passages do not contain is")
    out.append("  unfaithful; that is the whole point of the axis.")
    out.append("- **correct** - does the answer give what the REFERENCE says, on the substance")
    out.append("  asked? Extra detail is fine. Missing the thing asked for is not.")
    out.append("")
    out.append("Leave a label as `null` if you genuinely cannot decide - those are reported")
    out.append("separately rather than guessed, and a high abstention rate is itself a finding")
    out.append("about the question set.")
    out.append("")
    out.append("---")

    for r in picked:
        q = questions[r["id"]]
        out += ["", f"## {r['id']}  ({r['kind']}, {r['split']})", ""]
        out += [f"**Question.** {r['question']}", ""]
        out += ["**Passages the model was given.**", ""]
        for i, cid in enumerate(r["retrieved_ids"], start=1):
            chunk = by_id.get(cid)
            if chunk is None:
                out.append(f"> [P{i}] {cid} - not in the index")
                continue
            text = " ".join(chunk.text.split())
            out.append(f"> **[P{i}]** `{cid}` - {chunk.title} / {chunk.section}")
            out.append(">")
            out.append(f"> {text}")
            out.append(">")
        out += ["", "**Answer.**", "", r["answer"], ""]
        out += ["**Reference.**", "", q["reference_answer"], ""]
        out.append("---")

    WORKSHEET.parent.mkdir(parents=True, exist_ok=True)
    WORKSHEET.write_text("\n".join(out) + "\n")

    labels_path = LABEL_DIR / f"judge_labels_{args.labeller}.jsonl"
    if labels_path.exists() and not args.force:
        raise SystemExit(f"{labels_path.relative_to(ROOT)} exists - pass --force to replace")
    labels_path.write_text(
        "\n".join(
            json.dumps({"id": r["id"], "faithful": None, "correct": None, "note": ""})
            for r in picked
        )
        + "\n"
    )
    print(f"worksheet -> {WORKSHEET.relative_to(ROOT)}   ({len(picked)} answers, blind)")
    print(f"labels    -> {labels_path.relative_to(ROOT)}   fill in true/false, then `score`")


def kappa(human: list[bool], judge: list[bool]) -> float:
    """Cohen's kappa: agreement above what two raters would reach by chance alone.

    Raw agreement lies when one class dominates. Faithfulness is ~94% True, so a judge that
    simply answered True every time would score ~0.94 agreement while contributing nothing.
    Kappa subtracts that floor: 0 means no better than guessing at the base rate, 1 is
    perfect. Below about 0.4 is usually read as poor.
    """
    n = len(human)
    if n == 0:
        return float("nan")
    po = sum(h == j for h, j in zip(human, judge, strict=True)) / n
    pe = sum(
        (sum(1 for h in human if h == v) / n) * (sum(1 for j in judge if j == v) / n)
        for v in (True, False)
    )
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def read_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        row["id"]: row
        for row in (json.loads(x) for x in path.read_text().splitlines() if x.strip())
    }


def compare(name_a: str, a: dict, name_b: str, b: dict, axis: str, reasons: dict) -> None:
    pairs = [
        (qid, a[qid][axis], b[qid][axis])
        for qid in sorted(a)
        if qid in b and a[qid][axis] is not None and b[qid][axis] is not None
    ]
    if not pairs:
        print(f"  {name_a} vs {name_b:8} no overlapping labels")
        return
    left = [bool(x) for _, x, _ in pairs]
    right = [bool(y) for _, _, y in pairs]
    agree = sum(x == y for x, y in zip(left, right, strict=True))
    print(
        f"  {name_a} vs {name_b:8} {agree}/{len(pairs)} = {agree / len(pairs):.3f}"
        f"   kappa {kappa(left, right):+.3f}"
    )
    for qid, x, y in pairs:
        if bool(x) != bool(y):
            extra = reasons.get(qid, "")
            print(f"      {qid}  {name_a}={x!s:5} {name_b}={y!s:5}  {extra[:60]}")


def cmd_score(args) -> None:
    """Compare every labeller file against the judge, and against each other."""
    files = sorted(LABEL_DIR.glob(LABEL_GLOB))
    if not files:
        raise SystemExit("no label files - run `python -m scripts.label_judge sample` first")
    labellers = {f.stem.replace("judge_labels_", ""): read_labels(f) for f in files}

    run = latest_run()
    rows = {
        r["id"]: r
        for r in (json.loads(x) for x in run.read_text().splitlines() if x.strip())
        if "faithful" in r
    }
    any_ids = set().union(*(set(v) for v in labellers.values()))
    missing = sorted(any_ids - set(rows))
    if missing:
        print(f"!! labelled but not judged in {run.name}: {missing}")
        print("   the run was replaced since sampling; re-sample or keep the old run.\n")

    print(f"run={run.name}   labellers: {', '.join(labellers)}\n")
    for axis in ("faithful", "correct"):
        print(axis.upper())
        judge_view = {qid: {axis: rows[qid][axis]} for qid in rows}
        reasons = {qid: rows[qid][axis + "_reason"] for qid in rows}
        for name, labels in labellers.items():
            compare("judge", judge_view, name, labels, axis, reasons)
        names = list(labellers)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                compare(a, labellers[a], b, labellers[b], axis, {})
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="build a blind worksheet and an empty labels file")
    s.add_argument("-n", type=int, default=10, help="how many answers to label")
    s.add_argument("--force", action="store_true", help="overwrite an existing labels file")
    s.add_argument(
        "--labeller",
        default="you",
        help="who is labelling. Becomes the filename, so the provenance of a number in "
        "docs/results.md can always be traced back to who produced it",
    )
    s.set_defaults(func=cmd_sample)
    c = sub.add_parser("score", help="compare your labels with the judge")
    c.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
