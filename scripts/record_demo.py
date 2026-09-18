"""Run the real system over the landing page's questions and record what it says.

    python -m scripts.record_demo            # writes web/demo.json
    python -m scripts.record_demo --dry-run  # list the questions, spend nothing

**Why record rather than call live.** The common visit to a portfolio link is someone
clicking an example and reading the answer. Serving that from a file makes it instant,
free, and available when the day's budget is spent - and the live box underneath still
lets anyone ask their own question. The recordings are real runs, dated, with their
request ids and timings kept, which is the difference between a recording and a mock-up.

**The questions are not chosen by taste.** Every "works" entry scored correct AND faithful
in the most recent judged eval run. The refusal is one of the six deliberately
unanswerable questions. The failure is q011, whose gold passage sits at dense rank 66 - a
question this system gets wrong, on the page, on purpose, because a demo that only shows
wins is indistinguishable from one that was never measured.

**Two paths, on purpose.** The first version of this script ran everything through the
agent with routing on, which is neither what the eval measures nor what `/ask` serves -
three configurations, one page. Measured afterwards (`docs/results.md`, Stage 7): routing
wins two comparison questions on hit@5 and answers two questions it should refuse. So the
answers a visitor reads come from the **pipeline**, the configuration every published
number describes, and the agent appears only where routing IS the thing being shown: a
catalog question, which takes no model call at all, and a question naming a paper by an
alias the corpus titles do not contain.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from arxiv_rag import __version__
from arxiv_rag.agent.followup import Turn, resolve_followup
from arxiv_rag.agent.graph import build_graph, run_agent
from arxiv_rag.config import get_settings
from arxiv_rag.observability import current_stats, start_request
from arxiv_rag.retrieval.answer import answer_question
from arxiv_rag.retrieval.hybrid import build_hybrid
from arxiv_rag.retrieval.store import ChunkStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "demo.json"

# (kind, path, question). `path` is the configuration that answers it - see the module
# docstring for why they are not all the same.
DEMO = [
    ("works", "pipeline", "How does PlaNet search for a good action sequence?"),
    ("works", "pipeline", "What pieces make up the latent dynamics model in Dreamer?"),
    (
        "works",
        "pipeline",
        "Dreamer only imagines a fixed number of steps ahead. "
        "How does it avoid becoming shortsighted?",
    ),
    (
        "works",
        "pipeline",
        "Why does JEPA training collapse to trivial solutions, "
        "and what constraint was proposed against it?",
    ),
    ("works", "pipeline", "What is a recurrent state-space model, and why split the state in two?"),
    # No model call at all - the catalog node answers from the index's metadata, because
    # "which papers do you have" is a fact and a generator asked for it invents a list.
    ("catalog", "agent", "Which papers do you have indexed?"),
    # The corpus files this paper as "Revisiting Feature Prediction for Learning Visual
    # Representations from Video". The alias table is what makes the question work.
    ("scoped", "agent", "In V-JEPA, how is masking done?"),
    # Specific, plausible, and absent from all 49 papers. It refuses.
    ("refuses", "pipeline", "What learning rate was used to train Dreamer?"),
    # Gold at dense rank 66. On the page under its own heading.
    (
        "fails",
        "pipeline",
        "Both the original World Models paper and Dreamer learn a policy inside a "
        "learned model. How does the optimisation differ?",
    ),
]

# The conversation, recorded as two turns so the page can show the resolution.
FOLLOW_UP = (
    "How does PlaNet search for a good action sequence?",
    "and does it train a policy network?",
)


def sources_of(answer, by_id) -> list[dict]:
    return [
        {
            "chunk_id": cid,
            "arxiv_id": by_id[cid].arxiv_id,
            "title": by_id[cid].title,
            "section": by_id[cid].section,
            "url": f"https://arxiv.org/abs/{by_id[cid].arxiv_id}",
        }
        for cid in answer.retrieved_ids
        if cid in by_id
    ]


def record(answerers, question: str, by_id, kind: str, path: str, asked: str | None = None) -> dict:
    """One question, answered for real, with what it cost and which path ran it."""
    start_request()  # so the token accounting has somewhere to land
    answer = answerers[path](question)
    stats = current_stats()
    return {
        "kind": kind,
        "path": path,
        "asked": asked or question,
        "resolved": question if asked and asked != question else None,
        "answer": answer.text,
        "citations": answer.citations,
        "refused": answer.refused,
        "sources": sources_of(answer, by_id),
        "trace": answer.trace,
        "retrieve_ms": round(answer.retrieve_ms, 1),
        "generate_ms": round(answer.generate_ms, 1),
        "tokens_in": stats.tokens_in if stats else 0,
        "tokens_out": stats.tokens_out if stats else 0,
        "cost_usd": round(stats.cost_usd, 6) if stats else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="list the questions, spend nothing")
    args = parser.parse_args()

    if args.dry_run:
        for kind, path, question in DEMO:
            print(f"  {kind:8} {path:8} {question}")
        print(f"  {'follow':8} {'pipeline':8} {FOLLOW_UP[0]!r} -> {FOLLOW_UP[1]!r}")
        return

    settings = get_settings()
    store = ChunkStore.load(settings.index_dir)
    by_id = {c.chunk_id: c for c in store.chunks}
    retriever = build_hybrid(store, settings)
    graph = build_graph(retriever, settings, store=store)
    answerers = {
        "pipeline": lambda q: answer_question(q, retriever, settings),
        "agent": lambda q: run_agent(graph, q),
    }

    entries = []
    for kind, path, question in DEMO:
        print(f"  {kind:8} {path:8} {question[:56]}")
        entries.append(record(answerers, question, by_id, kind, path))

    # The conversation. Turn one is answered, then the follow-up is resolved against it
    # exactly as `/ask` would, and the resolution is kept so the page can show it.
    print(f"  follow   pipeline {FOLLOW_UP[1]}")
    first = record(answerers, FOLLOW_UP[0], by_id, "follow", "pipeline")
    history = [Turn(question=FOLLOW_UP[0], answer=first["answer"])]
    start_request()
    resolution = resolve_followup(history, FOLLOW_UP[1], settings)
    second = record(answerers, resolution.resolved, by_id, "follow", "pipeline", asked=FOLLOW_UP[1])
    entries.extend([first, second])

    total = sum(e["cost_usd"] for e in entries)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "version": __version__,
                "model": settings.llm_model,
                "chunks_indexed": len(store),
                "total_cost_usd": round(total, 6),
                "entries": entries,
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    print(f"\n{len(entries)} entries -> {OUT.relative_to(ROOT)}  (cost ${total:.4f})")


if __name__ == "__main__":
    main()
