"""Score retrievers on the eval set. No generation, no judge, no cost.

    python -m scripts.retrieval_eval              # every retriever, side by side
    python -m scripts.retrieval_eval --ranks      # plus where each gold chunk landed

hit@k, recall@k and MRR are computed from retrieved ids against gold ids. The answer text
never enters into them, so there is no reason to pay for generation while iterating on
retrieval - and Stage 3 has five more retriever variants to measure. `make eval` stays for
milestones, where answer quality and refusal matter.

Query embeddings are read from the disk cache, so a run is free and takes about a second.
"""

import argparse
import json
from pathlib import Path

from arxiv_rag.config import get_settings
from arxiv_rag.evaluation.metrics import flatten, hit_at_k, mean, recall_at_k, reciprocal_rank
from arxiv_rag.evaluation.timing import summarise
from arxiv_rag.retrieval.bm25 import BM25Index
from arxiv_rag.retrieval.embeddings import embed_query, embed_texts
from arxiv_rag.retrieval.filters import ChunkFilter
from arxiv_rag.retrieval.hierarchical import top_papers
from arxiv_rag.retrieval.hybrid import DenseRetriever, HybridRetriever
from arxiv_rag.retrieval.store import ChunkStore

ROOT = Path(__file__).resolve().parents[1]
DEEP_K = 100  # retrieve this deep for MRR, so "found it at rank 40" is visible


def load_questions(split: str = "all") -> list[dict]:
    """Answerable questions, optionally restricted to one split.

    Stage 3 reported every retrieval number on dev+test together, which is fine for
    *describing* a fixed retriever and wrong for *choosing* one: the Stage 4 RRF grid
    compared 30 configurations, and picking the winner on all 34 questions means the test
    split helped pick it. Tuning happens on dev; test is looked at once, at the end.
    """
    path = ROOT / "eval" / "questions.jsonl"
    questions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    answerable = [q for q in questions if q["answerable"]]
    if split == "all":
        return answerable
    return [q for q in answerable if q["split"] == split]


def score(name: str, retrieve, questions: list[dict], k: int) -> dict:
    """Run one retriever over every question and summarise. ``retrieve`` takes (question,
    k) and returns chunk ids, best first."""
    from time import perf_counter

    rows, timings = [], []
    for q in questions:
        started = perf_counter()
        ids = retrieve(q["question"], DEEP_K)
        timings.append((perf_counter() - started) * 1000)
        gold = q["gold_chunk_ids"]
        rows.append(
            {
                "id": q["id"],
                "hit@1": hit_at_k(ids, gold, 1),
                "hit@5": hit_at_k(ids, gold, k),
                "recall@5": recall_at_k(ids, gold, k),
                "rr": reciprocal_rank(ids, gold),
                "rank": next((i for i, c in enumerate(ids, 1) if c in set(flatten(gold))), None),
            }
        )
    latency = summarise(timings)
    return {
        "name": name,
        "hit@1": mean([float(r["hit@1"]) for r in rows]),
        "hit@5": mean([float(r["hit@5"]) for r in rows]),
        "recall@5": mean([r["recall@5"] for r in rows]),
        "mrr": mean([r["rr"] for r in rows]),
        "p50": latency["p50"],
        "p95": latency["p95"],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", action="store_true", help="per-question gold rank")
    parser.add_argument(
        "--paper-recall",
        action="store_true",
        help="phase 0 of two-stage retrieval: how often the gold paper survives a coarse "
        "pass that keeps the top N papers. Free; bounds what two-stage could ever reach",
    )
    parser.add_argument(
        "--coarse-depth",
        type=int,
        default=100,
        help="how many chunks the coarse pass looks at before grouping them by paper",
    )
    parser.add_argument(
        "--paper-embed",
        action="store_true",
        help="phase 0b: a coarse pass with an INDEPENDENT signal - embed each paper's "
        "title and opening text as one document and retrieve papers directly",
    )
    parser.add_argument(
        "--filter-ceiling",
        action="store_true",
        help="what perfect metadata filtering could buy: restrict each question to its own "
        "gold paper and re-measure. An oracle router, no model, no cost",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "dev", "test"],
        help="which questions to score. Tune on dev; `all` includes the test split and is "
        "for reporting a decided configuration, not for choosing one",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="add a cross-encoder reranking row (needs: pip install -e '.[rerank]')",
    )
    parser.add_argument("--rerank-depth", type=int, default=30, help="candidates to rescore")
    parser.add_argument(
        "--rerank-llm",
        action="store_true",
        help="add a listwise LLM reranking row (costs money: ~34 calls per run)",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="*",
        default=None,
        help="fusion depths to compare (default: 10, the configured depth, 30)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="sweep the RRF constant and per-retriever weights (free, no LLM)",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        nargs="*",
        default=[60, 20, 10, 5, 1, 0],
        help="RRF constants to try. 60 is Cormack et al.'s default; lower makes rank "
        "differences matter more, which favours a confident single-retriever find",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        default=["1,1", "1,0.7", "1,0.5", "1,0.3", "1,0"],
        help='dense,bm25 weight pairs. "1,0" must reproduce dense exactly - it is the '
        "known-answer check on the grid itself",
    )
    parser.add_argument(
        "--grid-depth", type=int, default=None, help="fusion depth for the grid (default: settings)"
    )
    args = parser.parse_args()

    settings = get_settings()
    store = ChunkStore.load(settings.index_dir)
    bm25 = BM25Index(store.chunks)
    questions = load_questions(args.split)
    k = settings.top_k
    if not questions:
        raise SystemExit(f"no answerable questions in split {args.split!r}")

    def dense(question: str, n: int) -> list[str]:
        return [h.chunk.chunk_id for h in store.search(embed_query(question, settings), k=n)]

    def sparse(question: str, n: int) -> list[str]:
        return [h.chunk.chunk_id for h in bm25.search(question, k=n)]

    results = [
        score("dense", dense, questions, k),
        score("bm25", sparse, questions, k),
    ]

    # Depth is a real knob, not a constant to guess once: measured on q030, gold lands at
    # fused rank 3 at depth 10 and rank 11 at depth 50, because every extra candidate that
    # BOTH retrievers found outscores one that only BM25 found.
    dense_retriever = DenseRetriever(store, settings)
    # The configured depth is always in the table. It was not, for all of Stage 3: the
    # default was [10, 30] while `fusion_depth` was 20, so the one configuration the
    # system actually ships never appeared in its own report, and the nearest row invited
    # being read as if it were that row.
    depths = args.depths if args.depths else sorted({10, settings.fusion_depth, 30})
    # Every row below uses the CONFIGURED rrf_k and weights, varying only depth. These
    # rows constructed HybridRetriever directly for one run and silently picked up the
    # library defaults (rrf_k=60, equal weights) while the header printed the configured
    # values - a report claiming to describe a configuration it was not using.
    configured = {"rrf_k": settings.rrf_k, "weights": settings.fusion_weight_list}
    for depth in depths:
        hybrid = HybridRetriever([dense_retriever, bm25], depth=depth, **configured)

        def fused(question: str, n: int, _h=hybrid) -> list[str]:
            return [h.chunk.chunk_id for h in _h.search(question, k=n)]

        mark = "*" if depth == settings.fusion_depth else ""
        results.append(score(f"hybrid d={depth}{mark}", fused, questions, k))

    grid_rows: list[dict] = []
    if args.grid:
        depth = args.grid_depth if args.grid_depth is not None else settings.fusion_depth
        for rrf_k in args.rrf_k:
            for spec in args.weights:
                weights = [float(x) for x in spec.split(",")]
                cfg = HybridRetriever(
                    [dense_retriever, bm25], depth=depth, rrf_k=rrf_k, weights=weights
                )

                def fused_cfg(question: str, n: int, _h=cfg) -> list[str]:
                    return [h.chunk.chunk_id for h in _h.search(question, k=n)]

                grid_rows.append(score(f"k={rrf_k} w={spec}", fused_cfg, questions, k))

    if args.rerank:
        from arxiv_rag.retrieval.rerank import CrossEncoderReranker, RerankingRetriever

        reranker = CrossEncoderReranker()
        base = HybridRetriever([dense_retriever, bm25], depth=args.rerank_depth, **configured)
        reranked = RerankingRetriever(base, reranker, depth=args.rerank_depth)

        # Warm the model before timing. The first call downloads and loads ~90MB, which
        # would land entirely in question 1 and make p50 meaningless.
        print(f"loading cross-encoder ({reranker.model_name}) ...")
        reranked.search("warm up the model", k=1)

        def with_rerank(question: str, n: int) -> list[str]:
            return [h.chunk.chunk_id for h in reranked.search(question, k=n)]

        results.append(score(f"+xenc d={args.rerank_depth}", with_rerank, questions, k))

    if args.rerank_llm:
        from arxiv_rag.retrieval.rerank import LLMListwiseReranker, RerankingRetriever

        llm_base = HybridRetriever([dense_retriever, bm25], depth=args.rerank_depth, **configured)
        llm_reranked = RerankingRetriever(
            llm_base, LLMListwiseReranker(settings), depth=args.rerank_depth
        )

        def with_llm(question: str, n: int) -> list[str]:
            return [h.chunk.chunk_id for h in llm_reranked.search(question, k=n)]

        print(f"listwise reranking with {settings.llm_model} ({len(questions)} calls) ...")
        results.append(score(f"+llm d={args.rerank_depth}", with_llm, questions, k))

    print(
        f"{len(questions)} answerable questions (split={args.split}), {len(store)} chunks, k={k}\n"
    )
    print(
        f"configured: depth={settings.fusion_depth} (*)  rrf_k={settings.rrf_k}  "
        f"weights={settings.fusion_weight_list or 'equal'}\n"
    )
    print(
        f"{'retriever':<14} {'hit@1':>7} {'hit@5':>7} {'recall@5':>9} {'MRR':>7} "
        f"{'p50 ms':>8} {'p95 ms':>8}"
    )
    for r in results:
        print(
            f"{r['name']:<14} {r['hit@1']:>7.3f} {r['hit@5']:>7.3f} {r['recall@5']:>9.3f} "
            f"{r['mrr']:>7.3f} {r['p50']:>8.1f} {r['p95']:>8.1f}"
        )

    if args.filter_ceiling:
        # The question the `filtered` route rests on: what could ANY filter buy?
        #
        # Each question is restricted to the paper its gold chunk lives in - an oracle
        # router that is always right, which no real router can beat. If this ceiling sits
        # on top of the unfiltered number, metadata filtering cannot help here whoever
        # chooses the filter, and the `filtered` route is architecture without benefit.
        # Deterministic and free: no model is involved in picking the filter.
        base = next(r for r in results if r["name"].startswith(f"hybrid d={settings.fusion_depth}"))
        oracle = HybridRetriever(
            [dense_retriever, bm25],
            depth=settings.fusion_depth,
            rrf_k=settings.rrf_k,
            weights=settings.fusion_weight_list,
        )
        rows = []
        for q in questions:
            gold_papers = {cid.split("::")[0] for cid in flatten(q["gold_chunk_ids"])}
            hits = oracle.search(
                q["question"], k=DEEP_K, chunk_filter=ChunkFilter(arxiv_ids=frozenset(gold_papers))
            )
            ids = [h.chunk.chunk_id for h in hits]
            rows.append(
                {
                    "id": q["id"],
                    "papers": len(gold_papers),
                    "hit@1": hit_at_k(ids, q["gold_chunk_ids"], 1),
                    "hit@5": hit_at_k(ids, q["gold_chunk_ids"], k),
                    "recall@5": recall_at_k(ids, q["gold_chunk_ids"], k),
                    "rr": reciprocal_rank(ids, q["gold_chunk_ids"]),
                }
            )
        print(f"\nFILTER CEILING   (oracle filter = the gold chunk's own paper, n={len(rows)})")
        print(f"{'':16} {'hit@1':>7} {'hit@5':>7} {'recall@5':>9} {'MRR':>7}")
        print(
            f"{'unfiltered':16} {base['hit@1']:>7.3f} {base['hit@5']:>7.3f} "
            f"{base['recall@5']:>9.3f} {base['mrr']:>7.3f}"
        )
        print(
            f"{'oracle filter':16} {mean([float(r['hit@1']) for r in rows]):>7.3f} "
            f"{mean([float(r['hit@5']) for r in rows]):>7.3f} "
            f"{mean([r['recall@5'] for r in rows]):>9.3f} "
            f"{mean([r['rr'] for r in rows]):>7.3f}"
        )
        by_id = {row["id"]: row for row in base["rows"]}
        rescued = [r["id"] for r in rows if r["hit@5"] and not by_id[r["id"]]["hit@5"]]
        lost = [r["id"] for r in rows if not r["hit@5"] and by_id[r["id"]]["hit@5"]]
        print(f"\n  rescued by filtering   {len(rescued)}  {rescued}")
        print(f"  lost to filtering      {len(lost)}  {lost}")
        multi = [r["id"] for r in rows if r["papers"] > 1]
        print(f"  questions whose gold spans >1 paper   {len(multi)}  {multi}")

    if args.paper_recall:
        # THE decision number for two-stage retrieval. The oracle ceiling says perfect
        # paper filtering is worth +0.148 hit@5; this says how much of that a real coarse
        # pass can reach, because a gold chunk whose paper misses the top N is unreachable
        # no matter how good the fine pass is. `1 - recall@N` is the floor on questions
        # two-stage would LOSE - the failure the oracle could not have by construction.
        coarse = HybridRetriever(
            [dense_retriever, bm25],
            depth=settings.fusion_depth,
            rrf_k=settings.rrf_k,
            weights=settings.fusion_weight_list,
        )
        deep = {q["id"]: coarse.search(q["question"], k=args.coarse_depth) for q in questions}
        gold_papers = {
            q["id"]: {cid.split("::")[0] for cid in flatten(q["gold_chunk_ids"])} for q in questions
        }
        # A question needing chunks from three papers is only served if ALL three survive.
        needed = {qid: len(papers) for qid, papers in gold_papers.items()}

        one = sum(1 for v in needed.values() if v == 1)
        many = sum(1 for v in needed.values() if v > 1)
        print(f"\nPAPER RECALL   (coarse pass over {args.coarse_depth} chunks, n={len(questions)})")
        print(f"  gold spans 1 paper: {one}, >1 paper: {many}")
        print(f"\n{'aggregate':10} " + " ".join(f"{'N=' + str(n):>8}" for n in (1, 2, 3, 5, 10)))
        for aggregate in ("max", "sum", "mean", "count"):
            cells = []
            for n in (1, 2, 3, 5, 10):
                ok = sum(
                    1
                    for q in questions
                    if gold_papers[q["id"]] <= set(top_papers(deep[q["id"]], n, aggregate))
                )
                cells.append(f"{ok / len(questions):>8.3f}")
            print(f"{aggregate:10} " + " ".join(cells))

        # Which questions the coarse pass would strand, at the shape most likely to ship.
        stranded = [
            q["id"]
            for q in questions
            if not gold_papers[q["id"]] <= set(top_papers(deep[q["id"]], 3, "max"))
        ]
        print(f"\n  stranded at N=3, max   {len(stranded)}  {stranded}")
        print(
            "  a stranded question cannot be rescued by any fine pass - this is the "
            "floor on what two-stage would lose"
        )

    if args.paper_embed:
        # Phase 0 showed the aggregated coarse pass captures NONE of the oracle's gain: all
        # five questions it rescues are stranded, and three working ones would break. The
        # diagnosis was derivation - a coarse pass built from chunk scores inherits the
        # chunk retrieval's blindness, and fails on exactly the questions that need it.
        #
        # This is the same idea with an INDEPENDENT signal: one embedding per paper, built
        # from its title and opening text, matched against the question directly. It can
        # surface a paper whose individual chunks all rank poorly - which is the case that
        # matters.
        import numpy as np

        docs, ids = [], []
        for arxiv_id in sorted({c.arxiv_id for c in store.chunks}):
            chunks = sorted(
                (c for c in store.chunks if c.arxiv_id == arxiv_id), key=lambda c: c.index
            )
            # Title plus the opening ~2 chunks: abstract and introduction, which is where a
            # paper says what it is for. Whole papers would dilute to nothing.
            body = " ".join(" ".join(c.text.split()) for c in chunks[:2])[:4000]
            docs.append(f"{chunks[0].title}. {body}")
            ids.append(arxiv_id)

        print(f"\nPAPER EMBEDDINGS   (1 vector per paper, {len(ids)} papers)")
        matrix = np.array(embed_texts(docs, settings), dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

        gold_papers = {
            q["id"]: {cid.split("::")[0] for cid in flatten(q["gold_chunk_ids"])} for q in questions
        }
        ranked = {}
        for q in questions:
            vector = np.array(embed_query(q["question"], settings), dtype=np.float32)
            vector /= np.linalg.norm(vector)
            order = np.argsort(-(matrix @ vector))
            ranked[q["id"]] = [ids[i] for i in order]

        print(f"\n{'':10} " + " ".join(f"{'N=' + str(n):>8}" for n in (1, 2, 3, 5, 10)))
        cells = []
        for n in (1, 2, 3, 5, 10):
            ok = sum(1 for q in questions if gold_papers[q["id"]] <= set(ranked[q["id"]][:n]))
            cells.append(f"{ok / len(questions):>8.3f}")
        print(f"{'abstract':10} " + " ".join(cells))

        # The only questions that matter: the five the oracle filter rescues. A coarse pass
        # that misses these captures none of the +0.148 hit@5, whatever its average says.
        rescued = ["q026", "q028", "q029", "q033", "q037"]
        print("\n  the five the oracle rescues - rank of each gold paper:")
        for qid in rescued:
            if qid not in ranked:
                continue
            positions = {p: ranked[qid].index(p) + 1 for p in sorted(gold_papers[qid])}
            worst = max(positions.values())
            verdict = "REACHED at N=3" if worst <= 3 else f"needs N>={worst}"
            print(f"    {qid}  {positions}  -> {verdict}")

    if grid_rows:
        depth = args.grid_depth if args.grid_depth is not None else settings.fusion_depth
        dense_hits = {r["id"] for r in results[0]["rows"] if r["hit@5"]}
        print(f"\nRRF GRID   (depth={depth}, dense hit@5 = {results[0]['hit@5']:.3f})")
        print(
            f"{'config':<16} {'hit@1':>7} {'hit@5':>7} {'recall@5':>9} {'MRR':>7} "
            f"{'+dense':>7} {'-dense':>7}"
        )
        # `+dense` / `-dense` are the point of this table. A config that gains four
        # questions and loses four has hit@5 identical to dense and is not the same
        # retriever - the aggregate hides exactly the trade this experiment is about.
        for r in sorted(grid_rows, key=lambda r: (-r["hit@5"], -r["mrr"])):
            got = {x["id"] for x in r["rows"] if x["hit@5"]}
            print(
                f"{r['name']:<16} {r['hit@1']:>7.3f} {r['hit@5']:>7.3f} {r['recall@5']:>9.3f} "
                f"{r['mrr']:>7.3f} {len(got - dense_hits):>7} {len(dense_hits - got):>7}"
            )
        best = max(grid_rows, key=lambda r: (r["hit@5"], r["mrr"]))
        gained = sorted({x["id"] for x in best["rows"] if x["hit@5"]} - dense_hits)
        lost = sorted(dense_hits - {x["id"] for x in best["rows"] if x["hit@5"]})
        print(f"\n  best by hit@5: {best['name']}")
        print(f"    gained vs dense: {gained}")
        print(f"    lost vs dense:   {lost}")

    # Which questions does each retriever win? This is the case for fusing them - if one
    # method were simply better everywhere, there would be nothing to fuse.
    by_id = {r["name"]: {row["id"]: row for row in r["rows"]} for r in results}
    names = [r["name"] for r in results]
    if len(names) >= 2:
        a, b = names[0], names[1]
        only_a = [q for q in by_id[a] if by_id[a][q]["hit@5"] and not by_id[b][q]["hit@5"]]
        only_b = [q for q in by_id[b] if by_id[b][q]["hit@5"] and not by_id[a][q]["hit@5"]]
        both = [q for q in by_id[a] if by_id[a][q]["hit@5"] and by_id[b][q]["hit@5"]]
        neither = [q for q in by_id[a] if not by_id[a][q]["hit@5"] and not by_id[b][q]["hit@5"]]
        print(f"\nhit@{k} overlap")
        print(f"  both          {len(both):>3}")
        print(f"  {a} only     {len(only_a):>3}  {only_a}")
        print(f"  {b} only      {len(only_b):>3}  {only_b}")
        print(f"  neither       {len(neither):>3}  {neither}")
        print(
            f"\n  union ceiling: {(len(both) + len(only_a) + len(only_b)) / len(questions):.3f} "
            f"- what perfect fusion of these two could reach"
        )

    print(f"\nplanted cases (gold rank, out of {DEEP_K} retrieved)")
    for qid in ("q030", "q034"):
        cells = []
        for r in results:
            row = next((x for x in r["rows"] if x["id"] == qid), None)
            if row:
                rank = str(row["rank"]) if row["rank"] else f">{DEEP_K}"
                cells.append(f"{r['name']}={rank}")
        print(f"  {qid}: " + "  ".join(cells))

    if args.ranks:
        print("\nper-question gold rank")
        print(f"  {'id':<6} " + " ".join(f"{r['name']:>8}" for r in results))
        for q in questions:
            cells = []
            for r in results:
                row = next(x for x in r["rows"] if x["id"] == q["id"])
                cells.append(f"{row['rank'] if row['rank'] else '-':>8}")
            print(f"  {q['id']:<6} " + " ".join(cells))


if __name__ == "__main__":
    main()
