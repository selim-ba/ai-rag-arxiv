"""Ingestion CLI: arXiv -> PDFs -> cleaned text -> chunks on disk.

Two ways to choose papers:

    python -m scripts.ingest --query 'all:"JEPA"' --limit 5
    python -m scripts.ingest --ids-file corpus.txt

The second is the real one. A curated list of ids in a file makes the index
reproducible; a query returns whatever the ranker liked on the day you ran it.
"""

import argparse
import logging
import sys
from pathlib import Path

from arxiv_rag.config import get_settings
from arxiv_rag.ingestion import arxiv_client, chunker, pdf_parser

log = logging.getLogger("ingest")


def load_corpus_ids(path: Path) -> list[str]:
    """Read arXiv ids from a corpus file.

    One id per line. Anything after ``#`` is a comment, so each line can record why the
    paper is in the index — which is most of the point of keeping the file.
    """
    ids = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into the local index")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", help='arXiv query, e.g. all:"JEPA"')
    source.add_argument("--ids-file", type=Path, help="Corpus file: one arXiv id per line")
    parser.add_argument("--limit", type=int, default=5, help="Only used with --query")
    parser.add_argument("--force", action="store_true", help="Re-process existing papers")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    if args.ids_file:
        wanted = load_corpus_ids(args.ids_file)
        log.info("corpus file lists %d papers", len(wanted))
        papers = arxiv_client.fetch_by_ids(wanted, delay_seconds=settings.arxiv_delay_seconds)
        missing = set(wanted) - {p.arxiv_id for p in papers}
        if missing:
            log.warning("arXiv returned nothing for %d id(s): %s", len(missing), sorted(missing))
    else:
        papers = arxiv_client.search(
            args.query, limit=args.limit, delay_seconds=settings.arxiv_delay_seconds
        )
    log.info("Found %d papers", len(papers))

    total_chunks = 0
    for paper in papers:
        chunk_path = settings.chunks_dir / f"{paper.arxiv_id}.jsonl"
        if chunk_path.exists() and not args.force:
            log.info("skip %s (already ingested)", paper.arxiv_id)
            continue

        log.info("processing %s — %s", paper.arxiv_id, paper.title[:70])
        pdf_path = settings.papers_dir / f"{paper.arxiv_id}.pdf"
        pdf_parser.download_pdf(paper.pdf_url, pdf_path)

        raw = pdf_parser.extract_text(pdf_path)
        text = pdf_parser.clean_text(raw)
        chunks = chunker.chunk_paper(paper, text, settings.chunk_size, settings.chunk_overlap)

        if not chunks:
            log.warning("no chunks produced for %s — look at the extracted text", paper.arxiv_id)
            continue

        with chunk_path.open("w") as fh:
            for chunk in chunks:
                fh.write(chunk.model_dump_json() + "\n")

        # Metadata alongside the chunks, so retrieval can show titles and links later.
        meta_path = settings.papers_dir / f"{paper.arxiv_id}.json"
        meta_path.write_text(paper.model_dump_json(indent=2))

        total_chunks += len(chunks)
        log.info("  -> %d chunks", len(chunks))

    log.info("done: %d new chunks", total_chunks)

    return 0


if __name__ == "__main__":
    sys.exit(main())
