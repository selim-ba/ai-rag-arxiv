"""Ingestion CLI: arXiv search -> PDFs -> cleaned text -> chunks on disk.

    python -m scripts.ingest --query "retrieval augmented generation" --limit 5

The plumbing is given. The four TODO lines call the functions you implement in Stage 1.
"""

import argparse
import logging
import sys

from arxiv_rag.config import get_settings
from arxiv_rag.ingestion import arxiv_client, chunker, pdf_parser

log = logging.getLogger("ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into the local index")
    parser.add_argument("--query", required=True, help='arXiv query, e.g. all:"RAG"')
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Re-process existing papers")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    # TODO(you): fetch papers from arXiv.
    papers = arxiv_client.search(args.query, limit=args.limit,
                                 delay_seconds=settings.arxiv_delay_seconds)
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

        # TODO(you): extract, clean, chunk.
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

    # Sanity check you should actually do, not just read:
    # open one .jsonl in data/chunks and read three chunks out loud. If they do not
    # read as coherent, self-contained passages, your chunker needs work — and no
    # amount of clever retrieval later will fix it.
    return 0


if __name__ == "__main__":
    sys.exit(main())
