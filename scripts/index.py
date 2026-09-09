"""Build the vector index from the chunks on disk.

    python -m scripts.index

Reads every chunk in ``data/chunks/``, embeds it (cached, so re-running is nearly free),
and writes the index to ``data/index/``.
"""

import argparse
import logging
import sys

from arxiv_rag.config import get_settings
from arxiv_rag.retrieval import embeddings, store

log = logging.getLogger("index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the vector index")
    parser.add_argument("--limit", type=int, help="Only index the first N chunks (for testing)")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    if not settings.openai_api_key:
        log.error("OPENAI_API_KEY is not set — check your .env")
        return 1

    chunks = store.load_chunks(settings.chunks_dir)
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        log.error("no chunks in %s — run scripts.ingest first", settings.chunks_dir)
        return 1

    total_tokens = sum(c.token_count for c in chunks)
    log.info(
        "embedding %d chunks (~%d tokens) with %s",
        len(chunks),
        total_tokens,
        settings.embedding_model,
    )

    vectors = embeddings.embed_texts([c.text for c in chunks], settings)

    import numpy as np

    index = store.ChunkStore(chunks, np.array(vectors, dtype=np.float32))
    index.save(settings.index_dir)

    log.info("done: %d chunks, %d dimensions", len(index), index.vectors.shape[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
