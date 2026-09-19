"""Commit the eval questions' embeddings, so the retrieval harness needs no API key.

    python -m scripts.export_query_cache      # writes eval/query_cache.json

**Why this exists.** `make retrieval-eval` was described as free and deterministic - no
model calls, pure arithmetic over a committed index. That was true only on a machine whose
embedding cache already held the 40 eval questions. CI, with no cache and no key, went
looking for the API on the first question and failed. The claim was warm-cache reasoning,
the same mistake this project keeps finding in its own latency numbers.

Embedding a question is cheap - about $0.0000004 - so the cost was never the point. The
point is that a harness which gates a build must not depend on a key, a network, or
anything that can answer differently on a different day. Committing the vectors makes the
whole retrieval evaluation reproducible by anyone who clones the repository.

About 1 MB for 40 questions. Regenerate it whenever `eval/questions.jsonl` changes.
"""

import json
from pathlib import Path

from arxiv_rag.config import Settings, get_settings
from arxiv_rag.retrieval.embeddings import cache_key, get_cache

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "questions.jsonl"
QUERY_CACHE = ROOT / "eval" / "query_cache.json"


def question_texts() -> list[str]:
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    return [q["question"] for q in rows]


def preload_query_cache(settings: Settings) -> int:
    """Put the committed vectors into this process's cache before anything embeds.

    Returns how many were loaded. Silent when the file is absent - a checkout that has not
    exported it yet should still work, just at the cost of an API call per question.
    """
    if not QUERY_CACHE.exists():
        return 0
    cache = get_cache(settings.embedding_cache_dir, settings.embedding_cache_writes)
    stored = json.loads(QUERY_CACHE.read_text())
    for key, vector in stored.items():
        if cache.get(key) is None:
            cache.put(key, vector)
    return len(stored)


def main() -> None:
    settings = get_settings()
    cache = get_cache(settings.embedding_cache_dir, writes_enabled=False)
    texts = question_texts()

    exported, missing = {}, []
    for text in texts:
        key = cache_key(text, settings.embedding_model)
        vector = cache.get(key)
        if vector is None:
            missing.append(text)
        else:
            exported[key] = vector

    if missing:
        # Deliberately a failure rather than a silent API call: this script is run to make
        # something reproducible, and quietly spending money to fill a gap would hide that
        # the local cache is not what it was assumed to be.
        print(f"!! {len(missing)} question(s) are not in the local embedding cache:")
        for text in missing[:5]:
            print(f"   {text[:72]}")
        print("\n   run `make retrieval-eval` once to populate it, then try again.")
        raise SystemExit(1)

    QUERY_CACHE.write_text(json.dumps(exported))
    size_mb = QUERY_CACHE.stat().st_size / 1e6
    print(
        f"{len(exported)} question embeddings ({settings.embedding_model}) -> "
        f"{QUERY_CACHE.relative_to(ROOT)}  ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
