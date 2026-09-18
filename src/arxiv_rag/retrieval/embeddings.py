"""Turning chunk text into vectors.

An embedding is a fixed-length list of floats positioning a piece of text in a space
where "nearby" means "similar in meaning". Two chunks about latent dynamics land close
together even with no words in common; that is what lets retrieval find a passage the
user did not know how to phrase.

The weakness is the mirror image, and you should reproduce it deliberately in this
stage: dense vectors are poor at *exact* terms. A rare token like "LoRA" or "V-JEPA"
carries little weight in a 1536-dimensional average, so the paper literally named after
it may not come back. That single failure is the entire motivation for hybrid search in
Stage 3.
"""

import hashlib
import json
import logging
import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from arxiv_rag.config import Settings
from arxiv_rag.llm import get_client as _client

log = logging.getLogger(__name__)


def cache_key(text: str, model: str) -> str:
    """Stable id for one (text, model) pair.

    The model is part of the key because embeddings from different models are not
    comparable — mixing them in one index silently produces nonsense results.
    """
    digest = hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()
    return digest[:32]


class EmbeddingCache:
    """Disk cache of embeddings, keyed by ``cache_key``.

    Re-embedding a chunk you have already paid for is money and time on fire, and you
    will re-run the indexing script many times.

    **Three details that are about latency, not correctness**, all found by measuring
    rather than by reading the code. At 919 entries this file is 28MB of JSON.

    1. *Loaded once per process, not once per call.* ``get_cache`` is ``lru_cache``d.
       Constructing it per query meant a 28MB read and parse before every single
       retrieval.
    2. *Saved only when something changed.* ``embed_texts`` called ``save()``
       unconditionally, so a query that was a pure cache hit still serialised and wrote
       28MB. Together with (1) that was 479ms per query against 0.9ms of actual search.
    3. *Written atomically.* ``write_text`` truncates the file and then fills it, so a
       crash - or two concurrent ``/ask`` requests - leaves a corrupt cache that fails to
       parse on the next start. Writing a temporary file and ``os.replace``-ing it is
       atomic on POSIX: readers see either the old file or the new one, never a partial.
    """

    def __init__(self, path: Path, writes_enabled: bool = True) -> None:
        self.path = path / "cache.json"
        self._data: dict[str, list[float]] = {}
        self._dirty = False
        # Reading is always allowed. A cache baked into a read-only image is still worth
        # every hit it serves; only persisting new entries is in question.
        self._writes_enabled = writes_enabled
        if self.path.exists():
            self._data = json.loads(self.path.read_text())
            log.info("embedding cache: %d entries", len(self._data))
        if not writes_enabled:
            log.info("embedding cache: writes disabled; new embeddings are not persisted")

    def get(self, key: str) -> list[float] | None:
        return self._data.get(key)

    def put(self, key: str, vector: list[float]) -> None:
        self._data[key] = vector
        self._dirty = True

    def save(self) -> None:
        """No-op when nothing was added, or when writes are off. Atomic when it happens.

        **A failed write here must not fail the request.** The cache saves money; it does
        not make an answer correct, and a service that returns 500 because an optimisation
        could not persist has traded a real failure for an imaginary one. The rule this
        draws is worth stating: a write the system depends on fails loudly, a write that
        only makes it cheaper degrades quietly and says so once.

        Once rejected, writes stay off for the life of the process. Retrying on every
        cache miss would turn one `OSError` into one per request, and a read-only
        filesystem does not become writable while you watch it.
        """
        if not self._dirty or not self._writes_enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data))
            os.replace(tmp, self.path)  # atomic: never leaves a half-written cache
        except OSError as exc:
            self._writes_enabled = False
            log.warning("embedding cache: read-only (%s); continuing without saving", exc)
            return
        self._dirty = False
        log.info("embedding cache: saved %d entries", len(self._data))


@lru_cache(maxsize=8)
def get_cache(directory: Path, writes_enabled: bool = True) -> EmbeddingCache:
    """One cache object per directory per process.

    The 28MB parse happens on first use and never again. Every caller must go through
    this rather than constructing ``EmbeddingCache`` directly, or the saving is split
    across two objects that do not see each other's writes.
    """
    return EmbeddingCache(directory, writes_enabled=writes_enabled)


def batched(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield ``items`` in lists of at most ``size``.

    Batching matters: one API call per chunk means 874 round trips. The embeddings
    endpoint accepts a list, so batching turns that into seven. Same tokens, same cost,
    a fraction of the wall-clock time.
    """
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    """Embed a list of texts, returning one vector per input, in order."""
    # Step 1 - open the cache and the client
    cache = get_cache(settings.embedding_cache_dir, settings.embedding_cache_writes)
    client = _client(settings)

    # Step 2 - find cache misses and their indices
    cache_misses = []
    indices = []
    for i, text in enumerate(texts):
        key = cache_key(text, settings.embedding_model)
        cached_vector = cache.get(key)
        if cached_vector is not None:
            # Cache hit, store the vector in the correct position
            indices.append((i, cached_vector))
        else:
            # Cache miss, store the index for later embedding
            cache_misses.append(text)
            indices.append((i, None))

    # Step 3 - embed the cache misses in batches
    for batch in batched(cache_misses, settings.embedding_batch_size):
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        for text, embedding in zip(
            batch, sorted(response.data, key=lambda d: d.index), strict=True
        ):
            key = cache_key(text, settings.embedding_model)
            cache.put(key, embedding.embedding)

    # Step 4 - save the cache
    cache.save()

    # Step 5 - construct the final list of embeddings in the original order
    embeddings = [None] * len(texts)
    for i, vector in indices:
        if vector is not None:
            embeddings[i] = vector
        else:
            key = cache_key(texts[i], settings.embedding_model)
            embeddings[i] = cache.get(key)

    return embeddings


def embed_query(query: str, settings: Settings) -> list[float]:
    """Embed a single search query. Just ``embed_texts`` of one."""
    return embed_texts([query], settings)[0]
