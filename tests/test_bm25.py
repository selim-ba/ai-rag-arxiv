"""Stage 3 - BM25. A hand-built corpus, so every ranking is checkable by reading it."""

from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.bm25 import BM25Index, tokenize


def chunk(index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"paper::{index}",
        arxiv_id="1234.5678",
        title="A paper",
        section="Method",
        text=text,
        index=index,
        token_count=len(text.split()),
    )


# 0: names V-JEPA           2: the word "features" many times
# 1: names I-JEPA           3: long, mentions features once
CORPUS = [
    chunk(0, "V-JEPA is trained on video with a feature prediction objective"),
    chunk(1, "I-JEPA is trained on images with a feature prediction objective"),
    chunk(2, "features features features features features"),
    chunk(3, " ".join(["padding"] * 60) + " features"),
]


def build() -> BM25Index:
    return BM25Index(CORPUS)


# -- tokenize ------------------------------------------------------------------------


def test_lowercases():
    assert tokenize("Dreamer LEARNS") == ["dreamer", "learns"]


def test_hyphenated_names_survive_whole():
    """The decision the whole module rests on.

    Splitting V-JEPA into "v" and "jepa" makes it indistinguishable from I-JEPA on the
    half that carries meaning, and adds a junk token that appears everywhere.
    """
    assert tokenize("V-JEPA and I-JEPA") == ["v-jepa", "and", "i-jepa"]
    assert tokenize("the MDN-RNN") == ["the", "mdn-rnn"]


def test_punctuation_is_dropped_but_words_are_kept():
    assert tokenize("predicts (masked) features.") == ["predicts", "masked", "features"]


def test_digits_are_kept():
    """'V-JEPA 2' must not become 'V-JEPA'."""
    assert tokenize("V-JEPA 2 and DreamerV3") == ["v-jepa", "2", "and", "dreamerv3"]


def test_empty_text_gives_no_tokens():
    assert tokenize("") == []


# -- search --------------------------------------------------------------------------


def test_rare_name_retrieves_the_right_document():
    """The q030 case in miniature: dense ranked V-JEPA's own paper 170th."""
    hits = build().search("V-JEPA", k=2)
    assert hits[0].chunk.chunk_id == "paper::0"


def test_sibling_names_are_not_confused():
    assert build().search("I-JEPA", k=1)[0].chunk.chunk_id == "paper::1"


def test_common_terms_do_not_decide_the_ranking():
    """Every document has 'features' or 'feature'; only one has 'video'."""
    assert build().search("video features", k=1)[0].chunk.chunk_id == "paper::0"


def test_repetition_saturates():
    """Five mentions must beat one, but not by five times - that is what k1 buys."""
    index = build()
    scores = {h.chunk.chunk_id: h.score for h in index.search("features", k=4)}
    assert scores["paper::2"] > scores["paper::3"]
    assert scores["paper::2"] < 5 * scores["paper::3"]


def test_shorter_documents_win_on_equal_evidence():
    """paper::3 mentions 'features' once inside 60 words of padding. b penalises it."""
    index = BM25Index([chunk(0, "features"), chunk(1, " ".join(["padding"] * 60) + " features")])
    hits = index.search("features", k=2)
    assert hits[0].chunk.chunk_id == "paper::0"


def test_unknown_term_returns_nothing_rather_than_raising():
    assert build().search("transformer", k=5) == []


def test_returns_fewer_than_k_when_few_documents_match():
    """Unlike the dense store, which always has 874 things to rank."""
    assert len(build().search("video", k=5)) == 1


def test_k_is_respected():
    assert len(build().search("feature prediction objective", k=1)) == 1


def test_results_are_sorted_best_first():
    hits = build().search("features feature prediction", k=4)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_returns_search_hits_like_the_dense_store():
    """Same shape as ChunkStore.search, so the two are interchangeable."""
    hit = build().search("V-JEPA", k=1)[0]
    assert hit.chunk.chunk_id == "paper::0"
    assert isinstance(hit.score, float)
