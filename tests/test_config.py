"""Stage 0 — these pass already. Read them to see how settings are meant to be used."""

from arxiv_rag.config import Settings


def test_defaults_are_sane():
    s = Settings(_env_file=None)
    assert s.chunk_size == 800
    assert s.chunk_overlap < s.chunk_size, "overlap >= size would loop forever"
    assert s.embedding_model.startswith("text-embedding")


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("PT_CHUNK_SIZE", "512")
    assert Settings(_env_file=None).chunk_size == 512


def test_derived_paths():
    s = Settings(_env_file=None, data_dir="/tmp/pt-test")
    assert s.papers_dir.name == "papers"
    assert s.chunks_dir.parent == s.data_dir
