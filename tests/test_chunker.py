"""Stage 1 — the chunker. This file is the spec; make it green.

``heading_of`` and ``count_tokens`` are given, so their tests pass already. The rest
fail until you implement them.
"""

import pytest

from arxiv_rag.ingestion.chunker import (
    chunk_text,
    count_tokens,
    heading_of,
    split_into_sections,
    strip_references,
)

SAMPLE = """Some preamble text before any heading.

Introduction

We introduce a thing.

2 Related Work

Others did things.

References

[1] Someone, "A paper", 2020.
"""


# --- given helpers: these already pass -------------------------------------------


def test_heading_of_recognises_known_headings():
    assert heading_of("Introduction") == "Introduction"
    assert heading_of("REFERENCES") == "REFERENCES"


def test_heading_of_recognises_numbered_headings():
    assert heading_of("2 Related Work") == "Related Work"
    assert heading_of("3.1 Experimental Setup") == "Experimental Setup"


def test_heading_of_rejects_prose():
    assert heading_of("We introduce a thing.") is None
    assert heading_of("Bananas") is None  # capitalised, but not a known heading


def test_count_tokens():
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0


# --- split_into_sections ----------------------------------------------------------


def test_sections_are_in_document_order():
    titles = [s.title for s in split_into_sections(SAMPLE)]
    assert titles == ["", "Introduction", "Related Work", "References"]


def test_preamble_has_empty_title():
    first = split_into_sections(SAMPLE)[0]
    assert first.title == ""
    assert first.body == "Some preamble text before any heading."


def test_body_excludes_the_heading_line():
    intro = split_into_sections(SAMPLE)[1]
    assert intro.body == "We introduce a thing."
    assert "Introduction" not in intro.body


def test_final_section_is_not_lost():
    """The classic bug: forgetting to flush the buffer after the loop."""
    last = split_into_sections(SAMPLE)[-1]
    assert last.title == "References"
    assert "Someone" in last.body


def test_empty_sections_are_dropped():
    text = "Introduction\n\nReal content.\n\nDiscussion\n\n"
    assert [s.title for s in split_into_sections(text)] == ["Introduction"]


# --- strip_references -------------------------------------------------------------


def test_strip_references_removes_bibliography():
    out = strip_references(SAMPLE)
    assert "Someone" not in out
    assert "References" not in out
    assert "Others did things." in out


def test_strip_references_is_a_noop_without_a_bibliography():
    text = "Introduction\n\nNo bibliography here."
    assert strip_references(text) == text.strip()


# --- chunk_text -------------------------------------------------------------------

LONG = ". ".join(f"sentence {i} contains the unique token w{i}" for i in range(200)) + "."


def test_empty_text_yields_no_chunks():
    assert chunk_text("", chunk_size=100, overlap=10) == []
    assert chunk_text("   \n  ", chunk_size=100, overlap=10) == []


def test_short_text_is_a_single_chunk():
    text = "A short paragraph that easily fits inside the budget."
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].strip() == text


def test_no_chunk_exceeds_the_budget():
    chunks = chunk_text(LONG, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 for c in chunks)


def test_no_content_is_lost():
    chunks = chunk_text(LONG, chunk_size=100, overlap=20)
    produced = set(" ".join(chunks).split())
    assert set(LONG.split()) <= produced


def test_consecutive_chunks_overlap():
    """With overlap, some text is deliberately duplicated, so the chunks together
    contain more tokens than the original. If this fails, you are splitting with no
    overlap at all — and a sentence landing on a boundary will be lost to retrieval."""
    chunks = chunk_text(LONG, chunk_size=100, overlap=20)
    assert sum(count_tokens(c) for c in chunks) > count_tokens(LONG)


def test_overlap_not_smaller_than_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text(LONG, chunk_size=100, overlap=100)
