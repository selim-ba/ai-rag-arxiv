"""Stage 1 — cleaning extracted PDF text. This is the spec for ``clean_text``."""

from arxiv_rag.ingestion.pdf_parser import clean_text


def test_dehyphenates_words_broken_across_lines():
    assert clean_text("retrie-\nval augmented") == "retrieval augmented"


def test_single_newline_is_a_line_wrap_not_a_paragraph_break():
    assert clean_text("the quick brown\nfox jumps") == "the quick brown fox jumps"


def test_blank_line_separates_paragraphs():
    assert clean_text("first para\n\nsecond para") == "first para\n\nsecond para"


def test_many_blank_lines_collapse_to_one():
    assert clean_text("first\n\n\n\n\nsecond") == "first\n\nsecond"


def test_runs_of_spaces_and_tabs_collapse():
    assert clean_text("a    b\t\tc") == "a b c"


def test_result_is_stripped():
    assert clean_text("\n\n  hello world  \n\n") == "hello world"


def test_empty_input():
    assert clean_text("") == ""


def test_realistic_fragment():
    raw = (
        "We propose a retrie-\nval augmented\ngeneration    approach.\n\n"
        "2   Related  Work\n\nPrior work on knowl-\nedge intensive tasks.\n"
    )
    cleaned = clean_text(raw)
    assert "retrieval augmented generation approach." in cleaned
    assert "knowledge intensive tasks." in cleaned
    assert "2 Related Work" in cleaned
