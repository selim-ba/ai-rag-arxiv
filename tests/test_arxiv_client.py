"""Stage 1 - parsing the arXiv Atom feed. No network: the fixture is a real response."""

from datetime import date

from arxiv_rag.ingestion.arxiv_client import normalise_arxiv_id, parse_atom_feed


def test_normalise_strips_version_and_url():
    assert normalise_arxiv_id("http://arxiv.org/abs/2005.11401v4") == "2005.11401"
    assert normalise_arxiv_id("2106.09685v2") == "2106.09685"
    assert normalise_arxiv_id("2106.09685") == "2106.09685"


def test_parses_all_entries(arxiv_atom_xml):
    papers = parse_atom_feed(arxiv_atom_xml)
    assert len(papers) == 2
    assert [p.arxiv_id for p in papers] == ["2005.11401", "2106.09685"]


def test_title_whitespace_is_collapsed(arxiv_atom_xml):
    """The fixture's first title is split across two lines, as arXiv really sends it."""
    paper = parse_atom_feed(arxiv_atom_xml)[0]
    assert paper.title == ("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks")
    assert "\n" not in paper.title


def test_abstract_is_collapsed_and_stripped(arxiv_atom_xml):
    paper = parse_atom_feed(arxiv_atom_xml)[0]
    assert paper.abstract.startswith("Large pre-trained language models")
    assert "\n" not in paper.abstract
    assert paper.abstract == paper.abstract.strip()


def test_authors_in_order(arxiv_atom_xml):
    paper = parse_atom_feed(arxiv_atom_xml)[0]
    assert paper.authors[:2] == ["Patrick Lewis", "Ethan Perez"]
    assert len(paper.authors) == 3


def test_published_date(arxiv_atom_xml):
    assert parse_atom_feed(arxiv_atom_xml)[0].published == date(2020, 5, 22)


def test_pdf_url_comes_from_the_feed(arxiv_atom_xml):
    paper = parse_atom_feed(arxiv_atom_xml)[0]
    assert paper.pdf_url.endswith("/pdf/2005.11401v4")


def test_categories(arxiv_atom_xml):
    papers = parse_atom_feed(arxiv_atom_xml)
    assert "cs.CL" in papers[0].categories
    assert "cs.LG" in papers[0].categories
    assert papers[1].categories == ["cs.CL"]


def test_empty_feed_returns_empty_list():
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_atom_feed(empty) == []


def test_normalise_strips_the_arxiv_prefix():
    """The spelling this codebase emits. `format_context` renders every passage header as
    `[P1] arXiv:2404.08471 | title`, so a model calling a tool with a paper id writes it
    back that way - and an id that fails to normalise silently becomes an unknown paper.
    """
    assert normalise_arxiv_id("arXiv:2404.08471v2") == "2404.08471"
    assert normalise_arxiv_id("ARXIV:1912.01603") == "1912.01603"


def test_normalise_is_idempotent_on_a_bare_id():
    assert normalise_arxiv_id("2404.08471") == "2404.08471"
    assert normalise_arxiv_id(normalise_arxiv_id("arXiv:2404.08471v2")) == "2404.08471"


def test_normalise_tolerates_surrounding_whitespace():
    assert normalise_arxiv_id("  2404.08471 ") == "2404.08471"
