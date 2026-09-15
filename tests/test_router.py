"""Stage 4 step 5 - the router's verification layer. Pure; no model, no network.

`verify_route` is where the model's route meets what the index actually contains. The
router is asked to do a membership check over a 49-line list, which is exactly the task
models are unreliable at, so the claim is reconciled with the index rather than trusted.
Seventh place in this codebase that a model's own report is checked against the source.
"""

from arxiv_rag.agent.router import RouteDecision, build_router_prompt, verify_route
from arxiv_rag.agent.tools import IndexedPaper

INDEXED = frozenset({"1912.01603", "2404.08471", "2506.09985"})

PAPERS = [
    IndexedPaper(arxiv_id="1912.01603", title="Dream to Control", chunk_count=14),
    IndexedPaper(arxiv_id="2404.08471", title="Revisiting Feature Prediction", chunk_count=9),
]


# -- rule 1: a scope-restricted search with no scope is not scoped ---------------------


def test_filtered_with_no_ids_becomes_retrieve():
    out = verify_route(RouteDecision(route="filtered", arxiv_ids=[]), INDEXED)
    assert out.route == "retrieve"


def test_filtered_with_ids_stays_filtered():
    out = verify_route(RouteDecision(route="filtered", arxiv_ids=["1912.01603"]), INDEXED)
    assert out.route == "filtered" and out.arxiv_ids == ["1912.01603"]


# -- rule 2: a named paper that is not in the index is a catalog question --------------


def test_filtered_on_an_unindexed_paper_becomes_catalog():
    """The model recognised "TD-MPC2" as a paper name and missed that it is absent.
    Searching for it returns nothing or everything; either way the honest answer is that
    the paper is not in this index."""
    out = verify_route(RouteDecision(route="filtered", arxiv_ids=["2310.16828"]), INDEXED)
    assert out.route == "catalog"


def test_the_unknown_ids_survive_onto_the_decision():
    """The catalog handler has to be able to name the paper it does not have."""
    out = verify_route(RouteDecision(route="filtered", arxiv_ids=["2310.16828"]), INDEXED)
    assert out.arxiv_ids == ["2310.16828"]


# -- rule 3: a mix keeps what exists ---------------------------------------------------


def test_a_mix_of_known_and_unknown_keeps_the_known():
    out = verify_route(
        RouteDecision(route="filtered", arxiv_ids=["1912.01603", "2310.16828"]), INDEXED
    )
    assert out.route == "filtered" and out.arxiv_ids == ["1912.01603"]


def test_two_indexed_ids_both_survive():
    """f03 compares V-JEPA with V-JEPA 2. Dropping one answers half the question."""
    out = verify_route(
        RouteDecision(route="filtered", arxiv_ids=["2404.08471", "2506.09985"]), INDEXED
    )
    assert sorted(out.arxiv_ids) == ["2404.08471", "2506.09985"]


# -- routes that make no membership claim pass through --------------------------------


def test_retrieve_is_untouched():
    out = verify_route(RouteDecision(route="retrieve", reason="unscoped"), INDEXED)
    assert out.route == "retrieve" and out.reason == "unscoped"


def test_catalog_is_untouched():
    out = verify_route(RouteDecision(route="catalog", reason="asks what is indexed"), INDEXED)
    assert out.route == "catalog"


def test_verify_does_not_mutate_its_input():
    """Pure. A verification that edits the thing it verifies makes the original
    unrecoverable, and the original is what a trace should show."""
    decision = RouteDecision(route="filtered", arxiv_ids=["2310.16828"])
    verify_route(decision, INDEXED)
    assert decision.route == "filtered" and decision.arxiv_ids == ["2310.16828"]


# -- the prompt carries the corpus ----------------------------------------------------


def test_the_catalog_is_injected_into_the_prompt():
    """Without it the router cannot tell an absent VALUE from an absent PAPER - the
    f06/c03 pair in eval/routes.jsonl turns on exactly that."""
    prompt = build_router_prompt(PAPERS)
    assert "1912.01603" in prompt and "Dream to Control" in prompt


def test_the_prompt_still_states_the_decisive_rule():
    prompt = build_router_prompt(PAPERS)
    assert "absent VALUE is not an absent PAPER" in prompt
