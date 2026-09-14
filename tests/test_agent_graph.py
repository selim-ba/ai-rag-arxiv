"""Stage 4 step 1 - a graph that changes nothing.

Fake retriever, monkeypatched generation: no index, no API key, no network. What is being
tested is the wiring - that nodes run in order, that partial updates merge correctly, and
that the reducer on `trace` appends rather than overwrites.
"""

import pytest

from arxiv_rag.agent import nodes as agent_nodes
from arxiv_rag.agent.grader import Grade
from arxiv_rag.agent.graph import build_graph, make_decide_after_grade, run_agent
from arxiv_rag.agent.state import AgentState, initial_state
from arxiv_rag.config import Settings
from arxiv_rag.ingestion.models import Chunk
from arxiv_rag.retrieval.answer import Answer
from arxiv_rag.retrieval.store import SearchHit

CHUNKS = [
    Chunk(
        chunk_id=f"1811.04551::{i}",
        arxiv_id="1811.04551",
        title="Learning Latent Dynamics",
        section="Method",
        text=f"passage {i}",
        index=i,
        token_count=5,
    )
    for i in range(3)
]


class FakeRetriever:
    def __init__(self, hits=None):
        self.hits = CHUNKS if hits is None else hits
        self.queries: list[str] = []

    def search(self, query, k=5, chunk_filter=None):
        self.queries.append(query)
        return [SearchHit(c, 1.0) for c in self.hits[:k]]


class FakeGrader:
    """A grader with a scripted opinion. No model, no network.

    `verdicts` is consumed one per call; the last one repeats forever, so a grader that
    never changes its mind is `FakeGrader([False])` - which is exactly the case the loop
    guard has to survive.
    """

    def __init__(self, verdicts=(True,), missing="the specific number is not stated"):
        self.verdicts = list(verdicts)
        self.missing = missing
        self.calls: list[str] = []

    def grade(self, question, hits):
        self.calls.append(question)
        relevant = self.verdicts[min(len(self.calls) - 1, len(self.verdicts) - 1)]
        if relevant:
            return Grade(relevant=True, evidence="a real quote from the passages")
        return Grade(relevant=False, missing=self.missing)


@pytest.fixture
def fake_rewrite(monkeypatch):
    """Replace the rewrite LLM call. Records what the node passed it."""
    seen = {}

    def rewrite_query(question, missing, settings):
        seen.setdefault("calls", []).append((question, missing))
        return f"{question} :: {missing}"

    monkeypatch.setattr(agent_nodes, "rewrite_query", rewrite_query)
    return seen


@pytest.fixture
def fake_generate(monkeypatch):
    """Replace the LLM call with something deterministic that records what it saw."""
    seen = {}

    def generate_answer(question, hits, settings):
        seen["question"] = question
        seen["hits"] = hits
        return Answer(
            question=question,
            text="an answer [1811.04551]",
            citations=["1811.04551"],
            refused=False,
            retrieved_ids=[h.chunk.chunk_id for h in hits],
        )

    monkeypatch.setattr(agent_nodes, "generate_answer", generate_answer)
    return seen


# -- state and reducers ---------------------------------------------------------------


def test_initial_state_fills_every_key_a_node_might_read():
    state = initial_state("why do world models collapse?")
    assert state["question"] == "why do world models collapse?"
    assert state["query"] == state["question"]
    assert state["attempts"] == 0
    assert state["hits"] == [] and state["trace"] == []


def test_trace_accumulates_across_nodes(fake_generate):
    """The reducer: every node appends one line, and all of them survive.

    Without `Annotated[list[str], operator.add]` each node's return would OVERWRITE the
    last's and the trace would have one entry. No error - just a shorter list.

    The count is the number of nodes that actually ran, so it moves whenever the graph
    grows: two before the grade node existed, three now on the no-retry path.
    """
    graph = build_graph(FakeRetriever(), Settings(), FakeGrader())
    final: AgentState = graph.invoke(initial_state("a question"))
    assert len(final["trace"]) == 3  # retrieve, grade, generate
    assert "retrieve" in final["trace"][0]


# -- wiring ---------------------------------------------------------------------------


def test_nodes_run_in_order_retrieve_then_generate(fake_generate):
    graph = build_graph(FakeRetriever(), Settings(), FakeGrader())
    graph.invoke(initial_state("a question"))
    assert fake_generate["hits"], "generate ran before retrieve, or got no hits"
    assert [h.chunk.chunk_id for h in fake_generate["hits"]] == [c.chunk_id for c in CHUNKS]


def test_generate_answers_the_original_question_not_the_rewritten_query(fake_generate):
    """A rewritten query is a retrieval device. Answering it instead of what was asked is
    a subtle way to be confidently off-topic - and there will be a rewrite node soon."""
    graph = build_graph(FakeRetriever(), Settings(), FakeGrader())
    state = initial_state("what did the user actually ask?")
    state["query"] = "some rewritten search string"
    graph.invoke(state)
    assert fake_generate["question"] == "what did the user actually ask?"


def test_the_retriever_is_given_the_query_not_the_question(fake_generate):
    """The mirror image: retrieval searches `query`, which is what a rewrite changes."""
    retriever = FakeRetriever()
    graph = build_graph(retriever, Settings(), FakeGrader())
    state = initial_state("what did the user actually ask?")
    state["query"] = "some rewritten search string"
    graph.invoke(state)
    assert retriever.queries == ["some rewritten search string"]


def test_run_agent_returns_an_answer_not_a_state(fake_generate):
    """Everything upstream cares about an Answer. The graph is an implementation detail."""
    answer = run_agent(build_graph(FakeRetriever(), Settings(), FakeGrader()), "a question")
    assert isinstance(answer, Answer)
    assert answer.citations == ["1811.04551"]
    assert answer.retrieved_ids == [c.chunk_id for c in CHUNKS]


def test_empty_retrieval_still_produces_an_answer(fake_generate):
    """No hits is a normal outcome - generate_answer refuses. It is not a graph failure."""
    answer = run_agent(build_graph(FakeRetriever(hits=[]), Settings(), FakeGrader()), "a question")
    assert isinstance(answer, Answer)


def test_chunk_filter_reaches_the_retriever(fake_generate):
    from arxiv_rag.retrieval.filters import ChunkFilter

    seen = {}

    class Recording(FakeRetriever):
        def search(self, query, k=5, chunk_filter=None):
            seen["filter"] = chunk_filter
            return super().search(query, k, chunk_filter)

    f = ChunkFilter(arxiv_ids=frozenset({"1811.04551"}))
    run_agent(build_graph(Recording(), Settings(), FakeGrader()), "a question", chunk_filter=f)
    assert seen["filter"] == f


def test_retrieval_timing_survives_onto_the_answer(fake_generate):
    """The agent must report the same fields as the pipeline, or the harness drops rows.

    Measured the hard way: `retrieve_ms` defaulted to 0.0 on the agent path, and
    `if r.get("retrieve_ms")` treated that as absent, so the whole latency row disappeared
    from the eval output without any error. Fourth falsy-zero bug in this project.
    """
    answer = run_agent(build_graph(FakeRetriever(), Settings(), FakeGrader()), "a question")
    assert answer.retrieve_ms > 0.0


def test_trace_records_how_long_retrieval_took(fake_generate):
    graph = build_graph(FakeRetriever(), Settings(), FakeGrader())
    final = graph.invoke(initial_state("a question"))
    assert "ms" in final["trace"][0]


# -- step 3: the loop ------------------------------------------------------------------


def test_a_relevant_grade_goes_straight_to_generate(fake_generate, fake_rewrite):
    """The common case. A grader that approves must cost exactly one retrieval."""
    retriever = FakeRetriever()
    graph = build_graph(retriever, Settings(), FakeGrader([True]))
    graph.invoke(initial_state("a question"))
    assert len(retriever.queries) == 1
    assert "calls" not in fake_rewrite, "rewrote a query the grader was happy with"


def test_a_failed_grade_rewrites_and_retrieves_again(fake_generate, fake_rewrite):
    retriever = FakeRetriever()
    graph = build_graph(retriever, Settings(max_retries=1), FakeGrader([False, True]))
    final = graph.invoke(initial_state("what data is V-JEPA trained on?"))
    assert len(retriever.queries) == 2, "the back-edge did not fire"
    assert retriever.queries[1] != retriever.queries[0]
    assert final["attempts"] == 1


def test_the_rewrite_node_is_told_what_was_missing(fake_generate, fake_rewrite):
    """The grader's `missing` field is the whole reason the retry is better than a repeat.

    Without it the rewrite is a blind paraphrase and the second retrieval is a coin flip.
    """
    grader = FakeGrader([False, True], missing="the size of the pretraining dataset")
    graph = build_graph(FakeRetriever(), Settings(), grader)
    graph.invoke(initial_state("what data is V-JEPA trained on?"))
    question, missing = fake_rewrite["calls"][0]
    assert question == "what data is V-JEPA trained on?"
    assert missing == "the size of the pretraining dataset"


def test_a_grader_that_never_approves_still_terminates(fake_generate, fake_rewrite):
    """The one that matters. The guard cannot depend on the grader changing its mind.

    Measured: catch 0.417, false alarm 0.091. This is not an oracle, and a loop written as
    "retry until the grader is satisfied" is a loop whose exit condition is a component
    known to be wrong a third of the time.
    """
    retriever = FakeRetriever()
    graph = build_graph(retriever, Settings(max_retries=1), FakeGrader([False]))
    final = graph.invoke(initial_state("a question"))
    assert len(retriever.queries) == 2  # original + one retry, then answer anyway
    assert final["answer"] is not None
    assert final["attempts"] == 1


def test_max_retries_zero_disables_the_loop_entirely(fake_generate, fake_rewrite):
    """The kill switch: the agent falls back to exactly the Stage 2 pipeline shape.

    Worth keeping working, because it is what an A/B of "is the loop earning anything"
    runs against.
    """
    retriever = FakeRetriever()
    graph = build_graph(retriever, Settings(max_retries=0), FakeGrader([False]))
    final = graph.invoke(initial_state("a question"))
    assert len(retriever.queries) == 1
    assert final["answer"] is not None


def test_two_retries_are_two_retries(fake_generate, fake_rewrite):
    retriever = FakeRetriever()
    graph = build_graph(retriever, Settings(max_retries=2), FakeGrader([False]))
    final = graph.invoke(initial_state("a question"))
    assert len(retriever.queries) == 3
    assert final["attempts"] == 2


def test_generate_still_answers_the_original_question_after_a_rewrite(fake_generate, fake_rewrite):
    """The rewritten string is a retrieval device. Answering it is how an agent ends up
    confidently addressing a question nobody asked."""
    graph = build_graph(FakeRetriever(), Settings(max_retries=1), FakeGrader([False, True]))
    graph.invoke(initial_state("what did the user actually ask?"))
    assert fake_generate["question"] == "what did the user actually ask?"


def test_the_grade_in_state_is_the_latest_one(fake_generate, fake_rewrite):
    """No reducer on `grade`, deliberately: the edge wants the current verdict, not a log."""
    graph = build_graph(FakeRetriever(), Settings(max_retries=1), FakeGrader([False, True]))
    final = graph.invoke(initial_state("a question"))
    assert final["grade"].relevant


def test_the_trace_shows_the_whole_lap(fake_generate, fake_rewrite):
    graph = build_graph(FakeRetriever(), Settings(max_retries=1), FakeGrader([False, True]))
    final = graph.invoke(initial_state("a question"))
    steps = [line.split()[0].split("(")[0] for line in final["trace"]]
    assert steps == ["retrieve", "grade", "rewrite", "retrieve", "grade", "generate"]


# -- the edge condition on its own ------------------------------------------------------


def _state(relevant: bool, attempts: int) -> AgentState:
    state = initial_state("a question")
    state["grade"] = Grade(relevant=relevant, missing="" if relevant else "something")
    state["attempts"] = attempts
    return state


def test_decide_routes_a_good_grade_to_generate():
    decide = make_decide_after_grade(Settings(max_retries=1))
    assert decide(_state(relevant=True, attempts=0)) == "generate"


def test_decide_routes_a_bad_grade_to_rewrite_while_budget_remains():
    decide = make_decide_after_grade(Settings(max_retries=1))
    assert decide(_state(relevant=False, attempts=0)) == "rewrite"


def test_decide_stops_when_the_budget_is_spent():
    decide = make_decide_after_grade(Settings(max_retries=1))
    assert decide(_state(relevant=False, attempts=1)) == "generate"


def test_decide_stops_when_the_budget_is_overspent():
    """`>=`, not `==`. A cap that only catches exact equality is not a cap."""
    decide = make_decide_after_grade(Settings(max_retries=1))
    assert decide(_state(relevant=False, attempts=7)) == "generate"


def test_decide_survives_a_missing_grade():
    """Defensive: a node order change that routes here before grading should not crash
    the graph into an unhandled KeyError."""
    decide = make_decide_after_grade(Settings(max_retries=1))
    state = initial_state("a question")
    assert decide(state) in {"generate", "rewrite"}


# -- instrumentation: the loop has to be visible from outside the graph -----------------


class ShiftingRetriever(FakeRetriever):
    """Returns different chunks on each call, so a retry is detectable in the ids."""

    def search(self, query, k=5, chunk_filter=None):
        self.queries.append(query)
        offset = len(self.queries) - 1
        rotated = CHUNKS[offset:] + CHUNKS[:offset]
        return [SearchHit(c, 1.0) for c in rotated[:k]]


def test_the_answer_reports_how_many_retries_happened(fake_generate, fake_rewrite):
    """Without this the harness cannot tell a loop that never fired from a loop that fired
    and achieved nothing - the two produce byte-identical metrics."""
    answer = run_agent(
        build_graph(FakeRetriever(), Settings(max_retries=1), FakeGrader([False])), "a question"
    )
    assert answer.attempts == 1


def test_no_retry_reports_zero_attempts(fake_generate, fake_rewrite):
    answer = run_agent(build_graph(FakeRetriever(), Settings(), FakeGrader([True])), "a question")
    assert answer.attempts == 0
    assert answer.first_retrieved_ids == []


def test_the_first_retrieval_is_kept_for_comparison(fake_generate, fake_rewrite):
    """`hits` is overwritten by the retry, so the before-picture has to be saved."""
    retriever = ShiftingRetriever()
    answer = run_agent(
        build_graph(retriever, Settings(max_retries=1), FakeGrader([False])), "a question"
    )
    assert len(retriever.queries) == 2
    assert answer.first_retrieved_ids == [c.chunk_id for c in CHUNKS]
    assert answer.retrieved_ids != answer.first_retrieved_ids


def test_the_snapshot_is_not_overwritten_by_later_laps(fake_generate, fake_rewrite):
    """Two retries, and the snapshot still shows lap one. The retrieve node returns the
    key only when `attempts` is 0; on later laps it is absent and LangGraph keeps the
    existing value."""
    retriever = ShiftingRetriever()
    answer = run_agent(
        build_graph(retriever, Settings(max_retries=2), FakeGrader([False])), "a question"
    )
    assert len(retriever.queries) == 3
    assert answer.first_retrieved_ids == [c.chunk_id for c in CHUNKS]


def test_the_pipeline_path_reports_no_loop():
    """`attempts=0` on the pipeline is the true value, not a missing one."""
    from arxiv_rag.retrieval.answer import Answer

    assert Answer(question="q", text="t").attempts == 0
