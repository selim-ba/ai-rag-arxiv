"""Retrieval metrics.

Pure functions: lists in, numbers out. No network, no models, no state — which is why
they are the easiest thing in the project to test and the hardest thing to argue with.

**Gold chunks are grouped, and the grouping carries meaning.** ``gold_groups`` is a list
of lists, read as *one chunk from each group*::

    [["1912.01603::7"], ["2010.02193::8"], ["2301.04104::5"]]   three facts, all needed
    [["1811.04551::4", "2107.08241::17"]]                       one fact, two sources

The second case is why the grouping exists. Papers cite their predecessors and surveys
describe everything, so the same fact usually lives in several chunks. "Does PlaNet train
a policy network?" is answered by the PlaNet paper *and* by the model-based RL survey; a
retriever that finds either one has done its job.

The flat list this replaced could not say that. Adding the survey chunk to a flat gold
list would have grown the denominator of ``recall_at_k`` from 1 to 2, so retrieving one
valid source out of two scored 0.5 — improving the labels would have degraded the metric.
Groups fix that: recall counts *groups satisfied*, so alternatives cost nothing and
genuinely complementary chunks still each have to be found.
"""


def flatten(gold_groups: list[list[str]]) -> list[str]:
    """Every acceptable gold chunk, ignoring the grouping."""
    return [chunk_id for group in gold_groups for chunk_id in group]


def hit_at_k(retrieved_ids: list[str], gold_groups: list[list[str]], k: int) -> bool:
    """Did *any* acceptable gold chunk make the top k?

    Grouping does not matter here — one relevant chunk anywhere in the top k is a hit.
    It is the metric that matters most in practice: the generator can ignore an
    irrelevant chunk, but it cannot invent one you failed to retrieve.
    """
    gold = set(flatten(gold_groups))
    return any(chunk_id in gold for chunk_id in retrieved_ids[:k])


def recall_at_k(retrieved_ids: list[str], gold_groups: list[list[str]], k: int) -> float:
    """What fraction of the gold *groups* were satisfied within the top k?

    A group is satisfied when at least one of its chunks was retrieved. So a question
    needing three separate facts scores 1/3 when one is found, and a question with two
    interchangeable sources scores 1.0 when either is found.

    This differs from ``hit_at_k`` whenever a question needs more than one passage —
    exactly the comparison questions in the eval set. Report both, and know which one
    you are quoting.

    Returns 0.0 for no groups, rather than dividing by zero.
    """
    if not gold_groups:
        return 0.0
    top = set(retrieved_ids[:k])
    satisfied = sum(1 for group in gold_groups if top & set(group))
    return satisfied / len(gold_groups)


def reciprocal_rank(retrieved_ids: list[str], gold_groups: list[list[str]]) -> float:
    """1 / (rank of the first acceptable gold chunk), or 0.0 if none was retrieved.

    Ranks are 1-based: a gold chunk in first place scores 1.0, second 0.5, third 0.333.
    Averaged over questions this is MRR, and it is what tells you whether a reranker has
    anything to work with — "the right chunk was there, but ranked fifth" is invisible to
    hit@5 and obvious in MRR.

    Note this is MRR over whatever list you pass in. Passing only the top 5 makes it
    MRR@5, and a gold chunk at rank 40 scores 0 rather than 0.025.
    """
    gold = set(flatten(gold_groups))
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    """Average, with an empty list giving 0.0 rather than an exception."""
    return sum(values) / len(values) if values else 0.0
