"""Retrieval metrics.

Pure functions: lists in, numbers out. No network, no models, no state — which is why
they are the easiest thing in the project to test and the hardest thing to argue with.

"""


def hit_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> bool:
    """Did *any* gold chunk make the top k?

    It is the one that matters most in practice: the generator can ignore an irrelevant
    chunk, but it cannot invent one you failed to retrieve.
    """
    for ids in retrieved_ids[:k]:
        if ids in gold_ids:
            return True
    return False


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """What fraction of the gold chunks made the top k?

    This is recall in the strict sense, and it differs from ``hit_at_k`` whenever a
    question needs more than one passage to answer — exactly the comparison questions
    your eval set contains. Report both, and know which one you are quoting.

    Returns 0.0 when there are no gold ids, rather than dividing by zero.
    """
    if hit_at_k(retrieved_ids, gold_ids, k):
        fraction = len(set(retrieved_ids[:k]) & set(gold_ids)) / len(gold_ids)
        return fraction
    else:
        return 0.0


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """1 / (rank of the first gold chunk), or 0.0 if none was retrieved.


    Ranks are 1-based: a gold chunk in first place scores 1.0, second place 0.5, third
    0.333. Averaged over questions this is MRR, and it is what tells you whether a
    reranker has anything to work with — "the right chunk was there, but ranked fifth"
    is invisible to hit@5 and obvious in MRR.
    """
    for i, id in enumerate(retrieved_ids, start=1):
        if id in gold_ids:
            return 1.0 / i
    return 0.0


def mean(values: list[float]) -> float:
    """Average, with an empty list giving 0.0 rather than an exception. Given to you."""
    return sum(values) / len(values) if values else 0.0
