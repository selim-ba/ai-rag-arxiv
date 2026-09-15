"""Latency summaries.

Reranking buys accuracy with milliseconds. You cannot report a trade-off you never
measured, so every row of the Stage 3 results table carries a latency column - including
the baseline row, which is why this lands before BM25 does.

**Why percentiles and not a mean.** Latency is right-skewed: most calls are fast, a few
are much slower (a cold cache, a slow API response, a garbage collection pause). A mean is
dragged upward by those outliers while hiding how bad they are, and nobody experiences the
mean. p50 is the typical request; p95 is the tail that generates complaints.

**Honest caveat at this scale.** Nearest-rank p95 of 40 samples is the 38th value, so
only two measurements sit above it. It is worth reporting and not worth over-reading: a
20ms move in p95 across runs is noise, and a single slow query may not move it at all.
That is why ``summarise`` also reports max - at this sample size, max is the honest
companion to p95.
"""

from math import ceil


def percentile(values: list[float], p: float) -> float:
    """The value below which ``p`` percent of ``values`` fall. ``p`` is 0-100.

    Use the **nearest-rank** method: sort, then take the element at index
    ``ceil(p/100 * n) - 1``, clamped into range. No interpolation.

    Interpolating between neighbours (what ``numpy.percentile`` does by default) invents a
    number that was never measured. For a latency report you want a real observation: "5%
    of queries took at least this long" is a true statement about a request that actually
    happened, and 47.3ms averaged from 44ms and 51ms is not.

    ``percentile(values, 50)`` is the median. Return 0.0 for an empty list rather than
    raising - a run with nothing to time is not an error.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    # ceil(0) - 1 == -1, and ordered[-1] is the LARGEST element - so p=0 would report the
    # slowest query as the fastest. Clamping both ends is the whole subtlety here.
    index = ceil(p / 100 * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def summarise(values: list[float]) -> dict[str, float]:
    """p50, p95 and max, rounded to a tenth of a millisecond.

    ``max`` is included because with n=40 it is the honest companion to p95: if they are
    far apart, one query is pathological and worth looking at by name rather than
    averaging away.
    """
    return {
        "p50": round(percentile(values, 50), 1),
        "p95": round(percentile(values, 95), 1),
        "max": round(max(values), 1) if values else 0.0,
    }
