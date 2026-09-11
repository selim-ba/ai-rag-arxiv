"""Stage 3 - latency percentiles. Nearest-rank, so every answer is a real measurement."""

from arxiv_rag.evaluation.timing import percentile, summarise

# 1..10, so the expected index is easy to check by hand
TEN = [float(i) for i in range(1, 11)]


def test_median_of_ten():
    """ceil(0.50 * 10) = 5 -> index 4 -> the 5th value."""
    assert percentile(TEN, 50) == 5.0


def test_p95_of_ten():
    """ceil(0.95 * 10) = 10 -> index 9 -> the largest value."""
    assert percentile(TEN, 95) == 10.0


def test_p90_of_ten():
    assert percentile(TEN, 90) == 9.0


def test_order_does_not_matter():
    assert percentile([9.0, 1.0, 5.0, 3.0, 7.0], 50) == 5.0


def test_p0_is_the_minimum():
    """ceil(0) = 0 -> index -1, which must clamp to 0, not wrap to the end."""
    assert percentile(TEN, 0) == 1.0


def test_p100_is_the_maximum():
    assert percentile(TEN, 100) == 10.0


def test_single_value():
    assert percentile([42.0], 95) == 42.0


def test_empty_returns_zero_not_an_exception():
    assert percentile([], 50) == 0.0


def test_result_is_always_a_real_measurement():
    """No interpolation: a latency report should quote a request that actually happened."""
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 75) in values
    assert percentile(values, 60) in values


def test_outlier_moves_p95_but_not_p50():
    """The point of percentiles over a mean: p50 ignores the tail, p95 shows it."""
    ten = [10.0] * 9 + [1000.0]  # ceil(0.95 * 10) = 10 -> the outlier is the p95
    assert percentile(ten, 50) == 10.0
    assert percentile(ten, 95) == 1000.0


def test_one_outlier_in_twenty_does_not_reach_p95():
    """Sample size decides what p95 can see, and it is worth knowing where the line is.

    ceil(0.95 * 20) = 19, so p95 is the 19th of 20 values and a single slow query sits
    above it, invisible. At n=40 (the eval set) p95 is the 38th, so the two slowest
    queries are above it. That is why `summarise` also reports max: with samples this
    small, max is the honest companion to p95.
    """
    twenty = [10.0] * 19 + [1000.0]
    assert percentile(twenty, 95) == 10.0
    assert max(twenty) == 1000.0


def test_summarise_reports_the_three_numbers():
    s = summarise([10.0, 20.0, 30.0, 1000.0])
    assert s["p50"] == 20.0
    assert s["max"] == 1000.0
    assert set(s) == {"p50", "p95", "max"}


def test_summarise_of_empty_does_not_raise():
    assert summarise([]) == {"p50": 0.0, "p95": 0.0, "max": 0.0}


def test_p_is_a_percentage_not_a_fraction():
    """The footgun: passing 0.95 meaning "95%" silently reports the fastest query.

    ceil(0.0095 * 10) = 1 -> index 0 -> the minimum. No error, just a latency report
    claiming the tail is as fast as the best case.
    """
    assert percentile(TEN, 95) == 10.0
    assert percentile(TEN, 0.95) == 1.0
