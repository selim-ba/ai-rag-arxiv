"""Stage 7 - what a call costs. No network, no model.

The budget is denominated in dollars, so these numbers decide when the demo stops
answering. That makes a wrong price a real failure rather than a cosmetic one.
"""

import logging

import pytest

from arxiv_rag.pricing import PRICES, cost_usd, rates


def test_a_typical_request_costs_about_half_a_tenth_of_a_cent():
    """3000 tokens of passages and prompt, 150 out, on gpt-4o-mini. The estimate that
    sized the daily budget - now checked against the table that will enforce it."""
    assert cost_usd("gpt-4o-mini", 3000, 150) == pytest.approx(0.00054, abs=1e-6)


def test_fifty_cents_buys_roughly_nine_hundred_questions():
    per_request = cost_usd("gpt-4o-mini", 3000, 150)
    assert 800 <= 0.50 / per_request <= 1000


def test_an_embedding_costs_almost_nothing():
    assert cost_usd("text-embedding-3-small", 20, 0) < 1e-6


def test_output_tokens_cost_more_than_input_ones():
    """Four times more on this model, which is why a verbose answer is not free."""
    assert cost_usd("gpt-4o-mini", 0, 1000) == pytest.approx(4 * cost_usd("gpt-4o-mini", 1000, 0))


def test_an_unknown_model_is_charged_at_the_most_expensive_rate(caplog):
    """Conservative on purpose. A budget that under-counts is a budget that does not
    exist, and the failure of a wrong guess should be "the demo stopped early", never
    "the bill was larger than the cap"."""
    with caplog.at_level(logging.WARNING):
        assert rates("gpt-6-does-not-exist") == max(PRICES.values())
    assert "no price" in caplog.text


def test_the_unknown_model_warning_is_not_repeated_per_request(caplog):
    """Once per model per process. A warning on every request is a log nobody reads.

    `caplog` collects for the whole test, not only inside `at_level`, so the first call's
    warning has to be cleared explicitly - otherwise this passes or fails for a reason
    that has nothing to do with the code."""
    with caplog.at_level(logging.WARNING):
        rates("another-unknown-model")
        assert "no price" in caplog.text

        caplog.clear()
        rates("another-unknown-model")
        assert caplog.text == ""
