"""What a call costs, in dollars, so the budget can be spent in the unit that matters.

**A snapshot, not a source of truth.** Prices change and this table does not notice. It
carries the date it was read and the page it came from, so the next person can check it in
thirty seconds instead of wondering. Everything derived from it - the cost on each log
line, the daily budget - is only as current as this dict.

Read 2026-09-18 from https://platform.openai.com/docs/pricing, in USD per 1M tokens.
"""

import logging

log = logging.getLogger(__name__)

PRICES: dict[str, tuple[float, float]] = {
    # model: (input per 1M tokens, output per 1M tokens)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    # Embeddings have no output side; the second number is there to keep one shape.
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

# What an unknown model is charged at. The most expensive row, deliberately: a budget that
# under-counts is a budget that does not exist, and the failure of a wrong guess should be
# "the demo stopped early", never "the bill was larger than the cap".
_FALLBACK = max(PRICES.values())

_warned: set[str] = set()


def rates(model: str) -> tuple[float, float]:
    """Input and output price per 1M tokens for a model, or the safest guess."""
    if model in PRICES:
        return PRICES[model]
    if model not in _warned:  # once per model per process, not once per request
        _warned.add(model)
        log.warning(
            "no price for %r; charging it at %s per 1M to stay conservative", model, _FALLBACK
        )
    return _FALLBACK


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Dollars for one call. Small numbers by design - a single `/ask` is about $0.0005."""
    price_in, price_out = rates(model)
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
