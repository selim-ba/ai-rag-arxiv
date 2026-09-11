"""Is a quoted span actually present in the text it was attributed to?

Written because the gold-label auditor fabricated quotes. Asked which chunks supported a
question, it returned confident, on-topic, verbatim-looking evidence - and three of those
quotes did not appear anywhere in the chunk they named. ``"the world model is fixed while
learning behaviors"`` is exactly what the answer needed; nobody wrote it.

This is the same failure as the fabricated arXiv citations, and it takes the same fix:
the model decides *which* passage, and code verifies the claim against the source it
already holds. A model is a good proposer and an unreliable reporter.

Matching cannot be exact string containment. Extracted PDF text carries ligatures
(``ﬁ`` for ``fi``), hard line breaks mid-sentence, and doubled spaces, and a model
quoting from it normalises all of that on the way out. So both sides are normalised, and
a near-match is accepted, while an invention is not.
"""

import re
import unicodedata
from difflib import SequenceMatcher

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "–": "-", "—": "-"}

MIN_WORDS = 4  # below this a "quote" is too short to confirm anything
MIN_RATIO = 0.75  # share of the quote that must be found as a contiguous run


def normalise(text: str) -> str:
    """Lowercase, fix ligatures, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_supported(quote: str, text: str) -> bool:
    """Does ``quote`` appear in ``text``, allowing for extraction noise?

    An empty or very short quote is not support: a model that cannot point at four
    consecutive words has not shown you anything.
    """
    needle, haystack = normalise(quote), normalise(text)
    if len(needle.split()) < MIN_WORDS:
        return False
    if needle in haystack:
        return True
    # Longest contiguous run of the quote that exists in the text. A real quote with a
    # dropped word still scores high; an invented sentence built from the chunk's
    # vocabulary does not, because its word ORDER was never in the source.
    match = SequenceMatcher(None, needle, haystack, autojunk=False).find_longest_match(
        0, len(needle), 0, len(haystack)
    )
    return match.size / len(needle) >= MIN_RATIO
