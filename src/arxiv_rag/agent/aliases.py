"""The names people use for these papers, which are not the names on the papers.

**Measured, and the reason this file exists.** The router routes by matching a question
against a catalog of arXiv ids and titles. With titles alone it scored route accuracy
0.833 and *filtered-with-the-right-ids 0 out of 4*: right route, wrong paper, every time.

    question says      title in the index
    -------------      ------------------------------------------------------------
    V-JEPA             Revisiting Feature Prediction for Learning Visual ... Video
    I-JEPA             Self-Supervised Learning from Images with a Joint-Embedding ...
    IRIS               Transformers are Sample-Efficient World Models

None of those titles contain the name the question uses. The router did the only thing
available and matched on surface similarity: "V-JEPA" went to the one title that does
contain that string - *V-JEPA 2* - and "IRIS" matched nothing, so it reported that IRIS is
not indexed. It is. The list just did not call it that.

**This is a specification, not derived data, and it cannot be derived.** Two attempts
failed for instructive reasons:

* *From the abstract*: the top capitalised candidates for V-JEPA are CNRS, PSL and LIGM -
  author affiliations - and for Genie it is Parker-Holder, an author surname.
* *By frequency*: the names nest. "V-JEPA" occurs 181 times in the **V-JEPA 2** paper
  because "V-JEPA 2" contains it; "Dreamer" occurs 115 times in **ARB4WM**, which
  benchmarks Dreamer, R2-Dreamer and Dreamer-Pro; and "DreamerV3" occurs **once** in its
  own paper, which calls itself Dreamer. Counting cannot tell an introduction from a
  citation when one name is a substring of another.

So a person decides, and `scripts/route_eval.py` measures whether the decision works -
the same arrangement as `eval/routes.jsonl`.

**Conservative on purpose.** A missing alias degrades to matching on titles, which is
today's behaviour. A wrong alias sends a confident search to the wrong paper, which is
worse than not filtering at all. Papers whose title already contains their common name are
listed anyway: it costs nothing and makes the match exact rather than fuzzy.
"""

# arXiv id -> the names a person might use, most common first.
PAPER_ALIASES: dict[str, tuple[str, ...]] = {
    # -- the ones the router got wrong, which is what this file is for
    "1811.04551": ("PlaNet",),
    "1912.01603": ("Dreamer", "DreamerV1", "Dream to Control"),
    "2010.02193": ("DreamerV2",),
    "2301.04104": ("DreamerV3",),
    "2301.08243": ("I-JEPA",),
    "2404.08471": ("V-JEPA",),
    "2506.09985": ("V-JEPA 2",),
    "2209.00588": ("IRIS",),
    "1803.10122": ("World Models (Ha and Schmidhuber)",),
    # -- title already carries the name; listed so the match is exact, not fuzzy
    "2402.15391": ("Genie",),
    "2605.09241": ("Sub-JEPA",),
    "2603.20111": ("Var-JEPA",),
    "2311.03622": ("TWIST",),
    "2410.08822": ("SOLD",),
    "2605.13013": ("JEDI",),
    "2608.05720": ("PhyLatent",),
    "2606.16605": ("ARB4WM",),
    "2605.25313": ("UWM-JEPA",),
    "2606.31232": ("Delta-JEPA",),
    "2608.29029": ("Flow-JEPA",),
    "2607.05238": ("Branch-JEPA",),
    "2607.04044": ("SiamJEPA",),
    "2608.17542": ("AC-MTM",),
    # -- named in their own text but not in their title
    "2210.11698": ("VSG",),
    "2303.14889": ("Iso-Dream",),
    "2305.18499": ("ContextWM",),
    "2106.13229": ("LatCo",),
}


def aliases_for(arxiv_id: str) -> tuple[str, ...]:
    """Known names for one paper, or an empty tuple. Missing is a safe default."""
    return PAPER_ALIASES.get(arxiv_id, ())
