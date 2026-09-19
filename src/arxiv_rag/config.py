"""Typed application configuration.

Everything configurable lives here, in one validated object. Nothing anywhere else in
the codebase should call ``os.getenv``.

Why this matters: if ``OPENAI_API_KEY`` is missing, you want to find out at startup with
a clear error naming the field — not twenty minutes into an ingestion run.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PT_",
        extra="ignore",
    )

    # This one keeps its conventional name, so it also picks up a key already exported
    # in your shell. validation_alias bypasses the PT_ prefix.
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Deliberately separate from llm_model. A judge sharing a family with the generator
    # tends to like its own phrasing, so you want to be able to swap in a stronger or
    # different model without touching generation. Same model for now, one env var away
    # from not being.
    judge_model: str = "gpt-4o-mini"

    # The grader drives control flow, so it is separable from both the generator and the
    # judge: it runs on every query (cost matters) and a wrong verdict costs a retry.
    grader_model: str = "gpt-4o-mini"

    # Separable for the same reason as the grader: the router runs on every request and a
    # wrong route costs a whole request, not a retry. Its job is a membership check over a
    # 49-line list, which is a different skill from generating or judging.
    router_model: str = "gpt-4o-mini"

    # Separable again: resolving a follow-up is a reference-tracking task over a short
    # transcript, and it runs on every turn after the first.
    followup_model: str = "gpt-4o-mini"

    # How many times the agent may rewrite the query and retrieve again before it gives
    # up and answers with whatever it has. The loop's hard cap, and the reason the agent
    # terminates even when the grader never changes its mind.
    #
    # Measured grader quality argues for a small number: catch rate 0.417, false-alarm
    # rate 0.091. At one retry a false alarm costs one extra retrieval and one extra
    # grader call on a question that was already fine; at five it costs five, and the
    # grader is not accurate enough to be worth paying that for.
    # ZERO, measured. Across two runs the retry loop rescued 1 question and lost 2. The
    # losses (q002, q034) are the grader's false alarms: it flagged retrievals that already
    # contained the gold chunk, the rewrite fired, and the second retrieval was worse. Its
    # false-alarm rate DOUBLED (0.091 -> 0.174) when retrieval improved - more good
    # retrievals means more chances to wrongly flag one.
    #
    # The loop stays wired and tested. Set this above 0 only together with a retry action
    # that is measured to help; rewriting is not it, because hybrid retrieval already
    # solved the vocabulary-mismatch failures a rewrite is for.
    max_retries: int = Field(default=0, ge=0, le=3)

    data_dir: Path = Path("data")

    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0)

    arxiv_delay_seconds: float = 3.0

    # Retrieval
    embedding_batch_size: int = Field(default=128, ge=1, le=2048)
    # The embedding cache is an optimisation, and in a container it is a write to a
    # filesystem that may be read-only and will not survive a restart either way. Turned
    # off in the image (`PT_EMBEDDING_CACHE_WRITES=false`); on everywhere else, where it
    # saves real money across eval runs.
    embedding_cache_writes: bool = True

    # Stage 7. What stops a public demo costing money. Measured: one `/ask` costs
    # $0.000517, so 30 cents is about 580 questions a day - generous for a portfolio link.
    # 0 or less disables the cap, for local runs.
    #
    # 30 rather than 50 because the two caps have to agree. The provider-side hard cap is
    # $10 a month; 50 cents a day spent every day is $15, so the app's own limit would stop
    # binding around the twentieth of the month and the provider's would take over. That is
    # the wrong failure: this one refuses with a defined 503 and a time to come back, the
    # provider's arrives as 429s in the middle of answers. $0.30 x 31 = $9.30, so the cap
    # that binds first is the one that explains itself.
    daily_budget_usd: float = 0.30
    # A visitor clicking three example questions is not abuse; a script asking a hundred
    # times a minute is. Burst covers the former, the rate bounds the latter.
    rate_limit_per_minute: float = 10.0
    rate_limit_burst: int = 5
    top_k: int = Field(default=5, ge=1, le=50)
    # How deep each retriever goes before reciprocal rank fusion. Measured across 34
    # questions the choice barely matters (hit@1 identical from 5 to 50), so this is a
    # reasonable default rather than a tuned one.
    fusion_depth: int = Field(default=20, ge=1, le=200)

    # The constant in RRF's `1/(k + rank)`. 60 comes from Cormack et al., SIGIR 2009,
    # where it was tuned for fusing dozens of TREC runs of comparable quality: a large k
    # flattens rank differences so that AGREEMENT between systems carries the signal.
    # With two retrievers of unequal quality that is the wrong prior - it lets a chunk
    # both retrievers ranked 8th beat one dense ranked 2nd. Measured in the Stage 4 grid.
    # Measured: 5. Any value in 0-10 gives identical hit@5 on both splits, and the
    # ordering of the tiny MRR differences between them INVERTS from dev to test - which
    # is what noise looks like. 5 is the middle of the agreeing range rather than an edge.
    # `fusion.DEFAULT_K` stays at 60: that is the literature's value and belongs to the
    # library. This is what this corpus measured, and belongs to the application.
    rrf_k: int = Field(default=5, ge=0, le=1000)

    # Per-retriever weights for fusion, as "dense,bm25". Empty means equal weights.
    # A string rather than a list because this arrives from the environment, and
    # PT_FUSION_WEIGHTS="1,0.3" is something you can type.
    # Measured on the dev split: "1,0.3" gains q003, q007 and q034 - three of the four
    # questions only BM25 finds - and loses NOTHING (+3/-0), landing one question short of
    # the 0.692 union ceiling. Equal weights score +4/-4: they gain all four and give four
    # back, which is why hybrid retrieval had been scoring the same as dense alone.
    #
    # The mechanism: RRF rewards agreement, so a chunk both retrievers ranked 8th outscores
    # one dense ranked 2nd. BM25 is the weaker retriever here (hit@5 0.471 vs 0.618) and
    # was voting at full strength.
    fusion_weights: str = "1,0.3"

    # Route POST /ask through the LangGraph agent instead of calling answer_question.
    # Off by default, deliberately: the agent and the pipeline are measured to produce
    # identical numbers with `max_retries=0`, so switching is a behaviour change nobody
    # can see in the eval set and a latency change every caller can. One env var
    # (PT_USE_AGENT=true) turns it on, and /ask reports which path served the request.
    use_agent: bool = False

    # The SDK's default is 600 seconds. One hung call would block a request for ten
    # minutes, and an agent request makes up to three calls. 30s is comfortably above the
    # measured generate p95 (1.9s) and the listwise reranker's p95 (7.8s), and far below
    # anything a caller would wait for.
    openai_timeout_seconds: float = Field(default=30.0, gt=0, le=600)

    # The SDK's own default, set explicitly so it is visible and tunable. Retries here are
    # exponential (0.5s -> 8s) and honour Retry-After; a hand-written loop on top would
    # multiply rather than replace them.
    openai_max_retries: int = Field(default=2, ge=0, le=5)

    log_level: str = "INFO"

    @property
    def fusion_weight_list(self) -> list[float] | None:
        """Parsed weights, or None for equal weighting.

        None and [1.0, 1.0] are the same fusion. None is returned for the empty setting so
        that "never configured" and "configured to be equal" stay distinguishable in a log
        line - the kind of distinction that took two debugging sessions to want.
        """
        if not self.fusion_weights.strip():
            return None
        return [float(part) for part in self.fusion_weights.split(",")]

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.data_dir / "embedding_cache"

    def ensure_dirs(self) -> None:
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. FastAPI dependency-injects this; scripts just call it."""
    return Settings()
