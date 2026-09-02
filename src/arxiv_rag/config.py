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

    data_dir: Path = Path("data")

    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0)

    arxiv_delay_seconds: float = 3.0
    log_level: str = "INFO"

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    def ensure_dirs(self) -> None:
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. FastAPI dependency-injects this; scripts just call it."""
    return Settings()
