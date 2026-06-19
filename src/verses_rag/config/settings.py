"""
Config layer for the Scripture+Article RAG app (SPEC §4.1, D5).

One Settings object, env/.env-driven. Nested blocks are overridable with a
double-underscore delimiter, e.g.:

    CORPUS_DIR=/data/RAG
    BIBLE__WINDOW_SIZE=7
    BIBLE__OVERLAP=2
    RETRIEVAL__TOP_K=5

Requires Pydantic v2:  pip install pydantic-settings
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, model_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

openai_api_key: SecretStr | None = None
anthropic_api_key: SecretStr | None = None


class BibleChunkingSettings(BaseModel):
    """Maps 1:1 to BibleProcessor's constructor args."""

    chunk_strategy: Literal["window", "verse"] = "window"
    window_size: int = 5
    overlap: int = 1
    include_verse_numbers: bool = False
    status: str = "active"

    @model_validator(mode="after")
    def _check_overlap(self) -> BibleChunkingSettings:
        if self.chunk_strategy == "window" and not (
            0 <= self.overlap < self.window_size
        ):
            raise ValueError("require 0 <= overlap < window_size")
        return self


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection_name: str = "verses_rag"


class EmbeddingSettings(BaseModel):
    dense_model: str = "BAAI/bge-large-en-v1.5"
    dense_dim: int = 1024
    sparse_model: str = "Qdrant/bm25"


class RetrievalSettings(BaseModel):
    """§4.5 candidate-fetch / rerank knobs (placeholders until those layers land)."""

    bm25_k: int = 10
    dense_k: int = 10
    top_k: int = 3


class RerankSettings(BaseModel):
#     """§4.5 stage-3 knobs; consumed by Reranker.from_settings."""

     model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
     device: Optional[str] = None     # None -> auto (mps on Apple Silicon)
     top_k: int = 5                   # candidates surviving rerank into grading


class ArticleChunkingSettings(BaseModel):
    """Maps 1:1 to ArticleProcessor's constructor args."""

    chunk_size: int = 800
    chunk_overlap: int = 120
    prepend_title: bool = True
    extract_refs: bool = True
    status: str = "active"

    @model_validator(mode="after")
    def _check_overlap(self) -> ArticleChunkingSettings:
        if not (0 <= self.chunk_overlap < self.chunk_size):
            raise ValueError("require 0 <= chunk_overlap < chunk_size")
        return self


class JudgeSettings(BaseModel):
    """Primary (OpenAI) + backup (Anthropic) for all judgment-critical roles.
    
    Applies to: grade, verify, route, classify (§4.8.1).
    primary_model / backup_model should be set to whatever is current at
    build time — verify against provider docs (R7).
    """

    # --- primary: OpenAI ---
    primary_provider: Literal["openai"] = "openai"
    primary_model: str = "gpt-4o-mini"          # fast, cheap; swap gpt-4o for harder tasks

    # --- backup: Anthropic ---
    backup_provider: Literal["anthropic"] = "anthropic"
    backup_model: str = "claude-haiku-4-5-20251001"   # fast + capable fallback

    # --- reliability knobs ---
    timeout: float = 10.0                # seconds; request exceeding this raises -> triggers fallback
    degraded_latency_ms: float = 3000.0  # health-check threshold: warn if probe > this ms
    max_tokens: int = 512                # judge calls are short structured outputs


class GenerationSettings(BaseModel):
    """Local Ollama generation model (§4.8.1 — high volume, stays local)."""

    model: str = "qwen3:1.7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0    # deterministic for grounded generation
    max_tokens: int = 1024


class GradeSettings(BaseModel):
    """Relevance grading knobs (grade_documents node)."""
    score_threshold: float = -8.0
    # Rerank score below which we skip the LLM and grade insufficient directly.
    # -8.0 is a reasonable default for ms-marco-MiniLM logits; tune via eval (Phase 5).
    max_retries: int = 2
    # Max retrieve→grade loops before the graph forces abstention.    


class LangSmithSettings(BaseModel):
    enabled: bool = False
    project: str = "verses-rag"
    endpoint: str = "https://api.smith.langchain.com"
    api_key: SecretStr | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    corpus_dir: Path = Path("/Volumes/X10 Pro/RAG")
    kjv_filename: str = "kjv.json"
    articles_subdir: str = "articles"

    bible: BibleChunkingSettings = BibleChunkingSettings()
    article: ArticleChunkingSettings = ArticleChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    qdrant: QdrantSettings = QdrantSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    rerank: RerankSettings = RerankSettings()
    judge: JudgeSettings = JudgeSettings()
    generation: GenerationSettings = GenerationSettings()
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    grade: GradeSettings = GradeSettings()
    langsmith: LangSmithSettings = LangSmithSettings()

    @property
    def kjv_path(self) -> Path:
        return self.corpus_dir / self.kjv_filename

    @property
    def articles_dir(self) -> Path:
        return self.corpus_dir / self.articles_subdir


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the .env is read once per process."""
    return Settings()


if __name__ == "__main__":
    s = get_settings()
    print("Resolved settings:")
    print(f"  kjv_path     = {s.kjv_path}")
    print(f"  articles_dir = {s.articles_dir}")
    print(f"  bible      = {s.bible.model_dump()}")
    print(f"  article    = {s.article.model_dump()}")
    print(f"  retrieval  = {s.retrieval.model_dump()}")
    print(f"  grade      = {s.grade.model_dump()}")
    print(f"  langsmith  = {s.langsmith.model_dump()}")
    
