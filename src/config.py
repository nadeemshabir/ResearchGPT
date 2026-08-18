"""Centralised configuration for ResearchGPT.

Every tunable parameter in the system is declared here and nowhere else.
Values resolve in this order: explicit constructor argument > environment
variable / ``.env`` entry > the default declared below.

This module deliberately owns all magic numbers so that evaluation sweeps
(chunk size, retrieval weights, top-k) can vary them from a single place.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLMProvider = Literal["groq", "openai", "gemini"]
PDFMethod = Literal["pymupdf", "pdfplumber", "pypdf2"]
CitationStyle = Literal["inline", "numbered", "apa"]
FusionMethod = Literal["weighted", "rrf"]
NormalisationMethod = Literal["minmax", "sum", "none"]

#: Model used when the caller does not name one explicitly.
DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    # gemini-1.5-* were retired and now return 404 from the API. Verified
    # working against a live key on 2026-08-05.
    "gemini": "gemini-2.5-flash",
}

#: Environment variable holding the credential for each provider.
API_KEY_ENV_VARS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _clean_env_values(cls, values: Any) -> Any:
        """Strip whitespace and inline comments from environment values.

        ``python-dotenv`` and Docker's ``--env-file`` do not parse the same
        file the same way. Given::

            EMBEDDING_MODEL= all-MiniLM-L6-v2 # the default

        dotenv yields ``all-MiniLM-L6-v2``; Docker passes the whole string,
        spaces, comment and all. That produced a container which loaded
        cleanly on a laptop and died on startup with::

            Repo id must use alphanumeric chars ...:
            'sentence-transformers/ all-MiniLM-L6-v2 # '

        Config that is valid in one runner and fatal in another is a trap, and
        a leading space in a model name is never intentional. Values are
        cleaned here so both parsers agree.

        A ``#`` is only treated as a comment when whitespace precedes it, so a
        value that legitimately contains one (an API key, a URL fragment) is
        left alone.
        """
        if not isinstance(values, dict):
            return values

        cleaned: dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, str):
                value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            cleaned[key] = value
        return cleaned

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------
    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    # GEMINI_API_KEY is the documented name; GOOGLE_API_KEY is accepted as a
    # fallback because earlier versions of this project read that instead.
    gemini_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    llm_provider: LLMProvider = "groq"
    llm_model: str | None = None
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2000, gt=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=4, ge=0, le=10)
    llm_retry_initial_backoff: float = Field(default=1.0, gt=0)
    llm_retry_max_backoff: float = Field(default=30.0, gt=0)

    #: Per-stage temperatures for the multi-agent pipeline. Extraction and
    #: citation must stay near-deterministic; synthesis is allowed some latitude.
    analyzer_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    synthesizer_temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    citation_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    critic_temperature: float = Field(default=0.4, ge=0.0, le=2.0)

    use_multi_agent: bool = True
    use_citations: bool = True
    use_smart_routing: bool = True
    citation_style: CitationStyle = "inline"

    #: Attach citations in code rather than asking the model to write them.
    #: Sources come from chunk metadata, so a citation cannot name a paper that
    #: was never retrieved. Measured: the LLM-written path cited only 30% of
    #: answers -- see docs/EVALUATION.md.
    use_deterministic_citations: bool = True
    #: Vocabulary overlap a paragraph must share with a chunk before that
    #: chunk is cited. Below this the paragraph gets no citation rather than a
    #: guessed one.
    citation_min_similarity: float = Field(default=0.18, ge=0.0, le=1.0)
    #: Cap on sources listed after one paragraph. Beyond three the citation
    #: group is longer than the sentence it supports.
    citation_max_per_paragraph: int = Field(default=3, ge=1, le=10)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    pdf_method: PDFMethod = "pymupdf"
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    tokenizer_encoding: str = "cl100k_base"

    #: Reject uploads larger than this before spending time parsing them.
    max_pdf_size_mb: float = Field(default=50.0, gt=0)
    #: A PDF yielding fewer characters than this is almost certainly a scan
    #: with no text layer; we fail loudly rather than index an empty document.
    min_extracted_chars: int = Field(default=200, ge=0)

    # ------------------------------------------------------------------
    # Embeddings and vector store
    # ------------------------------------------------------------------
    # 384-dim, fast, and what the shipped ChromaDB collection was built with.
    # 'allenai/specter' (768-dim, science-tuned) is the main alternative, but
    # switching requires re-ingesting: see EmbeddingDimensionMismatchError.
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, gt=0)
    #: ``None`` means auto-detect (cuda > mps > cpu).
    embedding_device: str | None = None

    chroma_db_path: Path = PROJECT_ROOT / "data" / "chroma_db"
    collection_name: str = "research_papers"

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    # Tuned on BEIR/SciFact (300 queries), not inherited from a tutorial.
    # A sweep from 0.0 to 1.0 peaked at 0.5/0.5 (nDCG@10 0.6972); the previous
    # 0.7/0.3 default scored 0.6873. See docs/EVALUATION.md.
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    #: How dense and BM25 result lists are merged. "weighted" combines
    #: normalised scores; "rrf" combines rank positions and ignores score
    #: magnitude entirely. Neither is benchmarked yet.
    fusion_method: FusionMethod = "weighted"
    #: How component scores are rescaled before weighted fusion. Ignored by RRF.
    normalisation: NormalisationMethod = "minmax"
    #: RRF damping constant. The standard value from Cormack et al. (2009).
    #: Larger flattens the contribution of top ranks; smaller sharpens it.
    #:
    #: Left at the literature standard deliberately. A sweep showed k<20 beats
    #: k=60 by ~1 nDCG point on SciFact, but the curve was non-monotonic
    #: (k=100 > k=60), implying ~±0.5 points of noise -- not enough to justify
    #: shipping a corpus-specific value as a general default. Lower it if you
    #: enable RRF and can measure on your own data.
    rrf_k: int = Field(default=60, gt=0)

    #: Candidates pulled from each searcher before fusion and reranking.
    top_k_retrieve: int = Field(default=50, gt=0)
    #: Results surfaced to the generator after reranking.
    top_k_rerank: int = Field(default=10, gt=0)

    # Off by default on measured evidence: on BEIR/SciFact the cross-encoder
    # cost 28x the latency (179ms -> 4998ms p95) to move nDCG@10 by +0.75
    # points while *lowering* Recall@5 by 1.1. Tuning semantic_weight achieved
    # more, for free. Enable it for corpora where it demonstrably helps, and
    # prefer a domain-matched cross-encoder. See docs/EVALUATION.md.
    use_reranking: bool = False
    use_query_processing: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str | None = None

    #: Cross-encoder logits are unbounded (roughly -11..+11 for ms-marco), so
    #: the floor differs by scoring mode. Anything below is treated as noise.
    min_rerank_score: float = -5.0
    #: Hybrid scores are min-max normalised to 0..1.
    #:
    #: This orders results but CANNOT judge whether anything is relevant at
    #: all: min-max rescales within the retrieved set, so the best of a bad
    #: lot always scores near 1.0. Use min_semantic_similarity for that.
    min_hybrid_score: float = Field(default=0.2, ge=0.0, le=1.0)

    #: Absolute relevance floor: raw cosine similarity between query and chunk,
    #: before any normalisation. Unlike the fused score this is comparable
    #: across queries, so it can answer "is anything here relevant?".
    #:
    #: Observed on this corpus: an on-topic query's best chunk scores ~0.49,
    #: an off-topic query's ~0.35. 0.40 sits between them.
    #:
    #: This threshold is under-measured -- it rests on a handful of probes, not
    #: a benchmark. Calibrating it is exactly what the unanswerable-question
    #: set in Milestone 2b is for. Raise it to refuse more readily, lower it to
    #: answer more readily.
    min_semantic_similarity: float = Field(default=0.40, ge=0.0, le=1.0)
    #: BM25 scores below this are dropped as non-matches.
    min_bm25_score: float = Field(default=0.01, ge=0.0)

    max_context_tokens: int = Field(default=4000, gt=0)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["plain", "json"] = "plain"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = str(value).upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}, got {value!r}")
        return level

    @model_validator(mode="after")
    def _check_invariants(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise chunking cannot advance."
            )
        if self.semantic_weight + self.keyword_weight <= 0:
            raise ValueError(
                "semantic_weight and keyword_weight cannot both be zero; "
                "at least one retrieval signal must contribute."
            )
        if self.top_k_rerank > self.top_k_retrieve:
            raise ValueError(
                f"top_k_rerank ({self.top_k_rerank}) cannot exceed top_k_retrieve "
                f"({self.top_k_retrieve}); you cannot rerank more candidates than you fetch."
            )
        return self

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_llm_model(self) -> str:
        """The model to use, falling back to the provider's default."""
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]

    def api_key_for(self, provider: str) -> str | None:
        """Return the plaintext API key for ``provider``, or ``None``.

        Gemini accepts either ``GEMINI_API_KEY`` (documented) or the legacy
        ``GOOGLE_API_KEY`` that earlier versions of this project used.
        """
        secret: SecretStr | None
        if provider == "groq":
            secret = self.groq_api_key
        elif provider == "openai":
            secret = self.openai_api_key
        elif provider == "gemini":
            secret = self.gemini_api_key or self.google_api_key
        else:
            return None
        return secret.get_secret_value() if secret else None

    def normalised_weights(self) -> tuple[float, float]:
        """Retrieval weights rescaled to sum to 1.0."""
        total = self.semantic_weight + self.keyword_weight
        return self.semantic_weight / total, self.keyword_weight / total


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that ``.env`` is parsed once. Call :func:`reload_settings` in
    tests that need to vary the environment.
    """
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read configuration. Intended for tests."""
    get_settings.cache_clear()
    return get_settings()
