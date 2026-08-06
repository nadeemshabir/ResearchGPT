"""Shared, expensive singletons.

The embedding model takes about six seconds to load and the vector store holds
an open handle. Building either per request would dominate a 2.7-second answer,
so both are constructed once at startup and injected.

Everything here is lazily built and cached, so importing the module costs
nothing -- which is what lets the test suite import the app without loading a
model.
"""

from __future__ import annotations

from functools import lru_cache

from api.jobs import JobRegistry
from src.generation.answer_generator import AnswerGenerator
from src.ingestion.database import VectorDatabase
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.retrieval_system import RetrievalSystem
from src.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_database() -> VectorDatabase:
    return VectorDatabase()


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingGenerator:
    return EmbeddingGenerator()


@lru_cache(maxsize=1)
def get_retrieval_system() -> RetrievalSystem:
    """Retrieval sharing the process-wide embedder and store."""
    return RetrievalSystem(embedder=get_embedder(), database=get_database())


@lru_cache(maxsize=2)
def get_generator(multi_agent: bool | None = None) -> AnswerGenerator:
    """An answer generator wired to the shared retrieval stack.

    Two are cached, one per pipeline mode. `answer_question` takes no
    per-call mode, so switching would otherwise mean mutating a shared object
    and leaking one request's configuration into the next. Constructing a
    second generator is cheap because retrieval and the LLM client are
    injected, not rebuilt.

    Smart routing is off: routing was never covered by the evaluation, so the
    API exposes the measured path by default.
    """
    return AnswerGenerator(
        use_multi_agent=multi_agent,
        use_smart_routing=False,
        retrieval_system=get_retrieval_system(),
    )


@lru_cache(maxsize=1)
def get_pipeline() -> IngestionPipeline:
    """Ingestion sharing the same embedder and store as retrieval.

    Sharing matters: two EmbeddingGenerator instances would load the model
    twice, and a second VectorDatabase handle on the same path invites
    write conflicts.
    """
    return IngestionPipeline(embedder=get_embedder(), database=get_database())


@lru_cache(maxsize=1)
def get_jobs() -> JobRegistry:
    return JobRegistry()


def warm_up() -> None:
    """Build everything now, so the first request is not the slow one."""
    logger.info("Warming up: loading embedding model and opening the store")
    get_generator()
    get_pipeline()
    logger.info("Warm-up complete")


def reset_caches() -> None:
    """Drop every singleton. For tests only."""
    for cached in (
        get_database,
        get_embedder,
        get_retrieval_system,
        get_generator,
        get_pipeline,
        get_jobs,
    ):
        cached.cache_clear()
