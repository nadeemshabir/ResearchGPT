"""Typed exceptions for ResearchGPT.

Boundaries (PDF parsing, the vector store, LLM providers) translate whatever
their underlying library raises into one of these, so callers can handle
failures by category instead of string-matching on messages.
"""

from __future__ import annotations


class ResearchGPTError(Exception):
    """Base class for every error this project raises deliberately."""


class ConfigurationError(ResearchGPTError):
    """Settings are missing or mutually inconsistent."""


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class IngestionError(ResearchGPTError):
    """A document could not be taken from file to indexed chunks."""


class PDFParseError(IngestionError):
    """A PDF could not be read."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path


class EncryptedPDFError(PDFParseError):
    """The PDF is password protected."""


class CorruptPDFError(PDFParseError):
    """The file is not a readable PDF."""


class EmptyPDFError(PDFParseError):
    """The PDF parsed successfully but carries no extractable text layer.

    Usually a scan or an image-only export; it needs OCR, which this pipeline
    does not perform.
    """


class PDFTooLargeError(PDFParseError):
    """The file exceeds ``Settings.max_pdf_size_mb``."""


class ChunkingError(IngestionError):
    """Text could not be split into usable chunks."""


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class VectorStoreError(ResearchGPTError):
    """The vector database rejected an operation."""


class EmbeddingDimensionMismatchError(VectorStoreError):
    """The active embedding model disagrees with the stored collection.

    Chroma reports this as an opaque dimensionality error at query time; we
    detect it up front so the fix (re-ingest, or switch back to the original
    model) is obvious.
    """

    def __init__(self, expected: int, actual: int, collection_model: str, active_model: str) -> None:
        super().__init__(
            f"Collection was built with '{collection_model}' ({expected}-dim) but the "
            f"active embedding model is '{active_model}' ({actual}-dim). "
            f"Either set EMBEDDING_MODEL={collection_model} or re-ingest your papers "
            f"into a fresh collection."
        )
        self.expected = expected
        self.actual = actual
        self.collection_model = collection_model
        self.active_model = active_model


class EmptyCollectionError(VectorStoreError):
    """A search ran against a collection with no documents in it."""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class RetrievalError(ResearchGPTError):
    """Retrieval failed."""


class NoRelevantContextError(RetrievalError):
    """Nothing in the corpus cleared the relevance threshold for this query.

    Raised rather than returning empty context, because an LLM handed an empty
    context window will answer from parametric memory — which is exactly the
    hallucination this system exists to avoid.
    """

    def __init__(self, query: str, *, candidates_considered: int = 0, threshold: float | None = None) -> None:
        detail = f" (considered {candidates_considered} candidates"
        if threshold is not None:
            detail += f", threshold {threshold}"
        detail += ")"
        super().__init__(f"No indexed content is relevant to {query!r}{detail}.")
        self.query = query
        self.candidates_considered = candidates_considered
        self.threshold = threshold


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------
class LLMError(ResearchGPTError):
    """Base class for LLM provider failures."""

    def __init__(self, message: str, *, provider: str | None = None, model: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class LLMProviderError(LLMError):
    """The provider returned an error that retrying will not fix."""


class LLMAuthenticationError(LLMError):
    """The API key is missing, malformed, or rejected."""


class LLMRateLimitError(LLMError):
    """The provider rate-limited the request. Retryable."""


class LLMTimeoutError(LLMError):
    """The request exceeded ``Settings.llm_timeout_seconds``. Retryable."""


class LLMResponseError(LLMError):
    """The provider returned a malformed or empty response."""


#: Failures worth retrying with backoff.
RETRYABLE_LLM_ERRORS: tuple[type[Exception], ...] = (LLMRateLimitError, LLMTimeoutError)
