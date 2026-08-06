"""Request and response models.

These are the API's contract, kept separate from the internal dicts the pipeline
passes around. A response model that mirrors an internal dict would make every
refactor a breaking API change.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobState(str, Enum):
    """Lifecycle of an ingestion job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# --- query ------------------------------------------------------------------


class QueryRequest(BaseModel):
    """A question to answer over the indexed corpus."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    multi_agent: bool | None = Field(
        default=None,
        description=(
            "Run the four-stage pipeline. Costs 4x the API calls and was not "
            "shown to be worth it; see docs/EVALUATION.md. Defaults to the "
            "server's configuration."
        ),
    )


class Source(BaseModel):
    """One paper that contributed to an answer."""

    paper_id: str
    title: str
    author: str = "Unknown"
    section: str = "Unknown"
    relevance_score: float = 0.0
    chunk_preview: str = ""


class CitedSource(BaseModel):
    """One chunk credited for a paragraph."""

    chunk_id: str
    paper_id: str
    title: str
    year: str
    section: str
    score: float = Field(
        description=(
            "Content-word overlap with the paragraph, 0-1. Exposed so a client "
            "can show how strong an attribution is rather than treating all "
            "attributions as equal."
        )
    )


class AttributedParagraph(BaseModel):
    """One paragraph of an answer and the chunks it draws on.

    ``sources`` is empty for a paragraph that matched nothing above the
    threshold, and for headings and lists (``structural``). Both are kept so a
    client can render the whole answer from this list alone.
    """

    text: str
    sources: list[CitedSource]
    structural: bool = False


class Chunk(BaseModel):
    """A retrieved passage, in full.

    The whole text is returned rather than a preview: a client highlighting the
    source behind a claim needs the passage a reader will actually read.
    """

    chunk_id: str
    paper_id: str
    title: str
    section: str
    text: str
    relevance_score: float


class QueryResponse(BaseModel):
    """An answer, its sources, and how it was produced."""

    question: str
    answer: str
    sources: list[Source]
    paragraphs: list[AttributedParagraph] = Field(
        default_factory=list,
        description=(
            "Paragraph-to-chunk links. Use these to highlight the passage "
            "behind a claim; parsing the citations back out of `answer` would "
            "be guesswork."
        ),
    )
    chunks: list[Chunk] = Field(
        default_factory=list,
        description="Every retrieved passage, keyed by the ids used in `paragraphs`.",
    )
    refused: bool = Field(
        description=(
            "True when the system declined to answer. A refusal is a correct "
            "outcome for a question the corpus cannot support, not an error."
        )
    )
    num_sources: int
    processing_time: float
    retrieval_time: float
    generation_time: float
    model: str
    generation_method: str
    citation_audit: dict[str, Any] | None = None


# --- papers -----------------------------------------------------------------


class Paper(BaseModel):
    """A paper in the store."""

    paper_id: str
    title: str
    author: str = "Unknown"
    num_chunks: int


class PaperList(BaseModel):
    papers: list[Paper]
    total_papers: int
    total_chunks: int


class IngestionJob(BaseModel):
    """Status of an upload.

    Ingestion takes seconds to minutes for a long paper, so the upload endpoint
    returns immediately with a job id and the work continues in the background.
    """

    job_id: str
    state: JobState
    filename: str
    paper_id: str | None = None
    detail: str | None = Field(
        default=None, description="Error message when the state is failed."
    )
    num_chunks: int | None = None
    num_sections: int | None = None
    num_pages: int | None = None
    seconds: float | None = None


class DeleteResponse(BaseModel):
    paper_id: str
    chunks_deleted: int


# --- health -----------------------------------------------------------------


class DependencyStatus(BaseModel):
    name: str
    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """Liveness plus a check of each dependency.

    ``status`` is ``degraded`` rather than ``unhealthy`` when the corpus is
    empty: the service is working correctly, it just has nothing to search.
    """

    status: str
    version: str
    dependencies: list[DependencyStatus]
    total_papers: int
    total_chunks: int


class ErrorResponse(BaseModel):
    """Uniform error body, so clients parse one shape."""

    error: str
    detail: str
