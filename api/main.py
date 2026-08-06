"""HTTP API for ResearchGPT.

    uvicorn api.main:app --reload

Endpoints
---------
``POST /query``          answer a question over the indexed corpus
``POST /query/stream``   the same, as server-sent events
``POST /papers``         upload a PDF; returns a job id
``GET  /papers``         list indexed papers
``GET  /papers/{id}``    one paper
``DELETE /papers/{id}``  remove a paper and its chunks
``GET  /jobs/{id}``      ingestion status
``GET  /health``         liveness and dependency checks

On error semantics: a **refusal is not an error**. When the corpus cannot
support a question the system says so, and that comes back as HTTP 200 with
``refused: true``. Mapping it to 4xx would make correct behaviour look like a
client mistake, and would hide the distinction from anything counting error
rates.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.dependencies import (
    get_database,
    get_generator,
    get_jobs,
    get_pipeline,
    warm_up,
)
from api.schemas import (
    DeleteResponse,
    DependencyStatus,
    HealthResponse,
    IngestionJob,
    JobState,
    Paper,
    PaperList,
    QueryRequest,
    QueryResponse,
)
from src.config import PROJECT_ROOT, get_settings
from src.exceptions import (
    IngestionError,
    LLMError,
    NoRelevantContextError,
    PDFParseError,
    ResearchGPTError,
)
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

VERSION = "0.2.0"

#: Rejected before anything is written to disk.
_ALLOWED_SUFFIXES = {".pdf"}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load models before the first request rather than during it."""
    setup_logging()
    await asyncio.to_thread(warm_up)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="ResearchGPT",
    version=VERSION,
    summary="Retrieval-augmented question answering over research papers",
    description=__doc__,
    lifespan=lifespan,
)


@app.exception_handler(ResearchGPTError)
async def handle_known_errors(_request: Request, exc: ResearchGPTError) -> JSONResponse:
    """Map the project's exception types onto status codes.

    Without this every failure is a 500 and clients cannot tell "your PDF is
    encrypted" from "the vector store is down".
    """
    if isinstance(exc, PDFParseError):
        status, kind = 422, "invalid_pdf"
    elif isinstance(exc, IngestionError):
        status, kind = 422, "ingestion_failed"
    elif isinstance(exc, LLMError):
        status, kind = 502, "llm_unavailable"
    else:
        status, kind = 500, "internal_error"

    logger.warning("%s: %s", kind, exc)
    return JSONResponse(status_code=status, content={"error": kind, "detail": str(exc)})


# --- query ------------------------------------------------------------------


def _to_response(question: str, result: dict[str, Any]) -> QueryResponse:
    metadata = result["metadata"]
    return QueryResponse(
        question=question,
        answer=result["answer"],
        sources=result.get("sources", []),
        paragraphs=result.get("paragraphs", []),
        chunks=result.get("chunks", []),
        refused=bool(metadata.get("refused", False)),
        num_sources=metadata.get("num_sources", 0),
        processing_time=metadata.get("processing_time", 0.0),
        retrieval_time=metadata.get("retrieval_time", 0.0),
        generation_time=metadata.get("generation_time", 0.0),
        model=metadata.get("model", "unknown"),
        generation_method=metadata.get("generation_method", "unknown"),
        citation_audit=metadata.get("citation_audit"),
    )


def _answer(request: QueryRequest) -> QueryResponse:
    """Answer a question, turning a refusal into a normal response."""
    generator = get_generator(request.multi_agent)

    try:
        result = generator.answer_question(request.question, top_k=request.top_k)
    except NoRelevantContextError as exc:
        return QueryResponse(
            question=request.question,
            answer=str(exc),
            sources=[],
            refused=True,
            num_sources=0,
            processing_time=0.0,
            retrieval_time=0.0,
            generation_time=0.0,
            model=generator.llm_client.model,
            generation_method="refused_at_retrieval",
        )
    return _to_response(request.question, result)


@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a question over the indexed papers.

    Runs in a worker thread: retrieval and generation are synchronous and
    CPU/network bound, and blocking the event loop would serialise every
    concurrent request behind the slowest one.
    """
    return await asyncio.to_thread(_answer, request)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/query/stream", tags=["query"])
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """Answer a question, reporting progress as server-sent events.

    **This streams stages, not tokens.** Events are ``retrieving``,
    ``generating``, then ``answer`` carrying the complete result. Token-level
    streaming would need `LLMClient` to expose the providers' streaming APIs,
    which it does not; claiming otherwise in the endpoint name would be a lie
    the client only discovers at runtime.

    Even stage events are worth having: generation is the slow part, and a
    client that knows retrieval finished can show something.
    """

    async def events() -> AsyncIterator[str]:
        started = time.perf_counter()
        yield _sse("retrieving", {"question": request.question, "top_k": request.top_k})

        task = asyncio.create_task(asyncio.to_thread(_answer, request))
        # Heartbeats keep proxies from closing an idle connection during a long
        # generation, and give the client something to render.
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=2.0)
            if not done:
                yield _sse("generating", {"elapsed": round(time.perf_counter() - started, 1)})

        try:
            response = task.result()
        except ResearchGPTError as exc:
            yield _sse("error", {"error": type(exc).__name__, "detail": str(exc)})
            return

        yield _sse("answer", response.model_dump())

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- papers -----------------------------------------------------------------


@app.get("/papers", response_model=PaperList, tags=["papers"])
async def list_papers() -> PaperList:
    """Every paper currently indexed."""
    database = get_database()
    papers = await asyncio.to_thread(database.list_papers)
    return PaperList(
        papers=[Paper(**p) for p in papers],
        total_papers=len(papers),
        total_chunks=sum(p["num_chunks"] for p in papers),
    )


@app.get("/papers/{paper_id}", response_model=Paper, tags=["papers"])
async def get_paper(paper_id: str) -> Paper:
    """One paper, by id."""
    papers = await asyncio.to_thread(get_database().list_papers)
    for paper in papers:
        if paper["paper_id"] == paper_id:
            return Paper(**paper)
    raise HTTPException(status_code=404, detail=f"No indexed paper with id {paper_id!r}")


def _ingest(job_id: str, path: Path, paper_id: str) -> None:
    """Run ingestion and record the outcome. Never raises: it has no caller."""
    jobs = get_jobs()
    jobs.update(job_id, state=JobState.RUNNING)
    started = time.perf_counter()
    try:
        stats = get_pipeline().process_paper(path, paper_id=paper_id)
        jobs.update(
            job_id,
            state=JobState.SUCCEEDED,
            num_chunks=stats.get("num_chunks"),
            num_sections=stats.get("num_sections"),
            num_pages=stats.get("num_pages"),
            seconds=round(time.perf_counter() - started, 2),
        )
        logger.info("Ingested %r via API (%s chunks)", paper_id, stats.get("num_chunks"))
    except (PDFParseError, ResearchGPTError) as exc:
        jobs.update(job_id, state=JobState.FAILED, detail=f"{type(exc).__name__}: {exc}")
        logger.warning("Ingestion failed for %r: %s", paper_id, exc)
    except Exception as exc:  # noqa: BLE001 - a background task must not vanish silently
        jobs.update(job_id, state=JobState.FAILED, detail=f"Unexpected error: {exc}")
        logger.exception("Unexpected ingestion failure for %r", paper_id)
    finally:
        path.unlink(missing_ok=True)


@app.post("/papers", response_model=IngestionJob, status_code=202, tags=["papers"])
async def upload_paper(
    background: BackgroundTasks,
    file: UploadFile,
    paper_id: str | None = None,
) -> IngestionJob:
    """Upload a PDF for ingestion.

    Returns 202 with a job id immediately; poll ``GET /jobs/{id}``. A 92-page
    paper takes about nine seconds to ingest, which is too long to hold a
    request open and long enough that a client needs progress.
    """
    filename = file.filename or "upload.pdf"
    if Path(filename).suffix.lower() not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported.")

    max_bytes = int(get_settings().max_pdf_size_mb * 1024 * 1024)
    # Written to a temp file rather than held in memory: a 50 MB upload per
    # concurrent request adds up, and the parser wants a path anyway.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        temp_path = Path(tmp.name)
        written = 0
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                temp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {get_settings().max_pdf_size_mb} MB limit.",
                )
            tmp.write(chunk)

    if written == 0:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    resolved_id = paper_id or Path(filename).stem
    job = get_jobs().create(filename=filename, paper_id=resolved_id)
    background.add_task(_ingest, job.job_id, temp_path, resolved_id)
    return job


@app.delete("/papers/{paper_id}", response_model=DeleteResponse, tags=["papers"])
async def delete_paper(paper_id: str) -> DeleteResponse:
    """Remove a paper and every chunk belonging to it."""
    deleted = await asyncio.to_thread(get_database().delete_paper, paper_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No indexed paper with id {paper_id!r}")
    return DeleteResponse(paper_id=paper_id, chunks_deleted=deleted)


# --- jobs -------------------------------------------------------------------


@app.get("/jobs/{job_id}", response_model=IngestionJob, tags=["papers"])
async def get_job(job_id: str) -> IngestionJob:
    """Status of an ingestion job."""
    job = get_jobs().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job with id {job_id!r}")
    return job


@app.get("/jobs", response_model=list[IngestionJob], tags=["papers"])
async def list_jobs() -> list[IngestionJob]:
    """Recent ingestion jobs, newest first."""
    return get_jobs().all()


# --- health -----------------------------------------------------------------


def _check_dependencies() -> tuple[list[DependencyStatus], int, int]:
    """Probe each dependency, reporting failures rather than raising."""
    checks: list[DependencyStatus] = []
    total_papers = total_chunks = 0

    try:
        stats = get_database().get_stats()
        total_papers = stats["total_papers"]
        total_chunks = stats["total_chunks"]
        checks.append(
            DependencyStatus(
                name="vector_store",
                ok=True,
                detail=f"{total_chunks} chunks from {total_papers} papers",
            )
        )
    except Exception as exc:  # noqa: BLE001 - health must report, not raise
        checks.append(DependencyStatus(name="vector_store", ok=False, detail=str(exc)))

    try:
        info = get_generator().llm_client.get_model_info()
        checks.append(
            DependencyStatus(
                name="llm",
                ok=True,
                detail=f"{info.get('provider')}/{info.get('model')}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(DependencyStatus(name="llm", ok=False, detail=str(exc)))

    # Configuration is checked, not just presence: an API key is only useful
    # for the provider actually selected.
    settings = get_settings()
    has_key = bool(settings.api_key_for(settings.llm_provider))
    checks.append(
        DependencyStatus(
            name="api_key",
            ok=has_key,
            detail=f"provider={settings.llm_provider}"
            + ("" if has_key else " (no key configured)"),
        )
    )

    return checks, total_papers, total_chunks


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness plus a probe of each dependency.

    An empty corpus reports ``degraded``, not ``unhealthy``: the service is
    working, it just has nothing to search. A load balancer should keep sending
    traffic; an operator should notice.
    """
    checks, total_papers, total_chunks = await asyncio.to_thread(_check_dependencies)

    if not all(check.ok for check in checks):
        status = "unhealthy"
    elif total_chunks == 0:
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        version=VERSION,
        dependencies=checks,
        total_papers=total_papers,
        total_chunks=total_chunks,
    )


# --- frontend ---------------------------------------------------------------
#
# The React bundle is served by this same app. One container serves the API and
# the UI, which is what a Hugging Face Space provides, and it removes CORS
# entirely -- the frontend calls same-origin paths.
#
# Mounted last so every API route above wins on a path collision.

_STATIC = PROJECT_ROOT / "static"


@app.get("/", include_in_schema=False)
async def root() -> Any:
    """Serve the app, or a pointer to the docs when no bundle is built."""
    index = _STATIC / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse(
        {
            "service": "ResearchGPT",
            "version": VERSION,
            "docs": "/docs",
            "detail": "No frontend bundle found. Build it with: cd frontend && npm run build",
        }
    )


if (_STATIC / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_STATIC / "assets"), name="assets")
