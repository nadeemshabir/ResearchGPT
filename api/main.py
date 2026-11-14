"""HTTP API for ResearchGPT.

    uvicorn api.main:app --reload

Endpoints
---------
``POST /query``          answer a question over the indexed corpus
``POST /query/stream``   the same, as server-sent events
``POST /papers``         upload and index a PDF
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

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.dependencies import (
    get_database,
    get_generator,
    get_jobs,
    get_pipeline,
    get_storage,
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
    PaperAlreadyExistsError,
    PaperStorageError,
    PDFParseError,
    ResearchGPTError,
)
from src.ingestion.ids import clean_paper_id
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
    if isinstance(exc, PaperAlreadyExistsError):
        status, kind = 409, "paper_exists"
    elif isinstance(exc, PaperStorageError):
        status, kind = 502, "storage_unavailable"
    elif isinstance(exc, PDFParseError):
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


def _paper_is_indexed(paper_id: str) -> bool:
    """Whether the vector store already holds chunks for this id."""
    return bool(get_database().get_by_paper_id(paper_id).get("ids"))


def _purge(paper_id: str) -> int:
    """Remove a paper's chunks and its stored PDF. Returns chunks deleted.

    Both halves run even when the first finds nothing: a half-finished upload
    can leave a PDF with no chunks, or chunks with no PDF, and a delete that
    cleaned only the half it happened to find would leave the other behind
    forever.
    """
    deleted = get_database().delete_paper(paper_id)
    get_storage().delete(paper_id)
    return deleted


def _safe_purge(paper_id: str) -> None:
    """Best-effort cleanup after a failed ingest."""
    try:
        _purge(paper_id)
    except Exception:  # noqa: BLE001 - the original failure is the one to report
        logger.debug("Cleanup after failed ingest of %r did not finish", paper_id, exc_info=True)


def _ingest(job_id: str, path: Path, paper_id: str, replace: bool) -> None:
    """Index one PDF and record the outcome on ``job_id``.

    Runs inside the request, not in a background task. On Cloud Run a
    background task keeps running only while the instance has CPU, and CPU is
    throttled to near zero once the response has been sent -- so a nine-second
    ingest started in ``BackgroundTasks`` could take minutes or be killed at
    the next scale-down. It also made the per-process job registry a lie: with
    more than one instance the client polling for its job hits whichever
    instance the load balancer picks, which is usually not the one holding it.

    Never raises: the job record carries the outcome.
    """
    jobs = get_jobs()
    jobs.update(job_id, state=JobState.RUNNING)
    started = time.perf_counter()
    try:
        if replace:
            # Chunk ids are ``<paper_id>_chunk_<n>`` and the store upserts by
            # id. Re-indexing a paper that now yields fewer chunks -- a
            # different chunk size, a shorter revision -- would leave the
            # surplus chunks of the old version in place, still searchable and
            # still citable. Deleting first is what makes a replace a replace.
            _purge(paper_id)

        stats = get_pipeline().process_paper(path, paper_id=paper_id)
        stored_uri = get_storage().save(paper_id, path)

        jobs.update(
            job_id,
            state=JobState.SUCCEEDED,
            num_chunks=stats.get("num_chunks"),
            num_sections=stats.get("num_sections"),
            num_pages=stats.get("num_pages"),
            seconds=round(time.perf_counter() - started, 2),
            stored_at=stored_uri,
        )
        logger.info("Ingested %r via API (%s chunks)", paper_id, stats.get("num_chunks"))
    except (PDFParseError, ResearchGPTError) as exc:
        # A paper that failed to index must not leave chunks behind: a partial
        # index answers questions from half a document without saying so.
        _safe_purge(paper_id)
        jobs.update(job_id, state=JobState.FAILED, detail=f"{type(exc).__name__}: {exc}")
        logger.warning("Ingestion failed for %r: %s", paper_id, exc)
    except Exception as exc:  # noqa: BLE001 - the job record is the only report
        _safe_purge(paper_id)
        jobs.update(job_id, state=JobState.FAILED, detail=f"Unexpected error: {exc}")
        logger.exception("Unexpected ingestion failure for %r", paper_id)


@app.post("/papers", response_model=IngestionJob, status_code=201, tags=["papers"])
async def upload_paper(
    file: UploadFile,
    paper_id: str | None = None,
    replace: bool = False,
) -> IngestionJob:
    """Upload a PDF and index it.

    The response comes back once the paper is searchable, carrying a finished
    job record. ``GET /jobs/{id}`` still works, and is still worth polling from
    a client that lost the response, but on a healthy request there is nothing
    left to wait for.

    Args:
        file: The PDF. Rejected above ``MAX_PDF_SIZE_MB``.
        paper_id: Override the id derived from the filename. The id becomes the
            citation key, so a readable one gives a readable citation.
        replace: Overwrite a paper already indexed under this id. Without it a
            collision is a 409, rather than a silent overwrite of somebody
            else's upload.
    """
    filename = file.filename or "upload.pdf"
    if Path(filename).suffix.lower() not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported.")

    resolved_id = clean_paper_id(paper_id or Path(filename).stem)

    # Checked before the body is read, so a client re-sending a 50 MB file it
    # has already uploaded is told so immediately.
    if not replace and await asyncio.to_thread(_paper_is_indexed, resolved_id):
        raise PaperAlreadyExistsError(resolved_id)

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

    job = get_jobs().create(filename=filename, paper_id=resolved_id)
    try:
        await asyncio.to_thread(_ingest, job.job_id, temp_path, resolved_id, replace)
    finally:
        temp_path.unlink(missing_ok=True)

    return get_jobs().get(job.job_id) or job


@app.delete("/papers/{paper_id}", response_model=DeleteResponse, tags=["papers"])
async def delete_paper(paper_id: str) -> DeleteResponse:
    """Remove a paper: its chunks and the PDF they were built from.

    Deleting only the chunks used to leave the uploaded PDF in storage for
    good, which is both a surprise -- the UI says the paper is gone -- and a
    bill that grows with every delete.
    """
    had_pdf = await asyncio.to_thread(get_storage().exists, paper_id)
    deleted = await asyncio.to_thread(_purge, paper_id)
    if deleted == 0 and not had_pdf:
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
        database = get_database()
        stats = database.get_stats()
        total_papers = stats["total_papers"]
        total_chunks = stats["total_chunks"]
        checks.append(
            DependencyStatus(
                name="vector_store",
                ok=True,
                detail=(
                    f"{total_chunks} chunks from {total_papers} papers "
                    f"({stats['mode']}: {stats['location']})"
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001 - health must report, not raise
        checks.append(DependencyStatus(name="vector_store", ok=False, detail=str(exc)))

    try:
        storage = get_storage()
        checks.append(
            DependencyStatus(
                name="paper_storage",
                ok=True,
                detail=f"{storage.backend}: {storage.location}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(DependencyStatus(name="paper_storage", ok=False, detail=str(exc)))

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

    An empty corpus is ``ok``. Every paper in this service arrives by upload,
    so an empty store is a new deployment, not a fault -- and reporting it as
    ``degraded`` would mean every fresh install started life looking broken.
    What matters is whether the store and the model are reachable, which is
    what the dependency checks say.
    """
    checks, total_papers, total_chunks = await asyncio.to_thread(_check_dependencies)
    status = "ok" if all(check.ok for check in checks) else "unhealthy"

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
