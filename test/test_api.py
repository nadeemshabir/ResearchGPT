"""Tests for the HTTP API.

Every dependency is replaced before the app starts, so no model loads, no store
opens and no API is called. `warm_up` is patched out for the same reason -- the
lifespan handler would otherwise pull in sentence-transformers on import.

The behaviours worth pinning are the ones a client depends on and cannot
discover from the schema: that a refusal is a 200 rather than a 4xx, that upload
returns before ingestion finishes, and that health degrades rather than fails on
an empty corpus.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from api import dependencies
from api.jobs import JobRegistry
from src.config import Settings

CHUNK = {
    "id": "c1",
    "text": "The Transformer uses multi-head self-attention instead of recurrence.",
    "metadata": {
        "paper_id": "attention_2017",
        "title": "Attention is All you Need",
        "section_title": "Model Architecture",
    },
    "similarity_score": 0.71,
    "hybrid_score": 0.88,
}

PAPERS = [
    {
        "paper_id": "attention_2017",
        "title": "Attention is All you Need",
        "author": "Vaswani et al.",
        "num_chunks": 19,
    },
    {
        "paper_id": "bert_2019",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "author": "Devlin et al.",
        "num_chunks": 41,
    },
]


class StubDatabase:
    def __init__(self) -> None:
        self.papers = [dict(p) for p in PAPERS]
        self.deleted: list[str] = []

    def list_papers(self) -> list[dict[str, Any]]:
        return [dict(p) for p in self.papers]

    def delete_paper(self, paper_id: str) -> int:
        for paper in self.papers:
            if paper["paper_id"] == paper_id:
                self.papers.remove(paper)
                self.deleted.append(paper_id)
                return int(paper["num_chunks"])
        return 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_papers": len(self.papers),
            "total_chunks": sum(p["num_chunks"] for p in self.papers),
        }


class StubLLM:
    provider = "stub"
    model = "stub-model"

    def get_model_info(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model}


class StubGenerator:
    """Returns a canned answer and records how it was called."""

    def __init__(self, multi_agent: bool | None = None):
        self.llm_client = StubLLM()
        self.multi_agent = multi_agent
        self.calls: list[dict[str, Any]] = []
        self.raises: Exception | None = None

    def answer_question(self, question: str, top_k: int = 5) -> dict[str, Any]:
        self.calls.append({"question": question, "top_k": top_k})
        if self.raises is not None:
            raise self.raises
        return {
            "answer": "Multi-head self-attention replaces recurrence. [Attention is All you Need, 2017]",
            "question": question,
            "sources": [
                {
                    "paper_id": "attention_2017",
                    "title": "Attention is All you Need",
                    "author": "Vaswani et al.",
                    "section": "Model Architecture",
                    "relevance_score": 0.88,
                    "chunk_preview": CHUNK["text"][:40],
                }
            ],
            "paragraphs": [
                {
                    "text": "Multi-head self-attention replaces recurrence.",
                    "structural": False,
                    "sources": [
                        {
                            "chunk_id": "c1",
                            "paper_id": "attention_2017",
                            "title": "Attention is All you Need",
                            "year": "2017",
                            "section": "Model Architecture",
                            "score": 0.34,
                        }
                    ],
                }
            ],
            "chunks": [
                {
                    "chunk_id": "c1",
                    "paper_id": "attention_2017",
                    "title": "Attention is All you Need",
                    "section": "Model Architecture",
                    "text": CHUNK["text"],
                    "relevance_score": 0.88,
                }
            ],
            "metadata": {
                "refused": False,
                "num_sources": 1,
                "processing_time": 0.4,
                "retrieval_time": 0.02,
                "generation_time": 0.35,
                "model": "stub-model",
                "generation_method": "multi-agent" if self.multi_agent else "single-shot",
                "citation_audit": {"total_citations": 1, "unknown_citations": []},
            },
        }


class StubPipeline:
    def __init__(self) -> None:
        self.processed: list[str] = []
        self.raises: Exception | None = None

    def process_paper(self, path: Any, paper_id: str | None = None) -> dict[str, Any]:
        if self.raises is not None:
            raise self.raises
        self.processed.append(paper_id or "unknown")
        return {"num_chunks": 12, "num_sections": 4, "num_pages": 9}


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace every singleton, and stop the lifespan loading models."""
    database = StubDatabase()
    pipeline = StubPipeline()
    jobs = JobRegistry()
    generators: dict[bool | None, StubGenerator] = {}

    def fake_generator(multi_agent: bool | None = None) -> StubGenerator:
        return generators.setdefault(multi_agent, StubGenerator(multi_agent))

    monkeypatch.setattr(dependencies, "warm_up", lambda: None)
    for module in ("api.main", "api.dependencies"):
        monkeypatch.setattr(f"{module}.get_database", lambda: database, raising=False)
        monkeypatch.setattr(f"{module}.get_pipeline", lambda: pipeline, raising=False)
        monkeypatch.setattr(f"{module}.get_jobs", lambda: jobs, raising=False)
        monkeypatch.setattr(f"{module}.get_generator", fake_generator, raising=False)
    monkeypatch.setattr("api.main.warm_up", lambda: None, raising=False)

    # Settings too, or /health reads the developer's .env: with a key present
    # it reports "ok", without one "unhealthy". Pinning it keeps the health
    # tests about the corpus, which is what they are checking.
    stub_settings = Settings(_env_file=None, groq_api_key="test-key", llm_provider="groq")  # type: ignore[call-arg]
    monkeypatch.setattr("api.main.get_settings", lambda: stub_settings, raising=False)

    return {
        "database": database,
        "pipeline": pipeline,
        "jobs": jobs,
        "generators": generators,
    }


@pytest.fixture
def client(stubs: dict[str, Any]) -> Any:
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


# --- query ------------------------------------------------------------------


def test_query_returns_an_answer_and_sources(client: Any) -> None:
    response = client.post("/query", json={"question": "How does attention work?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["sources"][0]["paper_id"] == "attention_2017"
    assert body["refused"] is False


def test_query_returns_paragraph_to_chunk_links(client: Any) -> None:
    """What makes click-to-highlight possible without parsing the answer text."""
    body = client.post("/query", json={"question": "How does attention work?"}).json()

    paragraph = body["paragraphs"][0]
    assert paragraph["sources"][0]["chunk_id"] == "c1"
    assert paragraph["sources"][0]["score"] > 0


def test_every_cited_chunk_id_resolves_to_a_returned_chunk(client: Any) -> None:
    """A dangling id would leave the UI with a citation it cannot open."""
    body = client.post("/query", json={"question": "How does attention work?"}).json()

    available = {c["chunk_id"] for c in body["chunks"]}
    cited = {s["chunk_id"] for p in body["paragraphs"] for s in p["sources"]}

    assert cited <= available


def test_chunks_carry_full_text_not_a_preview(client: Any) -> None:
    """A highlight needs the passage a reader will read, not the first 200 chars."""
    body = client.post("/query", json={"question": "How does attention work?"}).json()

    assert body["chunks"][0]["text"] == CHUNK["text"]


def test_query_passes_top_k_through(client: Any, stubs: dict[str, Any]) -> None:
    client.post("/query", json={"question": "How does attention work?", "top_k": 3})

    assert stubs["generators"][None].calls[0]["top_k"] == 3


def test_query_rejects_an_empty_question(client: Any) -> None:
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_rejects_an_out_of_range_top_k(client: Any) -> None:
    response = client.post("/query", json={"question": "hi there", "top_k": 500})

    assert response.status_code == 422


def test_a_refusal_is_a_200_not_an_error(client: Any, stubs: dict[str, Any]) -> None:
    """A refusal is correct behaviour, so it must not look like a client fault.

    Returning 4xx here would make "the corpus cannot answer that" indistinguishable
    from "your request was malformed" in any error-rate dashboard.
    """
    from src.exceptions import NoRelevantContextError

    stubs["generators"].setdefault(None, StubGenerator()).raises = NoRelevantContextError(
        "what is the capital of France?"
    )

    response = client.post("/query", json={"question": "what is the capital of France?"})

    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert response.json()["sources"] == []


def test_an_llm_outage_is_a_502(client: Any, stubs: dict[str, Any]) -> None:
    """A provider failure is upstream, not the caller's fault."""
    from src.exceptions import LLMTimeoutError

    stubs["generators"].setdefault(None, StubGenerator()).raises = LLMTimeoutError(
        "gemini request timed out after 60.0s"
    )

    response = client.post("/query", json={"question": "How does attention work?"})

    assert response.status_code == 502
    assert response.json()["error"] == "llm_unavailable"


def test_multi_agent_uses_a_separate_generator(client: Any, stubs: dict[str, Any]) -> None:
    """Mutating one shared generator would leak the mode into the next request."""
    client.post("/query", json={"question": "q1", "multi_agent": True})
    client.post("/query", json={"question": "q2", "multi_agent": False})

    assert stubs["generators"][True].calls == [{"question": "q1", "top_k": 5}]
    assert stubs["generators"][False].calls == [{"question": "q2", "top_k": 5}]


# --- streaming --------------------------------------------------------------


def test_stream_emits_stage_events_then_the_answer(client: Any) -> None:
    with client.stream(
        "POST", "/query/stream", json={"question": "How does attention work?"}
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = "".join(response.iter_text())

    assert "event: retrieving" in body
    assert "event: answer" in body


# --- papers -----------------------------------------------------------------


def test_list_papers_reports_totals(client: Any) -> None:
    body = client.get("/papers").json()

    assert body["total_papers"] == 2
    assert body["total_chunks"] == 60


def test_get_one_paper(client: Any) -> None:
    body = client.get("/papers/attention_2017").json()

    assert body["title"] == "Attention is All you Need"


def test_get_a_missing_paper_is_404(client: Any) -> None:
    assert client.get("/papers/does_not_exist").status_code == 404


def test_delete_removes_the_paper(client: Any, stubs: dict[str, Any]) -> None:
    response = client.delete("/papers/bert_2019")

    assert response.status_code == 200
    assert response.json()["chunks_deleted"] == 41
    assert stubs["database"].deleted == ["bert_2019"]


def test_delete_a_missing_paper_is_404(client: Any) -> None:
    assert client.delete("/papers/nope").status_code == 404


# --- upload -----------------------------------------------------------------


def test_upload_accepts_a_pdf_and_returns_a_job(client: Any) -> None:
    """202, not 200: ingestion has been accepted, not completed."""
    response = client.post(
        "/papers", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")}
    )

    assert response.status_code == 202
    assert response.json()["job_id"]
    assert response.json()["paper_id"] == "paper"


def test_upload_runs_ingestion_in_the_background(client: Any, stubs: dict[str, Any]) -> None:
    """TestClient runs background tasks before returning, so this is observable."""
    client.post("/papers", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")})

    assert stubs["pipeline"].processed == ["paper"]


def test_a_successful_job_records_its_statistics(client: Any, stubs: dict[str, Any]) -> None:
    job_id = client.post(
        "/papers", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")}
    ).json()["job_id"]

    body = client.get(f"/jobs/{job_id}").json()

    assert body["state"] == "succeeded"
    assert body["num_chunks"] == 12


def test_a_failed_ingestion_is_recorded_not_raised(client: Any, stubs: dict[str, Any]) -> None:
    """A background task has no caller, so a swallowed error would be invisible."""
    from src.exceptions import EncryptedPDFError

    stubs["pipeline"].raises = EncryptedPDFError("paper.pdf is password protected.")

    job_id = client.post(
        "/papers", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")}
    ).json()["job_id"]
    body = client.get(f"/jobs/{job_id}").json()

    assert body["state"] == "failed"
    assert "password protected" in body["detail"]


def test_a_custom_paper_id_is_honoured(client: Any, stubs: dict[str, Any]) -> None:
    client.post(
        "/papers?paper_id=attention_2017",
        files={"file": ("2312.10997v5.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert stubs["pipeline"].processed == ["attention_2017"]


def test_a_non_pdf_upload_is_rejected(client: Any) -> None:
    response = client.post("/papers", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 415


def test_an_empty_upload_is_rejected(client: Any) -> None:
    response = client.post("/papers", files={"file": ("empty.pdf", b"", "application/pdf")})

    assert response.status_code == 422


def test_an_oversized_upload_is_rejected(client: Any) -> None:
    """Streamed to disk in 1 MB pieces, so the limit trips before the file lands."""
    from src.config import get_settings

    oversized = b"x" * (int(get_settings().max_pdf_size_mb * 1024 * 1024) + 1024)

    response = client.post("/papers", files={"file": ("huge.pdf", oversized, "application/pdf")})

    assert response.status_code == 413


def test_an_unknown_job_is_404(client: Any) -> None:
    assert client.get("/jobs/deadbeef").status_code == 404


# --- health -----------------------------------------------------------------


def test_health_reports_ok_with_a_populated_corpus(client: Any) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["total_papers"] == 2
    assert {c["name"] for c in body["dependencies"]} == {
        "vector_store",
        "llm",
        "api_key",
    }


def test_health_is_degraded_not_unhealthy_on_an_empty_corpus(
    client: Any, stubs: dict[str, Any]
) -> None:
    """The service works; it just has nothing indexed.

    Reporting unhealthy would take a working instance out of a load balancer
    for a condition only an operator can fix.
    """
    stubs["database"].papers = []

    assert client.get("/health").json()["status"] == "degraded"


def test_health_is_unhealthy_when_the_store_fails(client: Any, stubs: dict[str, Any]) -> None:
    def boom() -> dict[str, Any]:
        raise RuntimeError("chroma is unreachable")

    stubs["database"].get_stats = boom  # type: ignore[method-assign]

    body = client.get("/health").json()

    assert body["status"] == "unhealthy"
    assert any(not c["ok"] for c in body["dependencies"])


def test_the_openapi_schema_is_served(client: Any) -> None:
    """Broken response models surface here rather than at the first request."""
    schema = client.get("/openapi.json").json()

    assert "/query" in schema["paths"]
    assert "/papers" in schema["paths"]
    assert "/health" in schema["paths"]
