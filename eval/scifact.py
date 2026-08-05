"""Load BEIR/SciFact and index it for evaluation.

SciFact pairs expert-written scientific claims with biomedical abstracts, with
human relevance judgments (qrels). Using it means the retrieval numbers rest on
expert labels rather than anything self-generated, and can be compared against
published BEIR baselines.

The index lives in its own ChromaDB path and collection, so running an
evaluation never touches the user's own corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, get_settings
from src.ingestion.chunker import TextChunker
from src.ingestion.database import VectorDatabase
from src.ingestion.embedder import EmbeddingGenerator
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Isolated from the application's store. Never point this at data/chroma_db.
EVAL_DB_PATH = PROJECT_ROOT / "data" / "eval_chroma"
COLLECTION_PREFIX = "scifact"

HF_CORPUS = "BeIR/scifact"
HF_QRELS = "BeIR/scifact-qrels"


@dataclass
class ScifactData:
    """The SciFact test split: documents, queries, and relevance judgments."""

    #: Document ID to ``{"title": ..., "text": ...}``.
    corpus: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Query ID to claim text. Only queries carrying judgments.
    queries: dict[str, str] = field(default_factory=dict)
    #: Query ID to the set of relevant document IDs.
    qrels: dict[str, set[str]] = field(default_factory=dict)

    def summary(self) -> str:
        judgments = sum(len(v) for v in self.qrels.values())
        per_query = judgments / len(self.qrels) if self.qrels else 0
        return (
            f"{len(self.corpus)} documents, {len(self.queries)} queries, "
            f"{judgments} judgments ({per_query:.2f} per query)"
        )


def load_scifact(split: str = "test", query_limit: int | None = None) -> ScifactData:
    """Download SciFact and return the split's corpus, queries, and qrels.

    Args:
        split: qrels split, ``"test"`` (300 queries, BEIR's evaluation split)
            or ``"train"``.
        query_limit: Keep only the first N queries. Useful for quick iteration;
            report full-split numbers.

    Returns:
        Queries are restricted to those with judgments, since a query with no
        qrels cannot be scored.
    """
    from datasets import load_dataset

    logger.info("Loading SciFact corpus from %s", HF_CORPUS)
    corpus_rows = load_dataset(HF_CORPUS, "corpus", split="corpus")
    query_rows = load_dataset(HF_CORPUS, "queries", split="queries")
    qrel_rows = load_dataset(HF_QRELS, split=split)

    qrels: dict[str, set[str]] = {}
    for row in qrel_rows:
        # BEIR uses graded scores elsewhere; SciFact is binary, but filter
        # explicitly so this loader stays correct for graded datasets too.
        if int(row["score"]) <= 0:
            continue
        qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    if query_limit is not None:
        kept = sorted(qrels, key=lambda q: int(q))[:query_limit]
        qrels = {qid: qrels[qid] for qid in kept}

    all_queries = {str(row["_id"]): row["text"] for row in query_rows}
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}

    missing = set(qrels) - set(queries)
    if missing:
        logger.warning("%d judged queries absent from the query file", len(missing))
        qrels = {qid: docs for qid, docs in qrels.items() if qid in queries}

    corpus = {
        str(row["_id"]): {"title": row["title"] or "", "text": row["text"] or ""}
        for row in corpus_rows
    }

    data = ScifactData(corpus=corpus, queries=queries, qrels=qrels)
    logger.info("SciFact %s: %s", split, data.summary())
    return data


def collection_name(chunk_size: int, embedding_model: str) -> str:
    """Name the collection after the settings that determine its contents.

    Chunk size and embedding model both change the stored vectors, so each
    combination gets its own collection. That makes chunk-size sweeps a matter
    of indexing once per size rather than rebuilding in place, and removes any
    chance of comparing a run against a stale index.
    """
    model_slug = embedding_model.replace("/", "-").replace("_", "-")
    return f"{COLLECTION_PREFIX}-{chunk_size}-{model_slug}"


def index_scifact(
    data: ScifactData,
    embedder: EmbeddingGenerator,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    force: bool = False,
) -> VectorDatabase:
    """Index the SciFact corpus into an isolated collection.

    Abstracts are short, so most become a single chunk; the chunker is still
    used so that the indexed representation matches what the application does
    to real papers.

    Args:
        data: Loaded SciFact.
        embedder: Embedding model. Its name is part of the collection name.
        chunk_size: Tokens per chunk. Defaults to ``Settings.chunk_size``.
        chunk_overlap: Defaults to ``Settings.chunk_overlap``.
        force: Re-index even when the collection is already populated.

    Returns:
        The populated store, ready to query.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    name = collection_name(chunk_size, embedder.model_name)
    database = VectorDatabase(db_path=EVAL_DB_PATH, collection_name=name)

    expected = len(data.corpus)
    if not force and database.count() > 0:
        indexed_docs = len(database.list_papers())
        if indexed_docs >= expected:
            logger.info(
                "Collection %r already holds %d documents (%d chunks); skipping indexing",
                name,
                indexed_docs,
                database.count(),
            )
            return database
        logger.warning(
            "Collection %r has %d/%d documents; re-indexing", name, indexed_docs, expected
        )
        database.reset()
    elif force and database.count() > 0:
        database.reset()

    database.verify_embedding_model(embedder.model_name, embedder.embedding_dim)
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info("Indexing %d SciFact documents into %r", expected, name)

    # Batch across documents: embedding 5k abstracts one at a time is dominated
    # by per-call overhead rather than compute.
    batch: list[dict[str, Any]] = []
    batch_docs: list[str] = []
    indexed = 0
    BATCH_DOCS = 256

    def flush() -> None:
        nonlocal batch, batch_docs, indexed
        if not batch:
            return
        embedded = embedder.embed_chunks(batch)
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for chunk in embedded:
            by_doc.setdefault(chunk["_doc_id"], []).append(chunk)
        for doc_id, chunks in by_doc.items():
            for chunk in chunks:
                chunk.pop("_doc_id", None)
            database.add_chunks(
                chunks=chunks,
                paper_id=doc_id,
                paper_metadata={"title": data.corpus[doc_id]["title"][:200]},
            )
        indexed += len(batch_docs)
        logger.info("Indexed %d/%d documents", indexed, expected)
        batch, batch_docs = [], []

    for doc_id, doc in data.corpus.items():
        # Title carries real signal for BM25 on scientific abstracts, so index
        # it as part of the text rather than metadata only.
        full_text = f"{doc['title']}\n\n{doc['text']}".strip()
        if not full_text:
            continue

        chunks = chunker.chunk_text(full_text, metadata={"doc_id": doc_id})
        for chunk in chunks:
            chunk["_doc_id"] = doc_id
        batch.extend(chunks)
        batch_docs.append(doc_id)

        if len(batch_docs) >= BATCH_DOCS:
            flush()

    flush()
    logger.info("Indexing complete: %d chunks in %r", database.count(), name)
    return database


def chunk_id_to_doc_id(chunk_id: str) -> str:
    """Recover the SciFact document ID from a stored chunk ID.

    Chunk IDs are ``{paper_id}_chunk_{n}``; SciFact document IDs are numeric,
    so splitting on the last ``_chunk_`` is unambiguous.
    """
    return chunk_id.rsplit("_chunk_", 1)[0]


def rank_documents(results: list[dict[str, Any]]) -> list[str]:
    """Collapse a ranked chunk list into a ranked document list.

    BEIR judges documents while this system retrieves chunks, so a document
    takes the rank of its best-scoring chunk and duplicates are dropped.
    Without this, a document split into three chunks would occupy three of the
    top ten slots and inflate precision.
    """
    seen: set[str] = set()
    ranked: list[str] = []
    for result in results:
        doc_id = result["metadata"].get("paper_id") or chunk_id_to_doc_id(result["id"])
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def eval_db_path() -> Path:
    """Where evaluation indexes live, for CLI messages."""
    return EVAL_DB_PATH
