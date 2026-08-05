"""ChromaDB-backed vector store.

Stores chunk text alongside its embedding and metadata. The collection records
which embedding model built it, so a later run with a different model fails
with an actionable message instead of an opaque dimensionality error from
Chroma at query time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from src.config import get_settings
from src.exceptions import EmbeddingDimensionMismatchError, VectorStoreError
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Collection metadata keys recording provenance of the stored vectors.
_MODEL_KEY = "embedding_model"
_DIM_KEY = "embedding_dim"


class VectorDatabase:
    """Persistent vector store for paper chunks."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        collection_name: str | None = None,
    ):
        """
        Args:
            db_path: Directory for the on-disk store. Defaults to
                ``Settings.chroma_db_path``.
            collection_name: Defaults to ``Settings.collection_name``.

        Raises:
            VectorStoreError: The store could not be opened or created.
        """
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else settings.chroma_db_path
        self.collection_name = collection_name or settings.collection_name

        self.db_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Opening vector store at %s (collection %r)", self.db_path, self.collection_name
        )

        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=str(self.db_path))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Research paper chunk embeddings"},
            )
        except Exception as exc:  # noqa: BLE001 - typed error for callers
            raise VectorStoreError(
                f"Could not open vector store at {self.db_path}: {exc}"
            ) from exc

        logger.info("Collection %r holds %d chunks", self.collection_name, self.count())

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    def verify_embedding_model(self, model_name: str, dimension: int) -> None:
        """Check the active embedding model against the one that built the store.

        On an empty collection this records the model instead of comparing, so
        the first ingestion establishes provenance.

        Raises:
            EmbeddingDimensionMismatchError: The stored vectors were produced
                by a model with a different output dimension.
        """
        metadata = self.collection.metadata or {}
        stored_model = metadata.get(_MODEL_KEY)
        stored_dim = metadata.get(_DIM_KEY)

        if self.count() == 0 or stored_dim is None:
            self._record_embedding_model(model_name, dimension)
            return

        if int(stored_dim) != int(dimension):
            raise EmbeddingDimensionMismatchError(
                expected=int(stored_dim),
                actual=int(dimension),
                collection_model=str(stored_model or "unknown"),
                active_model=model_name,
            )

        if stored_model and stored_model != model_name:
            # Same width, different model: vectors are not comparable, but this
            # is recoverable by switching back, so warn rather than fail.
            logger.warning(
                "Collection was built with %r but the active model is %r. "
                "Both are %d-dimensional so queries will run, but relevance "
                "will be poor. Re-ingest for correct results.",
                stored_model,
                model_name,
                dimension,
            )

    def _record_embedding_model(self, model_name: str, dimension: int) -> None:
        """Persist embedding provenance onto the collection metadata."""
        metadata = dict(self.collection.metadata or {})
        metadata[_MODEL_KEY] = model_name
        metadata[_DIM_KEY] = int(dimension)
        try:
            self.collection.modify(metadata=metadata)
        except Exception:  # noqa: BLE001 - provenance is best-effort
            logger.debug("Could not record embedding provenance", exc_info=True)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        paper_id: str,
        paper_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert embedded chunks for one paper.

        Args:
            chunks: Chunk records carrying ``text``, ``chunk_id``, and ``embedding``.
            paper_id: Unique paper identifier; chunk ids are namespaced under it.
            paper_metadata: Paper-level fields copied onto every chunk.

        Returns:
            Number of chunks written.

        Raises:
            VectorStoreError: A chunk lacks an embedding, or the write failed.
        """
        if not chunks:
            logger.warning("No chunks supplied for paper %r", paper_id)
            return 0

        missing = [c.get("chunk_id") for c in chunks if not c.get("embedding")]
        if missing:
            raise VectorStoreError(
                f"{len(missing)} chunk(s) for {paper_id!r} have no embedding "
                f"(first: chunk_id={missing[0]}). Run embed_chunks() before storing."
            )

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(f"{paper_id}_chunk_{chunk['chunk_id']}")
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])

            metadata: dict[str, Any] = {
                "paper_id": paper_id,
                "chunk_id": chunk["chunk_id"],
                "num_tokens": chunk.get("num_tokens", 0),
                "section_title": chunk.get("section_title", "Unknown"),
            }
            if paper_metadata:
                metadata.update(paper_metadata)
            if "metadata" in chunk:
                metadata.update(chunk["metadata"])

            metadatas.append(_sanitise_metadata(metadata))

        try:
            # upsert rather than add, so re-ingesting a paper replaces its
            # chunks instead of raising on duplicate ids.
            #
            # The casts exist because Chroma's stubs describe narrower types
            # than the runtime accepts; plain lists of floats and of scalar
            # metadata dicts are valid inputs.
            self.collection.upsert(
                ids=ids,
                embeddings=cast(Any, embeddings),
                documents=documents,
                metadatas=cast(Any, metadatas),
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"Failed to write {len(chunks)} chunks for {paper_id!r}: {exc}"
            ) from exc

        # Debug rather than info: bulk indexing calls this once per document,
        # and 5k INFO lines drown out everything else. Callers ingesting a
        # single paper log their own summary.
        logger.debug("Stored %d chunks for %r", len(chunks), paper_id)
        return len(chunks)

    def delete_paper(self, paper_id: str) -> int:
        """Remove every chunk belonging to ``paper_id``. Returns the count deleted."""
        existing = self.get_by_paper_id(paper_id)
        ids = existing.get("ids") or []
        if not ids:
            logger.warning("No chunks found for paper %r", paper_id)
            return 0

        try:
            self.collection.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to delete paper {paper_id!r}: {exc}") from exc

        logger.info("Deleted %d chunks for %r", len(ids), paper_id)
        return len(ids)

    def reset(self) -> None:
        """Drop and recreate the collection. Destroys all indexed data."""
        logger.warning("Resetting collection %r", self.collection_name)
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Research paper chunk embeddings"},
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to reset collection: {exc}") from exc

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Nearest-neighbour search.

        ``n_results`` is clamped to the collection size, because Chroma warns
        and truncates when asked for more than it holds.
        """
        available = self.count()
        if available == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        try:
            # Returned as a plain dict on purpose: callers should not have to
            # import Chroma's TypedDicts to consume this module.
            return cast(
                dict[str, Any],
                self.collection.query(
                    query_embeddings=cast(Any, [query_embedding]),
                    n_results=min(n_results, available),
                    where=cast(Any, filter_dict),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Vector query failed: {exc}") from exc

    def get_by_paper_id(self, paper_id: str) -> dict[str, Any]:
        """Return every stored chunk for one paper."""
        try:
            return cast(
                dict[str, Any], self.collection.get(where={"paper_id": paper_id})
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Lookup for paper {paper_id!r} failed: {exc}") from exc

    def get_all(self) -> dict[str, Any]:
        """Return the whole collection. Used to build the BM25 index."""
        try:
            return cast(dict[str, Any], self.collection.get())
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not read collection: {exc}") from exc

    def count(self) -> int:
        """Number of chunks stored."""
        try:
            return self.collection.count()
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not count collection: {exc}") from exc

    def list_papers(self) -> list[dict[str, Any]]:
        """Summarise the distinct papers in the store."""
        all_docs = self.get_all()
        metadatas = all_docs.get("metadatas") or []

        papers: dict[str, dict[str, Any]] = {}
        for metadata in metadatas:
            paper_id = metadata.get("paper_id", "unknown")
            if paper_id not in papers:
                papers[paper_id] = {
                    "paper_id": paper_id,
                    "title": metadata.get("title") or paper_id,
                    "author": metadata.get("author") or "Unknown",
                    "num_chunks": 0,
                }
            papers[paper_id]["num_chunks"] += 1
        return list(papers.values())

    def get_stats(self) -> dict[str, Any]:
        """Collection-level statistics, for the UI and health checks."""
        total_chunks = self.count()
        papers = self.list_papers()
        metadata = self.collection.metadata or {}

        stats: dict[str, Any] = {
            "total_chunks": total_chunks,
            "total_papers": len(papers),
            "collection_name": self.collection_name,
            "db_path": str(self.db_path),
            "embedding_model": metadata.get(_MODEL_KEY, "unknown"),
            "embedding_dim": metadata.get(_DIM_KEY),
        }
        if papers:
            stats["avg_chunks_per_paper"] = round(total_chunks / len(papers), 1)
        return stats


def _sanitise_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Coerce metadata into the scalar types Chroma accepts.

    Chroma rejects ``None`` and nested structures; unfiltered PDF metadata
    routinely contains both, which previously failed the whole insert.
    """
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
