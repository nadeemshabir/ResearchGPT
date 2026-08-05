"""Dense retrieval over chunk embeddings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.ingestion.database import VectorDatabase
from src.ingestion.embedder import EmbeddingGenerator
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SemanticSearcher:
    """Find chunks whose embeddings are nearest the query embedding."""

    def __init__(
        self,
        embedder: EmbeddingGenerator | None = None,
        database: VectorDatabase | None = None,
    ):
        """
        Args:
            embedder: Reuse a loaded model. One is created if omitted.
            database: Reuse an open store. One is opened if omitted.
        """
        self.embedder = embedder or EmbeddingGenerator()
        self.database = database or VectorDatabase()
        logger.debug("Semantic search ready over %d chunks", self.database.count())

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the ``top_k`` nearest chunks.

        Args:
            query: Natural-language query.
            top_k: Maximum results.
            filter_dict: Optional Chroma metadata filter.

        Returns:
            Results carrying ``id``, ``text``, ``metadata``, ``rank``,
            ``distance``, and ``similarity_score`` (0-1, higher is better).
            Empty when the query is blank or the collection is empty.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to semantic search")
            return []

        query_embedding = self.embedder.generate_single_embedding(query)
        raw = self.database.query(
            query_embedding=query_embedding, n_results=top_k, filter_dict=filter_dict
        )
        results = self._format_results(raw)

        logger.debug("Semantic search returned %d results for %r", len(results), query[:60])
        return results

    @staticmethod
    def _format_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten Chroma's nested response into a list of result dicts."""
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]

        results: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            result: dict[str, Any] = {
                "id": ids[index] if index < len(ids) else f"unknown_{index}",
                "text": document,
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "rank": index + 1,
            }
            if index < len(distances):
                distance = distances[index]
                # Chroma returns squared L2 distance (lower is better). Map it
                # to a bounded 0-1 similarity so it can be fused with BM25.
                result["distance"] = round(float(distance), 4)
                result["similarity_score"] = round(1.0 / (1.0 + float(distance)), 4)
            results.append(result)
        return results

    def search_with_context(
        self,
        query: str,
        top_k: int = 10,
        context_window: int = 1,
    ) -> list[dict[str, Any]]:
        """Search, then attach the neighbouring chunks of each hit.

        Useful when a match lands mid-argument and the surrounding text is
        needed to make the passage self-contained.
        """
        results = self.search(query, top_k=top_k)

        for result in results:
            paper_id = result["metadata"].get("paper_id")
            chunk_id = result["metadata"].get("chunk_id")
            if paper_id is None or chunk_id is None:
                result["context_before"] = []
                result["context_after"] = []
                continue

            neighbours = self._get_surrounding_chunks(paper_id, chunk_id, context_window)
            result["context_before"] = neighbours["before"]
            result["context_after"] = neighbours["after"]

        return results

    def _get_surrounding_chunks(
        self, paper_id: str, chunk_id: int, window: int
    ) -> dict[str, list[str]]:
        """Return up to ``window`` chunk texts either side of ``chunk_id``."""
        stored = self.database.get_by_paper_id(paper_id)
        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []

        ordered = sorted(
            zip(documents, metadatas, strict=False), key=lambda pair: pair[1].get("chunk_id", 0)
        )

        target_index: int | None = None
        for index, (_, metadata) in enumerate(ordered):
            if metadata.get("chunk_id") == chunk_id:
                target_index = index
                break

        if target_index is None:
            return {"before": [], "after": []}

        before = [text for text, _ in ordered[max(0, target_index - window) : target_index]]
        after = [
            text
            for text, _ in ordered[target_index + 1 : target_index + 1 + window]
        ]
        return {"before": before, "after": after}

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 5,
        aggregate: str = "max",
    ) -> list[dict[str, Any]]:
        """Search several phrasings and merge the hits.

        Args:
            queries: Query variations.
            top_k: Results per query, and the size of the merged output.
            aggregate: ``"max"``, ``"mean"``, or ``"sum"`` over per-query scores.

        Returns:
            Deduplicated results ranked by aggregated score, each carrying
            ``aggregated_score`` and ``num_hits``.
        """
        if aggregate not in ("max", "mean", "sum"):
            raise ValueError(f"aggregate must be 'max', 'mean', or 'sum', got {aggregate!r}")

        merged: dict[str, dict[str, Any]] = {}
        for query in queries:
            for result in self.search(query, top_k=top_k):
                entry = merged.setdefault(
                    result["id"], {"result": result, "scores": []}
                )
                entry["scores"].append(result.get("similarity_score", 0.0))

        aggregators: dict[str, Callable[[list[float]], float]] = {
            "max": max,
            "mean": lambda scores: sum(scores) / len(scores),
            "sum": sum,
        }
        aggregator = aggregators[aggregate]

        final: list[dict[str, Any]] = []
        for entry in merged.values():
            result = entry["result"]
            result["aggregated_score"] = round(aggregator(entry["scores"]), 4)
            result["num_hits"] = len(entry["scores"])
            final.append(result)

        final.sort(key=lambda r: r["aggregated_score"], reverse=True)
        for rank, result in enumerate(final, 1):
            result["rank"] = rank

        return final[:top_k]
