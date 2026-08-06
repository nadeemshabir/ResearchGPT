"""Lexical retrieval with BM25.

BM25 complements dense retrieval on exact terms that embeddings blur together:
model names, metric names, and numbers. The index lives in memory and is
rebuilt from the vector store, so it must be refreshed after ingestion.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.ingestion.database import VectorDatabase
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Split on non-alphanumerics but keep intra-word hyphens, so "self-attention"
#: survives as one token rather than becoming "self" and "attention".
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenise text for BM25."""
    return _TOKEN_PATTERN.findall(text.lower())


class KeywordSearcher:
    """BM25 search over the indexed chunks."""

    def __init__(
        self,
        database: VectorDatabase | None = None,
        min_score: float | None = None,
    ):
        """
        Args:
            database: Reuse an open store. One is opened if omitted.
            min_score: Drop results scoring below this. Defaults to
                ``Settings.min_bm25_score``.
        """
        self.database = database or VectorDatabase()
        self.min_score = (
            min_score if min_score is not None else get_settings().min_bm25_score
        )

        self.corpus: list[str] = []
        self.corpus_metadata: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        #: Chunk count the index was built from, used to detect staleness.
        self._indexed_count = -1

        self.refresh()

    def refresh(self, force: bool = False) -> bool:
        """Rebuild the BM25 index from the vector store.

        The index is an in-memory snapshot, so papers ingested after
        construction are invisible until this runs. :meth:`search` calls it
        automatically when the collection size changes.

        Args:
            force: Rebuild even when the chunk count is unchanged.

        Returns:
            ``True`` if the index was rebuilt.
        """
        current_count = self.database.count()
        if not force and current_count == self._indexed_count:
            return False

        stored = self.database.get_all()
        documents = stored.get("documents") or []
        ids = stored.get("ids") or []
        metadatas = stored.get("metadatas") or []

        if not documents:
            self.corpus = []
            self.corpus_metadata = []
            self.bm25 = None
            self._indexed_count = 0
            logger.debug("BM25 index empty: no documents in collection")
            return True

        self.corpus = documents
        self.corpus_metadata = [
            {
                "id": ids[i] if i < len(ids) else f"unknown_{i}",
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }
            for i in range(len(documents))
        ]
        self.bm25 = BM25Okapi([tokenize(doc) for doc in documents])
        self._indexed_count = current_count

        logger.info("Built BM25 index over %d chunks", len(documents))
        return True

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return the ``top_k`` best BM25 matches.

        Args:
            query: Search query. Quoted phrases are honoured by
                :meth:`search_with_phrases`, not here.
            top_k: Maximum results.

        Returns:
            Results with ``id``, ``text``, ``metadata``, ``rank``, and
            ``bm25_score``. Empty when the query is blank or nothing scores
            above ``min_score``.
        """
        self.refresh()

        if self.bm25 is None:
            logger.debug("BM25 search skipped: index is empty")
            return []
        if not query or not query.strip():
            logger.warning("Empty query passed to keyword search")
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for index in top_indices:
            score = float(scores[index])
            if score < self.min_score:
                continue
            results.append(
                {
                    "rank": len(results) + 1,
                    "text": self.corpus[index],
                    "bm25_score": round(score, 4),
                    "id": self.corpus_metadata[index]["id"],
                    "metadata": self.corpus_metadata[index]["metadata"],
                }
            )

        logger.debug("BM25 returned %d results for %r", len(results), query[:60])
        return results

    def search_with_phrases(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """BM25 search that boosts chunks containing quoted phrases verbatim.

        Each matched phrase multiplies the score by 1.5, so a chunk containing
        both the terms and the exact phrase outranks one with the terms
        scattered.
        """
        phrases = re.findall(r'"([^"]+)"', query)
        # Over-fetch, because boosting reorders the candidate set.
        results = self.search(query, top_k=top_k * 2)

        if phrases:
            logger.debug("Boosting exact phrases: %s", phrases)
            for result in results:
                text_lower = result["text"].lower()
                boost = 1.0
                for phrase in phrases:
                    if phrase.lower() in text_lower:
                        boost += 0.5
                result["phrase_boost"] = round(boost, 2)
                result["bm25_score"] = round(result["bm25_score"] * boost, 4)
            results.sort(key=lambda r: r["bm25_score"], reverse=True)

        results = results[:top_k]
        for rank, result in enumerate(results, 1):
            result["rank"] = rank
        return results

    def get_term_frequencies(self, query: str) -> dict[str, dict[str, float]]:
        """Document frequency of each query term across the corpus.

        Useful for diagnosing why a query returned nothing: a term present in
        zero documents explains it immediately.
        """
        self.refresh()
        if not self.corpus:
            return {}

        tokenized_corpus = [set(tokenize(doc)) for doc in self.corpus]
        frequencies: dict[str, dict[str, float]] = {}
        for term in tokenize(query):
            count = sum(1 for doc_tokens in tokenized_corpus if term in doc_tokens)
            frequencies[term] = {
                "doc_frequency": count,
                "doc_percentage": round(count / len(self.corpus) * 100, 2),
            }
        return frequencies
