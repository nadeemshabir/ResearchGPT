"""Retrieval orchestration: query processing, hybrid search, reranking."""

from __future__ import annotations

import time
from typing import Any

from src.config import FusionMethod, NormalisationMethod, get_settings
from src.exceptions import NoRelevantContextError
from src.ingestion.database import VectorDatabase
from src.ingestion.embedder import EmbeddingGenerator
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.reranker import Reranker
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RetrievalSystem:
    """Full retrieval pipeline over the indexed corpus."""

    def __init__(
        self,
        use_reranking: bool | None = None,
        use_query_processing: bool | None = None,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
        fusion: FusionMethod | None = None,
        normalisation: NormalisationMethod | None = None,
        embedder: EmbeddingGenerator | None = None,
        database: VectorDatabase | None = None,
    ):
        """
        Args:
            use_reranking: Apply the cross-encoder. Defaults to
                ``Settings.use_reranking``. Disabling it skips loading the
                model entirely, which is the fast path for evaluation sweeps.
            use_query_processing: Clean and expand queries first.
            semantic_weight: Dense weight for hybrid fusion.
            keyword_weight: BM25 weight for hybrid fusion.
            fusion: ``"weighted"`` or ``"rrf"``. See :class:`HybridSearcher`.
            normalisation: Score normaliser for weighted fusion; ignored by RRF.
            embedder: Shared embedding model.
            database: Shared vector store.
        """
        settings = get_settings()
        self.use_reranking = use_reranking if use_reranking is not None else settings.use_reranking
        self.use_query_processing = (
            use_query_processing
            if use_query_processing is not None
            else settings.use_query_processing
        )

        logger.info(
            "Initialising retrieval (reranking=%s, query_processing=%s)",
            self.use_reranking,
            self.use_query_processing,
        )

        self.hybrid_searcher = HybridSearcher(
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            fusion=fusion,
            normalisation=normalisation,
            embedder=embedder,
            database=database,
        )
        self.query_processor = QueryProcessor() if self.use_query_processing else None
        self.reranker = Reranker() if self.use_reranking else None

        logger.info("Retrieval system ready")

    @property
    def database(self) -> VectorDatabase:
        """The underlying vector store."""
        return self.hybrid_searcher.semantic_searcher.database

    def search(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        process_query: bool | None = None,
        rerank: bool | None = None,
    ) -> dict[str, Any]:
        """Retrieve and rank chunks for ``query``.

        Args:
            query: User query.
            top_k: Final result count. Defaults to ``Settings.top_k_rerank``.
            candidate_k: Candidates fetched before reranking. Defaults to
                ``Settings.top_k_retrieve``.
            process_query: Override query processing for this call.
            rerank: Override reranking for this call.

        Returns:
            ``{"results": [...], "metadata": {...}}``. ``metadata`` records
            timings and the query transformations applied, so retrieval can be
            audited without re-running it.
        """
        settings = get_settings()
        top_k = top_k or settings.top_k_rerank
        candidate_k = candidate_k or settings.top_k_retrieve
        process_query = process_query if process_query is not None else self.use_query_processing
        rerank = rerank if rerank is not None else self.use_reranking

        metadata: dict[str, Any] = {
            "original_query": query,
            "search_query": query,
            "search_method": "hybrid",
            "reranked": False,
            "top_k": top_k,
            "candidate_k": candidate_k,
        }

        if not query or not query.strip():
            metadata["error"] = "empty query"
            return {"results": [], "metadata": metadata}

        started = time.perf_counter()
        search_query = query

        if process_query and self.query_processor:
            processed = self.query_processor.process_query(query)
            search_query = processed["cleaned"] or query
            metadata.update(
                {
                    "search_query": search_query,
                    "query_intent": processed["intent"],
                    "key_terms": processed["key_terms"],
                    "query_variations": processed["variations"],
                }
            )

        retrieve_started = time.perf_counter()
        results = self.hybrid_searcher.search(
            query=search_query,
            top_k=candidate_k if rerank else top_k,
            return_component_scores=True,
        )
        metadata["retrieval_ms"] = round((time.perf_counter() - retrieve_started) * 1000, 1)
        metadata["candidates_retrieved"] = len(results)

        if not results:
            metadata["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
            logger.info("No candidates retrieved for %r", query[:60])
            return {"results": [], "metadata": metadata}

        if rerank and self.reranker and len(results) > 1:
            rerank_started = time.perf_counter()
            # The raw query is used deliberately: the cross-encoder expects
            # natural language, not the keyword-stripped search query.
            results = self.reranker.rerank(query=query, results=results, top_k=top_k)
            metadata["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000, 1)
            metadata["reranked"] = True
        else:
            results = results[:top_k]

        metadata["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "Retrieved %d results for %r in %.0fms",
            len(results),
            query[:60],
            metadata["total_ms"],
        )
        return {"results": results, "metadata": metadata}

    def get_relevant_chunks(
        self,
        query: str,
        max_tokens: int | None = None,
        min_score: float | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Assemble an LLM context window from the best-matching chunks.

        Chunks are added in rank order until ``max_tokens`` would be exceeded.

        Args:
            query: User query.
            max_tokens: Context budget. Defaults to ``Settings.max_context_tokens``.
            min_score: Relevance floor. Defaults to ``Settings.min_rerank_score``
                when reranking is on, ``Settings.min_hybrid_score`` otherwise,
                since the two score scales are unrelated.
            top_k: Chunks to consider. Defaults to ``Settings.top_k_rerank``.

        Returns:
            ``{"context": str, "chunks": [...], "metadata": {...}}``.

        Raises:
            NoRelevantContextError: Nothing cleared the threshold. Raised rather
                than returning empty context, because an LLM given no context
                answers from memory instead of from the corpus.
        """
        settings = get_settings()
        max_tokens = max_tokens or settings.max_context_tokens

        if min_score is None:
            min_score = (
                settings.min_rerank_score if self.use_reranking else settings.min_hybrid_score
            )

        search_result = self.search(query, top_k=top_k or settings.top_k_rerank)
        results = search_result["results"]

        # Gate 1: is anything relevant at all?
        #
        # This must use an *absolute* score. Hybrid scores are min-max
        # normalised within the retrieved set, so the top result approaches 1.0
        # no matter how poor the match, and cannot distinguish "good answer"
        # from "best of nothing". Raw cosine similarity is comparable across
        # queries, so it can. When reranking is on, the cross-encoder logit
        # serves the same purpose and is the stronger signal.
        if self.use_reranking:
            best_absolute = max((r.get("rerank_score", 0.0) for r in results), default=0.0)
            absolute_floor = settings.min_rerank_score
            absolute_name = "rerank_score"
        else:
            best_absolute = max((r.get("similarity_score", 0.0) for r in results), default=0.0)
            absolute_floor = settings.min_semantic_similarity
            absolute_name = "similarity_score"

        if not results or best_absolute < absolute_floor:
            logger.info(
                "Refusing %r: best %s %.4f is below the %.4f floor",
                query[:60],
                absolute_name,
                best_absolute,
                absolute_floor,
            )
            raise NoRelevantContextError(
                query,
                candidates_considered=len(results),
                threshold=absolute_floor,
            )

        # Gate 2: which of the relevant results are worth including?
        # The fused score is the right signal here, since this is a ranking
        # question rather than a relevance question.
        score_key = "rerank_score" if self.use_reranking else "hybrid_score"
        relevant = [r for r in results if r.get(score_key, 0.0) >= min_score]

        if not relevant:
            raise NoRelevantContextError(
                query,
                candidates_considered=len(results),
                threshold=min_score,
            )

        selected: list[dict[str, Any]] = []
        total_tokens = 0
        for result in relevant:
            # Fall back to a 4-chars-per-token estimate when the stored count
            # is missing (older rows predate the metadata field).
            chunk_tokens = result["metadata"].get("num_tokens") or max(1, len(result["text"]) // 4)
            if total_tokens + chunk_tokens > max_tokens:
                continue
            selected.append(result)
            total_tokens += chunk_tokens

        if not selected:
            # Every relevant chunk individually exceeds the budget; take the
            # best one anyway rather than refusing outright.
            selected = relevant[:1]
            total_tokens = selected[0]["metadata"].get("num_tokens", 0)
            logger.warning(
                "All relevant chunks exceed the %d-token budget; using the top chunk only",
                max_tokens,
            )

        context = "\n\n---\n\n".join(
            f"[Source {index}: {chunk['metadata'].get('title', 'Unknown')}"
            f" | Section: {chunk['metadata'].get('section_title', 'Unknown')}]\n{chunk['text']}"
            for index, chunk in enumerate(selected, 1)
        )

        logger.info(
            "Assembled context from %d/%d relevant chunks (%d tokens)",
            len(selected),
            len(relevant),
            total_tokens,
        )

        return {
            "context": context,
            "chunks": selected,
            "metadata": {
                "query": query,
                "num_chunks": len(selected),
                "num_relevant": len(relevant),
                "num_candidates": len(results),
                "total_tokens": total_tokens,
                "min_score": min_score,
                "score_key": score_key,
                "retrieval": search_result["metadata"],
            },
        }

    def multi_query_search(
        self,
        query: str,
        top_k: int | None = None,
        num_variations: int = 3,
    ) -> dict[str, Any]:
        """Search several rephrasings and merge by mean score.

        Chunks retrieved by more than one variation are usually more robustly
        relevant, so ``found_in_variations`` is reported alongside the score.
        """
        settings = get_settings()
        top_k = top_k or settings.top_k_rerank

        variations = (
            self.query_processor.generate_variations(query, num_variations)
            if self.query_processor
            else [query]
        )

        merged: dict[str, dict[str, Any]] = {}
        for variation in variations:
            for result in self.hybrid_searcher.search(variation, top_k=top_k * 2):
                entry = merged.setdefault(result["id"], {"result": result, "scores": []})
                entry["scores"].append(result.get("hybrid_score", 0.0))

        final: list[dict[str, Any]] = []
        for entry in merged.values():
            result = entry["result"]
            result["multi_query_score"] = round(sum(entry["scores"]) / len(entry["scores"]), 4)
            result["found_in_variations"] = len(entry["scores"])
            final.append(result)

        final.sort(key=lambda r: (r["found_in_variations"], r["multi_query_score"]), reverse=True)
        final = final[:top_k]
        for rank, result in enumerate(final, 1):
            result["rank"] = rank

        return {
            "results": final,
            "metadata": {
                "query": query,
                "variations": variations,
                "unique_results": len(merged),
            },
        }

    def search_with_filters(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Dense search restricted by chunk metadata.

        Only the semantic searcher supports filtering; BM25 has no metadata
        index, so hybrid fusion is skipped here.
        """
        top_k = top_k or get_settings().top_k_rerank
        results = self.hybrid_searcher.semantic_searcher.search(
            query=query, top_k=top_k, filter_dict=filters
        )
        return {
            "results": results,
            "metadata": {
                "query": query,
                "filters": filters,
                "search_method": "semantic (filtered)",
                "num_results": len(results),
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Retrieval configuration and corpus size."""
        return {
            "num_chunks": self.database.count(),
            "num_papers": len(self.database.list_papers()),
            "use_reranking": self.use_reranking,
            "use_query_processing": self.use_query_processing,
            "semantic_weight": round(self.hybrid_searcher.semantic_weight, 3),
            "keyword_weight": round(self.hybrid_searcher.keyword_weight, 3),
            "fusion": self.hybrid_searcher.fusion,
            "normalisation": self.hybrid_searcher.normalisation,
            "rrf_k": self.hybrid_searcher.rrf_k,
            "reranker_model": self.reranker.model_name if self.reranker else None,
        }
