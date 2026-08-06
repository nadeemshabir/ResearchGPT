"""Hybrid retrieval: fusion of dense and BM25 result lists.

Two strategies are implemented, because merging two rankings is not a solved
problem and the right choice is corpus-dependent:

**Weighted score fusion** normalises each retriever's scores onto a common
scale, then takes a weighted sum. It preserves how *confident* each retriever
was, but requires choosing a normaliser, and dense cosine (bounded 0-1) and
BM25 (unbounded) do not rescale onto each other cleanly.

**Reciprocal Rank Fusion** (Cormack et al., 2009) ignores scores entirely and
combines rank positions:  ``score(d) = sum_r w_r / (k + rank_r(d))``.
Being scale-free, it sidesteps normalisation altogether, which is why it is the
default in Elasticsearch, OpenSearch, Weaviate, and Qdrant. The cost is that it
is blind to confidence: a retriever that puts one document far ahead of the
field cannot say so, because rank 1 is rank 1 either way.

Neither has been benchmarked on this corpus. That is Milestone 2's job.
"""

from __future__ import annotations

from typing import Any

from src.config import FusionMethod, NormalisationMethod, get_settings
from src.ingestion.database import VectorDatabase
from src.ingestion.embedder import EmbeddingGenerator
from src.retrieval.keyword_search import KeywordSearcher
from src.retrieval.semantic_search import SemanticSearcher
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Terms suggesting the user wants exact lexical matching.
_TECHNICAL_TERMS = frozenset(
    {
        "algorithm",
        "model",
        "equation",
        "function",
        "method",
        "architecture",
        "training",
        "optimization",
        "optimisation",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "parameter",
        "benchmark",
        "dataset",
    }
)

#: Terms suggesting the user wants meaning-based matching.
_CONCEPTUAL_TERMS = frozenset(
    {
        "what",
        "why",
        "how",
        "explain",
        "describe",
        "compare",
        "difference",
        "relationship",
        "concept",
        "idea",
        "intuition",
        "motivation",
    }
)


class HybridSearcher:
    """Combine semantic and keyword retrieval under configurable weights."""

    def __init__(
        self,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
        fusion: FusionMethod | None = None,
        normalisation: NormalisationMethod | None = None,
        rrf_k: int | None = None,
        semantic_searcher: SemanticSearcher | None = None,
        keyword_searcher: KeywordSearcher | None = None,
        embedder: EmbeddingGenerator | None = None,
        database: VectorDatabase | None = None,
    ):
        """
        Args:
            semantic_weight: Dense weight. Defaults to ``Settings.semantic_weight``.
            keyword_weight: BM25 weight. Defaults to ``Settings.keyword_weight``.
            fusion: How the two result lists are merged. Defaults to
                ``Settings.fusion_method``.

                - ``"weighted"`` (default) normalises scores, then takes a
                  weighted sum. Preserves each retriever's confidence.
                - ``"rrf"`` fuses rank positions and ignores scores. Needs no
                  normalisation and no score calibration, but cannot express
                  that one retriever was far more certain than the other.
            normalisation: How component scores are rescaled before *weighted*
                fusion. Ignored when ``fusion="rrf"``. Defaults to
                ``Settings.normalisation``. No option here is benchmarked yet.

                - ``"minmax"`` (default) rescales each result set to span 0-1.
                  Conventional, but it always maps the worst candidate in a set
                  to exactly 0, discarding the signal that it was retrieved at
                  all. Kept as the default because it is what this system has
                  always used.
                - ``"sum"`` divides by the total, preserving relative gaps
                  between candidates and never zeroing a retrieved result.
                - ``"none"`` fuses raw scores, letting unbounded BM25 dominate
                  bounded cosine similarity. Useful only for ablations.
            rrf_k: RRF damping constant, default 60. Only used when
                ``fusion="rrf"``.
            semantic_searcher: Reuse an existing dense searcher.
            keyword_searcher: Reuse an existing BM25 searcher.
            embedder: Shared embedding model, if constructing searchers here.
            database: Shared vector store, if constructing searchers here.

        Raises:
            ValueError: Weights are out of range or both zero.
        """
        settings = get_settings()
        semantic_weight = (
            semantic_weight if semantic_weight is not None else settings.semantic_weight
        )
        keyword_weight = keyword_weight if keyword_weight is not None else settings.keyword_weight

        for name, value in (
            ("semantic_weight", semantic_weight),
            ("keyword_weight", keyword_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")

        total = semantic_weight + keyword_weight
        if total <= 0:
            raise ValueError("semantic_weight and keyword_weight cannot both be zero")

        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total
        self.fusion: FusionMethod = fusion or settings.fusion_method
        self.normalisation: NormalisationMethod = normalisation or settings.normalisation
        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

        if self.rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {self.rrf_k}")

        # Share one store and one embedding model across both searchers rather
        # than opening the store twice and loading the model twice.
        #
        # The store is opened lazily: when a caller supplies both searchers
        # there is nothing left to build, and opening a default VectorDatabase
        # anyway would connect to the *application's* collection, which is both
        # wasteful and wrong when the caller is pointing at another index.
        if semantic_searcher is None:
            self.semantic_searcher = SemanticSearcher(
                embedder=embedder, database=database or VectorDatabase()
            )
        else:
            self.semantic_searcher = semantic_searcher

        self.keyword_searcher = keyword_searcher or KeywordSearcher(
            database=self.semantic_searcher.database
        )

        logger.debug(
            "Hybrid search ready (fusion=%s, semantic=%.2f, keyword=%.2f, normalisation=%s)",
            self.fusion,
            self.semantic_weight,
            self.keyword_weight,
            self.normalisation,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int | None = None,
        return_component_scores: bool = False,
        weights: tuple[float, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve, fuse, and rank.

        Args:
            query: Search query.
            top_k: Results to return.
            retrieval_k: Candidates to pull from each searcher. Defaults to
                ``top_k * 2``, so fusion has room to reorder.
            return_component_scores: Attach the normalised per-method scores.
            weights: ``(semantic, keyword)`` overriding the instance weights for
                this call only. Passed explicitly rather than mutating
                ``self``, so concurrent callers cannot corrupt each other.

        Returns:
            Results ordered by ``hybrid_score``.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to hybrid search")
            return []

        retrieval_k = retrieval_k or top_k * 2
        semantic_w, keyword_w = weights or (self.semantic_weight, self.keyword_weight)

        semantic_results = self.semantic_searcher.search(query, top_k=retrieval_k)
        keyword_results = self.keyword_searcher.search(query, top_k=retrieval_k)

        fused = self._combine(
            semantic_results,
            keyword_results,
            semantic_w,
            keyword_w,
            return_component_scores,
        )
        fused.sort(key=lambda r: r["hybrid_score"], reverse=True)

        final = fused[:top_k]
        for rank, result in enumerate(final, 1):
            result["rank"] = rank

        logger.debug(
            "Hybrid search: %d semantic + %d keyword -> %d fused, returning %d",
            len(semantic_results),
            len(keyword_results),
            len(fused),
            len(final),
        )
        return final

    def _combine(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float,
        return_components: bool,
    ) -> list[dict[str, Any]]:
        """Merge the two result sets on chunk id, using the configured strategy."""
        if self.fusion == "rrf":
            return self._combine_rrf(
                semantic_results,
                keyword_results,
                semantic_weight,
                keyword_weight,
                return_components,
            )
        return self._combine_weighted(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight,
            return_components,
        )

    def _combine_rrf(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float,
        return_components: bool,
    ) -> list[dict[str, Any]]:
        """Fuse by rank position: ``sum_r w_r / (k + rank_r)``.

        A document missing from one retriever's list contributes nothing from
        that retriever, rather than being assigned a worst-case rank. This is
        standard RRF and it is why a document found by both retrievers reliably
        outranks one found by a single retriever, however confidently.

        The raw sum is rescaled by ``k + 1`` so that a document ranked first by
        both retrievers scores exactly 1.0. That is a monotonic transform, so
        the ordering is identical to raw RRF; it exists only so that
        ``hybrid_score`` keeps the same 0-1 meaning as the weighted path and
        the ``min_hybrid_score`` threshold stays valid. Without it, raw RRF
        scores top out near 0.016 and the default 0.2 threshold would silently
        discard every result.
        """
        k = self.rrf_k
        combined: dict[str, dict[str, Any]] = {}

        for weight, results, rank_key in (
            (semantic_weight, semantic_results, "semantic_rank"),
            (keyword_weight, keyword_results, "keyword_rank"),
        ):
            for position, result in enumerate(results, 1):
                # Trust enumeration order rather than a 'rank' field, which a
                # caller may have left stale after reordering.
                entry = combined.setdefault(
                    result["id"],
                    {"result": result, "rrf": 0.0, "semantic_rank": None, "keyword_rank": None},
                )
                entry["rrf"] += weight / (k + position)
                entry[rank_key] = position

        scale = k + 1
        fused: list[dict[str, Any]] = []
        for entry in combined.values():
            result = entry["result"]
            result["hybrid_score"] = round(entry["rrf"] * scale, 6)
            if return_components:
                result["semantic_rank"] = entry["semantic_rank"]
                result["keyword_rank"] = entry["keyword_rank"]
                result["found_by_both"] = (
                    entry["semantic_rank"] is not None and entry["keyword_rank"] is not None
                )
            fused.append(result)

        return fused

    def _combine_weighted(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float,
        return_components: bool,
    ) -> list[dict[str, Any]]:
        """Fuse by weighted sum of normalised scores."""
        semantic_scores = self._normalise(
            [r.get("similarity_score", 0.0) for r in semantic_results]
        )
        keyword_scores = self._normalise([r.get("bm25_score", 0.0) for r in keyword_results])

        combined: dict[str, dict[str, Any]] = {}

        for result, score in zip(semantic_results, semantic_scores, strict=True):
            combined[result["id"]] = {
                "result": result,
                "semantic_score": score,
                "keyword_score": 0.0,
            }

        for result, score in zip(keyword_results, keyword_scores, strict=True):
            if result["id"] in combined:
                combined[result["id"]]["keyword_score"] = score
            else:
                combined[result["id"]] = {
                    "result": result,
                    "semantic_score": 0.0,
                    "keyword_score": score,
                }

        fused: list[dict[str, Any]] = []
        for entry in combined.values():
            result = entry["result"]
            result["hybrid_score"] = round(
                entry["semantic_score"] * semantic_weight + entry["keyword_score"] * keyword_weight,
                6,
            )
            if return_components:
                result["semantic_score_normalized"] = round(entry["semantic_score"], 4)
                result["keyword_score_normalized"] = round(entry["keyword_score"], 4)
            fused.append(result)

        return fused

    def _normalise(self, scores: list[float]) -> list[float]:
        """Rescale raw component scores per ``self.normalisation``."""
        if not scores:
            return []

        if self.normalisation == "none":
            return list(scores)

        if self.normalisation == "sum":
            total = sum(scores)
            if total <= 0:
                return [0.0] * len(scores)
            return [score / total for score in scores]

        # minmax
        lowest, highest = min(scores), max(scores)
        if highest == lowest:
            # Every candidate is equally good; 1.0 keeps them all in contention,
            # whereas 0.0 would silently delete this signal from the fusion.
            return [1.0] * len(scores)
        return [(score - lowest) / (highest - lowest) for score in scores]

    def adaptive_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search with weights chosen from the query's surface features.

        Technical, jargon-heavy queries lean on BM25; open-ended conceptual
        questions lean on embeddings. The chosen weights are passed per-call,
        leaving instance state untouched.
        """
        features = self.analyse_query(query)

        if features["has_technical_terms"] and not features["is_conceptual"]:
            weights = (0.4, 0.6)
        elif features["is_conceptual"] and not features["has_technical_terms"]:
            weights = (0.8, 0.2)
        else:
            weights = (0.6, 0.4)

        logger.debug("Adaptive weights for %r: semantic=%.2f keyword=%.2f", query[:60], *weights)
        results = self.search(query, top_k=top_k, weights=weights)
        for result in results:
            result["adaptive_weights"] = {"semantic": weights[0], "keyword": weights[1]}
        return results

    @staticmethod
    def analyse_query(query: str) -> dict[str, Any]:
        """Surface features of a query, used to pick adaptive weights."""
        tokens = set(query.lower().split())
        return {
            "has_technical_terms": bool(tokens & _TECHNICAL_TERMS),
            "is_conceptual": bool(tokens & _CONCEPTUAL_TERMS),
            "length": len(query.split()),
            "has_quotes": '"' in query,
        }
