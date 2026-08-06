"""Cross-encoder reranking.

Bi-encoders embed query and document separately, so they miss fine-grained
interactions. A cross-encoder scores the pair jointly, which is far more
accurate and far more expensive: it is applied to a shortlist, never the corpus.

Scores are unbounded logits, roughly -11 to +11 for the ms-marco models. They
are not probabilities and are not comparable across models.
"""

from __future__ import annotations

from typing import Any

from src.config import get_settings
from src.exceptions import RetrievalError
from src.utils.device import resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Reranker:
    """Reorder a candidate list with a cross-encoder."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        """
        Args:
            model_name: Cross-encoder id. Defaults to ``Settings.reranker_model``.
            device: Torch device. Auto-detected if omitted.

        Raises:
            RetrievalError: The model could not be loaded.
        """
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.device = resolve_device(device or settings.reranker_device)

        logger.info("Loading reranker %s on %s", self.model_name, self.device)
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Could not load reranker {self.model_name!r}: {exc}") from exc
        logger.info("Reranker ready")

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
        score_key: str = "rerank_score",
    ) -> list[dict[str, Any]]:
        """Score every candidate against ``query`` and reorder by that score.

        Args:
            query: The original user query. Use the raw query here, not a
                processed variant: the cross-encoder was trained on natural
                language pairs and degrades on keyword-stripped input.
            results: Candidates, each carrying ``text``.
            top_k: Truncate after reordering. ``None`` keeps all.
            score_key: Field the score is written to.

        Returns:
            A new list ordered by descending score. Each result also gains
            ``original_rank`` so the reordering can be inspected.
        """
        if not results:
            return []

        pairs = [[query, result.get("text", "")] for result in results]
        try:
            scores = self.model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Reranking failed: {exc}") from exc

        # Copy so the caller's list order is not mutated underneath them.
        reranked = list(results)
        for result, score in zip(reranked, scores, strict=True):
            result[score_key] = round(float(score), 4)
            result.setdefault(
                "original_score",
                result.get("hybrid_score", result.get("similarity_score", 0.0)),
            )
            result.setdefault("original_rank", result.get("rank"))

        reranked.sort(key=lambda r: r[score_key], reverse=True)
        for rank, result in enumerate(reranked, 1):
            result["rank"] = rank

        if top_k is not None:
            reranked = reranked[:top_k]

        logger.debug("Reranked %d candidates, returning %d", len(results), len(reranked))
        return reranked

    def rerank_with_threshold(
        self,
        query: str,
        results: list[dict[str, Any]],
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank, then drop anything scoring below ``threshold``.

        Args:
            threshold: Minimum score. Defaults to ``Settings.min_rerank_score``.
                Because scores are unbounded logits, a sensible floor is
                negative; a threshold of 0.5 would discard nearly everything.
        """
        if threshold is None:
            threshold = get_settings().min_rerank_score

        reranked = self.rerank(query, results)
        kept = [r for r in reranked if r.get("rerank_score", 0.0) >= threshold]

        logger.debug(
            "Threshold %.2f kept %d/%d reranked results", threshold, len(kept), len(reranked)
        )
        return kept[:top_k] if top_k is not None else kept

    @staticmethod
    def score_summary(results: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate rerank scores, for diagnostics and evaluation reporting."""
        scores = [r["rerank_score"] for r in results if "rerank_score" in r]
        if not scores:
            return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

        ordered = sorted(scores)
        return {
            "count": len(scores),
            "min": round(ordered[0], 4),
            "max": round(ordered[-1], 4),
            "mean": round(sum(scores) / len(scores), 4),
            "median": round(ordered[len(ordered) // 2], 4),
        }
