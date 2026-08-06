"""Tests for score fusion, against hand-computed values.

The fusion maths decides the final ranking, and a mistake there is invisible:
results still come back, just in a worse order. The BEIR sweep gives indirect
evidence it is correct -- weight 0.0 and 1.0 reproduce pure BM25 and pure dense
exactly -- but that check takes twenty minutes and needs a downloaded corpus.
These run in milliseconds.

Sub-searchers are never constructed. `HybridSearcher._combine*` and `_normalise`
are pure functions of their inputs, so they are called directly on a searcher
built with stub components.
"""

from typing import Any

import pytest

from src.retrieval.hybrid_search import HybridSearcher


class _StubSearcher:
    """Stands in for a real searcher so no model or database is loaded."""

    def search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def refresh(self) -> None:
        pass


def make_searcher(**kwargs: Any) -> HybridSearcher:
    return HybridSearcher(
        semantic_searcher=_StubSearcher(),  # type: ignore[arg-type]
        keyword_searcher=_StubSearcher(),  # type: ignore[arg-type]
        **kwargs,
    )


def dense(*pairs: tuple[str, float]) -> list[dict[str, Any]]:
    return [{"id": i, "similarity_score": s} for i, s in pairs]


def bm25(*pairs: tuple[str, float]) -> list[dict[str, Any]]:
    return [{"id": i, "bm25_score": s} for i, s in pairs]


def scores(fused: list[dict[str, Any]]) -> dict[str, float]:
    return {r["id"]: r["hybrid_score"] for r in fused}


# --- normalisation ----------------------------------------------------------


def test_minmax_maps_to_zero_and_one() -> None:
    searcher = make_searcher(normalisation="minmax")

    assert searcher._normalise([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_minmax_on_identical_scores_keeps_them_in_contention() -> None:
    """Returning 0.0 would delete this retriever's signal from the fusion.

    Every candidate being equally good is information -- "all of these match" --
    not an absence of information.
    """
    searcher = make_searcher(normalisation="minmax")

    assert searcher._normalise([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]


def test_minmax_cannot_judge_absolute_relevance() -> None:
    """The property behind finding #8, pinned so it is not forgotten.

    minmax rescales *within* the retrieved set, so the top result normalises to
    1.0 whether it is an excellent match or a terrible one. Any refusal gate
    built on a normalised score is therefore broken by construction -- which is
    why the absolute gate reads raw cosine instead.
    """
    searcher = make_searcher(normalisation="minmax")

    excellent = searcher._normalise([0.95, 0.60, 0.30])
    terrible = searcher._normalise([0.11, 0.07, 0.02])

    assert excellent[0] == terrible[0] == 1.0


def test_sum_normalisation_produces_a_distribution() -> None:
    searcher = make_searcher(normalisation="sum")

    result = searcher._normalise([1.0, 3.0])

    assert result == [0.25, 0.75]
    assert sum(result) == pytest.approx(1.0)


def test_sum_normalisation_survives_non_positive_totals() -> None:
    """BM25 scores can be zero or negative; dividing by the sum would blow up."""
    searcher = make_searcher(normalisation="sum")

    assert searcher._normalise([0.0, 0.0]) == [0.0, 0.0]


def test_none_normalisation_passes_scores_through() -> None:
    searcher = make_searcher(normalisation="none")

    assert searcher._normalise([0.3, 7.5]) == [0.3, 7.5]


def test_normalising_an_empty_list_returns_empty() -> None:
    assert make_searcher()._normalise([]) == []


# --- weighted fusion --------------------------------------------------------


def test_weighted_fusion_matches_hand_computation() -> None:
    """minmax of [0.9, 0.5] is [1, 0]; of [8, 4] is [1, 0]. Weights 0.6/0.4.

    a: 1.0*0.6 + 0.0*0.4 = 0.6
    b: 0.0*0.6 + 1.0*0.4 = 0.4
    """
    searcher = make_searcher(normalisation="minmax")

    fused = searcher._combine(
        dense(("a", 0.9), ("b", 0.5)), bm25(("b", 8.0), ("a", 4.0)), 0.6, 0.4, False
    )

    assert scores(fused) == {"a": 0.6, "b": 0.4}


def test_a_document_found_by_only_one_retriever_scores_zero_for_the_other() -> None:
    searcher = make_searcher(normalisation="minmax")

    fused = searcher._combine(dense(("a", 0.9)), bm25(("b", 8.0)), 0.5, 0.5, False)

    # Single-element lists minmax to [1.0], so each contributes its own weight.
    assert scores(fused) == {"a": 0.5, "b": 0.5}


def test_full_semantic_weight_reproduces_the_dense_ranking() -> None:
    """The endpoint check that validated the fusion code during the sweep."""
    searcher = make_searcher(normalisation="minmax")

    fused = searcher._combine(
        dense(("a", 0.9), ("b", 0.5), ("c", 0.1)),
        bm25(("c", 9.0), ("b", 5.0), ("a", 1.0)),
        1.0,
        0.0,
        False,
    )
    order = [r["id"] for r in sorted(fused, key=lambda r: -r["hybrid_score"])]

    assert order == ["a", "b", "c"]


def test_full_keyword_weight_reproduces_the_bm25_ranking() -> None:
    searcher = make_searcher(normalisation="minmax")

    fused = searcher._combine(
        dense(("a", 0.9), ("b", 0.5), ("c", 0.1)),
        bm25(("c", 9.0), ("b", 5.0), ("a", 1.0)),
        0.0,
        1.0,
        False,
    )
    order = [r["id"] for r in sorted(fused, key=lambda r: -r["hybrid_score"])]

    assert order == ["c", "b", "a"]


def test_agreement_between_retrievers_beats_a_single_strong_hit() -> None:
    searcher = make_searcher(normalisation="minmax")

    fused = searcher._combine(
        dense(("both", 0.9), ("dense_only", 0.85)),
        bm25(("both", 9.0), ("bm25_only", 8.5)),
        0.5,
        0.5,
        False,
    )

    assert scores(fused)["both"] > scores(fused)["dense_only"]


def test_component_scores_are_attached_when_requested() -> None:
    searcher = make_searcher(normalisation="minmax")

    fused = searcher._combine(
        dense(("a", 0.9), ("b", 0.5)), bm25(("a", 8.0), ("b", 4.0)), 0.5, 0.5, True
    )

    assert "semantic_score_normalized" in fused[0]
    assert "keyword_score_normalized" in fused[0]


def test_fusion_of_two_empty_lists_returns_nothing() -> None:
    assert make_searcher()._combine([], [], 0.5, 0.5, False) == []


# --- reciprocal rank fusion -------------------------------------------------


def test_rrf_scores_a_document_ranked_first_by_both_as_one() -> None:
    """The rescale by (k+1) exists to make this exactly 1.0.

    Raw RRF tops out near 0.016 at k=60, which would put every result below the
    default `min_hybrid_score` of 0.2 and silently discard the entire list.
    """
    searcher = make_searcher(fusion="rrf", rrf_k=60)

    fused = searcher._combine(dense(("a", 0.9)), bm25(("a", 9.0)), 0.5, 0.5, False)

    assert scores(fused)["a"] == pytest.approx(1.0)


def test_rrf_ignores_score_magnitude() -> None:
    """The defining property: rank 1 is rank 1 however confident the retriever."""
    searcher = make_searcher(fusion="rrf", rrf_k=60)

    confident = searcher._combine(dense(("a", 0.99)), bm25(("a", 99.0)), 0.5, 0.5, False)
    hesitant = searcher._combine(dense(("a", 0.01)), bm25(("a", 0.1)), 0.5, 0.5, False)

    assert scores(confident) == scores(hesitant)


def test_rrf_matches_hand_computation() -> None:
    """k=10, weights 0.5/0.5, scale 11.

    b: dense rank 2 -> 0.5/12; bm25 rank 1 -> 0.5/11.  (0.0416667+0.0454545)*11
    """
    searcher = make_searcher(fusion="rrf", rrf_k=10)

    fused = searcher._combine(
        dense(("a", 0.9), ("b", 0.5)), bm25(("b", 8.0), ("a", 4.0)), 0.5, 0.5, False
    )
    expected_b = (0.5 / 12 + 0.5 / 11) * 11

    assert scores(fused)["b"] == pytest.approx(expected_b, abs=1e-6)


def test_rrf_ranks_a_document_found_by_both_above_one_found_by_one() -> None:
    searcher = make_searcher(fusion="rrf", rrf_k=60)

    fused = searcher._combine(
        dense(("both", 0.9), ("dense_only", 0.89)),
        bm25(("both", 9.0), ("bm25_only", 8.9)),
        0.5,
        0.5,
        False,
    )

    assert scores(fused)["both"] > scores(fused)["dense_only"]
    assert scores(fused)["both"] > scores(fused)["bm25_only"]


def test_rrf_uses_list_position_not_a_stale_rank_field() -> None:
    """A caller that reorders results may leave `rank` behind; position wins."""
    searcher = make_searcher(fusion="rrf", rrf_k=60)
    semantic = [
        {"id": "first", "similarity_score": 0.9, "rank": 99},
        {"id": "second", "similarity_score": 0.5, "rank": 1},
    ]

    fused = searcher._combine(semantic, [], 1.0, 0.0, False)

    assert scores(fused)["first"] > scores(fused)["second"]


def test_smaller_rrf_k_sharpens_the_gap_between_ranks() -> None:
    """k damps the rank discount; the sweep found smaller k slightly better."""
    fused_small = make_searcher(fusion="rrf", rrf_k=5)._combine(
        dense(("a", 0.9), ("b", 0.5)), [], 1.0, 0.0, False
    )
    fused_large = make_searcher(fusion="rrf", rrf_k=200)._combine(
        dense(("a", 0.9), ("b", 0.5)), [], 1.0, 0.0, False
    )

    gap_small = scores(fused_small)["a"] - scores(fused_small)["b"]
    gap_large = scores(fused_large)["a"] - scores(fused_large)["b"]

    assert gap_small > gap_large


def test_rrf_reports_which_retrievers_found_each_document() -> None:
    searcher = make_searcher(fusion="rrf", rrf_k=60)

    fused = searcher._combine(
        dense(("both", 0.9), ("dense_only", 0.5)), bm25(("both", 9.0)), 0.5, 0.5, True
    )
    by_id = {r["id"]: r for r in fused}

    assert by_id["both"]["found_by_both"] is True
    assert by_id["dense_only"]["found_by_both"] is False
    assert by_id["dense_only"]["keyword_rank"] is None
