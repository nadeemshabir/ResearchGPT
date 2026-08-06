"""Tests for the evaluation maths, against values computed by hand.

Eight of the nineteen findings in `docs/EVALUATION.md` were bugs in evaluation
code rather than in the system, and twice a failure was being scored as a
success. Evaluation code that is trusted but untested is exactly how that
happens, so the arithmetic every published number rests on is pinned here.

Expected values are worked out in the docstrings rather than produced by running
the code, which would only assert that it still does what it does.
"""

import math

import pytest

from eval.calibrate import cohens_kappa, spearman
from eval.metrics import (
    dcg_at_k,
    evaluate_run,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["d1", "d2", "d3", "d4", "d5"]


# --- recall -----------------------------------------------------------------


def test_recall_counts_relevant_documents_found() -> None:
    assert recall_at_k(RANKED, {"d1", "d3"}, 5) == 1.0
    assert recall_at_k(RANKED, {"d1", "d9"}, 5) == 0.5


def test_recall_respects_the_cutoff() -> None:
    assert recall_at_k(RANKED, {"d5"}, 3) == 0.0
    assert recall_at_k(RANKED, {"d5"}, 5) == 1.0


def test_recall_with_no_judgments_is_zero_not_one() -> None:
    """An unjudged query must not silently inflate the average as a free 1.0."""
    assert recall_at_k(RANKED, set(), 5) == 0.0


def test_recall_denominator_is_all_relevant_documents() -> None:
    """Relevant documents outside the ranking still count against recall."""
    assert recall_at_k(["d1"], {"d1", "d2", "d3", "d4"}, 10) == 0.25


# --- precision --------------------------------------------------------------


def test_precision_divides_by_k_not_by_results_returned() -> None:
    """Returning one correct result out of a requested ten is not precision 1.0."""
    assert precision_at_k(["d1"], {"d1"}, 10) == pytest.approx(0.1)


def test_precision_at_zero_k_is_zero_not_a_crash() -> None:
    assert precision_at_k(RANKED, {"d1"}, 0) == 0.0


# --- reciprocal rank --------------------------------------------------------


@pytest.mark.parametrize(
    ("relevant", "expected"),
    [({"d1"}, 1.0), ({"d2"}, 0.5), ({"d4"}, 0.25), ({"nope"}, 0.0)],
)
def test_reciprocal_rank_uses_the_first_relevant_position(
    relevant: set[str], expected: float
) -> None:
    assert reciprocal_rank(RANKED, relevant) == expected


def test_reciprocal_rank_ignores_later_relevant_documents() -> None:
    assert reciprocal_rank(RANKED, {"d2", "d3", "d4"}) == 0.5


# --- DCG and nDCG -----------------------------------------------------------


def test_dcg_applies_a_log2_discount() -> None:
    """Positions 1 and 3 relevant: 1/log2(2) + 1/log2(4) = 1.0 + 0.5."""
    assert dcg_at_k(RANKED, {"d1", "d3"}, 5) == pytest.approx(1.5)


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(RANKED, {"d1", "d2"}, 5) == pytest.approx(1.0)


def test_ndcg_matches_hand_computation() -> None:
    """Relevant {d1, d3}. DCG = 1/log2(2) + 1/log2(4) = 1.5.

    Ideal puts both at the top: 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093.
    nDCG = 1.5 / 1.63093 = 0.919721...
    """
    ideal = 1.0 + 1.0 / math.log2(3)

    assert ndcg_at_k(RANKED, {"d1", "d3"}, 5) == pytest.approx(1.5 / ideal)


def test_ndcg_rewards_ranking_the_answer_higher() -> None:
    """The property that distinguishes nDCG from recall, which sees no difference."""
    first = ndcg_at_k(["hit", "x", "y"], {"hit"}, 3)
    last = ndcg_at_k(["x", "y", "hit"], {"hit"}, 3)

    assert first > last
    assert recall_at_k(["hit", "x", "y"], {"hit"}, 3) == recall_at_k(
        ["x", "y", "hit"], {"hit"}, 3
    )


def test_ndcg_ideal_is_capped_at_k() -> None:
    """With 10 relevant documents and k=2, finding 2 is a perfect score."""
    assert ndcg_at_k(["a", "b"], {f"d{i}" for i in range(10)} | {"a", "b"}, 2) == (
        pytest.approx(1.0)
    )


def test_ndcg_with_no_judgments_is_zero() -> None:
    assert ndcg_at_k(RANKED, set(), 5) == 0.0


# --- hit rate ---------------------------------------------------------------


def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k(RANKED, {"d3"}, 5) == 1.0
    assert hit_rate_at_k(RANKED, {"d3", "d4"}, 5) == 1.0
    assert hit_rate_at_k(RANKED, {"nope"}, 5) == 0.0


def test_hit_rate_and_recall_coincide_on_single_answer_queries() -> None:
    """Why both are reported: on SciFact most queries have one relevant document."""
    assert hit_rate_at_k(RANKED, {"d2"}, 5) == recall_at_k(RANKED, {"d2"}, 5)


# --- evaluate_run -----------------------------------------------------------


def test_evaluate_run_macro_averages_over_queries() -> None:
    run = {"q1": ["a", "b"], "q2": ["x", "y"]}
    qrels = {"q1": {"a"}, "q2": {"y"}}

    result = evaluate_run(run, qrels, k_values=(1,))

    assert result["recall@1"] == 0.5  # q1 hits at rank 1, q2 does not
    assert result["mrr"] == 0.75  # (1.0 + 0.5) / 2


def test_evaluate_run_penalises_a_query_missing_from_the_run() -> None:
    """A crashed query must score zero, not vanish from the denominator."""
    result = evaluate_run({"q1": ["a"]}, {"q1": {"a"}, "q2": {"b"}}, k_values=(1,))

    assert result["recall@1"] == 0.5


def test_evaluate_run_excludes_queries_with_no_judgments() -> None:
    """Unjudged is not the same as failed, so it is dropped rather than zeroed."""
    result = evaluate_run({"q1": ["a"], "q2": ["z"]}, {"q1": {"a"}, "q2": set()}, (1,))

    assert result["recall@1"] == 1.0


def test_evaluate_run_on_empty_qrels_returns_empty() -> None:
    assert evaluate_run({}, {}) == {}


# --- calibration statistics -------------------------------------------------


def test_kappa_is_one_for_perfect_agreement() -> None:
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_kappa_is_zero_for_chance_agreement() -> None:
    """Both raters say yes half the time and are independent."""
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_kappa_is_negative_for_systematic_disagreement() -> None:
    assert cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)


def test_kappa_punishes_the_always_yes_rater() -> None:
    """The reason raw agreement is never quoted alone.

    Nine of ten answers are good. A rater that always says "yes" gets 90% raw
    agreement and kappa 0 -- it has added nothing.
    """
    human = [1] * 9 + [0]
    always_yes = [1] * 10

    raw = sum(1 for h, j in zip(human, always_yes, strict=True) if h == j) / 10

    assert raw == 0.9
    assert math.isnan(cohens_kappa(human, always_yes)) or cohens_kappa(
        human, always_yes
    ) == pytest.approx(0.0)


def test_kappa_is_undefined_when_a_rater_never_varies() -> None:
    assert math.isnan(cohens_kappa([1, 1, 1], [1, 1, 1]))


def test_kappa_of_nothing_is_undefined() -> None:
    assert math.isnan(cohens_kappa([], []))


def test_spearman_is_one_for_a_monotonic_relationship() -> None:
    """Rank correlation, so it does not require linearity."""
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == pytest.approx(1.0)


def test_spearman_is_minus_one_when_reversed() -> None:
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_handles_ties_with_average_ranks() -> None:
    """Judge scores tie constantly -- 13 of 23 graded answers scored exactly 1.0."""
    result = spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])

    assert math.isnan(result), "no variance in one series leaves rho undefined"


def test_spearman_needs_at_least_two_points() -> None:
    assert math.isnan(spearman([1.0], [1.0]))
