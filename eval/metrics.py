"""Standard information-retrieval metrics.

Pure functions over ranked ID lists, with no dependency on the rest of the
project, so they can be unit-tested against worked examples and reused for any
dataset that provides relevance judgments.

Conventions
-----------
``ranked`` is a list of document IDs ordered best-first, without duplicates.
``relevant`` is the set of IDs judged relevant for that query. Judgments are
treated as binary; BEIR's SciFact is binary, and graded relevance would need
``ndcg_at_k`` to take gains rather than a set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents appearing in the top ``k``.

    The headline RAG metric: a generator cannot use what retrieval never
    returned, so recall bounds end-to-end quality.

    Returns 0.0 when no documents are judged relevant, which keeps unjudged
    queries from silently inflating an average.
    """
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return found / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` that is relevant."""
    if k <= 0:
        return 0.0
    found = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return found / k


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the first relevant document.

    1.0 if the top hit is relevant, 0.5 if the second is, 0.0 if none are.
    Averaged over queries this is MRR, which captures how far a user has to
    read before finding something useful.
    """
    for position, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Discounted cumulative gain with binary gains and log2 discount."""
    return sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked[:k], 1)
        if doc_id in relevant
    )


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """DCG normalised by the best achievable DCG for this query.

    Rank-aware, so unlike recall it distinguishes finding the answer at
    position 1 from finding it at position 10. This is the metric BEIR
    leaderboards report, so it is the one to quote for comparability.
    """
    if not relevant:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / ideal


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant document is in the top ``k``, else 0.0.

    Reported alongside recall because on SciFact most queries have exactly one
    relevant document, where recall@k and hit-rate coincide; keeping both makes
    that visible rather than implied.
    """
    return 1.0 if any(doc_id in relevant for doc_id in ranked[:k]) else 0.0


#: Cutoffs reported by default. 10 is BEIR's convention for nDCG.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Score a full run: metrics macro-averaged over queries.

    Args:
        run: query ID to ranked document IDs, best first.
        qrels: query ID to the set of relevant document IDs.
        k_values: cutoffs for recall, precision, nDCG, and hit rate.

    Returns:
        Metric name to mean value, e.g. ``{"recall@5": 0.83, "mrr": 0.71}``.
        Averaged only over queries that carry judgments, so an incomplete run
        is penalised (missing queries score zero) but unjudged queries are
        excluded rather than counted as failures.
    """
    scored_queries = [qid for qid in qrels if qrels[qid]]
    if not scored_queries:
        return {}

    totals: dict[str, float] = {}
    for qid in scored_queries:
        ranked = run.get(qid, [])
        relevant = qrels[qid]

        for k in k_values:
            totals[f"recall@{k}"] = totals.get(f"recall@{k}", 0.0) + recall_at_k(ranked, relevant, k)
            totals[f"precision@{k}"] = totals.get(f"precision@{k}", 0.0) + precision_at_k(ranked, relevant, k)
            totals[f"ndcg@{k}"] = totals.get(f"ndcg@{k}", 0.0) + ndcg_at_k(ranked, relevant, k)
            totals[f"hit@{k}"] = totals.get(f"hit@{k}", 0.0) + hit_rate_at_k(ranked, relevant, k)
        totals["mrr"] = totals.get("mrr", 0.0) + reciprocal_rank(ranked, relevant)

    n = len(scored_queries)
    return {name: round(total / n, 4) for name, total in sorted(totals.items())}
