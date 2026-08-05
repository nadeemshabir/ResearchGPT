"""Parameter sweeps over BEIR/SciFact.

Answers the questions the fixed ablation cannot: is 0.7/0.3 actually the right
semantic/keyword split for this corpus, and does RRF's damping constant explain
why it underperforms weighted fusion here.

Usage:
    python -m eval.sweep weights          # semantic weight 0.0 -> 1.0
    python -m eval.sweep rrfk             # rrf_k in {5,10,20,40,60,100,200}
    python -m eval.sweep weights --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from eval.retrieval_eval import RESULTS_DIR, RetrievalEvaluator, RunConfig
from eval.scifact import index_scifact, load_scifact
from src.config import PROJECT_ROOT
from src.ingestion.embedder import EmbeddingGenerator
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

#: Semantic weight values; keyword weight is the complement.
WEIGHT_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

#: RRF damping constants. Smaller values sharpen the advantage of top ranks;
#: 60 is the constant from the original paper.
RRF_K_GRID: tuple[int, ...] = (5, 10, 20, 40, 60, 100, 200)

#: Metric the sweep optimises for when reporting a winner.
PRIMARY_METRIC = "ndcg@10"


def sweep_weights(evaluator: RetrievalEvaluator, normalisation: str) -> list[dict]:
    """Vary the semantic/keyword split across the full range.

    The endpoints are meaningful controls: 0.0 is pure BM25 fused through the
    hybrid code path and 1.0 is pure dense, so they should land near the
    standalone BM25 and dense rows of the main ablation. If they do not, the
    fusion code is doing something unintended.
    """
    rows: list[dict] = []
    for semantic in WEIGHT_GRID:
        keyword = round(1.0 - semantic, 2)
        config = RunConfig(
            name=f"w{semantic:.1f}",
            method="hybrid",
            semantic_weight=semantic,
            keyword_weight=keyword,
            fusion="weighted",
            normalisation=normalisation,
        )
        result = evaluator.run(config)
        rows.append(
            {
                "semantic_weight": semantic,
                "keyword_weight": keyword,
                **result.metrics,
                "p95_latency_ms": round(result.p95_latency_ms, 1),
            }
        )
        logger.info(
            "semantic=%.1f -> %s=%.4f", semantic, PRIMARY_METRIC, result.metrics[PRIMARY_METRIC]
        )
    return rows


def sweep_rrf_k(evaluator: RetrievalEvaluator) -> list[dict]:
    """Vary RRF's damping constant at fixed weights."""
    rows: list[dict] = []
    for k in RRF_K_GRID:
        config = RunConfig(name=f"rrf{k}", method="hybrid", fusion="rrf", rrf_k=k)
        result = evaluator.run(config)
        rows.append(
            {"rrf_k": k, **result.metrics, "p95_latency_ms": round(result.p95_latency_ms, 1)}
        )
        logger.info("rrf_k=%d -> %s=%.4f", k, PRIMARY_METRIC, result.metrics[PRIMARY_METRIC])
    return rows


def print_table(rows: list[dict], key: str) -> None:
    """Render a sweep as a Markdown table with the best row marked."""
    columns = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    best = max(rows, key=lambda r: r[PRIMARY_METRIC])

    print(f"| {key} | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for row in rows:
        marker = "  <-- best" if row is best else ""
        cells = " | ".join(f"{row[c]:.4f}" for c in columns)
        print(f"| {row[key]} | {cells} |{marker}")
    print(f"\nBest {PRIMARY_METRIC}: {best[PRIMARY_METRIC]:.4f} at {key}={best[key]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", choices=["weights", "rrfk"])
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N queries")
    parser.add_argument("--normalisation", default="minmax", choices=["minmax", "sum", "none"])
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()

    setup_logging()

    data = load_scifact(query_limit=args.limit)
    print(f"\nSciFact test: {data.summary()}\n")

    embedder = EmbeddingGenerator()
    database = index_scifact(data, embedder, chunk_size=args.chunk_size)
    evaluator = RetrievalEvaluator(data, embedder, database)

    if args.sweep == "weights":
        rows = sweep_weights(evaluator, args.normalisation)
        key = "semantic_weight"
        title = f"SEMANTIC/KEYWORD WEIGHT SWEEP ({args.normalisation})"
    else:
        rows = sweep_rrf_k(evaluator)
        key = "rrf_k"
        title = "RRF DAMPING CONSTANT SWEEP"

    print("\n" + "=" * 80)
    print(f"{title} - {len(data.queries)} queries")
    print("=" * 80 + "\n")
    print_table(rows, key)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = RESULTS_DIR / f"sweep-{args.sweep}-{stamp}.json"
    output.write_text(
        json.dumps(
            {
                "sweep": args.sweep,
                "dataset": "BEIR/scifact:test",
                "num_queries": len(data.queries),
                "normalisation": args.normalisation,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
