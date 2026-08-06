"""Retrieval ablations over BEIR/SciFact.

Runs each retrieval configuration against the same indexed corpus and the same
human relevance judgments, so differences in the output are attributable to the
configuration rather than to the data.

Usage:
    python -m eval.retrieval_eval --limit 50           # quick check
    python -m eval.retrieval_eval                      # full 300-query run
    python -m eval.retrieval_eval --configs bm25,rrf   # subset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from eval.metrics import DEFAULT_K_VALUES, evaluate_run
from eval.scifact import index_scifact, load_scifact, rank_documents
from src.config import PROJECT_ROOT, get_settings
from src.ingestion.embedder import EmbeddingGenerator
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.keyword_search import KeywordSearcher
from src.retrieval.reranker import Reranker
from src.retrieval.semantic_search import SemanticSearcher
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

Method = Literal["bm25", "dense", "hybrid"]


@dataclass
class RunConfig:
    """One retrieval configuration to evaluate.

    Every field is set explicitly and **nothing here reads from Settings**.
    That is deliberate: an ablation whose rows shift when someone edits `.env`
    is not reproducible, and the published numbers must stay reproducible
    regardless of local configuration. The shipped defaults appear as their own
    named config below rather than being inherited implicitly.
    """

    name: str
    method: Method
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    fusion: str = "weighted"
    normalisation: str = "minmax"
    rrf_k: int = 60
    rerank: bool = False
    #: Candidates fetched before reranking. Fixed across configs so that
    #: differences come from ranking, not from how much was retrieved.
    candidate_k: int = 50
    #: Results kept after ranking.
    top_k: int = 10

    def describe(self) -> str:
        if self.method == "bm25":
            base = "BM25 only"
        elif self.method == "dense":
            base = "Dense only"
        elif self.fusion == "rrf":
            base = f"Hybrid RRF (k={self.rrf_k}, w={self.semantic_weight:.1f}/{self.keyword_weight:.1f})"
        else:
            base = (
                f"Hybrid weighted {self.semantic_weight:.1f}/{self.keyword_weight:.1f}"
                f" ({self.normalisation})"
            )
        return base + (" + rerank" if self.rerank else "")


#: The ablation from the plan. Ordered so each row isolates one change.
DEFAULT_CONFIGS: dict[str, RunConfig] = {
    "bm25": RunConfig("bm25", method="bm25"),
    "dense": RunConfig("dense", method="dense"),
    "hybrid_minmax": RunConfig("hybrid_minmax", method="hybrid", normalisation="minmax"),
    "hybrid_sum": RunConfig("hybrid_sum", method="hybrid", normalisation="sum"),
    "rrf": RunConfig("rrf", method="hybrid", fusion="rrf"),
    # The configuration the application actually ships, post-tuning.
    "shipped": RunConfig(
        "shipped",
        method="hybrid",
        semantic_weight=0.5,
        keyword_weight=0.5,
        normalisation="minmax",
        rerank=False,
    ),
    "hybrid_minmax_rerank": RunConfig(
        "hybrid_minmax_rerank", method="hybrid", normalisation="minmax", rerank=True
    ),
    "rrf_rerank": RunConfig("rrf_rerank", method="hybrid", fusion="rrf", rerank=True),
}


@dataclass
class RunResult:
    """Metrics and timings for one configuration."""

    config: RunConfig
    metrics: dict[str, float]
    num_queries: int
    total_seconds: float
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        return ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "description": self.config.describe(),
            "config": vars(self.config),
            "num_queries": self.num_queries,
            "metrics": self.metrics,
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "total_seconds": round(self.total_seconds, 1),
        }


class RetrievalEvaluator:
    """Runs configurations against a fixed SciFact index.

    Searchers and the reranker are built once and reused across configurations,
    both because loading models repeatedly is slow and because it guarantees
    every configuration sees an identical index.
    """

    def __init__(self, data: Any, embedder: EmbeddingGenerator, database: Any):
        self.data = data
        self.semantic = SemanticSearcher(embedder=embedder, database=database)
        self.keyword = KeywordSearcher(database=database)
        self._reranker: Reranker | None = None

    @property
    def reranker(self) -> Reranker:
        """Loaded on first use, so runs without reranking never pay for it."""
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def _build_hybrid(self, config: RunConfig) -> HybridSearcher:
        """Construct the fusion searcher once per configuration.

        Building it per query would re-run weight validation 300 times and,
        more importantly, makes the measured latency reflect construction
        rather than retrieval.
        """
        return HybridSearcher(
            semantic_weight=config.semantic_weight,
            keyword_weight=config.keyword_weight,
            fusion=config.fusion,  # type: ignore[arg-type]
            normalisation=config.normalisation,  # type: ignore[arg-type]
            rrf_k=config.rrf_k,
            semantic_searcher=self.semantic,
            keyword_searcher=self.keyword,
        )

    def run(self, config: RunConfig) -> RunResult:
        """Evaluate one configuration over every judged query."""
        logger.info("Running config %r: %s", config.name, config.describe())

        hybrid = self._build_hybrid(config) if config.method == "hybrid" else None

        run: dict[str, list[str]] = {}
        latencies: list[float] = []
        started = time.perf_counter()

        for index, (qid, query) in enumerate(self.data.queries.items(), 1):
            query_started = time.perf_counter()

            if config.method == "bm25":
                results = self.keyword.search(query, top_k=config.candidate_k)
            elif config.method == "dense":
                results = self.semantic.search(query, top_k=config.candidate_k)
            else:
                assert hybrid is not None
                results = hybrid.search(query, top_k=config.candidate_k)

            if config.rerank and len(results) > 1:
                results = self.reranker.rerank(query=query, results=results, top_k=config.top_k)

            latencies.append((time.perf_counter() - query_started) * 1000)
            # Collapse chunks to documents: BEIR judges documents.
            run[qid] = rank_documents(results)[: config.top_k]

            if index % 50 == 0:
                logger.info("  %d/%d queries", index, len(self.data.queries))

        total = time.perf_counter() - started
        metrics = evaluate_run(run, self.data.qrels, k_values=DEFAULT_K_VALUES)

        return RunResult(
            config=config,
            metrics=metrics,
            num_queries=len(run),
            total_seconds=total,
            latencies_ms=latencies,
        )


def format_table(results: list[RunResult]) -> str:
    """Render results as a Markdown table, ready to paste into the docs."""
    columns = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    header = "| Configuration | " + " | ".join(columns) + " | p95 latency |"
    divider = "|---" * (len(columns) + 2) + "|"

    rows = [header, divider]
    for result in results:
        cells = [f"{result.metrics.get(col, 0.0):.4f}" for col in columns]
        rows.append(
            f"| {result.config.describe()} | "
            + " | ".join(cells)
            + f" | {result.p95_latency_ms:.0f} ms |"
        )
    return "\n".join(rows)


def _write_results(
    output: Path,
    args: argparse.Namespace,
    embedder: EmbeddingGenerator,
    data: Any,
    results: list[RunResult],
) -> None:
    """Write results so far, overwriting on each configuration.

    Called after every config rather than once at the end, so an interrupted
    run still leaves usable output on disk.
    """
    output.write_text(
        json.dumps(
            {
                "dataset": f"BEIR/scifact:{args.split}",
                "num_queries": len(data.queries),
                "num_documents": len(data.corpus),
                "embedding_model": embedder.model_name,
                "chunk_size": args.chunk_size or get_settings().chunk_size,
                "candidate_k": args.candidate_k,
                "complete": False,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "results": [r.as_dict() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N queries")
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help=f"Comma-separated subset of: {','.join(DEFAULT_CONFIGS)}",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk size")
    parser.add_argument("--candidate-k", type=int, default=50, help="Candidates before reranking")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    args = parser.parse_args()

    setup_logging()

    selected = list(DEFAULT_CONFIGS)
    if args.configs:
        selected = [name.strip() for name in args.configs.split(",")]
        unknown = [name for name in selected if name not in DEFAULT_CONFIGS]
        if unknown:
            print(f"Unknown config(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(DEFAULT_CONFIGS)}")
            return 1

    data = load_scifact(split=args.split, query_limit=args.limit)
    print(f"\nSciFact {args.split}: {data.summary()}\n")

    embedder = EmbeddingGenerator()
    database = index_scifact(data, embedder, chunk_size=args.chunk_size, force=args.force_reindex)

    evaluator = RetrievalEvaluator(data, embedder, database)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = RESULTS_DIR / f"retrieval-{args.split}-{stamp}.json"

    results: list[RunResult] = []
    for name in selected:
        config = DEFAULT_CONFIGS[name]
        config.candidate_k = args.candidate_k
        result = evaluator.run(config)
        results.append(result)

        # Report and checkpoint after every configuration. Reranked runs take
        # tens of minutes each, and a run that reveals nothing until the end
        # loses everything if it is interrupted.
        print(f"\n  {result.config.describe()}")
        for metric in ("recall@5", "recall@10", "mrr", "ndcg@10"):
            print(f"    {metric:<12} {result.metrics.get(metric, 0.0):.4f}")
        print(f"    {'p95 latency':<12} {result.p95_latency_ms:.0f} ms", flush=True)

        _write_results(output, args, embedder, data, results)

    print("\n" + "=" * 80)
    print(f"RETRIEVAL ABLATION - BEIR/SciFact {args.split} ({len(data.queries)} queries)")
    print("=" * 80 + "\n")
    print(format_table(results))
    print()
    print(f"Saved to {output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
