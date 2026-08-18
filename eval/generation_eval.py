"""Score generated answers with RAGAS.

Runs the real RAG pipeline over the test set, then scores the results.

Metrics, and what each one actually asks:

- **faithfulness** -- is every claim in the answer supported by the retrieved
  context? This is the hallucination metric.
- **answer_relevancy** -- does the answer address the question asked?
- **context_precision** -- were the retrieved chunks relevant, and ranked well?
- **context_recall** -- did retrieval find everything the reference answer needs?

Two things RAGAS does not cover are added here:

- **citation accuracy** -- do cited sources exist in what was retrieved? A
  citation naming a source that was never supplied is a fabricated reference.
- **refusal correctness** -- measured separately by `eval.refusal`.

The judge is a different model family from the generator, so it is not marking
its own work. Self-enhancement bias in LLM judges is well documented.

**These scores mean little on their own.** RAGAS metrics correlate with human
judgement at roughly 0.55. Read them next to the agreement figure from
`eval.calibrate`.

Usage:
    python -m eval.generation_eval --limit 10        # quick check
    python -m eval.generation_eval                   # full test set
    python -m eval.generation_eval --multi-agent     # score the 4-stage path
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from eval.console import setup_console
from eval.testset import TESTSET_PATH, TestQuestion, load
from src.config import PROJECT_ROOT, get_settings
from src.exceptions import LLMError, NoRelevantContextError
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_manager import CitationManager
from src.ingestion.database import VectorDatabase
from src.ingestion.embedder import EmbeddingGenerator
from src.retrieval.retrieval_system import RetrievalSystem
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

#: Default judge. Chosen to be a different provider from the default
#: generator, so the judge never marks its own work.
DEFAULT_JUDGE_PROVIDER = "openai"
DEFAULT_JUDGE_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
}


def build_judge(provider: str, model: str) -> Any:
    """Build the LLM that scores answers.

    Args:
        provider: ``"openai"``, ``"gemini"``, or ``"groq"``.
        model: Model id for that provider.

    Raises:
        RuntimeError: No API key for the chosen judge provider.
    """
    settings = get_settings()
    key = settings.api_key_for(provider)
    if not key:
        raise RuntimeError(
            f"No API key for judge provider {provider!r}. "
            f"Set the matching key in .env, or pass --judge-provider."
        )

    from langchain_core.language_models import BaseChatModel
    from pydantic import SecretStr
    from ragas.llms import LangchainLLMWrapper

    chat: BaseChatModel
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=model, api_key=SecretStr(key), temperature=0)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0)
    elif provider == "groq":
        from langchain_groq import ChatGroq

        chat = ChatGroq(model=model, api_key=SecretStr(key), temperature=0)
    else:
        raise RuntimeError(f"Unsupported judge provider: {provider!r}")

    return LangchainLLMWrapper(chat)


def build_judge_embeddings() -> Any:
    """Embeddings for RAGAS. Local, so judging costs no extra API calls."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=get_settings().embedding_model)
    )


def run_pipeline(generator: AnswerGenerator, questions: list[TestQuestion]) -> list[dict[str, Any]]:
    """Answer every question with the real pipeline and collect the traces."""
    records: list[dict[str, Any]] = []

    for index, item in enumerate(questions, 1):
        started = time.perf_counter()
        refused = False
        answer = ""
        contexts: list[str] = []
        source_titles: list[str] = []

        try:
            response = generator.answer_question(item.question, top_k=5)
            answer = response["answer"]
            refused = bool(response["metadata"].get("refused"))
            # Pull the exact chunk texts the generator saw, not a summary.
            retrieval = generator.retrieval_system.get_relevant_chunks(item.question, top_k=5)
            contexts = [c["text"] for c in retrieval["chunks"]]
            source_titles = sorted(
                {
                    c.get("metadata", {}).get("title", "")
                    for c in retrieval["chunks"]
                    if c.get("metadata", {}).get("title")
                }
            )
        except NoRelevantContextError:
            refused = True
            answer = "REFUSED: no relevant context found."
        except LLMError as exc:
            logger.warning("Generation failed for %s: %s", item.id, exc)
            answer = f"ERROR: {exc}"

        records.append(
            {
                "id": item.id,
                "question": item.question,
                "answer": answer,
                "contexts": contexts,
                "source_titles": source_titles,
                "reference": item.reference_answer,
                "reference_context": item.reference_context,
                "paper_id": item.paper_id,
                "refused": refused,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        if index % 5 == 0:
            logger.info("Answered %d/%d", index, len(questions))

    return records


def score_with_ragas(
    records: list[dict[str, Any]],
    judge_provider: str,
    judge_model: str,
) -> dict[str, Any]:
    """Score the traces with RAGAS.

    Refusals and errors are excluded: a refused question has no answer to
    judge, and scoring it as 0 would conflate "correctly declined" with
    "answered badly". Refusal is measured separately.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    scorable = [r for r in records if not r["refused"] and not r["answer"].startswith("ERROR:")]
    if not scorable:
        return {"error": "nothing scorable", "num_scored": 0}

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in scorable],
            "answer": [r["answer"] for r in scorable],
            "contexts": [r["contexts"] for r in scorable],
            "reference": [r["reference"] for r in scorable],
        }
    )

    logger.info(
        "Scoring %d answers with RAGAS (judge: %s/%s)",
        len(scorable),
        judge_provider,
        judge_model,
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=build_judge(judge_provider, judge_model),
        embeddings=build_judge_embeddings(),
        raise_exceptions=False,
    )

    # `evaluate` is typed as returning EvaluationResult | Executor, but the
    # non-streaming call used here always yields EvaluationResult.
    frame = cast(Any, result).to_pandas()
    scores: dict[str, Any] = {"num_scored": len(scorable)}
    # NaN marks a metric the judge failed on; kept as None rather than dropped
    # so per-question rows stay aligned with `scorable`.
    per_question: dict[str, list[float | None]] = {}

    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if metric in frame.columns:
            column = frame[metric].dropna()
            scores[metric] = round(float(column.mean()), 4) if len(column) else None
            scores[f"{metric}_n"] = int(len(column))
            per_question[metric] = [None if v != v else round(float(v), 4) for v in frame[metric]]

    # Attach per-question scores so calibration can sample specific rows.
    for offset, record in enumerate(scorable):
        record["scores"] = {m: vals[offset] for m, vals in per_question.items()}

    return scores


def score_citations(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Check citations against the context that was actually retrieved.

    ``unknown_citations`` is the number that matters: a citation naming a
    source never supplied to the model is a fabricated reference.
    """
    manager = CitationManager()
    total_citations = 0
    fabricated = 0
    answers_with_citations = 0

    scorable = [r for r in records if not r["refused"] and not r["answer"].startswith("ERROR:")]
    for record in scorable:
        # Titles as the model saw them. Passing paper_id here instead was a bug:
        # the model cites "ImageNet Classification with..." while paper_id is
        # "alexnet_2012", so nothing ever matched and citation_coverage was
        # pinned at 0.0 for every answer.
        sources = [{"title": t} for t in record.get("source_titles") or []]
        audit = manager.validate_citations(record["answer"], sources)
        total_citations += audit["total_citations"]
        fabricated += len(audit["unknown_citations"])
        if audit["total_citations"] > 0:
            answers_with_citations += 1
        record["citation_audit"] = audit

    return {
        "answers_scored": len(scorable),
        "answers_with_any_citation": answers_with_citations,
        "citation_rate": round(answers_with_citations / len(scorable), 4) if scorable else 0.0,
        "total_citations": total_citations,
        "fabricated_citations": fabricated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N")
    parser.add_argument("--multi-agent", action="store_true", help="Use the 4-stage pipeline")
    parser.add_argument("--testset", type=Path, default=TESTSET_PATH)
    parser.add_argument(
        "--generator-provider",
        default=None,
        choices=sorted(DEFAULT_JUDGE_MODELS),
        help=(
            "Provider that writes the answers. Defaults to Settings.llm_provider. "
            "Set this whenever --generator-model belongs to another provider; the "
            "model name alone does not switch providers."
        ),
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help=(
            "Model that writes the answers. Defaults to Settings.llm_model. "
            "Recorded in the results file, because answer quality depends on "
            "it and runs with different models are not comparable."
        ),
    )
    parser.add_argument(
        "--judge-provider",
        default=DEFAULT_JUDGE_PROVIDER,
        choices=sorted(DEFAULT_JUDGE_MODELS),
        help="Provider that scores the answers. Must differ from the generator.",
    )
    parser.add_argument("--judge-model", default=None, help="Judge model id")
    args = parser.parse_args()

    setup_console()
    setup_logging()

    if not args.testset.exists():
        print(f"No test set at {args.testset}. Run: python -m eval.testset")
        return 1

    questions = load(args.testset)
    if args.limit:
        questions = questions[: args.limit]

    mode = "multi-agent" if args.multi_agent else "single-shot"
    judge_model = args.judge_model or DEFAULT_JUDGE_MODELS[args.judge_provider]
    print(
        f"\nScoring {len(questions)} questions | generation: {mode} "
        f"| judge: {args.judge_provider}/{judge_model}\n"
    )

    database = VectorDatabase()
    embedder = EmbeddingGenerator()
    retrieval = RetrievalSystem(embedder=embedder, database=database)
    generator = AnswerGenerator(
        use_multi_agent=args.multi_agent,
        use_smart_routing=False,  # route everything through plain Q&A
        llm_provider=args.generator_provider,
        llm_model=args.generator_model,
        retrieval_system=retrieval,
    )
    print(f"generator: {generator.llm_client.provider}/{generator.llm_client.model}\n")

    records = run_pipeline(generator, questions)
    refused = sum(1 for r in records if r["refused"])
    errored = sum(1 for r in records if r["answer"].startswith("ERROR:"))

    ragas_scores = score_with_ragas(records, args.judge_provider, judge_model)
    citation_scores = score_citations(records)

    print("\n" + "=" * 72)
    print(f"GENERATION QUALITY - {mode}, {len(questions)} questions")
    print("=" * 72 + "\n")

    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        value = ragas_scores.get(metric)
        n = ragas_scores.get(f"{metric}_n", 0)
        shown = f"{value:.4f}" if isinstance(value, float) else "n/a"
        print(f"  {metric:20s} {shown:>8s}   (n={n})")

    print()
    print(f"  {'citation rate':20s} {citation_scores['citation_rate']:>8.4f}")
    print(f"  {'fabricated citations':20s} {citation_scores['fabricated_citations']:>8d}")
    print(f"  {'refused':20s} {refused:>8d}")
    print(f"  {'errors':20s} {errored:>8d}")
    print()
    print("  NOTE: these are LLM-judged. Read them next to the agreement")
    print("        figure from `python -m eval.calibrate`.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = RESULTS_DIR / f"generation-{mode}-{stamp}.json"
    output.write_text(
        json.dumps(
            {
                "mode": mode,
                "judge_provider": args.judge_provider,
                "judge_model": judge_model,
                "generator_provider": generator.llm_client.provider,
                "generator_model": generator.llm_client.model,
                "num_questions": len(questions),
                "num_refused": refused,
                "num_errors": errored,
                "ragas": ragas_scores,
                "citations": citation_scores,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
