"""Measure whether the system declines questions its corpus cannot answer.

A retrieval system that always answers is not a retrieval system, it is a
generator with extra steps. This measures the two failure directions
separately, because they trade off against each other:

- **false answer** -- an unanswerable question got answered. The system made
  something up. This is the dangerous direction.
- **false refusal** -- an answerable question got refused. The system is
  useless on questions it could have handled.

One number cannot capture both. A threshold of 1.0 refuses everything and
scores perfectly on the first; a threshold of 0.0 answers everything and scores
perfectly on the second.

This also sweeps ``min_semantic_similarity``, currently the weakest-supported
constant in the project: it was set to 0.40 from a handful of manual probes
rather than measured. The sweep replaces that guess with a number.

Setup:
    1. Write questions your papers cannot answer into
       ``eval/data/unanswerable.json`` (run with --template to create a
       starter file).
    2. Run the harness.

Usage:
    python -m eval.refusal --template     # write the starter file
    python -m eval.refusal                # measure at the current threshold
    python -m eval.refusal --sweep        # find the best threshold
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

from eval.console import setup_console
from eval.testset import TESTSET_PATH
from eval.testset import load as load_testset
from src.config import PROJECT_ROOT, get_settings
from src.exceptions import LLMError, NoRelevantContextError
from src.generation.answer_generator import AnswerGenerator
from src.ingestion.database import VectorDatabase
from src.ingestion.embedder import EmbeddingGenerator
from src.retrieval.retrieval_system import RetrievalSystem
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

UNANSWERABLE_PATH = PROJECT_ROOT / "eval" / "data" / "unanswerable.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

#: Thresholds tried by --sweep. Range chosen around the shipped 0.40.
THRESHOLD_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

TEMPLATE = {
    "note": (
        "Questions this corpus CANNOT answer. Aim for 20. Mix the three kinds "
        "below -- a set of only obvious off-topic questions makes the system "
        "look better than it is."
    ),
    "kinds": {
        "off_topic": "Nothing to do with the papers. Cooking, sport, geography.",
        "near_miss": (
            "Sounds like the papers but is not in them. A model that was not "
            "trained on these papers, or a metric they never report. These are "
            "the hard ones and the ones worth writing carefully."
        ),
        "unknowable": "No paper could answer it. Future events, opinions, private data.",
    },
    "questions": [
        {"question": "What is the capital city of France?", "kind": "off_topic"},
        {"question": "How long should I bake sourdough bread?", "kind": "off_topic"},
        {
            "question": "What accuracy did GPT-6 reach on the MMLU benchmark?",
            "kind": "near_miss",
        },
        {
            "question": "What was the carbon footprint of training AlexNet in kilograms?",
            "kind": "near_miss",
        },
        {
            "question": "Which model will be state of the art in 2030?",
            "kind": "unknowable",
        },
    ],
}


def write_template() -> None:
    """Create the starter file for hand-written unanswerable questions."""
    UNANSWERABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNANSWERABLE_PATH.write_text(json.dumps(TEMPLATE, indent=2), encoding="utf-8")
    print(f"\nWrote {UNANSWERABLE_PATH.relative_to(PROJECT_ROOT)}")
    print("\nIt has 5 examples. Add about 15 more, then run:")
    print("    python -m eval.refusal")
    print("\nThe 'near_miss' ones matter most. A question that merely sounds")
    print("like your papers is far harder to refuse than one about baking.")


def load_unanswerable() -> list[dict[str, str]]:
    """Read the hand-written unanswerable questions."""
    if not UNANSWERABLE_PATH.exists():
        raise FileNotFoundError(
            f"{UNANSWERABLE_PATH} does not exist. Create it with: python -m eval.refusal --template"
        )
    payload = json.loads(UNANSWERABLE_PATH.read_text(encoding="utf-8"))
    return [
        {"question": str(item["question"]), "kind": str(item.get("kind", "unspecified"))}
        for item in payload.get("questions", [])
        if item.get("question")
    ]


def refuses(
    retrieval: RetrievalSystem,
    question: str,
    top_k: int = 5,
    generator: AnswerGenerator | None = None,
) -> tuple[bool | None, float]:
    """Ask the system a question. Returns (refused, best raw similarity).

    ``refused`` is ``None`` when the request failed for infrastructure reasons.
    That third state matters: an earlier version returned ``False`` on
    :class:`~src.exceptions.LLMError`, so a timed-out request was scored as a
    successful answer. Two runs minutes apart then reported 1/15 and 0/15 false
    refusals for identical code, the difference being one Gemini 503. An API
    outage would have made the refusal rate look perfect.

    Errored questions are excluded from every rate and counted separately.

    Refusal happens at two points and both count. Retrieval raises when nothing
    clears the threshold; when it passes weak context through, the model still
    writes "the provided excerpts do not contain ...". Measuring only the first
    gate scores those prose refusals as fabricated answers -- on the first run
    of this harness that mistake turned 5 correct refusals into 4 reported
    failures.

    Passing ``generator=None`` measures the retrieval gate alone, which is what
    the threshold sweep needs and costs no API calls.
    """
    try:
        result = retrieval.get_relevant_chunks(question, top_k=top_k)
    except NoRelevantContextError:
        return True, 0.0
    except LLMError as exc:  # pragma: no cover - defensive
        logger.warning("Retrieval failed for %r: %s", question[:60], exc)
        return None, 0.0

    best = float(
        max(
            (chunk.get("similarity_score", 0.0) for chunk in result["chunks"]),
            default=0.0,
        )
    )

    if generator is None:
        return False, best

    try:
        response = generator.answer_question(question, top_k=top_k)
    except NoRelevantContextError:
        return True, best
    except LLMError as exc:
        logger.warning("Generation failed for %r: %s", question[:60], exc)
        return None, best

    return bool(response["metadata"].get("refused")), best


def evaluate(
    retrieval: RetrievalSystem,
    unanswerable: list[dict[str, str]],
    answerable: list[str],
    generator: AnswerGenerator | None = None,
) -> dict[str, Any]:
    """Run both question sets and count each failure direction."""
    unanswerable_rows: list[dict[str, Any]] = []
    for item in unanswerable:
        refused, score = refuses(retrieval, item["question"], generator=generator)
        unanswerable_rows.append(
            {
                "question": item["question"],
                "kind": item["kind"],
                "refused": refused,
                "best_similarity": round(score, 4),
            }
        )

    answerable_rows: list[dict[str, Any]] = []
    for question in answerable:
        refused, score = refuses(retrieval, question, generator=generator)
        answerable_rows.append(
            {
                "question": question,
                "refused": refused,
                "best_similarity": round(score, 4),
            }
        )

    # Infrastructure failures are excluded from the denominators. Counting a
    # timed-out request as either outcome corrupts the rate.
    scored_unanswerable = [r for r in unanswerable_rows if r["refused"] is not None]
    scored_answerable = [r for r in answerable_rows if r["refused"] is not None]
    errors = (len(unanswerable_rows) - len(scored_unanswerable)) + (
        len(answerable_rows) - len(scored_answerable)
    )

    correctly_refused = sum(1 for r in scored_unanswerable if r["refused"])
    wrongly_refused = sum(1 for r in scored_answerable if r["refused"])

    n_unanswerable = len(scored_unanswerable) or 1
    n_answerable = len(scored_answerable) or 1

    return {
        "threshold": get_settings().min_semantic_similarity,
        "num_unanswerable": len(scored_unanswerable),
        "num_answerable": len(scored_answerable),
        "num_errors": errors,
        "correct_refusals": correctly_refused,
        "false_answers": len(scored_unanswerable) - correctly_refused,
        "false_refusals": wrongly_refused,
        "refusal_rate": round(correctly_refused / n_unanswerable, 4),
        "false_refusal_rate": round(wrongly_refused / n_answerable, 4),
        "unanswerable": unanswerable_rows,
        "answerable": answerable_rows,
    }


def sweep(unanswerable: list[dict[str, str]], answerable: list[str]) -> list[dict[str, Any]]:
    """Score every threshold in the grid against both question sets.

    Similarity scores do not depend on the threshold, so each question is
    retrieved once and the thresholds are applied to the recorded scores. A
    real sweep would re-run retrieval per threshold and take nine times as long
    for identical numbers.
    """
    database = VectorDatabase()
    embedder = EmbeddingGenerator()
    retrieval = RetrievalSystem(embedder=embedder, database=database)

    print("Scoring questions once, then applying each threshold...\n")

    unanswerable_scores: list[tuple[str, float]] = []
    for item in unanswerable:
        _, score = refuses(retrieval, item["question"])
        unanswerable_scores.append((item["question"], score))

    answerable_scores: list[tuple[str, float]] = []
    for question in answerable:
        _, score = refuses(retrieval, question)
        answerable_scores.append((question, score))

    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        refused_bad = sum(1 for _, s in unanswerable_scores if s < threshold)
        refused_good = sum(1 for _, s in answerable_scores if s < threshold)
        n_bad = len(unanswerable_scores) or 1
        n_good = len(answerable_scores) or 1

        recall = refused_bad / n_bad  # unanswerable correctly refused
        specificity = 1 - refused_good / n_good  # answerable correctly answered
        balanced = (recall + specificity) / 2

        rows.append(
            {
                "threshold": threshold,
                "correct_refusals": refused_bad,
                "false_answers": n_bad - refused_bad,
                "false_refusals": refused_good,
                "refusal_rate": round(recall, 4),
                "false_refusal_rate": round(refused_good / n_good, 4),
                "balanced_accuracy": round(balanced, 4),
            }
        )
    return rows


def print_report(result: dict[str, Any]) -> None:
    """Print the two failure directions and the questions behind them."""
    print("\n" + "=" * 88)
    print(f"  REFUSAL BEHAVIOUR   (min_semantic_similarity = {result['threshold']})")
    print(f"  gates measured: {result.get('gates_measured', 'retrieval+model')}")
    print("=" * 88 + "\n")

    print(f"  Unanswerable questions: {result['num_unanswerable']}")
    print(
        f"    correctly refused     {result['correct_refusals']:3d}"
        f"   ({result['refusal_rate']:.1%})"
    )
    print(f"    ANSWERED ANYWAY       {result['false_answers']:3d}   <- made something up")

    print(f"\n  Answerable questions:   {result['num_answerable']}")
    print(
        f"    wrongly refused       {result['false_refusals']:3d}"
        f"   ({result['false_refusal_rate']:.1%})   <- lost a real answer"
    )

    errors = result.get("num_errors", 0)
    if errors:
        print(
            f"\n  {errors} question(s) failed on API errors and were excluded."
            f"\n  Rerun before quoting these rates -- the sample is that much smaller."
        )

    bad = [r for r in result["unanswerable"] if r["refused"] is False]
    if bad:
        print("\n  Unanswerable questions that were answered:")
        for row in bad[:10]:
            print(f"    [{row['best_similarity']:.3f}] ({row['kind']}) {row['question'][:66]}")

    lost = [r for r in result["answerable"] if r["refused"] is True]
    if lost:
        print("\n  Answerable questions that were refused:")
        for row in lost[:10]:
            print(f"    [{row['best_similarity']:.3f}] {row['question'][:74]}")


def print_sweep(rows: list[dict[str, Any]], current: float) -> None:
    """Print the threshold sweep and name the best cut-off."""
    print("\n" + "=" * 88)
    print("  THRESHOLD SWEEP")
    print("=" * 88 + "\n")
    print("  threshold   refused-correctly   false-answers   false-refusals   balanced")
    print("  " + "-" * 78)

    best = max(rows, key=lambda r: r["balanced_accuracy"])
    for row in rows:
        marker = " <-- best" if row is best else ""
        shipped = " (shipped)" if abs(row["threshold"] - current) < 1e-9 else ""
        print(
            f"  {row['threshold']:>9.2f}   {row['refusal_rate']:>17.1%}"
            f"   {row['false_answers']:>13d}   {row['false_refusals']:>14d}"
            f"   {row['balanced_accuracy']:>8.4f}{marker}{shipped}"
        )

    print(
        f"\n  Best balanced accuracy at {best['threshold']:.2f}"
        f" ({best['balanced_accuracy']:.4f}); shipped is {current:.2f}."
    )
    print("\n  Balanced accuracy weights both directions equally. If a wrong")
    print("  answer costs more than a missed one -- usually true for research")
    print("  tools -- pick a higher threshold than this row suggests.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", action="store_true", help="Write the starter file")
    parser.add_argument("--sweep", action="store_true", help="Sweep the threshold")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Measure the retrieval gate alone, skipping generation. Free and "
            "fast, but scores a prose refusal as a fabricated answer."
        ),
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider for the model gate. Defaults to Settings.llm_provider.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model for the model gate. Pass --provider too when it belongs elsewhere.",
    )
    parser.add_argument(
        "--answerable-limit",
        type=int,
        default=20,
        help="How many known-answerable questions to check for false refusals",
    )
    args = parser.parse_args()

    setup_console()
    setup_logging()

    if args.template:
        write_template()
        return 0

    try:
        unanswerable = load_unanswerable()
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1

    if len(unanswerable) < 5:
        print(f"\nOnly {len(unanswerable)} unanswerable questions. Aim for about 20.")
        print("Fewer than that and one question moves the rate by 20 points.")

    answerable: list[str] = []
    if TESTSET_PATH.exists():
        answerable = [q.question for q in load_testset(TESTSET_PATH)][: args.answerable_limit]
    else:
        print("\nNo test set found, so false refusals cannot be measured.")
        print("Build one with: python -m eval.testset")

    settings = get_settings()

    if args.sweep:
        rows = sweep(unanswerable, answerable)
        print_sweep(rows, settings.min_semantic_similarity)
        payload: dict[str, Any] = {"mode": "sweep", "rows": rows}
    else:
        database = VectorDatabase()
        embedder = EmbeddingGenerator()
        retrieval = RetrievalSystem(embedder=embedder, database=database)
        generator = (
            None
            if args.retrieval_only
            else AnswerGenerator(
                use_multi_agent=False,
                use_smart_routing=False,
                retrieval_system=retrieval,
                llm_provider=args.provider,
                llm_model=args.model,
            )
        )
        result = evaluate(retrieval, unanswerable, answerable, generator=generator)
        result["gates_measured"] = "retrieval" if args.retrieval_only else "retrieval+model"
        if generator is not None:
            result["model"] = f"{generator.llm_client.provider}/{generator.llm_client.model}"
        print_report(result)
        payload = {"mode": "single", **result}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = RESULTS_DIR / f"refusal-{payload['mode']}-{stamp}.json"
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
