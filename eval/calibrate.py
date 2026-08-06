"""Grade answers by hand, then measure how far the LLM judge agrees with you.

Every generation score in ``docs/EVALUATION.md`` comes from an LLM marking
another LLM's work. Published estimates put RAGAS-to-human correlation at
roughly 0.55, which is high enough to be useful and low enough that quoting a
faithfulness score without an agreement figure overstates what is known.

This tool closes that gap. It shows you answers one at a time, records your
verdict, and reports Cohen's kappa and Spearman correlation against the judge.

Two questions are asked per answer, chosen to map onto one RAGAS metric each:

- "Is every claim backed by the sources?"  ->  faithfulness
- "Does it answer the question asked?"     ->  answer_relevancy

Answers are sampled across the judge's score range, not taken from the top.
Sampling only high-scoring answers would measure agreement where agreement is
easy and tell you nothing about the cases that matter.

Progress is saved after every grade, so you can stop and resume.

Usage:
    python -m eval.calibrate                 # grade 30, resuming if started
    python -m eval.calibrate --n 50
    python -m eval.calibrate --report        # stats only, grade nothing
    python -m eval.calibrate --results eval/results/generation-....json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.console import setup_console
from src.config import PROJECT_ROOT

GRADES_PATH = PROJECT_ROOT / "eval" / "data" / "human_grades.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

#: Human question -> the RAGAS metric it is compared against.
DIMENSIONS = {
    "supported": "faithfulness",
    "answers_question": "answer_relevancy",
}

#: Judge score at or above this counts as the judge saying "good". Kappa is
#: also reported at the threshold that maximises it, since 0.5 is a convention
#: rather than a calibrated cut point.
DEFAULT_THRESHOLD = 0.5


def latest_results(directory: Path = RESULTS_DIR) -> Path | None:
    """Most recent generation results file, or ``None`` if there are none."""
    candidates = sorted(directory.glob("generation-*.json"))
    return candidates[-1] if candidates else None


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read scorable records, keeping only those the judge actually scored."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        record
        for record in payload.get("records", [])
        if record.get("scores") and not record.get("refused")
    ]


def stratified_sample(records: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Pick ``n`` records spread evenly across the judge's faithfulness range.

    Grading the top-scoring answers would measure agreement only where it is
    easy. Spreading the sample means the disagreements -- the informative
    cases -- are actually present.
    """
    if len(records) <= n:
        return list(records)

    ordered = sorted(records, key=lambda r: r["scores"].get("faithfulness") or 0.0)
    step = len(ordered) / n
    return [ordered[min(int(i * step), len(ordered) - 1)] for i in range(n)]


def load_grades() -> dict[str, Any]:
    """Read grades recorded so far."""
    if not GRADES_PATH.exists():
        return {}
    payload = json.loads(GRADES_PATH.read_text(encoding="utf-8"))
    return payload.get("grades", {})


def save_grades(grades: dict[str, Any], results_file: str) -> None:
    """Write grades after every verdict, so an interrupted session is not lost."""
    GRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRADES_PATH.write_text(
        json.dumps(
            {
                "results_file": results_file,
                "num_graded": len(grades),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "grades": grades,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _wrap(text: str, width: int = 92, indent: str = "  ") -> str:
    """Wrap text by words, preserving the blank lines between paragraphs."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        current = indent
        for word in paragraph.split():
            if len(current) + len(word) + 1 > width and current.strip():
                lines.append(current.rstrip())
                current = indent
            current += word + " "
        lines.append(current.rstrip())
    return "\n".join(lines)


def _ask(prompt: str) -> str | None:
    """Read one verdict. Returns ``None`` when the grader wants to stop."""
    while True:
        try:
            reply = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if reply in {"y", "n", "s", "q"}:
            return reply
        print("      Please type y, n, s (skip), or q (quit and save).")


def grade_one(record: dict[str, Any], position: int, total: int) -> dict[str, str] | None:
    """Show one answer and collect a verdict on each dimension."""
    print("\n" + "=" * 96)
    print(f"  ANSWER {position} of {total}      (id: {record['id']})")
    print("=" * 96)

    print("\nQUESTION")
    print(_wrap(record["question"]))

    print("\nANSWER")
    print(_wrap(record["answer"]))

    print("\nSOURCES THE SYSTEM WAS GIVEN")
    for index, context in enumerate(record.get("contexts", []), 1):
        excerpt = context[:700].rstrip()
        suffix = " ..." if len(context) > 700 else ""
        print(f"\n  [{index}]")
        print(_wrap(excerpt + suffix, indent="      "))

    print("\n" + "-" * 96)
    print("  y = yes    n = no    s = skip    q = quit and save")
    print("-" * 96)

    supported = _ask("\n  Is every claim in the answer backed by the sources above?  ")
    if supported is None or supported == "q":
        return None
    answers_question = _ask("  Does the answer address the question that was asked?       ")
    if answers_question is None or answers_question == "q":
        return None

    return {"supported": supported, "answers_question": answers_question}


# ----------------------------------------------------------------------------
# Agreement statistics
# ----------------------------------------------------------------------------


def cohens_kappa(human: list[int], judge: list[int]) -> float:
    """Agreement between two binary raters, corrected for chance.

    Raw percent agreement flatters any rater on a skewed set: if 90% of answers
    are good, always saying "good" scores 90%. Kappa subtracts the agreement
    expected from chance alone, so it stays honest.
    """
    n = len(human)
    if n == 0:
        return float("nan")

    observed = sum(1 for h, j in zip(human, judge, strict=True) if h == j) / n
    human_yes, judge_yes = sum(human) / n, sum(judge) / n
    expected = human_yes * judge_yes + (1 - human_yes) * (1 - judge_yes)
    if expected == 1.0:
        return float("nan")  # both raters constant; kappa undefined
    return (observed - expected) / (1 - expected)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, with ties handled by average ranks.

    Implemented here rather than pulled from scipy: it is a dozen lines and
    scipy is not otherwise required at runtime.
    """
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        result = [0.0] * n
        position = 0
        while position < n:
            end = position
            while end + 1 < n and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2 + 1
            for index in range(position, end + 1):
                result[order[index]] = average
            position = end + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    mean_x, mean_y = sum(rx) / n, sum(ry) / n
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry, strict=True))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry)
    )
    return numerator / denominator if denominator else float("nan")


def best_threshold(human: list[int], scores: list[float]) -> tuple[float, float]:
    """Judge cut-off that maximises kappa, and the kappa it reaches.

    A judge can rank well while being badly calibrated. Reporting the best
    achievable kappa separates "the judge is wrong" from "0.5 is the wrong
    place to cut".
    """
    best_kappa, best_cut = float("-inf"), DEFAULT_THRESHOLD
    for step in range(1, 20):
        cut = step / 20
        kappa = cohens_kappa(human, [1 if s >= cut else 0 for s in scores])
        if not math.isnan(kappa) and kappa > best_kappa:
            best_kappa, best_cut = kappa, cut
    return best_cut, best_kappa


def report(records: list[dict[str, Any]], grades: dict[str, Any]) -> None:
    """Print agreement between the human grades and the judge."""
    by_id = {record["id"]: record for record in records}

    print("\n" + "=" * 96)
    print("  JUDGE CALIBRATION")
    print("=" * 96)

    graded = {rid: g for rid, g in grades.items() if rid in by_id}
    if not graded:
        print("\n  No grades recorded yet. Run `python -m eval.calibrate` first.")
        return

    print(f"\n  Answers graded: {len(graded)}\n")

    for dimension, metric in DIMENSIONS.items():
        human: list[int] = []
        scores: list[float] = []
        for rid, grade in graded.items():
            verdict = grade.get(dimension)
            score = by_id[rid]["scores"].get(metric)
            if verdict in {"y", "n"} and isinstance(score, (int, float)):
                human.append(1 if verdict == "y" else 0)
                scores.append(float(score))

        print(f"  {dimension}  vs  RAGAS {metric}")
        if len(human) < 5:
            print(f"    only {len(human)} usable grades -- need at least 5\n")
            continue

        judge = [1 if s >= DEFAULT_THRESHOLD else 0 for s in scores]
        kappa = cohens_kappa(human, judge)
        rho = spearman([float(h) for h in human], scores)
        cut, tuned = best_threshold(human, scores)
        agreement = sum(1 for h, j in zip(human, judge, strict=True) if h == j) / len(human)

        yes_scores = [s for h, s in zip(human, scores, strict=True) if h == 1]
        no_scores = [s for h, s in zip(human, scores, strict=True) if h == 0]

        print(f"    n                     {len(human)}")
        print(f"    you said yes          {sum(human)} / {len(human)}")
        print(f"    raw agreement         {agreement:.2%}")
        print(f"    Cohen's kappa @0.50   {kappa:+.3f}   {_kappa_label(kappa)}")
        print(f"    Cohen's kappa @{cut:.2f}   {tuned:+.3f}   (best cut-off)")
        print(f"    Spearman rho          {rho:+.3f}")
        if yes_scores:
            print(f"    judge score, you=yes  {sum(yes_scores)/len(yes_scores):.3f}")
        if no_scores:
            print(f"    judge score, you=no   {sum(no_scores)/len(no_scores):.3f}")
        print()

    print("  How to read this:")
    print("    kappa above 0.6   the judge tracks you well; quote its scores")
    print("    kappa 0.4 to 0.6  useful for ranking runs, not for absolute claims")
    print("    kappa below 0.4   do not present the judge's scores as fact")
    print()
    print("  If the two mean scores at the bottom are far apart, the judge")
    print("  separates good from bad even when the 0.50 cut-off is misplaced.")


def _kappa_label(kappa: float) -> str:
    if math.isnan(kappa):
        return "undefined"
    if kappa >= 0.8:
        return "very strong"
    if kappa >= 0.6:
        return "strong"
    if kappa >= 0.4:
        return "moderate"
    if kappa >= 0.2:
        return "weak"
    return "poor"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30, help="How many answers to grade")
    parser.add_argument("--results", type=Path, default=None, help="Results file to grade")
    parser.add_argument("--report", action="store_true", help="Show stats, grade nothing")
    parser.add_argument("--seed-order", action="store_true", help="Grade in file order")
    args = parser.parse_args()

    setup_console()

    results_path = args.results or latest_results()
    if results_path is None or not results_path.exists():
        print("No generation results found. Run: python -m eval.generation_eval")
        return 1

    records = load_records(results_path)
    if not records:
        print(f"{results_path.name} has no scored answers to grade.")
        return 1

    print(f"\nUsing {results_path.name}  ({len(records)} scored answers)")

    grades = load_grades()
    sample = records if args.seed_order else stratified_sample(records, args.n)

    if args.report:
        report(records, grades)
        return 0

    pending = [r for r in sample if r["id"] not in grades]
    if not pending:
        print(f"\nAll {len(sample)} sampled answers are already graded.")
        report(records, grades)
        return 0

    print(f"Grading {len(pending)} answers ({len(grades)} already done).")
    print("Your progress is saved after every answer, so you can stop any time.")

    for position, record in enumerate(pending, 1):
        verdicts = grade_one(record, position, len(pending))
        if verdicts is None:
            print("\nStopped. Your grades are saved.")
            break
        if verdicts["supported"] == "s" or verdicts["answers_question"] == "s":
            continue
        grades[record["id"]] = verdicts
        save_grades(grades, results_path.name)

    print(f"\nSaved {len(grades)} grades to {GRADES_PATH.name}")
    report(records, grades)
    return 0


if __name__ == "__main__":
    sys.exit(main())
