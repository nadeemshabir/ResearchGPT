"""Build a question set from the indexed paper corpus.

Each question is generated from one chunk. That chunk becomes the question's
reference context, and the model also writes a reference answer from it. Those
two fields are what RAGAS needs for `context_recall`; the other metrics are
reference-free.

Why generate rather than hand-write: hand-labelling 200 questions is slow, and
the point of this set is to exercise the generation pipeline, not to serve as
ground truth for retrieval. Retrieval is measured separately against BEIR's
expert labels, where the labels genuinely matter.

The obvious weakness is that a model writes both the question and the reference.
That is why judge calibration exists -- see `eval/calibrate.py`.

Usage:
    python -m eval.testset --n 40
    python -m eval.testset --n 40 --force
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval.console import setup_console
from src.config import PROJECT_ROOT, get_settings
from src.exceptions import LLMError
from src.generation.llm_client import LLMClient
from src.ingestion.database import VectorDatabase
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

TESTSET_PATH = PROJECT_ROOT / "eval" / "data" / "testset.json"

#: Chunks shorter than this rarely contain a self-contained fact worth asking
#: about -- they are usually headings, captions, or reference-list fragments.
MIN_CHUNK_CHARS = 400

_PROMPT = """You are building an evaluation set for a research-paper question
answering system.

Read the passage below. Write ONE question that the passage answers, and the
answer to it.

Rules for the question:
- It must be answerable using ONLY this passage.
- It must be specific. Include the technical terms, model names, or numbers
  that appear in the passage.
- It must make sense on its own. Do NOT write "this paper", "the passage",
  "the authors", or "the above". Name the thing directly.
- Ask what a researcher would actually ask.

Rules for the answer:
- Answer only from the passage.
- Two or three sentences.
- Keep exact numbers and names.

PASSAGE:
{passage}

Reply with exactly this format and nothing else:
QUESTION: <your question>
ANSWER: <your answer>"""

#: Phrases that show the question depends on unseen context.
_CONTEXT_DEPENDENT = re.compile(
    r"\b(this paper|the paper|this passage|the passage|this study|the study|"
    r"the authors|this work|the above|this section|the text|this document|"
    r"this figure|this table|the following)\b",
    re.IGNORECASE,
)


@dataclass
class TestQuestion:
    """One generated question with its source chunk."""

    id: str
    question: str
    reference_answer: str
    reference_context: str
    paper_id: str
    section: str
    chunk_id: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse(reply: str) -> tuple[str, str] | None:
    """Pull the question and answer out of the model's reply."""
    q = re.search(r"QUESTION:\s*(.+?)(?:\n|$)", reply, re.IGNORECASE | re.DOTALL)
    a = re.search(r"ANSWER:\s*(.+)", reply, re.IGNORECASE | re.DOTALL)
    if not q or not a:
        return None
    question = " ".join(q.group(1).split()).strip()
    answer = " ".join(a.group(1).split()).strip()
    if not question or not answer:
        return None
    return question, answer


def _acceptable(question: str) -> tuple[bool, str]:
    """Reject questions that cannot stand on their own."""
    if len(question) < 20:
        return False, "too short"
    if not question.endswith("?"):
        return False, "not a question"
    match = _CONTEXT_DEPENDENT.search(question)
    if match:
        return False, f"context-dependent ({match.group(0)!r})"
    return True, ""


def sample_chunks(database: VectorDatabase, n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Pick ``n`` chunks, spread evenly across papers.

    Sampling per paper rather than uniformly matters because chunk counts are
    very uneven: llama3_2024 has 160 chunks and alexnet_2012 has 15, so uniform
    sampling would make the test set mostly about one paper.
    """
    stored = database.get_all()
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    ids = stored.get("ids") or []

    by_paper: dict[str, list[dict[str, Any]]] = {}
    for i, text in enumerate(documents):
        if len(text) < MIN_CHUNK_CHARS:
            continue
        meta = metadatas[i] if i < len(metadatas) else {}
        by_paper.setdefault(meta.get("paper_id", "unknown"), []).append(
            {
                "text": text,
                "chunk_id": ids[i] if i < len(ids) else f"chunk_{i}",
                "paper_id": meta.get("paper_id", "unknown"),
                "section": meta.get("section_title", "Unknown"),
            }
        )

    if not by_paper:
        return []

    rng = random.Random(seed)
    for chunks in by_paper.values():
        rng.shuffle(chunks)

    # Round-robin across papers until we have enough.
    selected: list[dict[str, Any]] = []
    papers = sorted(by_paper)
    position = 0
    while len(selected) < n:
        added = False
        for paper in papers:
            if position < len(by_paper[paper]):
                selected.append(by_paper[paper][position])
                added = True
                if len(selected) >= n:
                    break
        if not added:
            break
        position += 1

    return selected


def generate(n: int, seed: int = 42) -> list[TestQuestion]:
    """Generate ``n`` questions from the indexed corpus."""
    database = VectorDatabase()
    total = database.count()
    if total == 0:
        raise RuntimeError(
            "The corpus is empty. Index papers first: python scripts/index_papers.py"
        )

    # Over-sample: some chunks yield unusable questions and get dropped.
    chunks = sample_chunks(database, int(n * 1.6), seed=seed)
    logger.info("Sampled %d chunks from %d total", len(chunks), total)

    llm = LLMClient(temperature=0.3)
    questions: list[TestQuestion] = []
    rejected: dict[str, int] = {}

    for chunk in chunks:
        if len(questions) >= n:
            break
        try:
            reply = llm.generate(prompt=_PROMPT.format(passage=chunk["text"][:4000]))
        except LLMError as exc:
            logger.warning("Generation failed for %s: %s", chunk["chunk_id"], exc)
            rejected["llm error"] = rejected.get("llm error", 0) + 1
            continue

        parsed = _parse(reply)
        if parsed is None:
            rejected["unparseable"] = rejected.get("unparseable", 0) + 1
            continue

        question, answer = parsed
        ok, reason = _acceptable(question)
        if not ok:
            rejected[reason.split(" (")[0]] = rejected.get(reason.split(" (")[0], 0) + 1
            logger.debug("Rejected %r: %s", question[:60], reason)
            continue

        questions.append(
            TestQuestion(
                id=f"q{len(questions):03d}",
                question=question,
                reference_answer=answer,
                reference_context=chunk["text"],
                paper_id=chunk["paper_id"],
                section=chunk["section"],
                chunk_id=chunk["chunk_id"],
            )
        )
        if len(questions) % 10 == 0:
            logger.info("Generated %d/%d", len(questions), n)

    if rejected:
        logger.info("Rejected: %s", ", ".join(f"{k}={v}" for k, v in sorted(rejected.items())))
    return questions


def load(path: Path = TESTSET_PATH) -> list[TestQuestion]:
    """Read a saved test set."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TestQuestion(**item) for item in raw["questions"]]


def _display(path: Path) -> str:
    """Path relative to the project root when possible, else as given.

    ``Path.relative_to`` raises when the path is outside the root or when one
    side is relative, which a CLI argument often is.
    """
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40, help="How many questions to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing set")
    parser.add_argument("--out", type=Path, default=TESTSET_PATH)
    args = parser.parse_args()

    setup_console()
    setup_logging()

    if args.out.exists() and not args.force:
        existing = load(args.out)
        print(f"{_display(args.out)} already has {len(existing)} questions.")
        print("Use --force to regenerate.")
        return 0

    questions = generate(args.n, seed=args.seed)
    if not questions:
        print("No questions generated.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "generator_model": get_settings().resolved_llm_model,
                "num_questions": len(questions),
                "seed": args.seed,
                "questions": [q.as_dict() for q in questions],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    per_paper: dict[str, int] = {}
    for q in questions:
        per_paper[q.paper_id] = per_paper.get(q.paper_id, 0) + 1

    print(f"\nGenerated {len(questions)} questions -> {_display(args.out)}\n")
    for paper, count in sorted(per_paper.items(), key=lambda kv: -kv[1]):
        print(f"  {count:3d}  {paper}")
    print("\nExamples:")
    for q in questions[:3]:
        print(f"  [{q.paper_id}] {q.question}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
