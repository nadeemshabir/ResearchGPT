"""Download a small open-access paper corpus into ``data/raw/``.

Papers are not committed to the repository: they are large, and redistributing
them is a licensing question this project should not answer. Everything listed
here is openly licensed on arXiv, so anyone cloning the repo can rebuild the
same corpus by running this script.

Usage:
    python scripts/fetch_corpus.py [--out data/raw] [--force]
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: (arXiv id, filename stem). Foundational NLP/transformer papers, chosen so
#: the corpus is topically coherent enough for retrieval evaluation.
CORPUS: tuple[tuple[str, str], ...] = (
    ("1706.03762", "attention_is_all_you_need"),
    ("1810.04805", "bert"),
    ("2005.14165", "gpt3_few_shot_learners"),
    ("1907.11692", "roberta"),
    ("1910.01108", "distilbert"),
    ("2005.11401", "rag_knowledge_intensive_nlp"),
    ("1909.11942", "albert"),
    ("2004.04906", "dense_passage_retrieval"),
)

USER_AGENT = "ResearchGPT-corpus-fetcher/1.0 (+https://github.com/nadeemshabir/ResearchGPT)"


def download(arxiv_id: str, destination: Path, force: bool = False) -> bool:
    """Fetch one arXiv PDF. Returns ``True`` if a file was written."""
    if destination.exists() and not force:
        logger.info("Skipping %s (already present)", destination.name)
        return False

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.error("Could not download %s: %s", arxiv_id, exc)
        return False

    if not payload.startswith(b"%PDF-"):
        logger.error("%s did not return a PDF (got %d bytes)", arxiv_id, len(payload))
        return False

    destination.write_bytes(payload)
    logger.info("Saved %s (%.1f MB)", destination.name, len(payload) / (1024 * 1024))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Destination directory (default: data/raw)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist"
    )
    args = parser.parse_args()

    setup_logging()
    args.out.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for arxiv_id, stem in CORPUS:
        if download(arxiv_id, args.out / f"{stem}.pdf", force=args.force):
            downloaded += 1

    print(f"\n{downloaded} new file(s) in {args.out}")
    print("Index them with:  streamlit run app.py  (Upload tab)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
