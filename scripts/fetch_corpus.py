"""Download the evaluation corpus into ``data/raw/``.

**This is a local tool, not part of any deployment.** The Dockerfile used to
run it at build time and bake the resulting vector store into the image, which
meant every deployment shipped these eight papers to every user whether or not
anyone wanted them. The deployed service now starts empty and answers from
whatever its users upload.

The eight papers below are still what the evaluation sets in ``eval/data/``
were written against -- the questions, the hand grades and the unanswerable
set all refer to them -- so reproducing a number from ``docs/EVALUATION.md``
means fetching and indexing exactly this list. That is the only reason this
file still exists.

Papers are not committed to the repository: they are large, and redistributing
them is a licensing question this project should not answer. Everything listed
here is openly available from arXiv or NeurIPS proceedings.

Filenames are the paper ids. ``src.ingestion.ids.clean_paper_id`` turns
``attention_2017`` into exactly that, so no alias file is needed for a fresh
fetch.

Usage:
    python scripts/fetch_corpus.py [--out data/raw] [--force]
    python scripts/index_papers.py            # then index what was fetched
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

_ARXIV = "https://arxiv.org/pdf/{}"
_NEURIPS = "https://proceedings.neurips.cc/paper_files/paper/{year}/file/{hash}-Paper.pdf"

#: (paper_id, url). The paper_id becomes the filename and the citation key.
#: Changing this list invalidates the evaluation sets in ``eval/data/``, which
#: ask questions about these specific papers.
#:
#: AlexNet is the odd one out: it predates arXiv preprinting for this venue and
#: exists only in the NeurIPS proceedings.
CORPUS: tuple[tuple[str, str], ...] = (
    ("attention_2017", _ARXIV.format("1706.03762")),
    ("bert_2019", _ARXIV.format("1810.04805")),
    ("gpt3_2020", _ARXIV.format("2005.14165")),
    ("instructgpt_2022", _ARXIV.format("2203.02155")),
    ("rag_survey_2023", _ARXIV.format("2312.10997")),
    ("llama3_2024", _ARXIV.format("2407.21783")),
    ("deepseek_r1_2025", _ARXIV.format("2501.12948")),
    (
        "alexnet_2012",
        _NEURIPS.format(year=2012, hash="c399862d3b9d6b76c8436e924a68c45b"),
    ),
)

USER_AGENT = "ResearchGPT-corpus-fetcher/1.0 (+https://github.com/nadeemshabir/ResearchGPT)"


def download(url: str, destination: Path, force: bool = False) -> bool:
    """Fetch one PDF. Returns ``True`` if a file was written.

    Raises:
        RuntimeError: The download failed. A Docker build must not carry on and
            produce an image that is quietly missing a paper.
    """
    if destination.exists() and not force:
        logger.info("Skipping %s (already present)", destination.name)
        return False

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    if not payload.startswith(b"%PDF-"):
        raise RuntimeError(
            f"{url} did not return a PDF ({len(payload)} bytes). "
            f"arXiv rate-limits aggressively; try again in a minute."
        )

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
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    args = parser.parse_args()

    setup_logging()
    args.out.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for paper_id, url in CORPUS:
        if download(url, args.out / f"{paper_id}.pdf", force=args.force):
            downloaded += 1

    print(f"\n{downloaded} new file(s), {len(CORPUS)} papers in {args.out}")
    print("Index them with:  python scripts/index_papers.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
