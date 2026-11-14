"""Index PDFs from a directory into the application's vector store.

Bulk loading from the command line, for evaluation runs and for seeding a
development store. The deployed service does not use it: there, papers arrive
through ``POST /papers`` one at a time, and the store they go into is remote.

Pointed at a remote store (``CHROMA_MODE=cloud`` or ``http``) this writes to
whatever that configuration names -- including a live deployment. Check which
store the environment selects before running it with ``--reset``.

Usage:
    python scripts/index_papers.py                       # index data/raw
    python scripts/index_papers.py --skip Panipat.pdf    # exclude a file
    python scripts/index_papers.py --dry-run             # list, do not index
    python scripts/index_papers.py --reset               # wipe first

Paper IDs come from ``paper_ids.json`` in the source directory when present,
otherwise from the filename. IDs matter because they appear as the source name
in generated citations, so a readable ID gives a readable citation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.exceptions import PDFParseError, ResearchGPTError  # noqa: E402
from src.ingestion.ids import clean_paper_id  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

ALIAS_FILE = "paper_ids.json"


#: Kept as a name here because this script's docstring and the Dockerfile
#: both refer to it. The implementation lives in :mod:`src.ingestion.ids` so
#: that the upload endpoint derives ids exactly the same way.
clean_id = clean_paper_id


def load_aliases(directory: Path) -> dict[str, str]:
    """Read the optional filename-to-ID map from the source directory."""
    path = directory / ALIAS_FILE
    if not path.exists():
        return {}
    try:
        aliases = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable %s: %s", ALIAS_FILE, exc)
        return {}
    return {str(k): str(v) for k, v in aliases.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--skip", nargs="*", default=[], help="Filenames to exclude (exact match)")
    parser.add_argument("--dry-run", action="store_true", help="List files, index nothing")
    parser.add_argument("--reset", action="store_true", help="Clear the store first")
    args = parser.parse_args()

    setup_logging()

    if not args.dir.is_dir():
        # data/ is not in the repository, so on a fresh clone this is the
        # normal first state rather than a typo in --dir.
        print(f"No folder at {args.dir}.")
        print("Download the evaluation papers first:  make corpus")
        return 1

    skip = set(args.skip)
    aliases = load_aliases(args.dir)
    pdfs = [p for p in sorted(args.dir.glob("*.pdf")) if p.name not in skip]

    if not pdfs:
        print(f"No PDFs to index in {args.dir}")
        return 1

    print(f"\nFound {len(pdfs)} PDF(s) in {args.dir}")
    if skip:
        print(f"Skipping: {', '.join(sorted(skip))}")
    print()
    for pdf in pdfs:
        paper_id = aliases.get(pdf.name) or clean_id(pdf.stem)
        print(f"  {pdf.name}\n      -> id: {paper_id}")
    print()

    if args.dry_run:
        print("Dry run: nothing indexed.")
        return 0

    pipeline = IngestionPipeline()
    if args.reset:
        pipeline.database.reset()
        print("Store cleared.\n")

    succeeded, failed = 0, 0
    for index, pdf in enumerate(pdfs, 1):
        paper_id = aliases.get(pdf.name) or clean_id(pdf.stem)
        print(f"[{index}/{len(pdfs)}] {paper_id} ... ", end="", flush=True)
        try:
            stats = pipeline.process_paper(pdf, paper_id=paper_id)
            print(
                f"{stats['num_chunks']} chunks, "
                f"{stats['num_sections']} sections, "
                f"{stats['num_pages']} pages, "
                f"{stats['processing_time_seconds']}s"
            )
            succeeded += 1
        except (PDFParseError, ResearchGPTError) as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            failed += 1

    stats = pipeline.database.get_stats()
    print(f"\nIndexed {succeeded} paper(s), {failed} failed.")
    print(f"Store now holds {stats['total_chunks']} chunks from {stats['total_papers']} papers.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
