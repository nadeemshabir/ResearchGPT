"""Delete every indexed chunk from the vector store.

Needed after changing ``EMBEDDING_MODEL``: stored vectors have a fixed
dimension, and the app refuses to query a collection built by a different
model rather than returning silently wrong results.

Usage:
    python scripts/reset_db.py [--yes]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.exceptions import ResearchGPTError  # noqa: E402
from src.ingestion.database import VectorDatabase  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true", help="Skip the confirmation prompt"
    )
    args = parser.parse_args()

    setup_logging()

    try:
        database = VectorDatabase()
        stats = database.get_stats()
    except ResearchGPTError as exc:
        print(f"Could not open the vector store: {exc}")
        return 1

    print(f"Collection : {stats['collection_name']}")
    print(f"Location   : {stats['db_path']}")
    print(f"Papers     : {stats['total_papers']}")
    print(f"Chunks     : {stats['total_chunks']}")
    print(f"Built with : {stats['embedding_model']}")

    if stats["total_chunks"] == 0:
        print("\nAlready empty; nothing to do.")
        return 0

    if not args.yes:
        try:
            confirm = input("\nDelete all of this? Type 'yes' to confirm: ")
        except EOFError:
            # No terminal attached (CI, piped input). Refusing is the safe
            # default; --yes is the explicit opt-in.
            print("\nNot running interactively. Re-run with --yes to confirm.")
            return 1
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 1

    try:
        database.reset()
    except ResearchGPTError as exc:
        print(f"Reset failed: {exc}")
        return 1

    print("Vector store cleared. Re-ingest your papers to rebuild it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
