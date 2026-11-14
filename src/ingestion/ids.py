"""Turning a filename into a paper id.

The id is not an internal key. It is the name a reader sees in a citation, it
is half of every chunk id in the vector store, and it is the object name of the
stored PDF. So it has to be readable, stable, and safe to put in a path.

One implementation, used by both the bulk indexing script and the upload
endpoint. They used to differ, which meant the same PDF got one id when
indexed from the command line and another when uploaded through the API.
"""

from __future__ import annotations

import re

#: Long ids are truncated here. Chunk ids append ``_chunk_<n>`` to this, and a
#: 200-character filename produces an id nothing displays well.
MAX_ID_LENGTH = 96


def clean_paper_id(stem: str) -> str:
    """Turn a filename stem into a usable paper id.

    Strips duplicate markers like ``" (1)"``, collapses runs of punctuation
    and whitespace into single underscores, and lowercases.

    Returns ``"paper"`` for input that reduces to nothing -- a filename of
    only punctuation would otherwise produce an empty id, and an empty id
    silently collides with the next one.
    """
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_-").lower()
    stem = stem[:MAX_ID_LENGTH].strip("_-")
    return stem or "paper"
