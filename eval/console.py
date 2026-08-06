"""Console helpers for the evaluation CLIs.

These tools print text taken straight from research papers and from model
output, which routinely contains mathematical symbols, Greek letters, and
typographic dashes. On Windows the default console encoding is cp1252 and
printing any of those raises ``UnicodeEncodeError`` -- which would crash a run
*after* the expensive work was already done.
"""

from __future__ import annotations

import contextlib
import sys


def setup_console() -> None:
    """Make stdout and stderr tolerate any Unicode the corpus contains.

    Switches both streams to UTF-8 and replaces characters the terminal cannot
    render, so an unprintable glyph degrades to a placeholder instead of
    killing the process.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # A stream redirected to a pipe may refuse reconfiguration; there
            # is nothing useful to do about it, so carry on unchanged.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")
