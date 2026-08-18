"""Logging setup for ResearchGPT.

Library modules under ``src/`` call :func:`get_logger` and never configure
handlers or write to stdout directly. Entry points (the Streamlit app, scripts,
the future API) call :func:`setup_logging` exactly once at startup.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

ROOT_LOGGER_NAME = "researchgpt"

# Attached so that importing the package without configuring logging does not
# emit "No handlers could be found" warnings.
logging.getLogger(ROOT_LOGGER_NAME).addHandler(logging.NullHandler())

_configured = False

#: Fields already present on every LogRecord; anything else a caller passes via
#: ``extra=`` is treated as structured context worth serialising.
_RESERVED_RECORD_FIELDS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "asctime",
    "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_FIELDS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(
    level: str | None = None,
    fmt: str | None = None,
    *,
    force: bool = False,
) -> None:
    """Configure logging for the ``researchgpt`` logger tree.

    Idempotent: repeated calls are ignored unless ``force`` is set, so Streamlit
    re-running the script does not stack duplicate handlers.

    Args:
        level: Log level name. Defaults to ``Settings.log_level``.
        fmt: ``"plain"`` or ``"json"``. Defaults to ``Settings.log_format``.
        force: Replace existing handlers instead of returning early.
    """
    global _configured

    if _configured and not force:
        return

    # Imported lazily so that a configuration error surfaces from the caller's
    # entry point rather than at module import time.
    from src.config import get_settings

    settings = get_settings()
    level = (level or settings.log_level).upper()
    fmt = fmt or settings.log_format

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logger.addHandler(handler)

    # These libraries are extremely chatty at INFO and drown out our own logs.
    for noisy in ("httpx", "urllib3", "sentence_transformers", "chromadb", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Do not double-emit through the root logger.
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for ``name``, namespaced under ``researchgpt``.

    Accepts ``__name__`` directly: ``src.retrieval.hybrid_search`` becomes
    ``researchgpt.retrieval.hybrid_search``.
    """
    normalised = name[4:] if name.startswith("src.") else name
    if normalised in ("", ROOT_LOGGER_NAME):
        return logging.getLogger(ROOT_LOGGER_NAME)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{normalised}")
